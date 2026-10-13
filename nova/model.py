"""The full NOVA model: shared recurrent blocks, encoder, executor, confidence head and the training loss."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .operator import CrossExecutionConsensus, masked_softmax, sequence_nll
from .tasks import BOS, EOS, PAD, SEP


class GatedCell(nn.Module):
    """h_new = (1 - u) * n + u * h, with update gate u, reset gate r and candidate n. Optional FiLM conditioning."""

    def __init__(self, d_in: int, d: int, cond_dim: int = 0):
        super().__init__()
        self.wx = nn.Linear(d_in, 3 * d)
        self.wh = nn.Linear(d, 3 * d, bias=False)
        self.film = nn.Linear(cond_dim, 6 * d) if cond_dim else None  # FiLM: scale and shift from the rule z
        nn.init.zeros_(self.film.weight) if self.film is not None else None  # start FiLM as "no change"
        nn.init.zeros_(self.film.bias) if self.film is not None else None

    def film_params(self, cond: Optional[torch.Tensor]):
        if self.film is None or cond is None:
            return None
        return self.film(cond).chunk(2, dim=-1)

    def apply_film(self, xp: torch.Tensor, film) -> torch.Tensor:
        if film is None:
            return xp
        gamma, beta = film
        if xp.dim() == 3:
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
        return xp * (1 + gamma) + beta

    def project(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None, film=None) -> torch.Tensor:
        return self.apply_film(self.wx(x), film if film is not None else self.film_params(cond))

    def step(self, xp: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xr, xu, xn = xp.chunk(3, dim=-1)
        hr, hu, hn = self.wh(h).chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        u = torch.sigmoid(xu + hu)
        n = torch.tanh(xn + r * hn)
        return (1 - u) * n + u * h


class RecurrentLayer(nn.Module):
    def __init__(self, d_in: int, d: int, cond_dim: int = 0, reverse: bool = False):
        super().__init__()
        self.cell, self.reverse, self.d = GatedCell(d_in, d, cond_dim), reverse, d

    def forward(self, x, mask, cond=None, h0=None):
        N, L, _ = x.shape
        xp = self.cell.project(x, cond)
        h = x.new_zeros(N, self.d) if h0 is None else h0
        outs = [None] * L
        order = range(L - 1, -1, -1) if self.reverse else range(L)
        for t in order:
            h_new = self.cell.step(xp[:, t], h)
            h = torch.where(mask[:, t].unsqueeze(1), h_new, h)  # padding steps leave the state unchanged
            outs[t] = h
        return torch.stack(outs, dim=1), h


class BiRecurrent(nn.Module):
    def __init__(self, d: int, n_layers: int, cond_dim: int = 0):
        super().__init__()
        self.fwd = nn.ModuleList([RecurrentLayer(d, d, cond_dim) for _ in range(n_layers)])
        self.bwd = nn.ModuleList([RecurrentLayer(d, d, cond_dim, reverse=True) for _ in range(n_layers)])
        self.mix = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(n_layers)])
        # LayerNorm over features only (never the batch)
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])

    def forward(self, x, mask, cond=None):
        for f, b, mix, norm in zip(self.fwd, self.bwd, self.mix, self.norm):
            hf, _ = f(x, mask, cond)
            hb, _ = b(x, mask, cond)
            x = norm(x + mix(torch.cat([hf, hb], dim=-1)))
        return x * mask.unsqueeze(-1)


def masked_mean_max(x, mask):
    m = mask.unsqueeze(-1).to(x.dtype)
    mean = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
    mx = x.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
    mx = torch.where(mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig, embed: nn.Embedding):
        super().__init__()
        self.embed = embed
        self.rnn = BiRecurrent(cfg.d_model, cfg.n_layers)
        self.out = nn.Linear(2 * cfg.d_model, cfg.d_model)

    def forward(self, x, xm, y, ym):
        sep = torch.full_like(x[:, :1], SEP)
        tokens = torch.cat([x, sep, y], dim=1)
        mask = torch.cat([xm, torch.ones_like(xm[:, :1]), ym], dim=1)
        states = self.rnn(self.embed(tokens), mask)
        return self.out(masked_mean_max(states, mask))


class Executor(nn.Module):
    def __init__(self, cfg: ModelConfig, embed: nn.Embedding):
        super().__init__()
        d = cfg.d_model
        self.embed, self.read_option = embed, cfg.read_option
        self.reader = BiRecurrent(d, cfg.n_layers, cond_dim=d)
        self.init = nn.Linear(2 * d + d, d)
        self.writer = GatedCell(2 * d, d, cond_dim=d)
        self.query = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def _read(self, x, xm, z):
        states = self.reader(self.embed(x), xm, cond=z)
        h0 = torch.tanh(self.init(torch.cat([masked_mean_max(states, xm), z], dim=-1)))
        return states, h0

    def _context(self, h, states, xm):  # option A: content-based look-up (hand-written dot-product read)
        if self.read_option != "A":
            return torch.zeros_like(h)
        scores = torch.einsum("nd,nld->nl", self.query(h), states) / math.sqrt(h.shape[-1])
        attn = masked_softmax(scores, xm, dim=-1).to(states.dtype)
        return torch.einsum("nl,nld->nd", attn, states)

    def _logits(self, h, ctx):
        o = self.norm(torch.tanh(self.out(torch.cat([h, ctx], dim=-1))))
        return o @ self.embed.weight.t() + self.bias

    def _writer_parts(self, z):
        d = self.embed.weight.shape[1]
        w = self.writer.wx.weight
        film = self.writer.film_params(z)
        # token half of the input (can run for all steps at once)
        tok_part = lambda e: F.linear(e, w[:, :d], self.writer.wx.bias)
        ctx_part = lambda c: F.linear(c, w[:, d:])  # context half of the input (changes every step)
        return film, tok_part, ctx_part

    def score(self, x, xm, z, y, ym):
        """Return logits [N, L_out, V] for predicting y one token at a time, given the correct previous tokens."""
        states, h = self._read(x, xm, z)
        prev = torch.cat([torch.full_like(y[:, :1], BOS), y[:, :-1]], dim=1)
        film, tok_part, ctx_part = self._writer_parts(z)
        tok_in = tok_part(self.embed(prev))
        ctx = self._context(h, states, xm)
        logits = []
        for t in range(y.shape[1]):
            xp = self.writer.apply_film(tok_in[:, t] + ctx_part(ctx), film)
            h = torch.where(ym[:, t].unsqueeze(1), self.writer.step(xp, h), h)
            ctx = self._context(h, states, xm)  # one look-up per step, reused as the next step's input
            logits.append(self._logits(h, ctx))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def decode(self, x, xm, z, max_len: int):
        """Write the answer greedily. Returns tokens [N, T] (PAD after EOS) and per-token log-probabilities [N, T]."""
        states, h = self._read(x, xm, z)
        film, tok_part, ctx_part = self._writer_parts(z)
        tok = torch.full((x.shape[0],), BOS, dtype=torch.long, device=x.device)
        done = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        out, lps = [], []
        ctx = self._context(h, states, xm)
        for _ in range(max_len):
            h = self.writer.step(self.writer.apply_film(tok_part(self.embed(tok)) + ctx_part(ctx), film), h)
            ctx = self._context(h, states, xm)
            logp = F.log_softmax(self._logits(h, ctx).float(), -1)
            lp, nxt = logp.max(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            out.append(nxt)
            lps.append(torch.where(done, torch.zeros_like(lp), lp))
            done = done | (nxt == EOS)
            tok = nxt
            if done.all():
                break
        return torch.stack(out, 1), torch.stack(lps, 1)


class ConfidenceHead(nn.Module):  # turns fit statistics into "probability the answer is exactly right"
    N_FEATURES = 8

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(self.N_FEATURES, 32), nn.GELU(), nn.Linear(32, 1))
        self.register_buffer("temperature", torch.ones(()))  # fitted after training on IID validation (1 = unchanged)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # match the head's precision (float32 normally, float64 in gradcheck tests)
        feats = feats.to(self.mlp[0].weight.dtype)
        return self.mlp(feats).squeeze(-1) / self.temperature


def entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


class NovaModel(nn.Module):
    """Encoder + CEC operator + shared executor + confidence head. `mode` switches parts off (see operator.MODES)."""

    def __init__(self, cfg: ModelConfig, mode: Optional[str] = None, deep_supervision: bool = True, aux_loss: bool = True):
        super().__init__()
        self.cfg, self.mode = cfg, mode or cfg.mode
        self.deep_supervision, self.aux_loss = deep_supervision, aux_loss
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=cfg.d_model ** -0.5)
        self.encoder = Encoder(cfg, self.embed)
        self.executor = Executor(cfg, self.embed)  # Exec (shared by every stage and by Ablated NOVA)
        self.operator = CrossExecutionConsensus(cfg.d_model, cfg.tau_w, cfg.tau_v, cfg.eps_halt, cfg.delta_halt, cfg.clip_c)
        self.conf_head = ConfidenceHead()
        self.dropout = nn.Dropout(cfg.dropout)

    def _guesses(self, batch):
        B, K, _ = batch.support_inputs.shape
        h = self.embed.weight.new_zeros(B, K, self.cfg.d_model)
        b, k = batch.support_mask.nonzero(as_tuple=True)
        if b.numel():
            g = self.encoder(batch.support_inputs[b, k], batch.support_input_mask[b, k],
                             batch.support_outputs[b, k], batch.support_output_mask[b, k])
            h = h.index_put((b, k), self.dropout(g).to(h.dtype))
        return h

    def _features(self, op, mean_lp, q_task):
        T = max(1, op.rho_hist.shape[1] - 1)
        kv = op.w.gt(0).sum(1).float()
        f = torch.stack([op.rho_star, op.rho_loo, entropy(op.w), entropy(op.v), op.steps_used.float() / T,
                         kv / self.cfg.k_max, op.no_cross_check.float()], dim=-1)[q_task]
        # plus the answer's certainty; detached on purpose
        return torch.cat([f.float(), mean_lp.float().unsqueeze(-1)], dim=-1).detach()

    def loss(self, batch, targets, task_weights: Optional[torch.Tensor] = None):
        """Deep supervision over z_0..z_T + lambda_self * own-demo loss + lambda_cal * confidence loss. Returns (loss, info)."""
        h = self._guesses(batch)
        op = self.operator(h, batch, self.executor, self.embed.weight, self.cfg.t_max_train, halting=False, mode=self.mode)
        qb, qq = batch.query_mask.nonzero(as_tuple=True)
        zs = op.z_traj if self.deep_supervision else op.z_traj[-1:]
        n, S = qb.numel(), len(zs)
        x = batch.query_inputs[qb, qq].repeat(S, 1)
        xm = batch.query_input_mask[qb, qq].repeat(S, 1)
        y = targets.query_outputs[qb, qq].repeat(S, 1)
        ym = targets.query_output_mask[qb, qq].repeat(S, 1)
        z = torch.cat([zt[qb] for zt in zs], dim=0)
        logits = self.executor.score(x, xm, z, y, ym)
        ce = sequence_nll(logits, y, ym).view(S, n)
        gamma = torch.arange(1, S + 1, device=ce.device, dtype=torch.float32)  # rising weights 1..S
        gamma = gamma / gamma.sum()
        per_q = (gamma.unsqueeze(1) * ce).sum(0)
        B = batch.size
        nq = torch.zeros(B, device=ce.device, dtype=per_q.dtype).index_put((qb,), torch.ones_like(per_q), accumulate=True)
        # deterministic sum, also on GPU
        per_task = torch.zeros(B, device=ce.device, dtype=per_q.dtype).index_put((qb,), per_q, accumulate=True) / nq.clamp_min(1)
        if self.aux_loss:  # auxiliary loss: each guess should explain its own demonstration
            per_task = per_task + self.cfg.lambda_self * op.self_nll
        if callable(task_weights):  # residual-gated plasticity: the gate is computed from this pass's fit error
            task_weights = task_weights(op.rho_star.detach())  # detached, so the gate itself is not trained
        weights = torch.ones(B, device=ce.device, dtype=per_task.dtype) if task_weights is None else task_weights.to(per_task.dtype)
        main = (weights * per_task).sum() / weights.sum().clamp_min(1e-6)
        last = logits.view(S, n, *logits.shape[1:])[-1]
        # teacher-forced exact match = greedy exact match
        exact = ((last.argmax(-1) == y[:n]) | ~ym[:n]).all(-1).float()
        tok_lp = F.log_softmax(last.float(), -1).max(-1).values
        mean_lp = (tok_lp * ym[:n]).sum(-1) / ym[:n].sum(-1).clamp_min(1)
        conf_logit = self.conf_head(self._features(op, mean_lp, qb))
        cal = F.binary_cross_entropy_with_logits(conf_logit.float(), exact.float())
        total = main + self.cfg.lambda_cal * cal
        info = {"per_task_loss": per_task.detach(), "rho_star": op.rho_star.detach(), "exact": exact.mean().item(),
                "steps": op.steps_used.float().mean().item(), "executor_calls": op.executor_calls.detach()}
        return total, info

    @torch.no_grad()
    def predict(self, batch, max_steps: Optional[int] = None):
        """Greedy answers, confidence, trace and compute counters for every question."""
        h = self._guesses(batch)
        t_max = self.cfg.t_max_eval if max_steps is None else max_steps
        op = self.operator(h, batch, self.executor, self.embed.weight, t_max, halting=True, mode=self.mode)
        qb, qq = batch.query_mask.nonzero(as_tuple=True)
        toks, lps = self.executor.decode(batch.query_inputs[qb, qq], batch.query_input_mask[qb, qq], op.z_star[qb],
                                         batch.answer_cap(self.cfg.max_out_len))
        written = (toks != 0).float()
        mean_lp = (lps * written).sum(-1) / written.sum(-1).clamp_min(1)
        conf = torch.sigmoid(self.conf_head(self._features(op, mean_lp, qb)))
        # executor calls include the rows that write the answers
        calls = op.executor_calls + torch.zeros_like(op.executor_calls).index_add(0, qb, torch.ones_like(qb))
        return {"q_task": qb, "q_index": qq, "tokens": toks, "confidence": conf,
                "trace": {"R": op.R, "w": op.w, "v": op.v, "rho": op.rho_hist, "update_norms": op.update_norms,
                          "steps_used": op.steps_used, "halt_step": op.halt_step, "rho_loo": op.rho_loo},
                "compute": {"executor_calls": calls, "refine_steps": op.steps_used}}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
