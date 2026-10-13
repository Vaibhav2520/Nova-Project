"""Baselines written by hand: causal Transformer, Transformer-CM, recurrent model, NOVA variants and a non-learning heuristic."""

import dataclasses
import math
from collections import Counter
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import NovaModel, RecurrentLayer, count_parameters
from .operator import to_fp32
from .tasks import BOS, DEMO, EOS, PAD, SEP

MODEL_NAMES = ("nova", "ablated", "transformer", "transformer_cm", "recurrent", "heuristic",
               "nova_consensus_only", "nova_refine_only", "nova_no_persistent", "nova_shuffled_r", "nova_fixed_T",
               "nova_no_deep_sup", "nova_no_aux", "ablated_more_params")


def lm_rows(batch, targets=None) -> Tuple[List[List[int]], List[List[int]], torch.Tensor, torch.Tensor]:
    """For every real question: context = BOS x1 SEP y1 DEMO ... xq SEP, and target = the answer (if given)."""
    s_in, s_out = batch.support_inputs.tolist(), batch.support_outputs.tolist()
    s_mask, q_in = batch.support_mask.tolist(), batch.query_inputs.tolist()
    qb, qq = batch.query_mask.nonzero(as_tuple=True)
    real = lambda seq: [t for t in seq if t != PAD]
    contexts, answers = [], []
    for b, q in zip(qb.tolist(), qq.tolist()):
        ctx = [BOS]
        for k, ok in enumerate(s_mask[b]):
            if ok:
                ctx += real(s_in[b][k]) + [SEP] + real(s_out[b][k]) + [DEMO]
        contexts.append(ctx + real(q_in[b][q]) + [SEP])
        answers.append(real(targets.query_outputs[b, q].tolist()) if targets is not None else [])
    return contexts, answers, qb, qq


def _pad_rows(rows: List[List[int]], device) -> Tuple[torch.Tensor, torch.Tensor]:
    L = max(len(r) for r in rows)
    t = torch.full((len(rows), L), PAD, dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        t[i, :len(r)] = torch.tensor(r, device=device)
    return t, t != PAD


class LanguageModelBaseline(nn.Module):
    """Baselines that read the whole task as one sequence and predict the next token."""

    def logits(self, tokens, mask):
        raise NotImplementedError

    def loss(self, batch, targets, task_weights: Optional[torch.Tensor] = None):
        ctx, ans, qb, _ = lm_rows(batch, targets)
        full = [c + a for c, a in zip(ctx, ans)]
        tokens, mask = _pad_rows(full, batch.support_inputs.device)
        inp, nxt = tokens[:, :-1], tokens[:, 1:]
        labels = torch.full_like(nxt, -100)
        for i, (c, a) in enumerate(zip(ctx, ans)):
            labels[i, len(c) - 1:len(c) - 1 + len(a)] = nxt[i, len(c) - 1:len(c) - 1 + len(a)]
        logit = self.logits(inp, mask[:, :-1])
        tok_loss = F.cross_entropy(logit.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
        counted = (labels != -100).float()
        per_q = (tok_loss * counted).sum(1) / counted.sum(1).clamp_min(1)
        B = batch.size
        nq = torch.zeros(B, device=per_q.device, dtype=per_q.dtype).index_put((qb,), torch.ones_like(per_q), accumulate=True)
        # deterministic sum, also on GPU
        per_task = torch.zeros(B, device=per_q.device, dtype=per_q.dtype).index_put((qb,), per_q, accumulate=True) / nq.clamp_min(1)
        weights = torch.ones(B, device=per_q.device, dtype=per_q.dtype) if task_weights is None else task_weights.to(per_q.dtype)
        total = (weights * per_task).sum() / weights.sum().clamp_min(1e-6)
        pred_ok = ((logit.argmax(-1) == labels) | (labels == -100)).all(1).float()
        return total, {"per_task_loss": per_task.detach(), "exact": pred_ok.mean().item()}

    @torch.no_grad()
    def predict(self, batch, max_steps: Optional[int] = None):
        ctx, _, qb, qq = lm_rows(batch)
        dev = batch.support_inputs.device
        max_len = batch.answer_cap(self.cfg.max_out_len)
        lengths = torch.tensor([len(c) for c in ctx], device=dev)
        tokens = torch.full((len(ctx), int(lengths.max()) + max_len), PAD, dtype=torch.long, device=dev)
        for i, c in enumerate(ctx):
            tokens[i, :len(c)] = torch.tensor(c, device=dev)
        done = torch.zeros(len(ctx), dtype=torch.bool, device=dev)
        out, lps = [], []
        rows = torch.arange(len(ctx), device=dev)
        for _ in range(max_len):
            width = int(lengths.max())
            logit = self.logits(tokens[:, :width], tokens[:, :width] != PAD)
            last = F.log_softmax(logit[rows, lengths - 1].float(), -1)
            lp, nxt = last.max(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            out.append(nxt)
            lps.append(torch.where(done, torch.zeros_like(lp), lp))
            tokens[rows, lengths] = torch.where(done, tokens[rows, lengths], nxt)
            lengths = lengths + (~done).long()
            done = done | (nxt == EOS)
            if done.all():
                break
        toks, lps = torch.stack(out, 1), torch.stack(lps, 1)
        return _lm_output(self, toks, lps, qb, qq, batch.size)


def _lm_output(model, toks, lps, qb, qq, B):
    total_lp = lps.sum(-1)
    logit = (total_lp - torch.log1p(-total_lp.exp().clamp(max=1 - 1e-6))) / model.temperature
    calls = torch.zeros(B, dtype=torch.long, device=toks.device).index_add(0, qb, torch.ones_like(qb))
    return {"q_task": qb, "q_index": qq, "tokens": toks, "confidence": torch.sigmoid(logit), "trace": {},
            "compute": {"executor_calls": calls, "refine_steps": torch.zeros_like(calls)}}


class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, pos: str, dropout: float):
        super().__init__()
        assert d % heads == 0, "d_model must be divisible by n_heads"
        self.h, self.dh, self.pos = heads, d // heads, pos
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        slopes = torch.tensor([2 ** (-8 * (i + 1) / heads) for i in range(heads)])
        self.register_buffer("slopes", slopes, persistent=False)

    def _rope(self, x, pos=None):
        L, half = x.shape[-2], self.dh // 2
        freq = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
        if pos is None:
            ang = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(1) * freq
        else:
            ang = (pos.float().unsqueeze(-1) * freq).unsqueeze(1)
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x, mask):
        N, L, d = x.shape
        q, k, v = self.qkv(x).view(N, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.pos == "rope":
            q, k = self._rope(q), self._rope(k)
        scores = to_fp32(q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        i = torch.arange(L, device=x.device)
        if self.pos == "alibi":
            dist = (i.view(1, L) - i.view(L, 1)).clamp(max=0).float()
            scores = scores + self.slopes.view(1, -1, 1, 1).to(scores.dtype) * dist.to(scores.dtype)
        # True above the diagonal = future
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, -1e9)
        scores = scores.masked_fill(~mask.view(N, 1, 1, L), -1e9)
        attn = self.drop(torch.softmax(scores, dim=-1)).to(v.dtype)
        out = (attn @ v).transpose(1, 2).reshape(N, L, d)
        return self.proj(out)

    def cached(self, x, pos, key_mask, cache=None):
        """Attend from new tokens x [N, Ln, d] at positions pos [N, Ln] to every earlier token kept in `cache`.
        Gives the same result as forward() on the whole sequence, but only computes the new tokens."""
        N, Ln, d = x.shape
        q, k, v = self.qkv(x).view(N, Ln, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.pos == "rope":  # per-row positions, so left padding is harmless
            q, k = self._rope(q, pos), self._rope(k, pos)
        if cache is not None:
            k, v, kpos = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2), torch.cat([cache[2], pos], 1)
        else:
            kpos = pos
        scores = to_fp32(q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        rel = (kpos.unsqueeze(1) - pos.unsqueeze(2)).unsqueeze(1)
        if self.pos == "alibi":
            scores = scores + self.slopes.view(1, -1, 1, 1).to(scores.dtype) * rel.clamp(max=0).to(scores.dtype)
        scores = scores.masked_fill(rel > 0, -1e9)
        scores = scores.masked_fill(~key_mask.view(N, 1, 1, -1), -1e9)
        attn = torch.softmax(scores, dim=-1).to(v.dtype)
        out = (attn @ v).transpose(1, 2).reshape(N, Ln, d)
        return self.proj(out), (k, v, kpos)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d, cfg.n_heads, cfg.pos_encoding, cfg.dropout)
        self.mlp = nn.Sequential(nn.Linear(d, cfg.mlp_mult * d), nn.GELU(), nn.Linear(cfg.mlp_mult * d, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        return x + self.drop(self.mlp(self.ln2(x)))

    def cached(self, x, pos, key_mask, cache=None):  # evaluation only, so no dropout
        a, cache = self.attn.cached(self.ln1(x), pos, key_mask, cache)
        x = x + a
        return x + self.mlp(self.ln2(x)), cache


class CausalTransformer(LanguageModelBaseline):  # decoder-only Transformer; n_loops > 1 makes it Transformer-CM
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=cfg.d_model ** -0.5)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.register_buffer("temperature", torch.ones(()))

    def logits(self, tokens, mask):
        x = self.embed(tokens)
        for _ in range(self.cfg.n_loops):  # Transformer-CM repeats the same layers (shared weights)
            for block in self.blocks:
                x = block(x, mask)
        return self.ln_f(x) @ self.embed.weight.t() + self.bias

    @torch.no_grad()
    def predict(self, batch, max_steps: Optional[int] = None):
        """Same answers as LanguageModelBaseline.predict (tested), but each new token reuses the work done for earlier ones."""
        ctx, _, qb, qq = lm_rows(batch)
        dev = batch.support_inputs.device
        cap = batch.answer_cap(self.cfg.max_out_len)
        longest = max(len(c) for c in ctx) + cap
        # rows per chunk: keeps each attention map near 800 MB
        rows = max(1, int(2e8 // (self.cfg.n_heads * longest * longest)))
        toks, lps = [], []
        for s in range(0, len(ctx), rows):
            t, l = self._decode(ctx[s:s + rows], cap, dev)
            toks.append(t)
            lps.append(l)
        width = max(t.shape[1] for t in toks)
        toks = torch.cat([F.pad(t, (0, width - t.shape[1]), value=PAD) for t in toks])
        lps = torch.cat([F.pad(l, (0, width - l.shape[1]), value=0.0) for l in lps])
        return _lm_output(self, toks, lps, qb, qq, batch.size)

    def _decode(self, ctx, cap: int, dev):
        N, L = len(ctx), max(len(c) for c in ctx)
        tokens = torch.full((N, L), PAD, dtype=torch.long, device=dev)  # left-padded contexts (all rows end together)
        for i, c in enumerate(ctx):
            tokens[i, L - len(c):] = torch.tensor(c, device=dev)
        km = tokens != PAD
        pos = (km.long().cumsum(1) - 1).clamp_min(0)  # real tokens get positions 0, 1, 2, ... exactly as in training
        # one memory per block application (loops reuse weights)
        caches = [None] * (self.cfg.n_loops * len(self.blocks))
        x = self.embed(tokens)
        for i, block in enumerate(list(self.blocks) * self.cfg.n_loops):
            x, caches[i] = block.cached(x, pos, km, None)
        logits = self.ln_f(x[:, -1]) @ self.embed.weight.t() + self.bias
        last = pos[:, -1]
        done = torch.zeros(N, dtype=torch.bool, device=dev)
        out, lps = [], []
        for _ in range(cap):
            lp, nxt = F.log_softmax(logits.float(), -1).max(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            out.append(nxt)
            lps.append(torch.where(done, torch.zeros_like(lp), lp))
            done = done | (nxt == EOS)
            if done.all():
                break
            last = last + 1
            km = torch.cat([km, torch.ones(N, 1, dtype=torch.bool, device=dev)], 1)
            x = self.embed(nxt).unsqueeze(1)
            for i, block in enumerate(list(self.blocks) * self.cfg.n_loops):
                x, caches[i] = block.cached(x, last.unsqueeze(1), km, caches[i])
            logits = self.ln_f(x[:, 0]) @ self.embed.weight.t() + self.bias
        return torch.stack(out, 1), torch.stack(lps, 1)


class RecurrentBaseline(LanguageModelBaseline):  # gated recurrent language model with the same cell as NOVA
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=cfg.d_model ** -0.5)
        self.layers = nn.ModuleList([RecurrentLayer(cfg.d_model, cfg.d_model) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.register_buffer("temperature", torch.ones(()))

    def logits(self, tokens, mask):
        x = self.embed(tokens)
        for layer in self.layers:
            x, _ = layer(x, mask)
        return self.norm(x) @ self.embed.weight.t() + self.bias

    @torch.no_grad()
    def predict(self, batch, max_steps: Optional[int] = None):
        ctx, _, qb, qq = lm_rows(batch)
        tokens, mask = _pad_rows(ctx, batch.support_inputs.device)
        x, states = self.embed(tokens), []
        for layer in self.layers:
            x, h = layer(x, mask)
            states.append(h)
        rows = torch.arange(len(ctx), device=tokens.device)
        last = self.norm(x[rows, mask.sum(1) - 1]) @ self.embed.weight.t() + self.bias
        done = torch.zeros(len(ctx), dtype=torch.bool, device=tokens.device)
        out, lps = [], []
        for _ in range(batch.answer_cap(self.cfg.max_out_len)):
            lp, nxt = F.log_softmax(last.float(), -1).max(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            out.append(nxt)
            lps.append(torch.where(done, torch.zeros_like(lp), lp))
            done = done | (nxt == EOS)
            if done.all():
                break
            inp = self.embed(nxt)
            for i, layer in enumerate(self.layers):
                states[i] = layer.cell.step(layer.cell.project(inp, None), states[i])
                inp = states[i]
            last = self.norm(inp) @ self.embed.weight.t() + self.bias
        return _lm_output(self, torch.stack(out, 1), torch.stack(lps, 1), qb, qq, batch.size)


class HeuristicBaseline(nn.Module):
    """Copy the query, most frequent demo output, or a token lookup table; pick by leave-one-out accuracy on the demos."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("temperature", torch.ones(()))

    @staticmethod
    def _rules(pairs):
        outs = Counter(tuple(y) for _, y in pairs)
        frequent = list(outs.most_common(1)[0][0]) if outs else [EOS]
        table = {}
        for x, y in pairs:  # demos where output length = input length + EOS
            if len(y) == len(x) + 1:
                for a, b in zip(x, y):
                    table.setdefault(a, Counter())[b] += 1
        lookup = {a: c.most_common(1)[0][0] for a, c in table.items()}
        return {"copy": lambda x: list(x) + [EOS], "frequent": lambda x: list(frequent),
                "lookup": lambda x: [lookup.get(t, t) for t in x] + [EOS]}

    def loss(self, batch, targets, task_weights=None):
        return torch.zeros((), requires_grad=True), {"per_task_loss": torch.zeros(batch.size), "exact": 0.0}

    @torch.no_grad()
    def predict(self, batch, max_steps=None):
        real = lambda seq: [t for t in seq if t != PAD]
        s_in, s_out, s_mask = batch.support_inputs.tolist(), batch.support_outputs.tolist(), batch.support_mask.tolist()
        qb, qq = batch.query_mask.nonzero(as_tuple=True)
        answers, confs = [], []
        for b, q in zip(qb.tolist(), qq.tolist()):
            pairs = [(real(s_in[b][k]), real(s_out[b][k])) for k, ok in enumerate(s_mask[b]) if ok]
            score = Counter()
            for i in range(len(pairs)):
                rules = self._rules(pairs[:i] + pairs[i + 1:])
                for name, fn in rules.items():
                    score[name] += fn(pairs[i][0]) == pairs[i][1]
            best = max(("copy", "frequent", "lookup"), key=lambda n: score[n])  # best rule (ties: first in this order)
            answers.append(self._rules(pairs)[best](real(batch.query_inputs[b, q].tolist())))
            confs.append(score[best] / max(1, len(pairs)))  # confidence = leave-one-out accuracy
        toks, _ = _pad_rows(answers or [[EOS]], batch.support_inputs.device)
        calls = torch.zeros(batch.size, dtype=torch.long)
        return {"q_task": qb, "q_index": qq, "tokens": toks[:len(answers)],
                "confidence": torch.tensor(confs, dtype=torch.float32), "trace": {},
                "compute": {"executor_calls": calls, "refine_steps": calls}}


_NOVA_VARIANTS = {"nova": {}, "ablated": {"mode": "mean"}, "nova_consensus_only": {"mode": "consensus_only"},
                  "nova_refine_only": {"mode": "refine_only"}, "nova_no_persistent": {"mode": "no_persistent"},
                  "nova_shuffled_r": {"mode": "shuffled_r"}, "nova_fixed_T": {"mode": "fixed_T"},
                  "nova_no_deep_sup": {"deep_supervision": False}, "nova_no_aux": {"aux_loss": False},
                  "ablated_more_params": {"mode": "mean"}}  # widened ablation (width set by match_parameters)


def build_model(name: str, cfg: ModelConfig) -> nn.Module:
    """Create a model by name. Width and loops come from cfg (use match_parameters / match_compute to set them)."""
    if name in _NOVA_VARIANTS:
        return NovaModel(cfg, **_NOVA_VARIANTS[name])
    if name in ("transformer", "transformer_cm"):
        return CausalTransformer(cfg)
    if name == "recurrent":
        return RecurrentBaseline(cfg)
    if name == "heuristic":
        return HeuristicBaseline(cfg)
    raise ValueError(f"unknown model {name}; choose from {MODEL_NAMES}")


def used_parameters(model: nn.Module) -> int:
    unused = set()
    if isinstance(model, NovaModel) and model.mode in ("mean", "consensus_only"):  # these modes skip refinement
        unused = {n for n, _ in model.named_parameters() if n.startswith(("operator.upd", "operator.err_proj", "operator.alpha"))}
    return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n not in unused)


def match_parameters(name: str, cfg: ModelConfig, target: int, tolerance: float = 0.05) -> ModelConfig:
    """Search the width so the model is within +/-5% of `target` parameters (Transformer widths split into even heads)."""
    if name == "heuristic":
        return cfg
    # the widened ablation matches *used* parameters
    count = used_parameters if name == "ablated_more_params" else count_parameters
    best = None
    step = 2 * cfg.n_heads if name in ("transformer", "transformer_cm") else 2
    for d in range(step * 2, 1025, step):
        trial = dataclasses.replace(cfg, d_model=d)
        n = count(build_model(name, trial))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (trial, n)
        if n > target * (1 + tolerance):
            break
    if abs(best[1] - target) > tolerance * target:
        raise ValueError(f"{name}: cannot match {target} parameters within {tolerance:.0%} (best {best[1]})")
    return best[0]


def measure_flops(model: nn.Module, batch, targets) -> int:
    """Forward FLOPs of model.loss on one batch, measured with PyTorch's own FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as counter:
        model.loss(batch, targets)
    return int(counter.get_total_flops())


MATCH_LOG: dict = {}  # last measured matching numbers (read by train.train_run and written to the ledger)


def match_compute(cfg: ModelConfig, nova_flops: int, batch, targets, tolerance: float = 0.10) -> ModelConfig:
    """Choose n_loops so the looped Transformer uses about the same training FLOPs per batch as NOVA. Reports the ratio."""
    torch.manual_seed(0)
    one = measure_flops(CausalTransformer(dataclasses.replace(cfg, n_loops=1)), batch, targets)
    two = measure_flops(CausalTransformer(dataclasses.replace(cfg, n_loops=2)), batch, targets)
    per_loop = max(1, two - one)  # cost of one extra loop (embedding and output layer are paid only once)
    fixed = one - per_loop
    loops = max(1, round((nova_flops - fixed) / per_loop))
    ratio = (fixed + loops * per_loop) / max(1, nova_flops)
    if abs(ratio - 1) > tolerance:  # whole loops could not get within 10% (e.g. one loop already costs more than NOVA)
        print(f"warning: Transformer-CM compute ratio {ratio:.2f} is outside +/-{tolerance:.0%}; report it")
    MATCH_LOG["transformer_cm_compute_ratio"] = round(ratio, 3)
    return dataclasses.replace(cfg, n_loops=loops)
