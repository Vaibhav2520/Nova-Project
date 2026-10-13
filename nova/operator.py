"""Cross-Execution Consensus (CEC): the novel operator. Built only from plain PyTorch tensor operations."""

import dataclasses
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("full", "mean", "consensus_only", "refine_only", "fixed_T", "no_persistent", "shuffled_r")
CEC_WEIGHTS = {"full", "consensus_only", "fixed_T", "no_persistent", "shuffled_r"}
REFINES = {"full", "refine_only", "fixed_T", "no_persistent", "shuffled_r"}
DEBUG_FINITE = False  # tests switch this on to make the operator fail fast on NaN or Inf


class NumericalError(FloatingPointError):
    """A NaN or Inf appeared inside the operator."""


def to_fp32(x: torch.Tensor) -> torch.Tensor:
    """float16/bfloat16 -> float32. float32 and float64 are returned unchanged."""
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def check_finite(name: str, x: torch.Tensor) -> None:
    if DEBUG_FINITE and not torch.isfinite(x).all():
        raise NumericalError(f"non-finite values in {name}")


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax in float32 over entries where mask is True. All-masked rows give zeros, never NaN."""
    scores = to_fp32(scores).masked_fill(~mask, -1e9)  # big finite negative number instead of -inf (avoids NaN)
    probs = torch.softmax(scores, dim=dim) * mask
    return probs / probs.sum(dim, keepdim=True).clamp_min(1e-12)  # renormalise; an all-masked row stays all zeros


def sequence_nll(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood per row, averaged over real (unmasked) tokens. Computed in float32."""
    logp = F.log_softmax(to_fp32(logits), dim=-1)
    tok = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    m = mask.to(tok.dtype)
    return -(tok * m).sum(-1) / m.sum(-1).clamp_min(1.0)


def residual_feature(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """Embedding-space gap between what should be produced and what is expected to be produced (float32)."""
    probs = torch.softmax(to_fp32(logits), dim=-1)
    table = emb.to(probs.dtype)
    expected = probs @ table
    wanted = table[targets]
    m = mask.to(probs.dtype).unsqueeze(-1)
    return ((wanted - expected) * m).sum(1) / m.sum(1).clamp_min(1.0)


@dataclasses.dataclass
class OperatorOutput:
    z_star: torch.Tensor
    z_traj: List[torch.Tensor]
    w: torch.Tensor
    v: torch.Tensor
    R: torch.Tensor  # [B, K, K]; 0 where not computed
    pair_valid: torch.Tensor
    rho_hist: torch.Tensor  # [B, T+1]; -1 = not measured
    rho_star: torch.Tensor
    rho_loo: torch.Tensor
    steps_used: torch.Tensor
    halt_step: torch.Tensor
    update_norms: torch.Tensor
    executor_calls: torch.Tensor
    self_nll: torch.Tensor
    no_cross_check: torch.Tensor


class CrossExecutionConsensus(nn.Module):
    """Propose (encoder outputs h), cross-execute, weight by verification, refine with execution residuals, halt on fit."""

    def __init__(self, d_model: int, tau_w: float = 1.0, tau_v: float = 1.0, eps: float = 0.05,
                 delta: float = 0.005, clip_c: float = 1.0):
        super().__init__()
        self.err_proj = nn.Linear(d_model, d_model)
        self.upd = nn.Sequential(nn.Linear(3 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.alpha_logit = nn.Parameter(torch.zeros(()))
        self.tau_w, self.tau_v, self.eps, self.delta, self.clip_c = tau_w, tau_v, eps, delta, clip_c

    def _execute(self, executor, emb, batch, b_idx, j_idx, z_rows, need_err: bool):
        x = batch.support_inputs[b_idx, j_idx]
        xm = batch.support_input_mask[b_idx, j_idx]
        y = batch.support_outputs[b_idx, j_idx]
        ym = batch.support_output_mask[b_idx, j_idx]
        logits = executor.score(x, xm, z_rows, y, ym)
        nll = sequence_nll(logits, y, ym)
        err = self.err_proj(residual_feature(logits, y, ym, emb)) if need_err else None
        return nll, err

    @staticmethod
    def _clip(delta: torch.Tensor, c: float) -> torch.Tensor:
        norm = to_fp32(delta).norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return delta * (c / norm).clamp(max=1.0).to(delta.dtype)

    def forward(self, h: torch.Tensor, batch, executor, emb: torch.Tensor, t_max: int,
                halting: bool, mode: str = "full") -> OperatorOutput:
        """h: [B, K, d] rule guesses (one per demo). batch: TaskBatch. executor: shared Exec. emb: token embedding [V, d]."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        B, K, d = h.shape
        dev = h.device
        acc = torch.float64 if h.dtype == torch.float64 else torch.float32
        valid = batch.support_mask
        kv = valid.sum(1)
        calls = torch.zeros(B, dtype=torch.long, device=dev)
        eye = torch.eye(K, dtype=torch.bool, device=dev)
        both = valid.unsqueeze(2) & valid.unsqueeze(1)
        use_cec = mode in CEC_WEIGHTS
        pairs = both if use_cec else both & eye.unsqueeze(0)
        b_idx, k_idx, j_idx = pairs.nonzero(as_tuple=True)  # compaction: only real pairs are executed
        R = torch.zeros(B, K, K, device=dev, dtype=acc)
        if b_idx.numel():
            nll, _ = self._execute(executor, emb, batch, b_idx, j_idx, h[b_idx, k_idx], need_err=False)
            R = R.index_put((b_idx, k_idx, j_idx), nll.to(acc))
            calls = calls.index_add(0, b_idx, torch.ones_like(b_idx))
        check_finite("R", R)
        diag = R.diagonal(dim1=1, dim2=2)
        self_nll = (diag * valid).sum(1) / kv.clamp_min(1)
        off = both & ~eye.unsqueeze(0)
        if mode == "shuffled_r":  # control: same compute, scrambled evidence
            R_src, off_src = R.roll(1, 0), off.roll(1, 0)
            fallback = (R * off).sum((1, 2), keepdim=True) / off.sum((1, 2), keepdim=True).clamp_min(1)
            R_w = torch.where(off_src, R_src, fallback)
        else:
            R_w = R
        if use_cec:
            n_off = off.sum(2)
            a = -(R_w * off).sum(2) / n_off.clamp_min(1)  # trust score: minus mean error of guess k on other demos
            col = R_w.masked_fill(~off, float("inf"))
            b_score = -col.min(dim=1).values  # reliability score: minus best error any other guess achieves on demo j
            b_score = torch.where(off.any(1), b_score, torch.zeros_like(b_score))
            w = masked_softmax(a / self.tau_w, valid, dim=1)
            v = masked_softmax(b_score / self.tau_v, valid, dim=1)
        else:
            w = valid.to(acc) / kv.clamp_min(1).unsqueeze(1).to(acc)
            v = w.clone()
        check_finite("w", w)
        z0 = (w.unsqueeze(-1).to(h.dtype) * h).sum(1)  # z_0: trust-weighted blend of the guesses, [B, d]
        others = off.to(acc) * w.unsqueeze(2)
        loo = (others * R).sum(1) / others.sum(1).clamp_min(1e-12)
        rho_loo = (v * loo).sum(1)
        T = t_max if mode in REFINES else 0
        rho_hist = torch.full((B, T + 1), -1.0, device=dev, dtype=acc)
        norms = torch.zeros(B, max(T, 1), device=dev, dtype=acc)
        active = kv > 0
        halt_step = torch.zeros(B, dtype=torch.long, device=dev)
        steps = torch.zeros(B, dtype=torch.long, device=dev)
        z, traj = z0, [z0]
        best_z, best_rho = z0, torch.full((B,), float("inf"), device=dev, dtype=acc)
        prev_rho = torch.full((B,), float("inf"), device=dev, dtype=acc)
        rho_now = torch.zeros(B, device=dev, dtype=acc)
        alpha = torch.sigmoid(self.alpha_logit)
        for t in range(T + 1):
            rows = valid & active.unsqueeze(1)
            rb, rj = rows.nonzero(as_tuple=True)
            if rb.numel() == 0:
                break
            nll, err = self._execute(executor, emb, batch, rb, rj, z[rb], need_err=t < T)
            calls = calls.index_add(0, rb, torch.ones_like(rb))
            vw = v[rb, rj]
            # deterministic sum, also on GPU
            rho = torch.zeros(B, device=dev, dtype=acc).index_put((rb,), vw * nll.to(acc), accumulate=True)
            rho_now = torch.where(active, rho, rho_now)
            rho_hist[:, t] = torch.where(active, rho, rho_hist[:, t])
            better = active & (rho < best_rho)
            best_rho = torch.where(better, rho, best_rho)
            best_z = torch.where(better.unsqueeze(1), z, best_z)
            if halting and mode != "fixed_T":
                # fits, stalls, or out of steps
                stop = active & ((rho < self.eps) | (prev_rho - rho < self.delta) | (t == T))
            else:
                stop = active & torch.full_like(active, t == T)
            halt_step = torch.where(stop, torch.full_like(halt_step, t), halt_step)
            active = active & ~stop  # halting is final: stopped tasks never restart
            prev_rho = torch.where(active, rho, prev_rho)
            if t == T or not active.any():
                break
            e = torch.zeros(B, d, device=dev, dtype=err.dtype).index_put((rb,), vw.unsqueeze(1).to(err.dtype) * err, accumulate=True)
            base = z0 if mode == "no_persistent" else z  # no_persistent forgets earlier corrections
            delta = self._clip(self.upd(torch.cat([base, e.to(base.dtype), base * e.to(base.dtype)], -1)), self.clip_c)
            norms[:, t] = torch.where(active, to_fp32(delta).norm(dim=-1).to(acc), norms[:, t])
            z = torch.where(active.unsqueeze(1), base + alpha.to(base.dtype) * delta, z)
            steps = steps + active.long()
            traj.append(z)
            check_finite("z", z)
        z_star = best_z if halting else z  # evaluation uses the best-so-far rule; training uses the last rule
        rho_star = torch.where(torch.isfinite(best_rho), best_rho, rho_now) if halting else rho_now
        return OperatorOutput(z_star, traj, w, v, R, off, rho_hist, rho_star, rho_loo, steps, halt_step, norms, calls,
                              self_nll, kv < 2)
