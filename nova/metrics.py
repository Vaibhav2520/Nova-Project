"""Scores: exact match, GEI, calibration, risk-coverage, paired bootstrap, Spearman and continual-learning measures."""

import math
from typing import Dict, List, Optional, Sequence

import torch

from .tasks import EOS, PAD


def trim(tokens: Sequence[int]) -> List[int]:
    out = []
    for t in tokens:
        if t == PAD:
            continue
        out.append(t)
        if t == EOS:
            break
    return out


def exact_match(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    """A question is correct only if every token up to and including EOS matches."""
    return torch.tensor([trim(p) == trim(t) for p, t in zip(pred_tokens.tolist(), target_tokens.tolist())])


def task_accuracy(correct: torch.Tensor, q_task: torch.Tensor, n_tasks: int) -> torch.Tensor:
    """A task counts as solved only if all of its questions are correct."""
    wrong = torch.zeros(n_tasks, dtype=torch.long).index_add(0, q_task.cpu(), (~correct).long())
    return wrong == 0


# GEI: our reading of the brief; replaceable
def normalised_skill(acc_model: float, acc_heuristic: float, acc_transformer: float, eta: float = 1e-3) -> float:
    """(model - heuristic) / max(transformer - heuristic, eta). eta guards the case where the Transformer is no better."""
    return (acc_model - acc_heuristic) / max(acc_transformer - acc_heuristic, eta)


def domain_score(budgets: Sequence[float], skills: Sequence[float]) -> float:
    """Trapezoidal area of skill over log(budget), divided by the log-range so a constant skill s scores s."""
    logs = [math.log(b) for b in budgets]
    area = sum((logs[i + 1] - logs[i]) * (skills[i] + skills[i + 1]) / 2 for i in range(len(logs) - 1))
    return area / max(1e-12, logs[-1] - logs[0])


def gei(domain_scores: Dict[str, float], clip: float = 0.01, penalties: Sequence[float] = ()) -> float:
    """Geometric mean of clipped positive domain scores, multiplied by declared penalties. Always report raw numbers too."""
    vals = [max(s, clip) for s in domain_scores.values()]
    geo = math.exp(sum(math.log(v) for v in vals) / max(1, len(vals)))
    for p in penalties:
        geo *= p
    return geo


def budget_penalties(params: int, param_cap: int, peak_mem: int, mem_cap: int, updates: int, update_cap: int) -> List[float]:
    """Declared penalties: 0.9 per started 5% over the parameter cap, 0.5 over the memory cap, 0 over the update budget."""
    over = max(0.0, params / param_cap - 1.0)
    p_params = 0.9 ** math.ceil(over / 0.05) if over > 0 else 1.0
    p_mem = 0.5 if peak_mem > mem_cap else 1.0
    p_upd = 0.0 if updates > update_cap else 1.0  # an over-budget run is invalid
    return [p_params, p_mem, p_upd]


def steps_to_threshold(budgets: Sequence[float], accs: Sequence[float], threshold: float) -> Optional[float]:
    for b, a in zip(budgets, accs):
        if a >= threshold:
            return b
    return None


def ece(conf: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    """Average gap between confidence and accuracy, over 15 equal-width confidence bins."""
    conf, correct = conf.float().cpu(), correct.float().cpu()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        upper = conf <= edges[i + 1] if i == bins - 1 else conf < edges[i + 1]  # last bin includes 1.0
        inside = (conf >= edges[i]) & upper
        if inside.any():
            total += inside.float().mean().item() * abs(conf[inside].mean().item() - correct[inside].mean().item())
    return total


def brier(conf: torch.Tensor, correct: torch.Tensor) -> float:
    return ((conf.float().cpu() - correct.float().cpu()) ** 2).mean().item()


def risk_coverage(conf: torch.Tensor, correct: torch.Tensor) -> Dict[str, list]:
    """Coverage and error rate when answering only the top-k most confident questions, for every k. Plus AURC."""
    order = torch.argsort(conf.float().cpu(), descending=True)
    errs = (~correct.cpu().bool())[order].float()
    n = torch.arange(1, len(errs) + 1, dtype=torch.float32)
    risk = (torch.cumsum(errs, 0) / n).tolist()
    coverage = (n / max(1, len(errs))).tolist()
    return {"coverage": coverage, "risk": risk, "aurc": sum(risk) / max(1, len(risk))}


def selective_accuracy(conf: torch.Tensor, correct: torch.Tensor, coverage: float) -> float:
    k = max(1, int(round(coverage * len(conf))))
    top = torch.argsort(conf.float().cpu(), descending=True)[:k]
    return correct.cpu().float()[top].mean().item()


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Pick the temperature (log-spaced grid 0.05..20) that minimises binary cross-entropy on IID validation."""
    logits, labels = logits.float().cpu(), labels.float().cpu()
    grid = torch.logspace(math.log10(0.05), math.log10(20.0), 200)
    losses = [torch.nn.functional.binary_cross_entropy_with_logits(logits / t, labels).item() for t in grid]
    return float(grid[min(range(len(losses)), key=losses.__getitem__)])


def paired_bootstrap(a: torch.Tensor, b: torch.Tensor, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> Dict[str, float]:
    """Paired bootstrap of mean(a - b) over tasks, implemented with torch.randint. Returns mean, low and high."""
    diff = (a.float() - b.float()).cpu()
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(diff), (n_boot, len(diff)), generator=gen)
    means = diff[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([(1 - level) / 2, 1 - (1 - level) / 2])).tolist()
    return {"mean": diff.mean().item(), "low": lo, "high": hi, "excludes_zero": lo > 0 or hi < 0}


def _ranks(x: torch.Tensor) -> torch.Tensor:
    x = x.float().cpu()
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(len(x), dtype=torch.float32)
    for v in torch.unique(x):
        tie = x == v
        ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    rx, ry = _ranks(x), _ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = (rx.norm() * ry.norm()).item()
    return (rx @ ry).item() / denom if denom > 0 else 0.0


def seed_summary(values: Sequence[float]) -> Dict[str, float]:
    t = torch.tensor(list(values), dtype=torch.float32)
    return {"median": t.median().item(), "min": t.min().item(), "max": t.max().item(), "n": len(values)}


def continual_summary(history: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """history[family] = accuracy on that family at each evaluation point. Returns peak, final, forgetting per family."""
    out = {}
    for fam, accs in history.items():
        peak = max(accs) if accs else 0.0
        final = accs[-1] if accs else 0.0
        out[fam] = {"peak": peak, "final": final, "forgetting": peak - final}
    return out
