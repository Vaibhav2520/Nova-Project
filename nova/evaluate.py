"""Evaluation: scoring models, learning-curve tables, GEI, extrapolation surface, attacks and statistical controls."""

import copy
import random
import time
from typing import Callable, Dict, List, Optional, Sequence

import torch

from . import metrics as M
from . import tasks as T
from .splits import is_held_out, is_sealed


@torch.no_grad()
def evaluate_tasks(model, tasks: Sequence[T.Task], batch_size: int = 32, device="cpu", max_steps: Optional[int] = None) -> dict:
    """Run model.predict on tasks in batches. Returns accuracy, per-question correctness, confidence and compute."""
    model.eval()
    correct, conf, task_ok, steps, calls, depth = [], [], [], [], [], []
    start = time.perf_counter()
    for i in range(0, len(tasks), batch_size):
        chunk = tasks[i:i + batch_size]
        batch, target = T.collate(chunk, device)
        out = model.predict(batch, max_steps=max_steps)
        gold = target.query_outputs[out["q_task"], out["q_index"]]
        ok = M.exact_match(out["tokens"], gold)
        correct.append(ok)
        conf.append(out["confidence"].float().cpu())
        task_ok.append(M.task_accuracy(ok, out["q_task"].cpu(), len(chunk)))
        steps.append(out["compute"]["refine_steps"].cpu())
        calls.append(out["compute"]["executor_calls"].cpu())
        depth += [t.depth for t in chunk]
    elapsed = time.perf_counter() - start
    task_ok = torch.cat(task_ok)
    return {"task_acc": task_ok.float().mean().item(), "task_correct": task_ok, "query_correct": torch.cat(correct),
            "confidence": torch.cat(conf), "steps": torch.cat(steps), "executor_calls": torch.cat(calls),
            "depth": torch.tensor(depth), "seconds": elapsed, "ms_per_task": 1000 * elapsed / max(1, len(tasks))}


def calibration_report(result: dict) -> dict:
    c, ok = result["confidence"], result["query_correct"]
    return {"ece": M.ece(c, ok), "brier": M.brier(c, ok), "aurc": M.risk_coverage(c, ok)["aurc"],
            "sel_acc_50": M.selective_accuracy(c, ok, 0.5), "sel_acc_80": M.selective_accuracy(c, ok, 0.8)}


def adaptive_compute_report(result: dict) -> dict:
    return {"spearman_steps_depth": M.spearman(result["steps"].float(), result["depth"].float()),
            "mean_calls": result["executor_calls"].float().mean().item()}


def gei_from_curves(curves: Dict[str, Dict[str, List[float]]], budgets: Sequence[float], heuristic: Dict[str, float],
                    reference: str = "transformer", split_key: str = "ood") -> Dict[str, dict]:
    """curves[model][domain] = accuracy at each budget. heuristic[domain] = its constant accuracy. Returns GEI per model."""
    out = {}
    for model, by_domain in curves.items():
        scores = {}
        for domain, accs in by_domain.items():
            ref = curves[reference][domain]
            skills = [M.normalised_skill(a, heuristic[domain], r) for a, r in zip(accs, ref)]
            scores[domain] = M.domain_score(budgets, skills)
        out[model] = {"gei": M.gei(scores), "domain_scores": scores, "raw": by_domain}
    return out


def random_program(domain: str, depth: int, rng: random.Random, tries: int = 2000) -> T.Program:
    """Random valid program of exactly `depth` steps that is neither held out nor sealed (deeper than training is allowed)."""
    rules = T.RULES[domain]
    steps = [(n, p) for n, r in sorted(rules.items()) for p in r.params]
    for _ in range(tries):
        chain = tuple(rng.choice(steps) for _ in range(depth))
        if T._valid_chain(domain, chain) and not is_held_out(domain, chain) and not is_sealed(domain, chain):
            return chain
    raise RuntimeError(f"no valid {domain} program of depth {depth}")


def surface_regime(domain: str, mult: float) -> dict:
    if domain == "sequence":  # training maximum length is 16
        n = int(16 * mult)
        return {"len": [n, n]}
    if domain == "grid":  # training maximum side is 7
        s = min(14, int(7 * mult))  # capped so token counts stay manageable
        return {"side": [s, s], "objects": [1, 3], "distractor": False}
    n = min(T.NUM_VALUES - 1, int(12 * mult))  # training maximum is 12 nodes; node ids must fit in a token
    return {"nodes": [n, n], "model": "er"}


def surface_tasks(domain: str, mult: float, depth: int, n: int, k: int = 4, queries: int = 2) -> List[T.Task]:
    T.REGIMES[domain]["surface"] = surface_regime(domain, mult)
    rng = T.seeded_rng("surface", domain, mult, depth)
    return [T.make_task(domain, random_program(domain, depth, rng), "surface", rng, k, k, queries) for _ in range(n)]


def extrapolation_surface(model, domain: str, mults=(1, 2, 4, 8), depths=(1, 2, 3, 4, 5), n: int = 50, device="cpu") -> dict:
    """Accuracy over size x depth. The failure type (abrupt/gradual/unstable) is read from the resulting table."""
    table = {}
    for m in mults:
        for d in depths:
            try:  # some cells may be impossible (e.g. grids above the cap)
                res = evaluate_tasks(model, surface_tasks(domain, m, d, n), device=device)
                table[(m, d)] = {"acc": res["task_acc"], "finite_conf": bool(torch.isfinite(res["confidence"]).all())}
            except RuntimeError as err:
                table[(m, d)] = {"acc": None, "error": str(err)}
    return table


def _value_tokens(task: T.Task) -> List[int]:
    return sorted({t for x, y in task.supports + task.queries for t in x + y if t >= T.VALUE_OFFSET})


def attack_permute(task: T.Task, rng: random.Random) -> T.Task:
    t = copy.deepcopy(task)
    rng.shuffle(t.supports)
    return t


def attack_irrelevant(task: T.Task, donor: T.Task, rng: random.Random) -> T.Task:
    t = copy.deepcopy(task)
    t.supports = t.supports + donor.supports[:2]
    rng.shuffle(t.supports)
    return t


def attack_contradict(task: T.Task, donor: T.Task, rng: random.Random) -> T.Task:
    t = copy.deepcopy(task)
    i = rng.randrange(len(t.supports))
    t.supports[i] = (t.supports[i][0], donor.supports[0][1])
    return t


def attack_duplicate(task: T.Task, rng: random.Random) -> T.Task:
    t = copy.deepcopy(task)
    t.supports.append(t.supports[rng.randrange(len(t.supports))])
    return t


def attack_relabel(task: T.Task, rng: random.Random) -> T.Task:
    """Consistently swap value tokens across demos, questions and answers (a conjugated rule, so the answer stays defined)."""
    vals = _value_tokens(task)
    shuffled = vals[:]
    rng.shuffle(shuffled)
    m = dict(zip(vals, shuffled))
    f = lambda seq: [m.get(x, x) for x in seq]
    t = copy.deepcopy(task)
    t.supports = [(f(x), f(y)) for x, y in t.supports]
    t.queries = [(f(x), f(y)) for x, y in t.queries]
    return t


def attack_decoy(task: T.Task, rng: random.Random) -> T.Task:
    """Add isolated noise pixels or extra edges to every input and recompute the true answers with the hidden program."""
    if task.domain == "sequence":
        return copy.deepcopy(task)
    t = copy.deepcopy(task)
    def change(x):
        if task.domain == "grid":
            g = T.decode_grid(x)
            r, c = rng.randrange(len(g)), rng.randrange(len(g[0]))
            if g[r][c] == 0:
                g[r][c] = 9
            value = g
        else:
            g = T.decode_graph(x)
            u, v = rng.sample(range(g.n), 2)
            if (u, v) not in g.edges and not (task.program[-1][0] == "topo"):  # keep DAGs acyclic
                g.edges.append((u, v))
            value = g
        return T.encode_input(task.domain, value), T.encode_output(task.domain, T.run_program(task.domain, task.program, value))
    t.supports = [change(x) for x, _ in t.supports]
    t.queries = [change(x) for x, _ in t.queries]
    return t


ATTACKS: Dict[str, Callable] = {"permute": attack_permute, "duplicate": attack_duplicate, "relabel": attack_relabel,
                                # add the post-inspection attack here (and date it in the ledger)
                                "decoy": attack_decoy}
DONOR_ATTACKS: Dict[str, Callable] = {"irrelevant": attack_irrelevant, "contradict": attack_contradict}


def attack_suite(model, tasks: Sequence[T.Task], device="cpu", seed: int = 0) -> Dict[str, dict]:
    """Accuracy on clean tasks, under every attack, and on hard tasks with compute capped at one refinement step."""
    rng = random.Random(seed)
    out = {"clean": evaluate_tasks(model, tasks, device=device)}
    for name, fn in ATTACKS.items():
        out[name] = evaluate_tasks(model, [fn(t, rng) for t in tasks], device=device)
    for name, fn in DONOR_ATTACKS.items():
        donors = list(tasks[1:]) + list(tasks[:1])
        out[name] = evaluate_tasks(model, [fn(t, d, rng) for t, d in zip(tasks, donors)], device=device)
    hard = [t for t in tasks if t.depth >= 2] or list(tasks)
    out["compute_cap_1"] = evaluate_tasks(model, hard, device=device, max_steps=1)
    return {k: {"task_acc": v["task_acc"], "ece": M.ece(v["confidence"], v["query_correct"])} for k, v in out.items()}


def label_shuffle(tasks: Sequence[T.Task]) -> List[T.Task]:
    out = []
    for i, t in enumerate(tasks):
        donor = tasks[(i + 1) % len(tasks)]
        c = copy.deepcopy(t)
        c.supports = [(x, donor.supports[j % len(donor.supports)][1]) for j, (x, _) in enumerate(c.supports)]
        out.append(c)
    return out  # accuracy on these must collapse towards chance


def support_ablation(tasks: Sequence[T.Task]) -> List[T.Task]:
    out = []
    for t in tasks:
        c = copy.deepcopy(t)
        c.supports = []
        out.append(c)
    return out  # few-shot inference must drop on these


OOD_KEYS = ("comp_ood", "struct_ood")  # the public OOD splits used for the headline GEI


def curves_from_records(records: Sequence[dict], keys: Sequence[str] = OOD_KEYS) -> Dict[str, Dict[str, List[float]]]:
    """curves[model][domain] = mean accuracy over `keys` at each learning-curve point, averaged over seeds."""
    acc: Dict[str, Dict[str, List[List[float]]]] = {}
    for r in records:
        if r.get("end_state") != "completed" or not r.get("curve"):  # skip unfinished runs (they stay in the ledger)
            continue
        curve = [sum(p[k] for k in keys) / len(keys) for p in r["curve"]]
        acc.setdefault(r["model"], {}).setdefault(r["domain"], []).append(curve)
    return {m: {d: [sum(c[i] for c in cs) / len(cs) for i in range(min(map(len, cs)))] for d, cs in by.items()}
            for m, by in acc.items()}


def budgets_from_records(records: Sequence[dict]) -> List[float]:
    for r in records:
        if r.get("curve"):
            return [p["step"] for p in r["curve"]]
    return []
