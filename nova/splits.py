"""Train and test splits, the sealed family-OOD set, and split_audit(), which fails loudly on any leak."""

import hashlib
import json
import time
from typing import Dict, List, Optional

from . import tasks as T

SPLITS = ("train", "iid_val", "comp_ood", "struct_ood", "family_ood")
EVAL_SPLITS = ("iid_val", "comp_ood", "struct_ood")
REGIME_OF = {"train": "train", "iid_val": "train", "comp_ood": "train", "struct_ood": "large", "family_ood": "train"}


class SplitLeakageError(RuntimeError):
    """A forbidden overlap between training and test material was found."""


class SealedSplitError(PermissionError):
    """The family-OOD split may only be opened by final_eval()."""


def _probe_inputs(domain: str, n: int = 12, tag: str = "filter"):
    rng = T.seeded_rng("probe", tag, domain)
    regime = T.REGIMES[domain]["train"]
    if domain == "graph":
        return [T.sample_graph(rng, regime, dag=i % 2 == 0) for i in range(n)]
    return [T.sample_input(domain, rng, regime, (("copy", ()),)) for _ in range(n)]


def _behaviour(domain: str, program: T.Program, probes) -> str:
    outs = []
    for x in probes:
        try:
            outs.append(T.encode_output(domain, T.run_program(domain, program, x)))
        except Exception as err:
            outs.append(["error", type(err).__name__])
    return hashlib.sha256(json.dumps(outs).encode()).hexdigest()


def is_sealed(domain: str, program: T.Program) -> bool:
    return T.SEALED_FAMILIES[domain] in T.program_families(domain, program)


def is_held_out(domain: str, program: T.Program) -> bool:
    """True for held-out compositions, parameters and colour maps. The training sampler can never produce these."""
    names = [n for n, _ in program]
    fams = T.program_families(domain, program)
    pairs = set(zip(names, names[1:]))
    if domain == "sequence":
        if ("sort", "rotate") in pairs:
            return True
        if pairs & {("map_affine", "dedup"), ("dedup", "map_affine")}:  # these two commute, so both orders are held out
            return True
        if tuple(fams) in {("rotate", "map", "stable_sort"), ("deduplicate", "stable_sort", "bracket")}:
            return True
        return any(n == "map_affine" and p[0] in (4, 7) for n, p in program)
    if domain == "grid":
        steps = list(zip(fams, names))
        for (f1, n1), (f2, n2) in zip(steps, steps[1:]):
            # symmetry and recolour-by-size commute, so both orders are held out
            if {f1, n2} == {"symmetry", "recolour_largest"} or {n1, f2} == {"recolour_largest", "symmetry"}:
                return True
        return any(n == "recolour_map" and p[0] in (4, 5) for n, p in program)
    query = names[-1]
    if query == "reach" and "reverse" in names:  # reverse anywhere before reach (it commutes with drop_colour)
        return True
    return query == "largest_comp" and "drop_colour" in names


def split_programs(domain: str, max_depth: int) -> Dict[str, List[T.Program]]:
    """Assign every program of the grammar to its splits by the predicates above."""
    depth = min(max_depth, 2) if domain == "grid" else max_depth
    every = T.enumerate_programs(domain, depth)
    familiar = [p for p in every if not is_sealed(domain, p) and not is_held_out(domain, p)]
    probes = _probe_inputs(domain)
    seen = {_behaviour(domain, p, probes) for p in familiar}
    distinct = lambda p: _behaviour(domain, p, probes) not in seen  # held-out program that truly does something new
    return {
        "train": familiar, "iid_val": familiar, "struct_ood": familiar,
        "comp_ood": [p for p in every if is_held_out(domain, p) and not is_sealed(domain, p) and distinct(p)],
        "family_ood": [p for p in every if is_sealed(domain, p) and distinct(p)],
    }


class Splits:
    def __init__(self, domain: str, max_depth: int = 3, k_min: int = 2, k_max: int = 8, queries: int = 4,
                 eval_tasks: int = 1000, ledger_path: Optional[str] = None):
        if domain not in T.RULES:
            raise ValueError(f"unknown domain {domain}")
        self.domain, self.max_depth = domain, max_depth
        self.k_min, self.k_max, self.queries = k_min, k_max, queries
        self.n_eval, self.ledger_path = eval_tasks, ledger_path
        self.programs = split_programs(domain, max_depth)
        self._cache: Dict[str, List[T.Task]] = {}
        self._excluded: Optional[set] = None

    def _task(self, split: str, *key) -> T.Task:
        rng = T.seeded_rng("nova", self.domain, split, *key)
        program = rng.choice(self.programs[split])
        return T.make_task(self.domain, program, REGIME_OF[split], rng, self.k_min, self.k_max, self.queries)

    def eval_set(self, split: str, n: Optional[int] = None, unseal: bool = False) -> List[T.Task]:
        """The first n tasks of an evaluation split. Always identical for the same domain and settings."""
        if split == "family_ood" and not unseal:
            raise SealedSplitError("family_ood is sealed; use final_eval()")
        if split == "train":
            raise ValueError("use train_task() for training data")
        if split not in self._cache:
            self._cache[split] = [self._task(split, i) for i in range(self.n_eval)]
        return self._cache[split][: (n or self.n_eval)]

    def excluded_hashes(self) -> set:
        if self._excluded is None:
            self._excluded = {t.instance_hash() for s in EVAL_SPLITS for t in self.eval_set(s)}
        return self._excluded

    def train_task(self, seed: int, index: int) -> T.Task:
        """Deterministic training task. Any task identical to an evaluation task is skipped (no duplicates by construction)."""
        attempt = 0
        while True:
            task = self._task("train", seed, index, attempt)
            if task.instance_hash() not in self.excluded_hashes():
                return task
            attempt += 1

    def manifest(self) -> dict:
        sig = lambda s: sorted(T.program_signature(self.domain, p, {"name": REGIME_OF[s], **T.REGIMES[self.domain][REGIME_OF[s]]})
                               for p in self.programs[s])
        body = {"domain": self.domain, "generator": T.GENERATOR_VERSION, "max_depth": self.max_depth,
                "sizes": [self.k_min, self.k_max, self.queries, self.n_eval],
                "splits": {s: hashlib.sha256("".join(sig(s)).encode()).hexdigest() for s in SPLITS},
                "counts": {s: len(self.programs[s]) for s in SPLITS}}
        body["manifest_hash"] = "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        return body

    def final_eval_set(self, reason: str) -> List[T.Task]:
        """Open family_ood once, writing who opened it and why into the ledger."""
        if self.ledger_path:
            from .checkpoint import append_jsonl  # imported here to avoid a circular import
            append_jsonl(self.ledger_path, {"event": "family_ood_unsealed", "domain": self.domain,
                                            "reason": reason, "time": time.time()})
        return self.eval_set("family_ood", unseal=True)


def split_audit(splits: Splits, samples: int = 2000, report_path: Optional[str] = None) -> dict:
    """Check the splits for every kind of leak. Raises SplitLeakageError on any problem. No result is valid until it passes."""
    d = splits.domain
    problems, report = [], {"domain": d, "checks": {}}
    progs = splits.programs
    train = set(progs["train"])
    # 1. program-level intersections: test programs must never be training programs
    for s in ("comp_ood", "family_ood"):
        overlap = train & set(progs[s])
        report["checks"][f"train_x_{s}_programs"] = len(overlap)
        if overlap:
            problems.append(f"{len(overlap)} {s} programs also in train")
    bad_train = [p for p in progs["train"] if is_sealed(d, p) or is_held_out(d, p)]
    report["checks"]["train_programs_breaking_predicates"] = len(bad_train)
    if bad_train:
        problems.append("training contains held-out or sealed programs")
    # 2. behaviour-level check: a training program must not behave exactly like a held-out one
    probes = _probe_inputs(d, n=24, tag="audit")  # a larger, different probe set than the filter uses
    train_behaviour = {_behaviour(d, p, probes) for p in progs["train"]}
    for s in ("comp_ood", "family_ood"):
        same = [T.describe(p) for p in progs[s] if _behaviour(d, p, probes) in train_behaviour]
        report["checks"][f"{s}_behaviour_collisions"] = len(same)
        if same:
            problems.append(f"{s} programs behave like training programs: {same[:3]}")
    # 3. instance-level collisions between training samples and evaluation sets
    train_tasks = [splits.train_task(seed=0, index=i) for i in range(samples)]
    train_hashes = {t.instance_hash() for t in train_tasks}
    for s in EVAL_SPLITS:
        hashes = {t.instance_hash() for t in splits.eval_set(s)}
        report["checks"][f"train_x_{s}_instances"] = len(train_hashes & hashes)
        if train_hashes & hashes:
            problems.append(f"exact duplicate instances between train and {s}")
    # 4. near-duplicates: same program and first input differing in at most one token
    by_sig: Dict[str, list] = {}
    for t in train_tasks:
        by_sig.setdefault(t.signature, []).append(t.supports[0][0])
    near = 0
    for t in splits.eval_set("iid_val"):
        x = t.supports[0][0]
        for y in by_sig.get(t.signature, []):
            if len(x) == len(y) and sum(a != b for a, b in zip(x, y)) <= 1:
                near += 1
                break
    rate = near / max(1, len(splits.eval_set("iid_val")))
    report["checks"]["iid_val_near_duplicate_rate"] = round(rate, 4)
    if rate > 0.01:
        problems.append(f"near-duplicate rate {rate:.3f} above 0.01")
    # 5. signature completeness: changing any field must change the signature
    program = progs["train"][0]
    regime = {"name": "train", **T.REGIMES[d]["train"]}
    base = T.program_signature(d, program, regime)
    changed = {k: T.program_signature(d, program, {**regime, k: "CHANGED"}) != base for k in regime}
    changed["program"] = T.program_signature(d, progs["train"][1], regime) != base
    changed["domain"] = T.program_signature("other", program, regime) != base
    report["checks"]["signature_fields_complete"] = all(changed.values())
    if not all(changed.values()):
        problems.append(f"signature ignores fields: {[k for k, v in changed.items() if not v]}")
    report["manifest"] = splits.manifest()
    report["passed"] = not problems
    report["problems"] = problems
    if report_path:
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)
    if problems:
        raise SplitLeakageError("; ".join(problems))
    return report
