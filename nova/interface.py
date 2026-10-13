"""ResearchBackend: a typed, versioned, framework-free service boundary used by the console and by pure Python."""

import copy
import hashlib
import json
import time
from typing import Dict, List, Optional

import torch

from . import __version__, code_hash
from . import tasks as T
from .baselines import build_model
from .checkpoint import CheckpointError, CorruptCheckpointError, load_checkpoint
from .config import from_dict
from .model import count_parameters

REQUEST_SCHEMA, RESPONSE_SCHEMA = "nova.request/1.1", "nova.response/1.1"
ACCEPTED = ("nova.request/1.0", "nova.request/1.1")  # versions we can read (1.0 is migrated)
DOMAINS = ("sequence", "grid", "graph")
TARGET_KEYS = ("output", "target", "answer", "outputs", "label")  # keys that would leak an answer into a question
DEFAULT_SETTINGS = {"seed": 0, "deterministic": True, "device": "auto", "max_steps": 6,
                    "confidence_threshold": 0.5, "trace_level": "summary"}
ERROR_CODES = ("E_SCHEMA_VERSION", "E_SCHEMA", "E_SHAPE", "E_TOKEN_RANGE", "E_TARGET_IN_QUERY", "E_EMPTY_SUPPORT",
               "E_DEVICE_UNAVAILABLE", "E_CHECKPOINT_CORRUPT", "E_CHECKPOINT_INCOMPATIBLE", "E_MEMORY_BUDGET",
               "E_NUMERICAL", "E_INTERNAL", "E_UNKNOWN_MODEL")


class BackendError(Exception):
    def __init__(self, code: str, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.code, self.message, self.field = code, message, field

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "field": self.field}


def request_hash(request: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def migrate(request: dict) -> dict:
    req = copy.deepcopy(request)
    if req.get("schema_version") == "nova.request/1.0":
        req.setdefault("settings", {}).setdefault("trace_level", "summary")
        req["schema_version"] = REQUEST_SCHEMA
    return req


def _check_ints(values, lo, hi, field):
    if not isinstance(values, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        raise BackendError("E_SCHEMA", "expected a list of integers", field)
    if any(v < lo or v > hi for v in values):
        raise BackendError("E_TOKEN_RANGE", f"values must be in [{lo}, {hi}]", field)


def _check_value(domain: str, value, field: str, is_output: bool):
    if domain == "sequence":
        _check_ints(value, 0, T.NUM_VALUES - 1 if is_output else T.NUM_SYMBOLS - 1, field)
        if not value and not is_output:
            raise BackendError("E_SHAPE", "empty input sequence", field)
    elif domain == "grid":
        if not isinstance(value, list) or not value or not all(isinstance(r, list) and r for r in value):
            raise BackendError("E_SHAPE", "a grid must be a non-empty list of non-empty rows", field)
        if len({len(r) for r in value}) != 1:
            raise BackendError("E_SHAPE", "grid rows must all have the same length", field)
        if len(value) > 30 or len(value[0]) > 30:
            raise BackendError("E_SHAPE", "grids are limited to 30x30", field)
        for i, row in enumerate(value):
            _check_ints(row, 0, 9, f"{field}[{i}]")
    else:
        if is_output:
            _check_ints(value, 0, T.NUM_VALUES - 1, field)
            return
        if not isinstance(value, dict) or not {"n", "edges", "colours", "src", "dst"} <= set(value):
            raise BackendError("E_SCHEMA", "a graph needs n, edges, colours, src and dst", field)
        n = value["n"]
        if not isinstance(n, int) or not 2 <= n < T.NUM_VALUES:
            raise BackendError("E_SHAPE", f"n must be in [2, {T.NUM_VALUES - 1}]", field)
        _check_ints(value["colours"], 0, 3, field + ".colours")
        if len(value["colours"]) != n:
            raise BackendError("E_SHAPE", "need one colour per node", field)
        for e in value["edges"]:
            if not (isinstance(e, list) and len(e) == 2):
                raise BackendError("E_SHAPE", "edges must be [u, v] pairs", field)
            _check_ints(e, 0, n - 1, field + ".edges")
        _check_ints([value["src"], value["dst"]], 0, n - 1, field + ".src/dst")


def validate_request(request) -> dict:
    if not isinstance(request, dict):
        raise BackendError("E_SCHEMA", "request must be a JSON object")
    if request.get("schema_version") not in ACCEPTED:
        raise BackendError("E_SCHEMA_VERSION", f"unsupported schema_version {request.get('schema_version')!r}", "schema_version")
    req = migrate(request)
    allowed = {"schema_version", "request_id", "checkpoint", "models", "settings", "task"}
    extra = set(req) - allowed
    if extra:  # refuse unknown fields so typos are caught
        raise BackendError("E_SCHEMA", f"unknown fields {sorted(extra)}")
    settings = {**DEFAULT_SETTINGS, **req.get("settings", {})}
    if set(settings) - set(DEFAULT_SETTINGS):
        raise BackendError("E_SCHEMA", f"unknown settings {sorted(set(settings) - set(DEFAULT_SETTINGS))}", "settings")
    if not isinstance(settings["max_steps"], int) or not 0 <= settings["max_steps"] <= 8:
        raise BackendError("E_SCHEMA", "max_steps must be an integer in [0, 8]", "settings.max_steps")
    if not isinstance(settings["confidence_threshold"], (int, float)) or not 0 <= settings["confidence_threshold"] <= 1:
        raise BackendError("E_SCHEMA", "confidence_threshold must be in [0, 1]", "settings.confidence_threshold")
    if settings["device"] not in ("auto", "cpu", "cuda"):
        raise BackendError("E_SCHEMA", "device must be auto, cpu or cuda", "settings.device")
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise BackendError("E_DEVICE_UNAVAILABLE", "CUDA was requested but is not available", "settings.device")
    if settings["trace_level"] not in ("none", "summary", "full"):
        raise BackendError("E_SCHEMA", "trace_level must be none, summary or full", "settings.trace_level")
    task = req.get("task")
    if not isinstance(task, dict) or task.get("domain") not in DOMAINS:
        raise BackendError("E_SCHEMA", f"task.domain must be one of {DOMAINS}", "task.domain")
    support, query = task.get("support"), task.get("query")
    if not isinstance(support, list) or not isinstance(query, list):
        raise BackendError("E_SCHEMA", "task.support and task.query must be lists", "task")
    if not support:
        raise BackendError("E_EMPTY_SUPPORT", "at least one demonstration is required", "task.support")
    if len(support) > 8 or not 1 <= len(query) <= 8:
        raise BackendError("E_SHAPE", "use 1-8 demonstrations and 1-8 questions", "task")
    for i, q in enumerate(query):
        if not isinstance(q, dict) or "input" not in q:
            raise BackendError("E_SCHEMA", "each query needs an input", f"task.query[{i}]")
        if any(k in q for k in TARGET_KEYS):  # an answer was smuggled into a question
            raise BackendError("E_TARGET_IN_QUERY", "queries must not carry answers", f"task.query[{i}]")
        if set(q) - {"input"}:
            raise BackendError("E_SCHEMA", f"unknown query fields {sorted(set(q) - {'input'})}", f"task.query[{i}]")
        _check_value(task["domain"], q["input"], f"task.query[{i}].input", is_output=False)
    for i, s in enumerate(support):
        if not isinstance(s, dict) or set(s) != {"input", "output"}:
            raise BackendError("E_SCHEMA", "each demonstration needs exactly input and output", f"task.support[{i}]")
        _check_value(task["domain"], s["input"], f"task.support[{i}].input", is_output=False)
        _check_value(task["domain"], s["output"], f"task.support[{i}].output", is_output=True)
    req["settings"] = settings
    req.setdefault("models", ["nova"])
    if not isinstance(req["models"], list) or not req["models"]:
        raise BackendError("E_SCHEMA", "models must be a non-empty list", "models")
    return req


def _to_value(domain: str, v, is_output: bool):
    if domain == "graph" and not is_output:
        return T.Graph(v["n"], [tuple(e) for e in v["edges"]], list(v["colours"]), v["src"], v["dst"])
    return v


def tensorize(req: dict, device) -> T.TaskBatch:  # validated request -> TaskBatch (questions carry dummy answers only)
    d, task = req["task"]["domain"], req["task"]
    sup = [(T.encode_input(d, _to_value(d, s["input"], False)), _encode_answer(d, s["output"])) for s in task["support"]]
    qry = [(T.encode_input(d, _to_value(d, q["input"], False)), [T.EOS]) for q in task["query"]]
    batch, _ = T.collate([T.Task(d, (), "request", sup, qry)], device)
    return batch


def _encode_answer(domain: str, value) -> List[int]:
    return T.encode_grid(value) + [T.EOS] if domain == "grid" else [T.vt(v) for v in value] + [T.EOS]


def _summ(x: torch.Tensor, level: str):
    return None if level == "none" else [[round(float(v), 4) for v in row] if isinstance(row, list) else round(float(row), 4)
                                         for row in x.tolist()]


class ResearchBackend:
    """Separates validation, tensorisation, inference, metrics, tracing and serialisation. Never trains."""

    def __init__(self, memory_cap_bytes: int = 2 * 2 ** 30):
        self.models: Dict[str, torch.nn.Module] = {}
        self.identity: Dict[str, dict] = {}
        self.memory_cap = memory_cap_bytes

    def register(self, name: str, model: torch.nn.Module, identity: Optional[dict] = None) -> dict:
        model.eval()
        self.models[name] = model
        self.identity[name] = identity or {"id": f"memory:{name}", "code_hash": code_hash(), "params": count_parameters(model)}
        return self.identity[name]

    def load_checkpoint(self, ref: str) -> dict:
        try:
            p = load_checkpoint(ref, map_location="cpu", required=("schema", "model_name", "model", "config", "config_hash", "code_hash"))
            cfg = from_dict(p["config"])
            model = build_model(p["model_name"], cfg.model)
            model.load_state_dict(p["model"])
        except CorruptCheckpointError as err:
            return {"status": "error", "errors": [BackendError("E_CHECKPOINT_CORRUPT", str(err)).as_dict()]}
        except (CheckpointError, KeyError, RuntimeError, ValueError) as err:
            return {"status": "error", "errors": [BackendError("E_CHECKPOINT_INCOMPATIBLE", str(err)).as_dict()]}
        ident = {"id": ref, "step": p.get("step"), "config_hash": p["config_hash"], "code_hash": p["code_hash"],
                 "params": count_parameters(model)}
        return {"status": "ok", "model": p["model_name"], "identity": self.register(p["model_name"], model, ident)}

    def validate(self, request: dict) -> dict:
        try:
            req = validate_request(request)
            self._check_models(req)
            self._check_memory(req)
            return {"schema_version": RESPONSE_SCHEMA, "status": "ok", "request": req, "errors": []}
        except BackendError as err:
            return {"schema_version": RESPONSE_SCHEMA, "status": "error", "errors": [err.as_dict()]}

    def infer(self, request: dict) -> dict:
        return self._run(request, compare=False)

    def compare(self, request: dict) -> dict:
        return self._run(request, compare=True)

    def trace(self, request: dict) -> dict:
        req = copy.deepcopy(request) if isinstance(request, dict) else request
        if isinstance(req, dict):
            req.setdefault("settings", {})["trace_level"] = "full"
        return self._run(req, compare=False)

    def health(self) -> dict:
        cuda = torch.cuda.is_available()
        return {"schema_version": RESPONSE_SCHEMA, "status": "ok", "version": __version__, "code_hash": code_hash(),
                "torch": torch.__version__, "cuda": cuda, "device": torch.cuda.get_device_name(0) if cuda else "cpu",
                "models": {k: v for k, v in self.identity.items()}, "memory_cap_bytes": self.memory_cap}

    def infer_many(self, requests: List[dict]) -> List[dict]:
        """Requests with the same models, settings and domain run in one batch; each still gets its own response."""
        out: List[Optional[dict]] = [None] * len(requests)
        groups: Dict[str, list] = {}
        for i, r in enumerate(requests):
            v = self.validate(r)
            if v["status"] != "ok":
                out[i] = {**v, "request_id": r.get("request_id") if isinstance(r, dict) else None}
                continue
            req = v["request"]
            key = json.dumps([req["models"], req["settings"], req["task"]["domain"]], sort_keys=True)
            groups.setdefault(key, []).append((i, req))
        for members in groups.values():
            batches = [tensorize(req, "cpu") for _, req in members]
            merged = _merge(batches)
            results = self._predict(members[0][1], merged)
            for pos, (i, req) in enumerate(members):
                out[i] = self._respond(req, {m: _select(r, pos) for m, r in results.items()}, compare=True)
        return out

    def _check_models(self, req: dict) -> None:
        missing = [m for m in req["models"] if m not in self.models]
        if missing:
            raise BackendError("E_UNKNOWN_MODEL", f"models not loaded: {missing}", "models")

    def _check_memory(self, req: dict) -> None:
        t = req["task"]
        n_tok = sum(len(json.dumps(s)) for s in t["support"] + t["query"])
        k = len(t["support"])
        width = max((getattr(getattr(m, "cfg", None), "d_model", 0) for m in self.models.values()), default=0)
        estimate = 4 * width * n_tok * max(1, k * k) * 8  # bytes: float32 x width x tokens x cross pairs x layers/steps
        if estimate > self.memory_cap:
            raise BackendError("E_MEMORY_BUDGET", f"estimated {estimate} bytes exceeds the cap {self.memory_cap}", "task")

    def _device(self, req) -> torch.device:
        want = req["settings"]["device"]
        return torch.device("cuda" if want == "cuda" or (want == "auto" and torch.cuda.is_available()) else "cpu")

    def _predict(self, req: dict, batch) -> dict:
        s, dev = req["settings"], self._device(req)
        results = {}
        for name in req["models"]:
            model = self.models[name].to(dev)
            model.eval()
            before = _fingerprint(model)
            torch.manual_seed(s["seed"])
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                out = model.predict(batch.to(dev), max_steps=s["max_steps"])
            out["latency_ms"] = 1000 * (time.perf_counter() - start)
            out["peak_mem_bytes"] = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0
            if _fingerprint(model) != before:
                raise BackendError("E_INTERNAL", "model parameters changed during inference")
            if not torch.isfinite(out["confidence"]).all():
                raise BackendError("E_NUMERICAL", "non-finite confidence")
            results[name] = out
        return results

    def _run(self, request, compare: bool) -> dict:  # validate -> tensorise -> infer -> serialise
        rid = request.get("request_id") if isinstance(request, dict) else None
        try:  # every failure becomes a structured error, never a stack trace
            req = validate_request(request)
            if not compare:
                req["models"] = req["models"][:1]
            self._check_models(req)
            self._check_memory(req)
            batch = tensorize(req, "cpu")
            results = self._predict(req, batch)
            return self._respond(req, results, compare)
        except BackendError as err:
            return {"schema_version": RESPONSE_SCHEMA, "request_id": rid, "status": "error", "errors": [err.as_dict()]}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"schema_version": RESPONSE_SCHEMA, "request_id": rid, "status": "error",
                    "errors": [BackendError("E_MEMORY_BUDGET", "out of GPU memory").as_dict()]}
        except Exception as err:
            return {"schema_version": RESPONSE_SCHEMA, "request_id": rid, "status": "error",
                    "errors": [BackendError("E_INTERNAL", f"{type(err).__name__}: {err}").as_dict()]}

    def _respond(self, req: dict, results: dict, compare: bool) -> dict:
        s, d = req["settings"], req["task"]["domain"]
        body = {}
        for name, out in results.items():
            order = sorted(range(len(out["q_index"])), key=lambda i: int(out["q_index"][i]))
            conf = [round(float(out["confidence"][i]), 4) for i in order]
            entry = {"predictions": [T.decode_value(d, out["tokens"][i].tolist()) for i in order],
                     "confidence": conf, "abstained": [c < s["confidence_threshold"] for c in conf],
                     "compute": {"executor_calls": int(out["compute"]["executor_calls"].sum()),
                                 "refine_steps": int(out["compute"]["refine_steps"].sum()),
                                 "latency_ms": round(out.get("latency_ms", 0.0), 2), "peak_mem_bytes": int(out.get("peak_mem_bytes", 0))},
                     "checkpoint": self.identity.get(name, {})}
            tr = out.get("trace") or {}
            if tr and s["trace_level"] != "none":
                entry["trace"] = {"steps_used": int(tr["steps_used"][0]), "halt_step": int(tr["halt_step"][0]),
                                  "rho": [round(float(x), 4) for x in tr["rho"][0].tolist() if float(x) >= 0],
                                  "w": _summ(tr["w"][0], "summary"), "v": _summ(tr["v"][0], "summary")}
                if s["trace_level"] == "full":
                    entry["trace"].update(R=_summ(tr["R"][0], "full"), update_norms=_summ(tr["update_norms"][0], "full"),
                                          rho_loo=round(float(tr["rho_loo"][0]), 4))
            body[name] = entry
        return {"schema_version": RESPONSE_SCHEMA, "request_id": req.get("request_id"), "request_hash": request_hash(req),
                "status": "ok", "results": body, "errors": []}


def _fingerprint(model) -> float:  # cheap parameter fingerprint (detects any training during inference)
    with torch.no_grad():
        return float(sum(p.double().abs().sum().item() + p.double().sum().item() for p in model.parameters()))


def _merge(batches: List[T.TaskBatch]) -> T.TaskBatch:
    import torch.nn.functional as F
    def cat(name):
        xs = [getattr(b, name) for b in batches]
        dims = [max(x.shape[i] for x in xs) for i in range(1, xs[0].dim())]
        padded = []
        for x in xs:
            pad = []
            for i in reversed(range(1, x.dim())):
                pad += [0, dims[i - 1] - x.shape[i]]
            padded.append(F.pad(x, pad, value=0))  # PAD = 0 and False = 0
        return torch.cat(padded, 0)
    names = [f.name for f in T.dataclasses.fields(T.TaskBatch) if f.name != "meta"]
    meta = {"schema": "nova.taskbatch/1", "merged": len(batches)}
    return T.TaskBatch(**{n: cat(n) for n in names}, meta=meta)


def _select(out: dict, task_index: int) -> dict:
    keep = (out["q_task"] == task_index).nonzero(as_tuple=True)[0]
    pick = lambda x: x[task_index:task_index + 1]
    trace = {k: pick(v) for k, v in (out.get("trace") or {}).items()}
    return {"q_task": torch.zeros_like(out["q_task"][keep]), "q_index": out["q_index"][keep], "tokens": out["tokens"][keep],
            "confidence": out["confidence"][keep], "trace": trace,
            "compute": {k: pick(v) for k, v in out["compute"].items()}, "latency_ms": out.get("latency_ms", 0.0)}


def run_hidden_eval(batch_path: str, backend: ResearchBackend, model_name: str = "nova", max_steps: int = 6) -> List[dict]:
    """Entry point for the hidden evaluation: load serialized TaskBatch tensors and return predictions, confidence, trace, compute."""
    data = torch.load(batch_path, map_location="cpu", weights_only=True)
    fields = {f.name for f in T.dataclasses.fields(T.TaskBatch)} - {"meta"}
    if not fields <= set(data):
        raise BackendError("E_SCHEMA", f"TaskBatch file is missing {sorted(fields - set(data))}")
    batch = T.TaskBatch(**{k: data[k] for k in fields}, meta={"schema": "nova.taskbatch/1"})
    model = backend.models[model_name]
    model.eval()
    with torch.inference_mode():
        out = model.predict(batch, max_steps=max_steps)
    rows = []
    for i in range(len(out["q_task"])):
        b = int(out["q_task"][i])
        rows.append({"task": b, "query": int(out["q_index"][i]), "tokens": out["tokens"][i].tolist(),
                     "confidence": float(out["confidence"][i]),
                     "trace": {k: v[b].tolist() for k, v in (out.get("trace") or {}).items() if k in ("steps_used", "halt_step")},
                     "compute": {k: int(v[b]) for k, v in out["compute"].items()}})
    return rows


def task_to_request(task: T.Task, request_id: str, models=("nova",)) -> Optional[dict]:
    """Turn a generated task into a valid request. Returns None when an answer contains special tokens (e.g. brackets)."""
    d = task.domain
    def inp(x):
        if d == "grid":
            return T.decode_grid(x)
        if d == "graph":
            g = T.decode_graph(x)
            return {"n": g.n, "edges": [list(e) for e in g.edges], "colours": g.colours, "src": g.src, "dst": g.dst}
        return [t - T.VALUE_OFFSET for t in x]
    def out(y):
        body = [t for t in y if t != T.EOS]
        if d == "grid":
            return T.decode_grid(body)
        if any(t < T.VALUE_OFFSET for t in body):
            raise ValueError("special tokens")
        return [t - T.VALUE_OFFSET for t in body]
    try:
        support = [{"input": inp(x), "output": out(y)} for x, y in task.supports]
    except ValueError:  # e.g. bracket outputs, which contain special tokens
        return None
    return {"schema_version": REQUEST_SCHEMA, "request_id": request_id, "models": list(models),
            "settings": {"max_steps": 3, "trace_level": "summary"}, "task": {"domain": d, "support": support,
            "query": [{"input": inp(x)} for x, _ in task.queries]}}


MUTATIONS = {  # ways to break a valid request; each must be rejected with a structured error
    "no_version": lambda r: {k: v for k, v in r.items() if k != "schema_version"},
    "future_version": lambda r: {**r, "schema_version": "nova.request/9.0"},
    "unknown_field": lambda r: {**r, "debug": True},
    "empty_support": lambda r: {**r, "task": {**r["task"], "support": []}},
    "target_in_query": lambda r: {**r, "task": {**r["task"], "query": [{**r["task"]["query"][0], "output": [1]}]}},
    "bad_domain": lambda r: {**r, "task": {**r["task"], "domain": "audio"}},
    "bad_steps": lambda r: {**r, "settings": {**r["settings"], "max_steps": 99}},
    "not_a_list": lambda r: {**r, "task": {**r["task"], "query": "oops"}},
    "too_many_demos": lambda r: {**r, "task": {**r["task"], "support": r["task"]["support"] * 9}},
    "token_range": lambda r: {**r, "task": {**r["task"], "query": [{"input": _out_of_range(r["task"]["query"][0]["input"])}]}},
}


def _out_of_range(value):
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [[999] + value[0][1:]] + value[1:]
    if isinstance(value, dict):
        return {**value, "colours": [99] + value["colours"][1:]}
    return [999] + list(value[1:])


def reliability_suite(backend: ResearchBackend, tasks: List[T.Task], n_valid: int = 1000, n_invalid: int = 1000,
                      seed: int = 0) -> dict:
    """Replay valid and mutated invalid requests. Reports confusion, crashes, determinism, malformed replies, traces."""
    import random as _r
    rng = _r.Random(seed)
    valid = [r for i, t in enumerate(tasks) if (r := task_to_request(t, f"v{i}"))]
    valid = (valid * (n_valid // max(1, len(valid)) + 1))[:n_valid]
    names = sorted(MUTATIONS)
    invalid = [(names[i % len(names)], MUTATIONS[names[i % len(names)]](json.loads(json.dumps(rng.choice(valid)))))
               for i in range(n_invalid)]
    counts = {"valid_accepted": 0, "valid_rejected": 0, "invalid_rejected": 0, "invalid_accepted": 0, "crashes": 0,
              "malformed": 0, "deterministic_mismatch": 0, "memory_violations": 0}
    accepted_ids = []
    def call(req):
        try:  # a crash is any exception escaping the backend
            resp = backend.infer(req)
        except Exception:
            counts["crashes"] += 1
            return None
        if not isinstance(resp, dict) or resp.get("status") not in ("ok", "error") or "schema_version" not in resp:
            counts["malformed"] += 1
        if any(e.get("code") == "E_MEMORY_BUDGET" for e in resp.get("errors", [])):
            counts["memory_violations"] += 1
        return resp
    for i, req in enumerate(valid):
        resp = call(req)
        ok = resp is not None and resp.get("status") == "ok"
        counts["valid_accepted" if ok else "valid_rejected"] += 1
        if ok:
            accepted_ids.append(req["request_id"] + f"#{i}")
        if ok and i % 20 == 0:  # re-run 5% of them
            again = call(req)
            if again is None or again.get("results", {}).get("nova", {}).get("predictions") != resp["results"]["nova"]["predictions"]:
                counts["deterministic_mismatch"] += 1
    by_mutation: Dict[str, int] = {}
    for name, req in invalid:
        resp = call(req)
        rejected = resp is not None and resp.get("status") == "error"
        counts["invalid_rejected" if rejected else "invalid_accepted"] += 1
        if not rejected:
            by_mutation[name] = by_mutation.get(name, 0) + 1
    # fixed rule: the 20 smallest sha256 values
    traces = sorted(accepted_ids, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:20]
    return {**counts, "accepted_invalid_by_mutation": by_mutation, "trace_sample": traces,
            "passed": counts["crashes"] == 0 and counts["malformed"] == 0 and counts["invalid_accepted"] == 0}
