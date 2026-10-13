"""Training: determinism, hardware and precision policy, deterministic data, continual gates, replay, checkpoints and ledger."""

import dataclasses
import math
import os
import platform
import random
import time
from typing import Callable, List, Optional

import torch

from . import code_hash
from . import tasks as T
from .baselines import build_model, match_compute, match_parameters, measure_flops
from .checkpoint import (CKPT_SCHEMA, CheckpointError, append_jsonl, capture_rng, load_checkpoint, restore_rng,
                         save_checkpoint, sha256_file)
from .config import RunConfig, config_hash, to_dict
from .evaluate import evaluate_tasks
from .model import NovaModel, count_parameters
from .splits import EVAL_SPLITS, Splits


class DivergedError(FloatingPointError):
    """Training produced a non-finite loss."""


def set_determinism(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for repeatable GPU matrix products
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False  # no automatic kernel search (it changes results between runs)


def pick_device(preference: str = "auto") -> torch.device:
    if preference == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def amp_dtype(policy: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda" or policy == "off":
        return None
    if policy == "bf16" or (policy == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16  # otherwise float16 (with a GradScaler)


def environment() -> dict:
    env = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()}
    if torch.cuda.is_available():
        env.update({"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                    "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)})
    return env


# Data streams are index-based, so resume is exact.
class StaticData:
    def __init__(self, splits: Splits, seed: int):
        self.splits, self.seed = splits, seed

    def task(self, index: int) -> T.Task:
        return self.splits.train_task(self.seed, index)

    def eval_sets(self, n: int) -> dict:
        return {s: self.splits.eval_set(s, n) for s in EVAL_SPLITS}


class StreamData:
    def __init__(self, cfg: RunConfig, seed: int, ramp: int = 2000):
        d = cfg.data
        self.splits = {dom: Splits(dom, d.max_depth, d.k_min, d.k_max, d.queries, d.eval_tasks) for dom in T.RULES}
        self.seed, self.ramp = seed, ramp
        self.seg_len = max(1, cfg.train.updates * cfg.train.batch_tasks // len(T.STREAM_SEGMENTS))
        self.cfg = cfg
        self._evals: dict = {}
        self._excluded: Optional[set] = None

    def _programs(self, domain: str, family: str):
        return [p for p in self.splits[domain].programs["train"] if family in T.program_families(domain, p)]

    def _family_task(self, domain, family, *key):
        rng = T.seeded_rng("stream", domain, family, *key)
        d = self.cfg.data
        return T.make_task(domain, rng.choice(self._programs(domain, family)), "train", rng, d.k_min, d.k_max, d.queries)

    def task(self, index: int) -> T.Task:
        rng = T.seeded_rng("stream-pos", self.seed, index)
        domain, family = T.stream_choice(index, self.seg_len, self.ramp, rng)
        attempt = 0
        while True:
            t = self._family_task(domain, family, "train", self.seed, index, attempt)
            if t.instance_hash() not in self.excluded():
                return t
            attempt += 1

    def eval_sets(self, n: int) -> dict:
        for domain, family, _ in T.STREAM_SEGMENTS:
            key = f"{domain}:{family}"
            if key not in self._evals:
                self._evals[key] = [self._family_task(domain, family, "eval", i) for i in range(n)]
        return self._evals

    def excluded(self) -> set:
        if self._excluded is None:
            self._excluded = {t.instance_hash() for ts in self.eval_sets(self.cfg.train.eval_tasks_per_budget).values() for t in ts}
        return self._excluded


class Replay:
    def __init__(self, kind: str, cap: int):
        self.kind, self.cap = kind, cap
        self.items: List[list] = []
        self.seen = 0

    def add(self, index: int, priority: float, rng: random.Random) -> None:
        if self.kind == "none" or self.cap <= 0:
            return
        self.seen += 1
        if len(self.items) < self.cap:
            self.items.append([index, priority])
        elif self.kind == "reservoir":
            j = rng.randrange(self.seen)
            if j < self.cap:
                self.items[j] = [index, priority]
        else:  # priority: keep the most surprising tasks
            low = min(range(len(self.items)), key=lambda i: self.items[i][1])
            if priority > self.items[low][1]:
                self.items[low] = [index, priority]

    def sample(self, n: int, rng: random.Random) -> List[int]:
        return [rng.choice(self.items)[0] for _ in range(min(n, len(self.items)))] if self.items else []

    def state_dict(self) -> dict:
        return {"kind": self.kind, "cap": self.cap, "items": self.items, "seen": self.seen}

    def load_state_dict(self, s: dict) -> None:
        self.items, self.seen = [list(i) for i in s["items"]], s["seen"]


def loo_view(batch, targets=None):  # leave-one-out view: the last real demonstration becomes the question
    """Used by the loss-gated control: how well does a baseline predict one demo from the others?"""
    kv = batch.support_mask.sum(1).clamp_min(1)
    last = kv - 1
    rows = torch.arange(batch.size, device=kv.device)
    mask = batch.support_mask.clone()
    mask[rows, last] = False
    q_in = batch.support_inputs[rows, last].unsqueeze(1)
    new = dataclasses.replace(batch, support_mask=mask, query_inputs=q_in, query_input_mask=q_in != T.PAD,
                              query_mask=(kv > 1).unsqueeze(1))  # tasks with a single demo get no question
    y = batch.support_outputs[rows, last].unsqueeze(1)
    return new, T.TargetBatch(y, y != T.PAD)


def resolve_model_config(name: str, cfg: RunConfig, sample_tasks: List[T.Task]) -> RunConfig:
    """Match every learned baseline to NOVA's parameter count (+/-5%); match Transformer-CM to NOVA's training FLOPs."""
    target = count_parameters(NovaModel(cfg.model))
    model = cfg.model
    if name in ("transformer", "transformer_cm", "recurrent", "ablated_more_params"):
        model = match_parameters("transformer" if name == "transformer_cm" else name, model, target)
    if name == "transformer_cm":
        batch, tgt = T.collate(sample_tasks)
        torch.manual_seed(0)
        nova_flops = measure_flops(NovaModel(cfg.model), batch, tgt)
        model = match_compute(model, nova_flops, batch, tgt)
    return dataclasses.replace(cfg, model=model, train=dataclasses.replace(cfg.train, model_name=name))


class Trainer:
    def __init__(self, cfg: RunConfig, out_dir: Optional[str] = None, device=None, experiment: str = "adhoc",
                 data=None, run_name: Optional[str] = None):
        self.cfg, tr = cfg, cfg.train
        self.device = torch.device(device) if device else pick_device()
        set_determinism(tr.seed, tr.deterministic)
        self.out_dir = out_dir or tr.out_dir
        self.run_name = run_name or f"{tr.model_name}_{cfg.data.domain}_s{tr.seed}"
        self.ckpt_path = os.path.join(self.out_dir, "checkpoints", self.run_name, "last.pt")
        self.ledger = os.path.join(self.out_dir, "ledger", "experiments.jsonl")
        self.experiment = experiment
        if data is not None:
            self.data = data
        elif cfg.data.domain == "stream":
            self.data = StreamData(cfg, tr.seed)
        else:
            d = cfg.data
            self.data = StaticData(Splits(d.domain, d.max_depth, d.k_min, d.k_max, d.queries, d.eval_tasks, self.ledger), tr.seed)
        self.model = build_model(tr.model_name, cfg.model).to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        lr = tr.lr * (tr.lowlr_scale if tr.gate == "lowlr" else 1.0)
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=tr.weight_decay) if params else None
        warm, total = max(1, tr.warmup), max(1, tr.updates)
        # warm-up then cosine
        sched_fn = lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, sched_fn) if self.opt else None
        self.amp = amp_dtype(tr.amp, self.device)
        # only float16 needs scaling
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp == torch.float16)
        self.micro = tr.micro_batch or tr.batch_tasks
        self.replay = Replay(tr.replay, tr.replay_cap)
        self.step, self.examples, self.curve, self.events = 0, 0, [], []
        self.last_save = time.time()

    def state(self) -> dict:
        return {"schema": CKPT_SCHEMA, "model_name": self.cfg.train.model_name, "model": self.model.state_dict(),
                "optimizer": self.opt.state_dict() if self.opt else {}, "scheduler": self.sched.state_dict() if self.sched else {},
                "scaler": self.scaler.state_dict(), "rng": capture_rng(),
                "data_state": {"step": self.step, "replay": self.replay.state_dict(), "micro": self.micro},
                "config": to_dict(self.cfg), "config_hash": config_hash(self.cfg), "code_hash": code_hash(),
                "step": self.step, "examples_seen": self.examples, "curve": self.curve, "events": self.events}

    def save(self) -> dict:
        manifest = save_checkpoint(self.ckpt_path, self.state())
        self.last_save = time.time()
        return manifest

    def resume(self) -> bool:
        if not os.path.exists(self.ckpt_path):
            return False
        try:  # a checkpoint from other settings (or a damaged one) is set aside: never used, never fatal
            p = load_checkpoint(self.ckpt_path, map_location=self.device, expect_config_hash=config_hash(self.cfg))
        except CheckpointError as err:
            stale = f"{self.ckpt_path}.unusable-{int(time.time())}"
            os.replace(self.ckpt_path, stale)
            if os.path.exists(self.ckpt_path + ".json"):
                os.replace(self.ckpt_path + ".json", stale + ".json")
            self.events.append({"step": 0, "event": "checkpoint_set_aside", "reason": str(err), "moved_to": stale})
            print(f"  [{self.run_name}] old checkpoint not usable ({err}); moved aside, starting fresh", flush=True)
            return False
        self.model.load_state_dict(p["model"])
        if self.opt:
            self.opt.load_state_dict(p["optimizer"])
            self.sched.load_state_dict(p["scheduler"])
        self.scaler.load_state_dict(p["scaler"])
        self.replay.load_state_dict(p["data_state"]["replay"])
        self.micro = p["data_state"]["micro"]
        self.step, self.examples = p["step"], p["examples_seen"]
        self.curve, self.events = list(p["curve"]), list(p["events"])
        restore_rng(p["rng"])
        return True

    def _gate(self, batch, target) -> Optional[object]:
        tr = self.cfg.train
        clamp = lambda x: ((x - tr.gate_eps) / tr.gate_kappa).clamp(0.0, 1.0)  # gate shape: 0 below eps, rising to 1
        if tr.gate == "rgp":
            if not isinstance(self.model, NovaModel):
                raise ValueError("gate 'rgp' needs a NOVA model; use 'loss' for baselines")
            return lambda rho: clamp(rho.float())  # computed inside the forward pass from the fit error
        if tr.gate == "loss":
            with torch.no_grad():
                v_batch, v_tgt = loo_view(batch)
                _, info = self.model.loss(v_batch, v_tgt)
            return clamp(info["per_task_loss"].float())
        return None

    def _batch_indices(self, step: int) -> List[int]:
        B = self.cfg.train.batch_tasks
        new = list(range(step * B, (step + 1) * B))
        n_rep = int(self.cfg.train.replay_frac * B) if self.replay.kind != "none" else 0
        rep = self.replay.sample(n_rep, random.Random(step))
        return new[:B - len(rep)] + rep

    def train_step(self) -> float:
        frozen = self.cfg.train.gate == "frozen" and isinstance(self.data, StreamData) and self.examples >= self.data.seg_len
        # the heuristic never learns; "frozen" stops learning after the first stream segment
        if self.opt is None or frozen:
            self.step += 1
            self.examples += self.cfg.train.batch_tasks
            return 0.0  # no update, so adaptation can only happen in context
        indices = self._batch_indices(self.step)
        tasks = [self.data.task(i) for i in indices]
        while True:  # retry with smaller micro-batches after out-of-memory errors
            try:
                loss = self._accumulate(tasks, indices)
                break
            except torch.cuda.OutOfMemoryError:
                if self.micro == 1:
                    raise
                torch.cuda.empty_cache()
                self.opt.zero_grad(set_to_none=True)
                self.micro = max(1, self.micro // 2)  # halve the micro-batch (the effective batch stays the same)
                self.events.append({"step": self.step, "event": "oom", "micro": self.micro})
        if not math.isfinite(loss):
            raise DivergedError(f"non-finite loss at step {self.step}")
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        self.step += 1
        self.examples += len(tasks)
        return loss

    def _accumulate(self, tasks, indices) -> float:
        self.opt.zero_grad(set_to_none=True)
        total, rng = 0.0, random.Random(self.step)
        for i in range(0, len(tasks), self.micro):
            chunk, idx = tasks[i:i + self.micro], indices[i:i + self.micro]
            batch, target = T.collate(chunk, self.device)
            gate = self._gate(batch, target)
            with torch.autocast(self.device.type, dtype=self.amp or torch.float32, enabled=self.amp is not None):
                loss, info = self.model.loss(batch, target, task_weights=gate)
            share = len(chunk) / len(tasks)
            self.scaler.scale(loss * share).backward()  # gradients add up to the full-batch gradient
            total += loss.item() * share
            prio = info.get("rho_star", info["per_task_loss"]).float().cpu().tolist()
            for j, p in zip(idx, prio):
                self.replay.add(j, p, rng)
        return total

    def evaluate_now(self) -> dict:
        sets = self.data.eval_sets(self.cfg.train.eval_tasks_per_budget)
        point = {"step": self.step, "examples": self.examples}
        for name, tasks in sets.items():
            res = evaluate_tasks(self.model, tasks, batch_size=min(8, self.cfg.train.batch_tasks), device=self.device)
            point[name] = res["task_acc"]
        self.model.train()
        return point

    def run(self, max_steps: Optional[int] = None) -> dict:
        tr = self.cfg.train
        resumed = self.resume()
        record = {"run_id": self.run_name, "experiment": self.experiment, "model": tr.model_name, "domain": self.cfg.data.domain,
                  "seed": tr.seed, "config_hash": config_hash(self.cfg), "code_hash": code_hash(), "environment": environment(),
                  "start_state": f"resumed:{self.step}" if resumed else "fresh", "start_time": time.time()}
        if isinstance(self.data, StaticData):
            record["split_manifest_hash"] = self.data.splits.manifest()["manifest_hash"]
        self.model.train()
        stop_at = tr.updates if max_steps is None else min(tr.updates, self.step + max_steps)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.time()
            while self.step < stop_at:
                loss = self.train_step()
                if tr.log_every and self.step % tr.log_every == 0:
                    print(f"  [{self.run_name}] step {self.step}/{tr.updates}  loss {loss:.3f}  {(time.time() - started) / 60:.1f} min", flush=True)
                if self.step in tr.budgets:
                    self.curve.append(self.evaluate_now())
                    self.save()
                elif time.time() - self.last_save > tr.checkpoint_minutes * 60:
                    self.save()
            manifest = self.save()
            record.update(end_state="completed" if self.step >= tr.updates else "paused", checkpoint={"path": self.ckpt_path, "sha256": manifest["sha256"]})
        except KeyboardInterrupt:  # stopped by the user: never reported as success
            record.update(end_state="cancelled", failure_reason="KeyboardInterrupt")
            raise
        except DivergedError as err:
            record.update(end_state="diverged", failure_reason=str(err))
            raise
        except Exception as err:
            record.update(end_state="failed", failure_reason=f"{type(err).__name__}: {err}")
            raise
        finally:  # always write the ledger line
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            record.update(end_time=time.time(), step=self.step, examples=self.examples, params=count_parameters(self.model),
                          peak_mem_bytes=peak, micro_batch=self.micro, events=self.events, curve=self.curve)
            append_jsonl(self.ledger, record)
        return record


def train_run(cfg: RunConfig, out_dir: str = "runs", experiment: str = "adhoc", device=None, max_steps=None) -> dict:
    """Match the model fairly, train it (resuming if possible), and return the ledger record."""
    d = cfg.data
    # the stream is matched on sequence tasks (sizes do not depend on domain)
    domain = "sequence" if d.domain == "stream" else d.domain
    sample = [Splits(domain, d.max_depth, d.k_min, d.k_max, d.queries, 8).train_task(cfg.train.seed, i) for i in range(4)]
    from .baselines import MATCH_LOG
    MATCH_LOG.clear()
    cfg = resolve_model_config(cfg.train.model_name, cfg, sample)
    trainer = Trainer(cfg, out_dir=out_dir, device=device, experiment=experiment)
    if MATCH_LOG:  # Transformer-CM: record how close the compute match really is
        trainer.events.append({"step": 0, "event": "compute_match", **MATCH_LOG})
    return trainer.run(max_steps=max_steps)
