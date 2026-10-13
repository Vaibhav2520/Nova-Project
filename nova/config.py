"""All settings for a NOVA run in one place, plus a fingerprint (hash) of those settings."""

import dataclasses
import hashlib
import json

CONFIG_SCHEMA = "nova.config/1"


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 96  # number of different tokens (80 values + 16 special tokens)
    d_model: int = 96  # width: chosen so NOVA has about 1.0M parameters (the preregistered size)
    n_layers: int = 2
    n_heads: int = 4
    mlp_mult: int = 4
    dropout: float = 0.0
    k_max: int = 8
    t_max_train: int = 4
    t_max_eval: int = 6
    read_option: str = "A"  # "A" = executor can look back at the input, "B" = it cannot
    tau_w: float = 1.0
    tau_v: float = 1.0
    eps_halt: float = 0.05  # stop refining when the fit error drops below this
    delta_halt: float = 0.005  # stop refining when the fit error improves by less than this
    clip_c: float = 1.0
    mode: str = "full"  # which parts of the operator are switched on (see operator.MODES)
    pos_encoding: str = "alibi"
    n_loops: int = 1  # how many times Transformer layers are repeated (Transformer-CM uses > 1)
    # hard ceiling on answer length (the real limit comes from the demos, see TaskBatch.answer_cap)
    max_out_len: int = 480
    lambda_self: float = 0.1  # weight of the "each guess explains its own demo" loss
    lambda_cal: float = 0.1  # weight of the confidence-head loss


@dataclasses.dataclass(frozen=True)
class DataConfig:
    domain: str = "sequence"
    k_min: int = 2
    k_max: int = 8
    queries: int = 4
    eval_tasks: int = 1000
    max_depth: int = 3


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    model_name: str = "nova"
    seed: int = 0
    updates: int = 10000
    batch_tasks: int = 32
    micro_batch: int = 0  # tasks per forward pass (0 = same as batch_tasks)
    lr: float = 1e-3
    warmup: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: str = "auto"  # mixed precision: "auto", "off", "bf16" or "fp16"
    checkpoint_minutes: float = 15.0
    budgets: tuple = (156, 312, 625, 1250, 2500, 5000, 10000)  # steps where the learning curve is measured
    eval_tasks_per_budget: int = 200
    out_dir: str = "runs"
    deterministic: bool = True
    gate: str = "none"  # continual learning gate: none, rgp, loss, lowlr or frozen
    replay: str = "none"  # replay memory: none, reservoir or priority
    replay_cap: int = 512
    replay_frac: float = 0.1
    gate_eps: float = 0.1  # residual below which a task barely changes the weights (RGP)
    gate_kappa: float = 1.0  # how quickly the gate opens above gate_eps
    lowlr_scale: float = 0.5  # learning-rate factor of the lower-learning-rate control (set to the average RGP gate)
    log_every: int = 0  # print progress every this many steps (0 = quiet)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)


def to_dict(cfg) -> dict:
    """Convert a settings dataclass into a dict of plain Python values."""
    return dataclasses.asdict(cfg)


def canonical_json(obj) -> str:
    """Turn a dict into JSON with sorted keys and no spaces, so the same settings always give the same text."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=list)


def config_hash(cfg) -> str:
    """Return a sha256 fingerprint of the settings."""
    data = cfg if isinstance(cfg, dict) else to_dict(cfg)
    return "sha256:" + hashlib.sha256(canonical_json(data).encode()).hexdigest()


def from_dict(data: dict) -> RunConfig:
    """Rebuild a RunConfig from a dict, for example one stored inside a checkpoint."""
    train = dict(data["train"])
    train["budgets"] = tuple(train["budgets"])  # JSON stores tuples as lists, so turn it back into a tuple
    return RunConfig(model=ModelConfig(**data["model"]), data=DataConfig(**data["data"]), train=TrainConfig(**train))


def replace(cfg: RunConfig, **groups) -> RunConfig:
    """Return a copy of cfg where e.g. model={'d_model': 64} overrides only those fields."""
    model = dataclasses.replace(cfg.model, **groups.get("model", {}))
    data = dataclasses.replace(cfg.data, **groups.get("data", {}))
    train = dataclasses.replace(cfg.train, **groups.get("train", {}))
    return RunConfig(model=model, data=data, train=train)


def smoke_config(model_name: str = "nova", domain: str = "sequence") -> RunConfig:
    """Tiny settings that run end to end on a CPU in seconds."""
    return RunConfig(
        model=ModelConfig(d_model=32, n_layers=1, n_heads=2, k_max=4, t_max_train=2, t_max_eval=3, max_out_len=48),
        data=DataConfig(domain=domain, k_min=2, k_max=4, queries=2, eval_tasks=16, max_depth=2),
        train=TrainConfig(model_name=model_name, updates=20, batch_tasks=4, warmup=5, budgets=(10, 20),
                          eval_tasks_per_budget=8, checkpoint_minutes=1.0, amp="off"),
    )


def reduced_config(model_name: str = "nova", domain: str = "sequence") -> RunConfig:
    """Reduced settings for the end-to-end experiment that must finish in 90 GPU minutes."""
    return RunConfig(
        model=ModelConfig(d_model=96, n_layers=1, t_max_train=3, t_max_eval=4),
        data=DataConfig(domain=domain, eval_tasks=200),
        train=TrainConfig(model_name=model_name, updates=600, batch_tasks=16, warmup=60, log_every=100,
                          budgets=(75, 150, 300, 600), eval_tasks_per_budget=40),
    )
