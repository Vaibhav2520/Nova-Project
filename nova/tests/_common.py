"""Small shared helpers so every test file builds the same tiny models and tasks."""

import functools
import os
import tempfile

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # repeatable GPU maths if a GPU is present

from nova import config as C
from nova import tasks as T
from nova.baselines import build_model
from nova.splits import Splits

CFG = C.smoke_config().model


@functools.lru_cache(maxsize=None)
def splits(domain: str = "sequence") -> Splits:
    return Splits(domain, max_depth=2, k_min=2, k_max=4, queries=2, eval_tasks=6)


def tasks(n: int = 3, domain: str = "sequence", start: int = 0):
    sp = splits(domain)
    return [sp.train_task(0, i) for i in range(start, start + n)]


def batch(n: int = 3, domain: str = "sequence", start: int = 0):
    return T.collate(tasks(n, domain, start))


def model(name: str = "nova", seed: int = 0, double: bool = False, **cfg_changes):
    torch.manual_seed(seed)
    cfg = C.replace(C.RunConfig(model=CFG), model=cfg_changes).model if cfg_changes else CFG
    m = build_model(name, cfg)
    return m.double() if double else m


def tmpdir() -> str:
    return tempfile.mkdtemp(prefix="nova_test_")
