"""Crash-safe checkpoints (atomic save, integrity check, compatibility check) and the JSON Lines ledger."""

import hashlib
import json
import os
import random
import time

import torch

CKPT_SCHEMA = "nova.checkpoint/1"
REQUIRED_KEYS = ("schema", "model", "optimizer", "scheduler", "scaler", "rng", "data_state",
                 "config", "config_hash", "code_hash", "step", "examples_seen")


class CheckpointError(Exception):
    """Something is wrong with a checkpoint."""


class CorruptCheckpointError(CheckpointError):
    """The checkpoint file is unreadable or does not match its recorded sha256."""


class IncompatibleCheckpointError(CheckpointError):
    """The checkpoint is readable but cannot be used with this code or these settings."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _fsync_write(path: str, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def capture_rng() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(_as_tuple(state["python"]))  # saved tuples may come back as lists
    torch.set_rng_state(state["torch"].cpu())  # must be a CPU tensor, even after map_location="cuda"
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _as_tuple(x):
    return tuple(_as_tuple(i) for i in x) if isinstance(x, (list, tuple)) else x


def save_checkpoint(path: str, payload: dict) -> dict:
    """Atomically save payload: write a temp file, fsync it, then rename. A sidecar manifest stores its sha256."""
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise IncompatibleCheckpointError(f"refusing to save, missing: {missing}")
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"  # same folder, so the rename is atomic
    torch.save(payload, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    manifest = {"schema": CKPT_SCHEMA, "sha256": sha256_file(path), "step": payload["step"],
                "config_hash": payload["config_hash"], "code_hash": payload["code_hash"], "time": time.time()}
    _fsync_write(path + ".json.tmp", json.dumps(manifest, indent=2).encode())
    os.replace(path + ".json.tmp", path + ".json")
    return manifest


def load_checkpoint(path: str, map_location="cpu", expect_config_hash=None, required=REQUIRED_KEYS) -> dict:
    """Load a checkpoint after checking it exists, matches its sha256, has the right schema and all required parts."""
    if not os.path.exists(path):
        raise CorruptCheckpointError(f"checkpoint not found: {path}")
    manifest_path = path + ".json"
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        if manifest.get("sha256") != sha256_file(path):
            raise CorruptCheckpointError("checkpoint bytes do not match the recorded sha256")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as err:
        raise CorruptCheckpointError(f"cannot read checkpoint: {type(err).__name__}: {err}") from err
    if not isinstance(payload, dict) or payload.get("schema") != CKPT_SCHEMA:
        raise IncompatibleCheckpointError(f"unsupported checkpoint schema: {payload.get('schema') if isinstance(payload, dict) else None}")
    missing = [k for k in required if k not in payload]
    if missing:
        raise IncompatibleCheckpointError(f"checkpoint is missing: {missing}")
    if expect_config_hash and payload.get("config_hash") != expect_config_hash:
        raise IncompatibleCheckpointError("checkpoint was made with different settings")
    return payload


def append_jsonl(path: str, record: dict) -> None:
    """Append one JSON object as a line and fsync it, so ledger lines are never half-written or lost."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
