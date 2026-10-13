"""NOVA package: the Cross-Execution Consensus model, its baselines, puzzles, training and tools."""

import hashlib
import os

__version__ = "0.1.0"


def code_hash() -> str:
    """Return a sha256 fingerprint of every .py file in the nova package (sorted, so order never changes)."""
    digest = hashlib.sha256()
    root = os.path.dirname(os.path.abspath(__file__))
    for folder, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(folder, name)
                digest.update(os.path.relpath(path, root).encode())
                with open(path, "rb") as handle:
                    digest.update(handle.read())
    return "sha256:" + digest.hexdigest()
