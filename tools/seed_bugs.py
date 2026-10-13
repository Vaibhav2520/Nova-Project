"""Mutation drill: plant each of the brief's eight seeded-bug types in a copy of nova/ and check that a test fails."""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MUTATIONS = [  # (name, file, original text, broken text) for each bug type in the brief
    ("future-token leak through an incorrect mask", "nova/baselines.py",
     "diagonal=1)", "diagonal=2)"),
    ("detached state blocks long-horizon credit", "nova/operator.py",
     "base + alpha.to(base.dtype) * delta, z)", "base + alpha.to(base.dtype) * delta, z).detach()"),
    ("resume without optimizer state", "nova/train.py",
     'self.opt.load_state_dict(p["optimizer"])', "pass"),
    ("batch-dependent prediction from normalising over the wrong axis", "nova/model.py",
     "x = norm(x + mix(torch.cat([hf, hb], dim=-1)))",
     "x = norm(x + mix(torch.cat([hf, hb], dim=-1))); x = x - x.mean(0, keepdim=True)"),
    ("mixed-precision underflow in routing/halting probabilities", "nova/operator.py",
     "scores = to_fp32(scores).masked_fill(~mask, -1e9)", "scores = scores.to(torch.bfloat16).masked_fill(~mask, -1e4)"),
    ("train/eval mismatch from stochastic behaviour", "nova/model.py",
     "h = h.index_put((b, k), self.dropout(g).to(h.dtype))", "h = h.index_put((b, k), F.dropout(g, 0.2, training=True).to(h.dtype))"),
    ("split leakage from an incomplete program signature", "nova/tasks.py",
     '"program": [[n, list(p)] for n, p in program], "regime": regime}', '"program": [[n, list(p)] for n, p in program], "regime": {}}'),
    # needs a second edit (EXTRA) so the request is really accepted
    ("backend accepts a target-bearing inference request", "nova/interface.py",
     "if any(k in q for k in TARGET_KEYS):", "if False:"),
]
EXTRA = {"backend accepts a target-bearing inference request": ("nova/interface.py",
         'if set(q) - {"input"}:', 'if set(q) - {"input", "output"}:')}


def apply(root: str, path: str, old: str, new: str) -> None:
    full = os.path.join(root, path)
    with open(full) as fh:
        text = fh.read()
    if text.count(old) != 1:
        raise RuntimeError(f"mutation target not found in {path}: {old[:60]}")
    with open(full, "w") as fh:
        fh.write(text.replace(old, new))


def run_drill() -> list:
    results = []
    for name, path, old, new in MUTATIONS:
        tmp = tempfile.mkdtemp(prefix="nova_bug_")
        shutil.copytree(os.path.join(ROOT, "nova"), os.path.join(tmp, "nova"), ignore=shutil.ignore_patterns("__pycache__"))
        apply(tmp, path, old, new)
        if name in EXTRA:
            apply(tmp, *EXTRA[name])
        proc = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", "discover", "-s", "nova/tests", "-t", "."],
                              cwd=tmp, capture_output=True, text=True, timeout=1200)
        failed = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+)", proc.stderr, re.M)))
        results.append({"bug": name, "caught": proc.returncode != 0, "failing_tests": failed})
        shutil.rmtree(tmp, ignore_errors=True)
    return results


if __name__ == "__main__":
    rows = run_drill()
    for r in rows:
        mark = "CAUGHT " if r["caught"] else "MISSED "
        print(f"{mark} {r['bug']}\n         by: {', '.join(r['failing_tests'][:4]) or '-'}")
    missed = [r for r in rows if not r["caught"]]
    print(f"\n{len(rows) - len(missed)}/{len(rows)} seeded bugs caught")
    sys.exit(1 if missed else 0)
