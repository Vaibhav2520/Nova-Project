"""Dependency audit: only the Python standard library and PyTorch (plus the display bridge in the notebook)."""

import ast
import json
import os
import sys

ALLOWED = set(sys.stdlib_module_names) | {"torch", "nova", "__future__"}
NOTEBOOK_ONLY = {"IPython", "google"}  # display bridge, allowed only in the notebook layer (and console.py)
FORBIDDEN_ATTRS = {"numpy", "from_numpy"}  # e.g. tensor.numpy() or torch.from_numpy(...)


class DependencyViolation(RuntimeError):
    """A forbidden import or call was found."""


def audit_source(text: str, where: str, display_ok: bool = False) -> list:
    tree = ast.parse(text)
    allowed = ALLOWED | (NOTEBOOK_ONLY if display_ok else set())
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            names = []
        for name in names:
            if name.split(".")[0] not in allowed:
                problems.append(f"{where}:{node.lineno}: forbidden import {name}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"{where}:{node.lineno}: forbidden call .{node.attr}")
    return problems


def audit_package(root: str) -> dict:
    problems = []
    for folder, _, files in os.walk(root):
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(folder, name)
                with open(path) as fh:
                    problems += audit_source(fh.read(), path, display_ok=name == "console.py")
    return {"passed": not problems, "problems": problems}


def audit_notebook(path: str) -> dict:
    with open(path) as fh:
        nb = json.load(fh)
    problems = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            # drop magics
            lines = [l for l in "".join(cell["source"]).splitlines() if not l.lstrip().startswith(("%", "!"))]
            problems += audit_source("\n".join(lines), f"cell{i}", display_ok=True)
    return {"passed": not problems, "problems": problems}


def run_audit(package_root: str, notebook: str = None) -> dict:
    report = audit_package(package_root)
    if notebook and os.path.exists(notebook):
        nb = audit_notebook(notebook)
        report = {"passed": report["passed"] and nb["passed"], "problems": report["problems"] + nb["problems"]}
    if not report["passed"]:
        raise DependencyViolation("\n".join(report["problems"]))
    return report
