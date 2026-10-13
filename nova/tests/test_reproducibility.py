"""Reproducibility and statistical tests: same-seed replay, seed sensitivity, dependency audit and control transforms."""

import os
import unittest

import torch

from nova import audit as A
from nova import config as C
from nova import evaluate as E
from nova import metrics as M
from nova.train import Trainer
from nova.tests import _common as H

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def losses(seed: int, steps: int = 3):
    cfg = C.replace(C.smoke_config("nova"), train={"seed": seed, "updates": steps, "budgets": (100,)})
    t = Trainer(cfg, out_dir=H.tmpdir(), run_name="r")
    return [t.train_step() for _ in range(steps)]


class TestReproducibility(unittest.TestCase):
    def test_same_seed_replay_is_exact(self):
        self.assertEqual(losses(0), losses(0))

    def test_different_seeds_differ(self):
        self.assertNotEqual(losses(0), losses(1))

    def test_dependency_audit_passes(self):
        nb = os.path.join(ROOT, "NOVA_Research.ipynb")
        self.assertTrue(A.run_audit(os.path.join(ROOT, "nova"), nb)["passed"])

    def test_label_shuffle_breaks_the_rule(self):
        ts = H.tasks(3)
        shuffled = E.label_shuffle(ts)
        changed = sum(a.supports[0][1] != b.supports[0][1] for a, b in zip(ts, shuffled))
        self.assertGreater(changed, 0)

    def test_support_ablation_removes_demos(self):
        self.assertTrue(all(len(t.supports) == 0 for t in E.support_ablation(H.tasks(2))))

    def test_statistics_are_repeatable(self):
        a, b = torch.rand(30), torch.rand(30)
        self.assertEqual(M.paired_bootstrap(a, b, 500, seed=3), M.paired_bootstrap(a, b, 500, seed=3))
        self.assertEqual(M.seed_summary([1, 2, 3])["median"], 2.0)


if __name__ == "__main__":
    unittest.main()
