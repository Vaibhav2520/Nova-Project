"""Split tests: the audit passes, catches planted leaks, enforces sealing, and generation is deterministic."""

import json
import os
import unittest
from unittest import mock

from nova import splits as S
from nova import tasks as T
from nova.tests import _common as H


def fresh(domain="sequence", ledger=None):
    return S.Splits(domain, max_depth=2, k_min=2, k_max=4, queries=2, eval_tasks=6, ledger_path=ledger)


class TestSplits(unittest.TestCase):
    def test_audit_passes_on_every_domain(self):
        for d in ("sequence", "grid", "graph"):
            with self.subTest(domain=d):
                self.assertTrue(S.split_audit(fresh(d), samples=60)["passed"])

    def test_audit_catches_a_program_leak(self):
        sp = fresh()
        sp.programs["train"] = sp.programs["train"] + sp.programs["comp_ood"][:1]
        with self.assertRaises(S.SplitLeakageError):
            S.split_audit(sp, samples=20)

    def test_audit_catches_a_behaviour_leak(self):
        sp = fresh()
        # identity map = the training "copy"
        sp.programs["comp_ood"] = sp.programs["comp_ood"] + [(("map_affine", (1, 0)),)]
        with self.assertRaises(S.SplitLeakageError):
            S.split_audit(sp, samples=20)

    def test_audit_catches_an_incomplete_signature(self):
        original = T.program_signature
        def lazy(domain, program, regime):
            return original(domain, program, {})
        with mock.patch.object(T, "program_signature", lazy):
            with self.assertRaises(S.SplitLeakageError):
                S.split_audit(fresh(), samples=20)

    def test_sealed_split_needs_final_eval(self):
        ledger = os.path.join(H.tmpdir(), "ledger.jsonl")
        sp = fresh(ledger=ledger)
        with self.assertRaises(S.SealedSplitError):
            sp.eval_set("family_ood")
        self.assertEqual(len(sp.final_eval_set("unit test")), 6)
        with open(ledger) as fh:
            self.assertEqual(json.loads(fh.readline())["event"], "family_ood_unsealed")

    def test_generation_is_deterministic(self):
        a, b = fresh(), fresh()
        self.assertEqual(a.train_task(0, 5).instance_hash(), b.train_task(0, 5).instance_hash())
        self.assertNotEqual(a.train_task(0, 5).instance_hash(), a.train_task(0, 6).instance_hash())

    def test_training_skips_evaluation_instances(self):
        sp = fresh()
        clash = sp._task("train", 0, 3, 0).instance_hash()
        sp._excluded = sp.excluded_hashes() | {clash}
        self.assertNotEqual(sp.train_task(0, 3).instance_hash(), clash)

    def test_manifest_is_stable(self):
        self.assertEqual(fresh().manifest()["manifest_hash"], fresh().manifest()["manifest_hash"])

    def test_held_out_rules_cover_both_orders(self):
        self.assertTrue(S.is_held_out("sequence", (("dedup", ()), ("map_affine", (3, 1)))))
        self.assertTrue(S.is_held_out("sequence", (("map_affine", (3, 1)), ("dedup", ()))))


if __name__ == "__main__":
    unittest.main()
