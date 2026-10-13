"""Integration and contract tests: fresh import, exact resume, ledger outcomes, backend contracts and the console."""

import json
import os
import subprocess
import sys
import unittest
from unittest import mock

import torch

from nova import config as C
from nova import console as CON
from nova import tasks as T
from nova.interface import ResearchBackend, run_hidden_eval
from nova.train import DivergedError, Trainer
from nova.tests import _common as H

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def request(**task_changes):
    r = json.loads(json.dumps(CON.EXAMPLE_REQUEST))
    r["settings"]["max_steps"] = 2
    r["task"].update(task_changes)
    return r


def backend():
    be = ResearchBackend()
    for n in ("nova", "transformer", "recurrent", "ablated"):
        be.register(n, H.model(n))
    return be


def tiny_cfg(**train):
    return C.replace(C.smoke_config("nova"), train={"updates": 6, "budgets": (100,), **train})


class TestIntegration(unittest.TestCase):
    def test_fresh_runtime_import(self):
        code = "import nova, nova.model, nova.train, nova.interface, nova.console, nova.evaluate; print(nova.code_hash())"
        out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertTrue(out.stdout.strip().startswith("sha256:"))

    def test_resume_reproduces_next_step_exactly(self):
        """Two steps are compared, plus every weight: the first loss after resume is computed before the optimizer
        update, so a missing optimizer state only shows up in the second loss (found by the seeded-bug drill)."""
        out = H.tmpdir()
        a = Trainer(tiny_cfg(), out_dir=out, run_name="r")
        a.run(max_steps=3)
        expected = [a.train_step(), a.train_step()]
        b = Trainer(tiny_cfg(), out_dir=out, run_name="r")
        self.assertTrue(b.resume())
        self.assertEqual([b.train_step(), b.train_step()], expected)
        for (name, pa), pb in zip(a.model.state_dict().items(), b.model.state_dict().values()):
            self.assertTrue(torch.equal(pa, pb), name)

    def test_checkpoint_from_other_settings_is_set_aside(self):
        out = H.tmpdir()
        Trainer(tiny_cfg(), out_dir=out, run_name="s").run(max_steps=2)
        t = Trainer(tiny_cfg(lr=5e-4), out_dir=out, run_name="s")
        self.assertFalse(t.resume())
        self.assertEqual(t.events[-1]["event"], "checkpoint_set_aside")
        self.assertTrue(os.path.exists(t.events[-1]["moved_to"]))

    def test_interruption_is_never_reported_as_success(self):
        out = H.tmpdir()
        t = Trainer(tiny_cfg(), out_dir=out, run_name="x")
        with mock.patch.object(t, "train_step", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                t.run()
        with open(t.ledger) as fh:
            self.assertEqual(json.loads(fh.readlines()[-1])["end_state"], "cancelled")

    def test_divergence_is_recorded(self):
        t = Trainer(tiny_cfg(), out_dir=H.tmpdir(), run_name="d")
        with mock.patch.object(t, "train_step", side_effect=DivergedError("nan")):
            with self.assertRaises(DivergedError):
                t.run()
        with open(t.ledger) as fh:
            line = json.loads(fh.readlines()[-1])
        self.assertEqual(line["end_state"], "diverged")
        self.assertTrue(line["failure_reason"])

    def test_backend_rejects_answers_inside_questions(self):
        r = request()
        r["task"]["query"][0]["output"] = [14, 2, 18, 24]
        resp = backend().infer(r)
        self.assertEqual(resp["errors"][0]["code"], "E_TARGET_IN_QUERY")

    def test_schema_versions(self):
        be = backend()
        old = request()
        old["schema_version"] = "nova.request/1.0"
        del old["settings"]["trace_level"]
        self.assertEqual(be.infer(old)["status"], "ok")
        new = request()
        new["schema_version"] = "nova.request/2.0"
        self.assertEqual(be.infer(new)["errors"][0]["code"], "E_SCHEMA_VERSION")

    def test_structured_errors_never_raise(self):
        be = backend()
        cases = [None, "text", {}, {"schema_version": "nova.request/1.1"}, request(support=[]),
                 request(query=[{"input": [999]}]), request(domain="audio"), {**request(), "surprise": 1}]
        for r in cases:
            resp = be.infer(r)
            self.assertEqual(resp["status"], "error", r)
            self.assertIn(resp["errors"][0]["code"], ("E_SCHEMA", "E_SCHEMA_VERSION", "E_EMPTY_SUPPORT", "E_TOKEN_RANGE"))

    def test_idempotent_and_no_training_during_inference(self):
        be = backend()
        before = {k: v.clone() for k, v in be.models["nova"].state_dict().items()}
        a, b = be.compare(request()), be.compare(request())
        strip = lambda r: json.dumps({m: {k: v for k, v in x.items() if k != "compute"} for m, x in r["results"].items()}, sort_keys=True)
        self.assertEqual(strip(a), strip(b))
        for k, v in be.models["nova"].state_dict().items():
            self.assertTrue(torch.equal(v.cpu(), before[k].cpu()))  # compare on the CPU: the backend may use the GPU

    def test_batching_does_not_change_answers(self):
        be = backend()
        r2 = request(support=[{"input": [5, 5, 1], "output": [1, 5, 5]}, {"input": [2, 9], "output": [2, 9]}], query=[{"input": [3, 3, 8]}])
        merged = be.infer_many([request(), r2])
        single = [be.compare(request()), be.compare(r2)]
        for m, s in zip(merged, single):
            self.assertEqual(m["results"]["nova"]["predictions"], s["results"]["nova"]["predictions"])

    def test_checkpoint_loading_contract(self):
        out = H.tmpdir()
        t = Trainer(tiny_cfg(), out_dir=out, run_name="c")
        t.run(max_steps=2)
        be = ResearchBackend()
        self.assertEqual(be.load_checkpoint(t.ckpt_path)["status"], "ok")
        with open(t.ckpt_path, "r+b") as fh:
            fh.seek(100)
            fh.write(b"garbage!")
        self.assertEqual(be.load_checkpoint(t.ckpt_path)["errors"][0]["code"], "E_CHECKPOINT_CORRUPT")

    def test_hidden_eval_entry_point(self):
        b, _ = H.batch(2)
        path = os.path.join(H.tmpdir(), "hidden.pt")
        torch.save({f.name: getattr(b, f.name) for f in T.dataclasses.fields(T.TaskBatch) if f.name != "meta"}, path)
        rows = run_hidden_eval(path, backend())
        self.assertEqual(len(rows), int(b.query_mask.sum()))
        self.assertTrue({"tokens", "confidence", "trace", "compute"} <= set(rows[0]))

    def test_reliability_suite_small(self):
        from nova.interface import reliability_suite
        be = ResearchBackend()
        be.register("nova", H.model("nova"))
        tasks = [t for d in ("sequence", "grid", "graph") for t in H.splits(d).eval_set("iid_val", 4)]
        rep = reliability_suite(be, tasks, n_valid=12, n_invalid=30)
        self.assertEqual(rep["crashes"], 0)
        self.assertEqual(rep["invalid_accepted"], 0, rep["accepted_invalid_by_mutation"])
        self.assertEqual(rep["valid_rejected"], 0)
        self.assertEqual(rep["deterministic_mismatch"], 0)
        self.assertLessEqual(len(rep["trace_sample"]), 20)

    def test_console_end_to_end(self):
        c = CON.Console(backend())
        good = c.handle("run", json.dumps(request()))
        self.assertTrue(good["ok"])
        self.assertIn("<svg", good["html"])
        self.assertIn("transformer", good["html"])
        bad = c.handle("run", "{not json")
        self.assertFalse(bad["ok"])
        self.assertIn("E_SCHEMA", bad["html"])
        self.assertIn("nova.runrecord/1", json.dumps(c.record(), default=str))

    def test_train_and_eval_modes_are_separate(self):
        m = H.model(dropout=0.3)
        b, tg = H.batch(2)
        m.eval()
        self.assertTrue(torch.equal(m.predict(b)["tokens"], m.predict(b)["tokens"]))
        m.train()
        self.assertNotEqual(m.loss(b, tg)[0].item(), m.loss(b, tg)[0].item())


if __name__ == "__main__":
    unittest.main()
