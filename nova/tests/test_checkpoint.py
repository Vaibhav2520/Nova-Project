"""Checkpoint tests: round trip, atomic write, corrupt/partial/incompatible files, RNG restore and portability."""

import os
import random
import unittest

import torch

from nova import checkpoint as CK
from nova.tests import _common as H


def payload(**changes):
    m = H.model()
    base = {"schema": CK.CKPT_SCHEMA, "model": m.state_dict(), "optimizer": {}, "scheduler": {}, "scaler": {},
            "rng": CK.capture_rng(), "data_state": {"step": 0}, "config": {}, "config_hash": "sha256:x",
            "code_hash": "sha256:y", "step": 0, "examples_seen": 0}
    base.update(changes)
    return base


class TestCheckpoint(unittest.TestCase):
    def test_round_trip_and_atomic_write(self):
        path = os.path.join(H.tmpdir(), "c.pt")
        p = payload()
        CK.save_checkpoint(path, p)
        self.assertFalse(os.path.exists(path + ".tmp"))
        self.assertTrue(os.path.exists(path + ".json"))
        back = CK.load_checkpoint(path)
        for k, v in p["model"].items():
            torch.testing.assert_close(back["model"][k], v)

    def test_corrupt_file_is_refused(self):
        path = os.path.join(H.tmpdir(), "c.pt")
        CK.save_checkpoint(path, payload())
        with open(path, "r+b") as fh:
            fh.seek(200)
            fh.write(b"\x00\x01\x02\x03")
        with self.assertRaises(CK.CorruptCheckpointError):
            CK.load_checkpoint(path)

    def test_missing_file_is_refused(self):
        with self.assertRaises(CK.CorruptCheckpointError):
            CK.load_checkpoint(os.path.join(H.tmpdir(), "nope.pt"))

    def test_partial_checkpoint_is_refused(self):
        p = payload()
        del p["optimizer"]
        with self.assertRaises(CK.IncompatibleCheckpointError):
            CK.save_checkpoint(os.path.join(H.tmpdir(), "c.pt"), p)
        path = os.path.join(H.tmpdir(), "raw.pt")
        torch.save(p, path)
        with self.assertRaises(CK.IncompatibleCheckpointError):
            CK.load_checkpoint(path)

    def test_wrong_schema_and_settings_are_refused(self):
        path = os.path.join(H.tmpdir(), "c.pt")
        torch.save(payload(schema="nova.checkpoint/0"), path)
        with self.assertRaises(CK.IncompatibleCheckpointError):
            CK.load_checkpoint(path)
        CK.save_checkpoint(path, payload())
        with self.assertRaises(CK.IncompatibleCheckpointError):
            CK.load_checkpoint(path, expect_config_hash="sha256:other")

    def test_rng_capture_and_restore(self):
        state = CK.capture_rng()
        a = (random.random(), torch.rand(3))
        CK.restore_rng(state)
        b = (random.random(), torch.rand(3))
        self.assertEqual(a[0], b[0])
        torch.testing.assert_close(a[1], b[1])

    def test_rng_survives_save_and_load(self):
        path = os.path.join(H.tmpdir(), "c.pt")
        CK.save_checkpoint(path, payload())
        p = CK.load_checkpoint(path)
        CK.restore_rng(p["rng"])
        x = torch.rand(2)
        CK.restore_rng(p["rng"])
        torch.testing.assert_close(x, torch.rand(2))

    def test_rng_restore_after_loading_onto_gpu(self):  # regression test: found on the first Colab GPU run
        if not torch.cuda.is_available():
            self.skipTest("no GPU here")
        path = os.path.join(H.tmpdir(), "c.pt")
        CK.save_checkpoint(path, payload())
        p = CK.load_checkpoint(path, map_location="cuda")
        CK.restore_rng(p["rng"])
        a = torch.rand(2)
        CK.restore_rng(p["rng"])
        torch.testing.assert_close(a, torch.rand(2))

    def test_cpu_gpu_portability(self):
        if not torch.cuda.is_available():
            self.skipTest("no GPU here; the CPU path is covered by every other test")
        path = os.path.join(H.tmpdir(), "c.pt")
        p = payload(model={k: v.cuda() for k, v in H.model().state_dict().items()})
        CK.save_checkpoint(path, p)
        back = CK.load_checkpoint(path, map_location="cpu")
        self.assertTrue(all(v.device.type == "cpu" for v in back["model"].values()))


if __name__ == "__main__":
    unittest.main()
