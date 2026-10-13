"""Mask and information-flow tests: no leaks from questions, answers, future tokens, padding or hidden fields."""

import unittest

import torch

from nova import tasks as T
from nova.operator import masked_softmax
from nova.tests import _common as H


class TestMasks(unittest.TestCase):
    def test_query_inputs_never_reach_the_rule(self):  # I7: the rule depends only on demonstrations
        m = H.model()
        b, _ = H.batch(2)
        h = m._guesses(b)
        a = m.operator(h, b, m.executor, m.embed.weight, 2, halting=True).z_star
        b.query_inputs = torch.randint(16, 40, b.query_inputs.shape)
        c = m.operator(h, b, m.executor, m.embed.weight, 2, halting=True).z_star
        torch.testing.assert_close(a, c, atol=0, rtol=0)

    def test_taskbatch_cannot_carry_answers(self):
        names = {f.name for f in T.dataclasses.fields(T.TaskBatch)}
        self.assertFalse(any("query_output" in n for n in names))
        b, tg = H.batch(1)
        self.assertIsInstance(tg, T.TargetBatch)

    def test_meta_never_used_by_models(self):
        for name in ("nova", "transformer"):
            m = H.model(name).eval()
            b, _ = H.batch(2)
            a = m.predict(b)["tokens"]
            b.meta = {"signatures": ["fake"], "domains": ["fake"], "program": "copy"}
            self.assertTrue(torch.equal(a, m.predict(b)["tokens"]), name)

    def test_transformer_has_no_future_leak(self):
        m = H.model("transformer").eval()
        tok = torch.randint(16, 60, (2, 12))
        mask = torch.ones_like(tok, dtype=torch.bool)
        a = m.logits(tok, mask)
        tok2 = tok.clone()
        tok2[:, 6:] = torch.randint(16, 60, (2, 6))
        b = m.logits(tok2, mask)
        torch.testing.assert_close(a[:, :6], b[:, :6])

    def test_executor_padding_invariance(self):
        m = H.model(double=True)
        b, tg = H.batch(1)
        x, xm = b.support_inputs[:, 0], b.support_input_mask[:, 0]
        y, ym = b.support_outputs[:, 0], b.support_output_mask[:, 0]
        z = torch.randn(1, m.cfg.d_model, dtype=torch.float64)
        a = m.executor.score(x, xm, z, y, ym)
        xp = torch.cat([x, torch.zeros(1, 5, dtype=torch.long)], 1)
        b2 = m.executor.score(xp, xp != 0, z, y, ym)
        torch.testing.assert_close(a, b2, atol=1e-9, rtol=0)

    def test_encoder_padding_invariance(self):
        m = H.model(double=True)
        x = torch.tensor([[20, 21, 22]]); y = torch.tensor([[30, 31, T.EOS]])
        a = m.encoder(x, x != 0, y, y != 0)
        xp = torch.cat([x, torch.zeros(1, 4, dtype=torch.long)], 1)
        b = m.encoder(xp, xp != 0, y, y != 0)
        torch.testing.assert_close(a, b, atol=1e-9, rtol=0)

    def test_all_masked_softmax_row(self):
        p = masked_softmax(torch.randn(2, 4), torch.tensor([[True, False, True, False], [False] * 4]))
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue((p[1] == 0).all())
        torch.testing.assert_close(p[0].sum(), torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
