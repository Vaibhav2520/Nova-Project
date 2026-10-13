"""Shape tests for every model, batch equivalence, variable lengths and parameter matching."""

import unittest

import torch

from nova import tasks as T
from nova.baselines import build_model, match_parameters
from nova.model import count_parameters
from nova.tests import _common as H

MODELS = ("nova", "ablated", "transformer", "recurrent", "heuristic", "nova_shuffled_r", "nova_no_persistent")


class TestShapes(unittest.TestCase):
    def test_every_model_loss_and_predict(self):
        for domain in ("sequence", "grid", "graph"):
            b, tg = H.batch(2, domain)
            n_q = int(b.query_mask.sum())
            for name in MODELS:
                with self.subTest(model=name, domain=domain):
                    m = H.model(name)
                    loss, info = m.loss(b, tg)
                    self.assertEqual(loss.dim(), 0)
                    self.assertTrue(torch.isfinite(loss))
                    m.eval()
                    out = m.predict(b)
                    self.assertEqual(out["tokens"].shape[0], n_q)
                    c = out["confidence"]
                    self.assertTrue(((c >= 0) & (c <= 1)).all())

    def test_batch_equivalence(self):
        ts = H.tasks(3)
        for name in ("nova", "transformer", "recurrent"):
            with self.subTest(model=name):
                m = H.model(name, double=True).eval()
                together = m.predict(T.collate(ts)[0])
                for i, t in enumerate(ts):
                    alone = m.predict(T.collate([t])[0])
                    rows = (together["q_task"] == i).nonzero(as_tuple=True)[0]
                    w = alone["tokens"].shape[1]
                    self.assertTrue(torch.equal(together["tokens"][rows][:, :w], alone["tokens"][:, :w]))

    def test_variable_lengths_in_one_batch(self):
        short = H.tasks(1)[0]
        sp = H.splits()
        long = sp.eval_set("struct_ood")[0]
        b, tg = T.collate([short, long])
        loss, _ = H.model().loss(b, tg)
        self.assertTrue(torch.isfinite(loss))

    def test_wrong_input_rank_is_refused(self):
        b, tg = H.batch(2)
        b.query_inputs = b.query_inputs[:, 0]
        with self.assertRaises((RuntimeError, IndexError, ValueError)):
            H.model().loss(b, tg)

    def test_transformer_kv_cache_gives_identical_answers(self):
        from nova.baselines import LanguageModelBaseline  # the slow reference decoder (re-reads the whole sequence)
        for pos in ("alibi", "rope"):
            for loops in (1, 2):
                with self.subTest(pos=pos, loops=loops):
                    m = H.model("transformer", double=True, pos_encoding=pos, n_loops=loops).eval()
                    b, _ = H.batch(3)  # tasks of different lengths (left padding is exercised)
                    fast, slow = m.predict(b), LanguageModelBaseline.predict(m, b)
                    w = min(fast["tokens"].shape[1], slow["tokens"].shape[1])
                    self.assertTrue(torch.equal(fast["tokens"][:, :w], slow["tokens"][:, :w]))
                    torch.testing.assert_close(fast["confidence"], slow["confidence"])

    def test_parameter_matching_within_5_percent(self):
        target = count_parameters(H.model("nova"))
        for name in ("transformer", "recurrent"):
            n = count_parameters(build_model(name, match_parameters(name, H.CFG, target)))
            self.assertLessEqual(abs(n - target) / target, 0.05, name)


if __name__ == "__main__":
    unittest.main()
