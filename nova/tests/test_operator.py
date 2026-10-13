"""Operator tests: shapes, the invariants I1-I8, edge cases, fail-fast behaviour and NovaState."""

import unittest

import torch

from nova import operator as O
from nova import tasks as T
from nova.state import NovaState
from nova.tests import _common as H


def run_op(m, b, t_max=3, halting=True, mode=None, h=None):
    h = m._guesses(b) if h is None else h
    return m.operator(h, b, m.executor, m.embed.weight, t_max, halting=halting, mode=mode or m.mode), h


class TestOperator(unittest.TestCase):
    def test_output_shapes(self):
        m = H.model()
        b, _ = H.batch(3)
        op, h = run_op(m, b)
        B, K, d = h.shape
        self.assertEqual(op.z_star.shape, (B, d))
        self.assertEqual(op.R.shape, (B, K, K))
        self.assertEqual(op.w.shape, (B, K))
        self.assertEqual(op.v.shape, (B, K))
        self.assertEqual(op.rho_hist.shape, (B, 4))

    def test_I1_permutation_invariance(self):
        m = H.model(double=True)
        ts = H.tasks(2)
        flipped = [T.Task(t.domain, t.program, t.signature, t.supports[::-1], t.queries, t.depth) for t in ts]
        a, _ = run_op(m, T.collate(ts)[0])
        b, _ = run_op(m, T.collate(flipped)[0])
        torch.testing.assert_close(a.z_star, b.z_star, atol=1e-9, rtol=0)
        torch.testing.assert_close(a.rho_star, b.rho_star, atol=1e-9, rtol=0)

    def test_I2_padding_invariance(self):
        m = H.model(double=True)
        t = H.tasks(1)[0]
        big = H.tasks(1, start=7)[0]
        long = T.Task(big.domain, big.program, big.signature, big.supports * 2, big.queries, big.depth)
        alone, _ = run_op(m, T.collate([t])[0])
        padded, _ = run_op(m, T.collate([t, long])[0])
        torch.testing.assert_close(alone.z_star[0], padded.z_star[0], atol=1e-9, rtol=0)

    def test_I3_convex_weights(self):
        m = H.model()
        b, _ = H.batch(3)
        op, h = run_op(m, b, t_max=0)
        for w in (op.w, op.v):
            self.assertTrue((w >= 0).all())
            torch.testing.assert_close(w.sum(1), torch.ones(3))
            self.assertTrue((w[~b.support_mask] == 0).all())
        torch.testing.assert_close(op.z_star, (op.w.unsqueeze(-1) * h).sum(1))

    def test_I4_bounded_update(self):
        m = H.model()
        op, _ = run_op(m, H.batch(3)[0], t_max=4, halting=False)
        self.assertTrue((op.update_norms <= m.operator.clip_c + 1e-5).all())

    def test_I5_best_so_far_monotone(self):
        m = H.model(double=True)
        b, _ = H.batch(3)
        rhos = [run_op(m, b, t_max=t, mode="fixed_T")[0].rho_star for t in range(5)]
        for a, c in zip(rhos, rhos[1:]):
            self.assertTrue((c <= a + 1e-12).all())

    def test_I6_degenerate_equals_mean_pooling(self):
        m = H.model()
        b, _ = H.batch(3)
        op, h = run_op(m, b, mode="mean")
        mean = (h * b.support_mask.unsqueeze(-1)).sum(1) / b.support_mask.sum(1, keepdim=True)
        torch.testing.assert_close(op.z_star, mean)
        self.assertTrue((op.steps_used == 0).all())

    def test_I8_halting_is_absorbing(self):
        m = H.model()
        op, _ = run_op(m, H.batch(3)[0], t_max=4)
        for i in range(3):
            hs = int(op.halt_step[i])
            self.assertLessEqual(int(op.steps_used[i]), hs)
            self.assertTrue((op.rho_hist[i, hs + 1:] == -1).all())

    def test_nan_injection_fails_fast(self):
        m = H.model()
        b, _ = H.batch(2)
        h = m._guesses(b).detach().clone()
        h[0, 0, 0] = float("nan")
        O.DEBUG_FINITE = True
        try:
            with self.assertRaises(O.NumericalError):
                run_op(m, b, h=h)
        finally:
            O.DEBUG_FINITE = False

    def test_all_masked_task_gives_no_nan(self):
        m = H.model()
        b, _ = H.batch(2)
        b.support_mask[1] = False
        op, _ = run_op(m, b)
        self.assertTrue(torch.isfinite(op.z_star).all() and torch.isfinite(op.rho_star).all())
        self.assertTrue((op.w[1] == 0).all())
        self.assertEqual(int(op.executor_calls[1]), 0)

    def test_single_demo_sets_no_cross_check(self):
        m = H.model()
        b, _ = H.batch(1)
        b.support_mask[0, 1:] = False
        op, _ = run_op(m, b)
        self.assertTrue(bool(op.no_cross_check[0]))
        self.assertAlmostEqual(float(op.w[0, 0].detach()), 1.0, places=6)

    def test_invalid_mode_and_shapes(self):
        m = H.model()
        b, _ = H.batch(2)
        with self.assertRaises(ValueError):
            run_op(m, b, mode="bogus")
        with self.assertRaises(RuntimeError):
            run_op(m, b, h=torch.zeros(2, b.support_inputs.shape[1], 7))

    def test_float64_and_maximum_k(self):
        m = H.model(double=True)
        t = H.tasks(1)[0]
        big = T.Task(t.domain, t.program, t.signature, (t.supports * 4)[:8], t.queries, t.depth)
        op, _ = run_op(m, T.collate([big])[0])
        self.assertEqual(op.z_star.dtype, torch.float64)
        self.assertEqual(op.R.shape[-1], 8)

    def test_novastate_lifecycle(self):
        s = NovaState.init(2, 4, 3)
        s.z = torch.randn(2, 4, requires_grad=True)
        c = s.clone()
        self.assertFalse(c.z.data_ptr() == s.z.data_ptr())
        self.assertFalse(s.detach().z.requires_grad)
        restored = NovaState.load_state_dict(s.state_dict())
        torch.testing.assert_close(restored.z, s.z)
        s.reset()
        self.assertTrue((s.z == 0).all() and (s.rho_hist == -1).all())
        with self.assertRaises(ValueError):
            NovaState.load_state_dict({"z": s.z})


if __name__ == "__main__":
    unittest.main()
