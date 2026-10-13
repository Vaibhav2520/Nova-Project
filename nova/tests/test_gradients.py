"""Gradient and numerical tests: gradcheck, finite differences, gradient flow, AMP agreement and extreme inputs."""

import unittest

import torch

from nova.tests import _common as H


def op_out(m, b, h, t_max=2, halting=False, mode="full"):
    return m.operator(h, b, m.executor, m.embed.weight, t_max, halting=halting, mode=mode)


class TestGradients(unittest.TestCase):
    def test_gradcheck_operator_float64(self):
        m = H.model(double=True)
        b, _ = H.batch(2)
        h = m._guesses(b).detach().requires_grad_(True)
        f = lambda hh: op_out(m, b, hh).z_star
        self.assertTrue(torch.autograd.gradcheck(f, (h,), eps=1e-6, atol=1e-4, fast_mode=True))

    @unittest.skip("second-order gradients are not claimed for CEC (documented in the test report)")
    def test_gradgradcheck(self):
        pass

    def test_finite_difference_spot_check(self):
        m = H.model(double=True)
        b, tg = H.batch(2)
        p = m.operator.upd[0].weight
        loss = lambda: m.loss(b, tg)[0]
        m.zero_grad()
        loss().backward()
        g = p.grad[0, 0].item()
        eps = 1e-6
        with torch.no_grad():
            p[0, 0] += eps
            up = loss().item()
            p[0, 0] -= 2 * eps
            down = loss().item()
            p[0, 0] += eps
        self.assertAlmostEqual(g, (up - down) / (2 * eps), delta=1e-5 + 1e-3 * abs(g))

    def test_gradient_reaches_encoder_through_refinement(self):
        m = H.model("nova_no_deep_sup")
        m.aux_loss = False  # no auxiliary loss either, so z_T is the only path to the encoder
        b, tg = H.batch(2)
        loss, _ = m.loss(b, tg)
        loss.backward()
        g = sum(p.grad.abs().sum().item() for p in m.encoder.parameters() if p.grad is not None)
        self.assertGreater(g, 0.0)

    def test_gradient_flow_per_step(self):
        m = H.model()
        b, _ = H.batch(2)
        h = m._guesses(b)
        out = op_out(m, b, h, t_max=3)
        for z in out.z_traj:
            z.retain_grad()
        out.z_traj[-1].pow(2).sum().backward()
        norms = [z.grad.norm().item() for z in out.z_traj]
        self.assertTrue(all(torch.isfinite(torch.tensor(norms))))
        self.assertGreater(norms[0], 0.0)

    def test_amp_keeps_weights_in_float32_and_agrees(self):
        m = H.model()
        b, _ = H.batch(3)
        with torch.no_grad():
            h = m._guesses(b)
            ref = op_out(m, b, h, t_max=2, halting=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                amp = op_out(m, b, h, t_max=2, halting=True)
        self.assertEqual(amp.w.dtype, torch.float32)
        self.assertEqual(amp.R.dtype, torch.float32)
        torch.testing.assert_close(amp.w, ref.w, atol=0.05, rtol=0)  # tolerance from the bf16 mantissa (8 bits)
        torch.testing.assert_close(amp.v, ref.v, atol=0.05, rtol=0)

    def test_extreme_inputs_stay_finite(self):
        sp = H.splits()
        b, tg = H.T.collate(sp.eval_set("struct_ood", 2))
        m = H.model()
        loss, _ = m.loss(b, tg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
