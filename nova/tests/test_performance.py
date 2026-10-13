"""Performance tests: latency after warm-up, profiler, complexity guard, memory budget, OOM recovery, adaptive savings."""

import json
import time
import unittest
import warnings

import torch
from torch.utils.flop_counter import FlopCounterMode

from nova import console as CON
from nova.interface import ResearchBackend
from nova.tests import _common as H


def flops(fn) -> int:
    with FlopCounterMode(display=False) as c:
        fn()
    return c.get_total_flops()


class TestPerformance(unittest.TestCase):
    def test_latency_after_warm_up(self):
        m = H.model().eval()
        b, _ = H.batch(2)
        for _ in range(2):
            m.predict(b)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            m.predict(b)
            times.append(time.perf_counter() - start)
        self.assertTrue(all(0 < t < 30 for t in times))

    def test_profiler_names_top_operations(self):
        m = H.model()
        b, tg = H.batch(2)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
            m.loss(b, tg)
        top = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5)
        self.assertIn("aten::", top)

    def test_complexity_stays_within_declared_bound(self):
        m = H.model()
        z = torch.randn(1, m.cfg.d_model)
        def cost(L):
            x = torch.randint(16, 40, (1, L))
            return flops(lambda: m.executor.score(x, x != 0, z, x, x != 0))
        ratio = cost(64) / cost(16)
        bound = 16 * 1.1  # declared O(L^2) bound for read option A, with 10% slack
        if ratio > bound:
            warnings.warn(f"executor cost grew {ratio:.1f}x for 4x length (bound {bound})")
        self.assertLessEqual(ratio, bound)

    def test_oversized_request_is_refused_and_service_recovers(self):
        be = ResearchBackend(memory_cap_bytes=10_000)
        be.register("nova", H.model())
        r = json.loads(json.dumps(CON.EXAMPLE_REQUEST))
        r["models"] = ["nova"]
        self.assertEqual(be.infer(r)["errors"][0]["code"], "E_MEMORY_BUDGET")
        be.memory_cap = 2 ** 31
        self.assertEqual(be.infer(r)["status"], "ok")

    def test_peak_memory_under_cap_on_gpu(self):
        if not torch.cuda.is_available():
            self.skipTest("no GPU: peak CUDA memory cannot be measured here")
        m = H.model().cuda()
        b, tg = H.T.collate(H.splits().eval_set("struct_ood", 6), "cuda")
        torch.cuda.reset_peak_memory_stats()
        m.loss(b, tg)[0].backward()
        self.assertLess(torch.cuda.max_memory_allocated(), 12 * 2 ** 30)

    def test_adaptive_compute_saves_real_work(self):
        m = H.model().eval()
        b, _ = H.batch(3)
        m.operator.eps = 1e9  # every task "fits" at once, so adaptive NOVA halts at step 0
        adaptive_calls = m.predict(b, max_steps=4)["compute"]["executor_calls"].sum()
        adaptive_flops = flops(lambda: m.predict(b, max_steps=4))
        m.mode = "fixed_T"
        fixed_calls = m.predict(b, max_steps=4)["compute"]["executor_calls"].sum()
        fixed_flops = flops(lambda: m.predict(b, max_steps=4))
        self.assertLess(int(adaptive_calls), int(fixed_calls))
        self.assertLess(adaptive_flops, fixed_flops)


if __name__ == "__main__":
    unittest.main()
