"""XPU check + microbench of the fused hadamard_fwht kernel against the matmul
reference. Run on a GPU node inside the plugin venv:

    python tests/test_hadamard_xpu.py
"""
import math
import time

import torch
import xetla_pt_ext  # noqa: F401  registers torch.ops.xetla_int2.*
import ternsycl_pt_ext  # noqa: F401

dev = torch.device("xpu")
torch.manual_seed(0)
BLOCK = 1024


def h_matrix(block):
    h = torch.ones(1, 1)
    while h.shape[0] < block:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(block)


H = h_matrix(BLOCK).to(dev)
op = torch.ops.ternsycl.hadamard_fwht_run

for rows, K in [(1, 5120), (1, 6144), (1, 17408), (63, 5120), (63, 17408), (512, 5120)]:
    x = (torch.randn(rows, K, device=dev) * 3).to(torch.float16)
    signs = torch.where(torch.rand(K, device=dev) > 0.5, 1.0, -1.0).to(torch.float16)
    ref_f = ((x.float() * signs.float()).reshape(-1, BLOCK) @ H).reshape(rows, K)
    ref_i = ((x.float().reshape(-1, BLOCK) @ H).reshape(rows, K) * signs.float())
    y_f = op(x, signs, BLOCK, False)
    y_i = op(x, signs, BLOCK, True)
    torch.xpu.synchronize()
    ef = (y_f.float() - ref_f).abs().max().item() / ref_f.abs().max().item()
    ei = (y_i.float() - ref_i).abs().max().item() / ref_i.abs().max().item()
    # timing: fused vs matmul path (as the plugin does it, fp32)
    def bench(fn, n=200):
        for _ in range(10):
            fn()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.xpu.synchronize()
        return (time.perf_counter() - t0) / n * 1e6
    t_fused = bench(lambda: op(x, signs, BLOCK, False))
    t_mm = bench(lambda: ((x.float() * signs.float()).reshape(-1, BLOCK) @ H)
                 .reshape(rows, K).to(torch.float16))
    print(f"rows={rows:4d} K={K:6d}  rel.err fwd={ef:.2e} inv={ei:.2e}   "
          f"fused {t_fused:7.1f} us   matmul {t_mm:7.1f} us   x{t_mm / t_fused:.1f}")
    assert ef < 2e-3 and ei < 2e-3, "fused FWHT mismatch"
print("OK")
