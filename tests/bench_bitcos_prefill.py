"""Sweep the BITCOS prefill tile against the int2 prefill kernel.

Re-run with XETLA_BITCOS_PREFILL_CFG=0..5 to pick the M>1 tile.
"""
import os
import sys
import time

import torch
import xetla_pt_ext  # noqa: F401

sys.path.insert(0, "/data/nfs_home/egeorgan/cpu_ternary_vllm/xetla_vllm_plugin")
from xetla_vllm_plugin import (  # noqa: E402
    INT2_F16_GROUP_SIZE, pack_bitcos, pack_ternary_to_int2,
)

GS = INT2_F16_GROUP_SIZE
DEV = "xpu"
SHAPES = [("qkv", 4096, 6144), ("o", 4096, 4096),
          ("gate_up", 4096, 24576), ("down", 12288, 4096)]


def timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    M = int(os.environ.get("BENCH_M", "512"))
    cfg = os.environ.get("XETLA_BITCOS_PREFILL_CFG", "0")
    z = 0.25
    g = torch.Generator().manual_seed(0)
    iters = 5 if M >= 4096 else 20
    tot_i2 = tot_bc = 0.0
    for name, K, N in SHAPES:
        r = torch.rand(K, N, generator=g)
        sign = torch.where(torch.rand(K, N, generator=g) < 0.5, -1.0, 1.0)
        codes = torch.where(r < z, torch.zeros(K, N), sign).to(torch.int8)
        scale = (torch.rand(K // GS, N, generator=g) + 0.5).to(torch.float16)
        A = (torch.rand(M, K, generator=g) * 2 - 1).to(torch.float16)
        a_d, s_d = A.to(DEV), scale.to(DEV)
        i2 = pack_ternary_to_int2(codes).to(DEV)
        b_d = pack_bitcos(codes)[0].to(DEV)

        t_i2 = timeit(lambda: torch.ops.xetla_int2.int2_fp16_dpas_gemm_run(
            a_d, i2, s_d, None), iters)
        t_bc = timeit(lambda: torch.ops.xetla_int2.bitcos_fp16_upcvt_gemm_run(
            a_d, b_d, s_d, None, None), iters)
        tot_i2 += t_i2
        tot_bc += t_bc
        print(f"cfg={cfg} M={M:<5} {name:<8} K={K:<6} N={N:<6} "
              f"int2_dpas={t_i2:>9.3f}ms  bitcos={t_bc:>9.3f}ms  "
              f"ratio={t_bc / t_i2:>6.2f}x")
    print(f"cfg={cfg} M={M:<5} {'TOTAL':<8} {'':<6} {'':<6} "
          f"int2_dpas={tot_i2:>9.3f}ms  bitcos={tot_bc:>9.3f}ms  "
          f"ratio={tot_bc / tot_i2:>6.2f}x")


if __name__ == "__main__":
    main()
