"""Per-shape decode GEMV comparison: BITCOS vs the int2 baseline.

Times the real Bonsai 8B linear shapes so the BITCOS dispatch tiers are chosen
from measurement rather than from the synthetic square shmoo.
"""
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

# (name, K, N) for Bonsai 8B at TP=1.
SHAPES = [
    ("qkv_proj", 4096, 6144),
    ("o_proj", 4096, 4096),
    ("gate_up_proj", 4096, 24576),
    ("down_proj", 12288, 4096),
    ("lm_head", 4096, 32768),
]


def timeit(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    z = 0.4
    g = torch.Generator().manual_seed(0)
    print(f"{'shape':<14} {'K':>6} {'N':>6} {'int2':>8} {'bitcosL1':>9} "
          f"{'bitcosL4':>9} {'best':>9} {'vs int2':>8}")
    tot_i2 = tot_bc = 0.0
    for name, K, N in SHAPES:
        r = torch.rand(K, N, generator=g)
        sign = torch.where(torch.rand(K, N, generator=g) < 0.5, -1.0, 1.0)
        codes = torch.where(r < z, torch.zeros(K, N), sign).to(torch.int8)
        scale = (torch.rand(K // GS, N, generator=g) + 0.5).to(torch.float16)
        A = (torch.rand(1, K, generator=g) * 2 - 1).to(torch.float16)

        a_d = A.to(DEV)
        s_d = scale.to(DEV)
        i2 = pack_ternary_to_int2(codes).to(DEV)
        buf, ranks = pack_bitcos(codes)
        b_d = buf.to(DEV)
        r_d = ranks.to(DEV)

        t_i2 = timeit(lambda: torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run(
            a_d, i2, s_d, None))
        t_l1 = timeit(lambda: torch.ops.xetla_int2.bitcos_fp16_upcvt_gemm_run(
            a_d, b_d, s_d, None, None))
        t_l4 = timeit(lambda: torch.ops.xetla_int2.bitcos_fp16_upcvt_gemm_run(
            a_d, b_d, s_d, r_d, None))
        best = min(t_l1, t_l4)
        tot_i2 += t_i2
        tot_bc += best
        print(f"{name:<14} {K:>6} {N:>6} {t_i2:>8.4f} {t_l1:>9.4f} "
              f"{t_l4:>9.4f} {best:>9.4f} {t_i2 / best:>7.2f}x")
    print(f"{'TOTAL':<14} {'':>6} {'':>6} {tot_i2:>8.4f} {'':>9} {'':>9} "
          f"{tot_bc:>9.4f} {tot_i2 / tot_bc:>7.2f}x")


if __name__ == "__main__":
    main()
