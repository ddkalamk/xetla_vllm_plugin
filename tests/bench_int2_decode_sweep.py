"""Sweep the int2 decode tile per real Bonsai shape, out of cache.

The in-tree int2 tier table was measured on a discrete B70 (380-435 GiB/s);
Lunar Lake has an order of magnitude less bandwidth, so the winning tile is not
expected to carry over. Rotates over >=BENCH_GB of distinct weight buffers.
"""
import os
import sys
import time

import torch
import xetla_pt_ext  # noqa: F401

sys.path.insert(0, "/data/nfs_home/egeorgan/cpu_ternary_vllm/xetla_vllm_plugin")
from xetla_vllm_plugin import (  # noqa: E402
    INT2_F16_GROUP_SIZE, pack_ternary_to_int2,
)

GS = INT2_F16_GROUP_SIZE
DEV = "xpu"
SHAPES = [("qkv", 4096, 6144), ("o", 4096, 4096), ("gate_up", 4096, 24576),
          ("down", 12288, 4096), ("lm_head", 4096, 151680)]
CFGS = {0: (32, 1), 1: (32, 2), 2: (32, 4), 3: (32, 8),
        4: (64, 1), 5: (64, 2), 6: (64, 4), 7: (64, 8),
        8: (128, 1), 9: (128, 2), 10: (128, 4), 11: (128, 8)}
TARGET_BYTES = float(os.environ.get("BENCH_GB", "2.0")) * 1e9


def sets_for(t):
    flat = t.reshape(-1)
    numel = flat.numel()
    stride = ((numel + 63) // 64) * 64
    n = max(1, int(TARGET_BYTES / (numel * t.element_size())))
    big = torch.zeros(n * stride, dtype=t.dtype, device=t.device)
    for i in range(n):
        big[i * stride:i * stride + numel] = flat
    return [big[i * stride:i * stride + numel].view(t.shape)
            for i in range(n)], big


def rotate_time(op, sets, iters=200, warmup=20):
    n = len(sets)
    for i in range(warmup):
        op(sets[i % n])
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        op(sets[i % n])
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    z = float(os.environ.get("BENCH_Z", "0.368"))
    cfg = int(os.environ.get("XETLA_INT2_DECODE_CFG", "-1"))
    g = torch.Generator().manual_seed(0)
    tag = f"cfg={cfg} (wg_n={CFGS[cfg][0]},LS={CFGS[cfg][1]})" if cfg >= 0 \
        else "table"
    for name, K, N in SHAPES:
        r = torch.rand(K, N, generator=g)
        sign = torch.where(torch.rand(K, N, generator=g) < 0.5, -1.0, 1.0)
        codes = torch.where(r < z, torch.zeros(K, N), sign).to(torch.int8)
        scale = (torch.rand(K // GS, N, generator=g) + 0.5).to(torch.float16)
        A = (torch.rand(1, K, generator=g) * 2 - 1).to(torch.float16)
        a_d, s_d = A.to(DEV), scale.to(DEV)
        i_sets, big = sets_for(pack_ternary_to_int2(codes).to(DEV))
        t = sorted(
            rotate_time(lambda b: torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run(
                a_d, b, s_d, None), i_sets) for _ in range(3))[1]
        # int2 is a flat 2 bits per weight, plus the fp16 group scales.
        gb = (K * N / 4 + (K // GS) * N * 2) / 1e9
        print(f"{name:<8} K={K:<6} N={N:<7} {tag:<26} {t:>8.2f}us "
              f"{gb / (t * 1e-6):>7.1f} GB/s")
        del i_sets, big, a_d, s_d
        torch.xpu.empty_cache()


if __name__ == "__main__":
    main()
