"""Decode GEMV tuning for BITCOS against int2, with an out-of-cache footprint.

A single weight buffer for these shapes fits in cache, so timing it measures
cache bandwidth and the ranking of tiles comes out wrong. Every measurement
here rotates over enough distinct weight buffers to exceed BENCH_GB, which is
what decode actually does: stream the whole model once per token.
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
SHAPES = [("qkv", 4096, 6144), ("o", 4096, 4096), ("gate_up", 4096, 24576),
          ("down", 12288, 4096), ("lm_head", 4096, 151680)]
CFGS = {0: (32, 1, 64), 1: (64, 1, 64), 2: (128, 1, 64), 3: (256, 1, 64),
        4: (32, 4, 64), 5: (64, 4, 64), 6: (128, 4, 64), 7: (256, 4, 64),
        8: (128, 1, 128), 9: (128, 4, 128), 10: (256, 4, 128),
        11: (32, 8, 64), 12: (64, 8, 64), 13: (128, 8, 64), 14: (256, 8, 64)}
TARGET_BYTES = float(os.environ.get("BENCH_GB", "2.0")) * 1e9


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


def sets_for(t):
    """Views into one big allocation, so the timing loop streams >=BENCH_GB
    without the allocator churn that repeated clone()/free() introduces.

    The per-set stride is padded to 64 elements: the BITCOS buffer ends in an
    odd pad word, and an unaligned base makes the block loads unusable.
    """
    flat = t.reshape(-1)
    numel = flat.numel()
    stride = ((numel + 63) // 64) * 64
    n = max(1, int(TARGET_BYTES / (numel * t.element_size())))
    big = torch.zeros(n * stride, dtype=t.dtype, device=t.device)
    for i in range(n):
        big[i * stride:i * stride + numel] = flat
    sets = [big[i * stride:i * stride + numel].view(t.shape) for i in range(n)]
    return sets, big


def main():
    z = float(os.environ.get("BENCH_Z", "0.24"))
    cfg = int(os.environ.get("XETLA_BITCOS_DECODE_CFG", "-1"))
    _s = os.environ.get("BENCH_SLICES", "")
    slices = int(_s) if _s else None
    g = torch.Generator().manual_seed(0)

    for name, K, N in SHAPES:
        r = torch.rand(K, N, generator=g)
        sign = torch.where(torch.rand(K, N, generator=g) < 0.5, -1.0, 1.0)
        codes = torch.where(r < z, torch.zeros(K, N), sign).to(torch.int8)
        scale = (torch.rand(K // GS, N, generator=g) + 0.5).to(torch.float16)
        A = (torch.rand(1, K, generator=g) * 2 - 1).to(torch.float16)
        a_d, s_d = A.to(DEV), scale.to(DEV)

        buf, ranks = pack_bitcos(codes, slices=slices)
        r_d = ranks.to(DEV)
        b_sets, b_big = sets_for(buf.to(DEV))
        bc_op = (lambda b: torch.ops.xetla_int2.bitcos_fp16_upcvt_gemm_run(
            a_d, b, s_d, r_d, None))
        nb = len(b_sets)

        if cfg < 0:
            i_sets, i_big = sets_for(pack_ternary_to_int2(codes).to(DEV))
            i2_op = (lambda b: torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run(
                a_d, b, s_d, None))
            # Interleave so clock drift hits both sides equally.
            bc, i2 = [], []
            for _ in range(3):
                bc.append(rotate_time(bc_op, b_sets))
                i2.append(rotate_time(i2_op, i_sets))
            t_bc, t_i2 = sorted(bc)[1], sorted(i2)[1]
            del i_sets, i_big
            print(f"{name:<8} K={K:<6} N={N:<7} LS={ranks.shape[0] + 1} "
                  f"sets={nb:<3} int2={t_i2:>7.2f}us bitcos={t_bc:>7.2f}us  "
                  f"{t_i2 / t_bc:>5.2f}x")
        else:
            t_bc = sorted(rotate_time(bc_op, b_sets) for _ in range(3))[1]
            wgn, ls, sgk = CFGS[cfg]
            print(f"{name:<8} K={K:<6} N={N:<7} cfg={cfg} "
                  f"(wg_n={wgn},LS={ls},sg_k={sgk}) sets={nb:<3} "
                  f"{t_bc:>7.2f}us")
        del b_sets, b_big, a_d, s_d, r_d
        torch.xpu.empty_cache()


if __name__ == "__main__":
    main()
