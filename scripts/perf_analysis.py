"""Roofline analysis for the int2 kernels at real model shapes.

    python scripts/perf_analysis.py [--model 235b|30b] [--iters 50]

Decode is weight-stationary and memory bound: every token streams the whole
weight matrix once, so the figure of merit is achieved read bandwidth, not
FLOP/s. Reported GB/s counts packed weights + scales + activations + output,
and is compared against a measured device copy bandwidth rather than a vendor
peak number, so the ratio is self-consistent on whatever card this runs on.
"""
from __future__ import annotations

import argparse
import time

import torch
import xetla_pt_ext  # noqa: F401  -- registers torch.ops.xetla_int2.*

GS = 128  # scale group along K

# (label, K, N) for one card, tp=1. Qwen3-235B-A22B: hidden 4096, 64 q heads,
# 4 kv heads, head_dim 128, moe_intermediate 1536, 128 experts, top-8.
SHAPES = {
    "235b": [
        ("qkv_proj", 4096, 9216),
        ("o_proj", 8192, 4096),
        ("expert_w13", 4096, 3072),
        ("expert_w2", 1536, 4096),
        ("lm_head_fp16", 4096, 151936),
    ],
    "8b": [
        # Qwen3-8B: hidden 4096, 32 q heads, 8 kv heads, head_dim 128,
        # intermediate 12288, 36 layers.
        ("qkv_proj", 4096, 6144),
        ("o_proj", 4096, 4096),
        ("gate_up", 4096, 24576),
        ("down_proj", 12288, 4096),
        # CAT-Q leaves lm_head in fp16, so the fp16 column is the real cost
        ("lm_head", 4096, 151936),
    ],
    "30b": [
        ("qkv_proj", 2048, 4096),
        ("o_proj", 4096, 2048),
        ("expert_w13", 2048, 1536),
        ("expert_w2", 768, 2048),
    ],
}


def _sync():
    torch.xpu.synchronize()


def _time(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters


def copy_bandwidth(mb=512, iters=30):
    """Device read+write bandwidth, used as the practical roofline."""
    n = mb * 1024 * 1024 // 2
    src = torch.empty(n, dtype=torch.float16, device="xpu")
    dst = torch.empty_like(src)
    s = _time(lambda: dst.copy_(src), iters)
    return 2 * src.numel() * 2 / s / 1e9  # read + write


def int2_gemm_bytes(m, k, n):
    return (k * n // 4)  + (k // GS) * n * 2 + m * k * 2 + m * n * 2


def bench_int2(m, k, n, iters):
    a = torch.randn(m, k, device="xpu", dtype=torch.float16)
    packed = torch.randint(0, 255, (k // 16, n), dtype=torch.int32, device="xpu")
    scale = torch.rand(k // GS, n, device="xpu", dtype=torch.float16) + 0.5
    op = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
    try:
        op(a, packed, scale, None)
    except Exception as e:  # shape guards (K divisibility, N alignment)
        return None, str(e).split("\n")[0][:70]
    s = _time(lambda: op(a, packed, scale, None), iters)
    return s, None


def bench_fp16_gemm(m, k, n, iters):
    a = torch.randn(m, k, device="xpu", dtype=torch.float16)
    w = torch.randn(k, n, device="xpu", dtype=torch.float16)
    s = _time(lambda: torch.mm(a, w), iters)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="235b", choices=sorted(SHAPES))
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--batch", type=int, default=1, help="decode batch (M)")
    p.add_argument("--tp", type=int, default=1,
                   help="shard each shape as tensor parallel would, to show "
                        "how much efficiency narrower tiles cost")
    a = p.parse_args()

    print(f"device      : {torch.xpu.get_device_name(0)}")
    roof = copy_bandwidth()
    print(f"copy bandwidth (read+write) : {roof:8.1f} GB/s   <- practical roofline")
    print()
    print(f"{'layer':<14}{'K':>7}{'N':>8}{'M':>4}{'int2 us':>10}{'GB/s':>9}"
          f"{'%roof':>7}{'fp16 us':>10}{'speedup':>9}")
    print("-" * 78)

    tot_int2 = tot_fp16 = 0.0
    for label, k, n in SHAPES[a.model]:
        # column-parallel layers split N, row-parallel split K
        if a.tp > 1:
            if label in ("o_proj", "expert_w2"):
                k //= a.tp
            else:
                n //= a.tp
        s, err = bench_int2(a.batch, k, n, a.iters)
        if err:
            print(f"{label:<14}{k:>7}{n:>8}{a.batch:>4}   skipped: {err}")
            continue
        gbs = int2_gemm_bytes(a.batch, k, n) / s / 1e9
        f16 = bench_fp16_gemm(a.batch, k, n, a.iters)
        tot_int2 += s
        tot_fp16 += f16
        print(f"{label:<14}{k:>7}{n:>8}{a.batch:>4}{s * 1e6:>10.1f}{gbs:>9.1f}"
              f"{100 * gbs / roof:>6.0f}%{f16 * 1e6:>10.1f}{f16 / s:>8.2f}x")

    print("-" * 78)
    print(f"{'total':<14}{'':>19}{tot_int2 * 1e6:>10.1f}{'':>16}"
          f"{tot_fp16 * 1e6:>10.1f}{tot_fp16 / max(tot_int2, 1e-9):>8.2f}x")


if __name__ == "__main__":
    main()
