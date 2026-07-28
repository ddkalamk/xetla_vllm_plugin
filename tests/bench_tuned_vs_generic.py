"""Compare achieved M=1 bandwidth for shapes that HAVE tuned dispatch entries
(Bonsai-8B) against shapes that fall into the generic fallback tiers
(Bonsai-27B), to size up the autotuning opportunity.

    python tests/bench_tuned_vs_generic.py
"""

from __future__ import annotations

import time

import torch

from xetla_vllm_plugin import (
    INT2_F16_GROUP_SIZE,
    pack_ternary_to_int2,
    quantize_to_ternary_f16,
)

import xetla_pt_ext  # noqa: F401

ROWS = [
    # model, K, N, dispatch tier used by csrc/int2_fp16_upcvt_kernel.sycl
    ("8B", 4096, 6144, "TUNED (32,1,4)"),
    ("8B", 4096, 24576, "TUNED (32,1,4)"),
    ("8B", 12288, 4096, "TUNED (32,1,8)"),
    ("8B", 4096, 151680, "TUNED (32,1,4)"),
    ("27B", 5120, 34816, "generic (128,1,1)"),
    ("27B", 17408, 5120, "generic (64,1,4)"),
    ("27B", 5120, 16384, "generic (64,1,2)"),
    ("27B", 6144, 5120, "generic (64,1,4)"),
    ("27B", 5120, 14336, "generic (64,1,2)"),
    ("27B", 5120, 248320, "generic (128,1,1)"),
]


def packed(K: int, N: int):
    gs = INT2_F16_GROUP_SIZE
    g = torch.Generator(device="cpu").manual_seed(K + N)
    c = torch.randint(-1, 2, (K // gs, gs, N), generator=g, dtype=torch.int8)
    s = (torch.rand(K // gs, 1, N, generator=g) * 0.05 + 0.005).to(torch.float16)
    w = (c.to(torch.float16) * s).reshape(K, N)
    cc, ss = quantize_to_ternary_f16(w, gs)
    return pack_ternary_to_int2(cc).to("xpu").contiguous(), ss.to("xpu").contiguous()


def bench(K: int, N: int, iters: int = 40):
    up = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
    w, s = packed(K, N)
    x = (torch.randn(1, K, device="xpu") * 0.5).to(torch.float16)
    for _ in range(8):
        up(x, w, s, None)
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        up(x, w, s, None)
    torch.xpu.synchronize()
    us = (time.perf_counter() - t) / iters * 1e6
    nbytes = K * N // 4 + (K // 128) * N * 2 + K * 2 + N * 2
    return us, nbytes / us / 1e3


def main() -> None:
    if not torch.xpu.is_available():
        raise SystemExit("XPU not available")
    print(f"{'model':6} {'K':>6} {'N':>7} {'dispatch tier':>20} {'us':>8} {'GB/s':>7}")
    for model, K, N, tier in ROWS:
        us, bw = bench(K, N)
        print(f"{model:6} {K:>6} {N:>7} {tier:>20} {us:>8.1f} {bw:>7.1f}")


if __name__ == "__main__":
    main()
