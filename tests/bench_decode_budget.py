"""Sum the real (un-synchronized) cost of every int2 GEMM in one Bonsai-27B
decode step, to see how much of the per-token time is kernel time and how much
is framework overhead.

    python tests/bench_decode_budget.py
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

# (name, count per token, K, N)   -- Bonsai-27B: 64 layers, 48 GDN + 16 full attn
LAYERS = [
    ("mlp.gate_up_proj", 64, 5120, 34816),
    ("mlp.down_proj", 64, 17408, 5120),
    ("linear_attn.in_proj_qkvz", 48, 5120, 16384),
    ("linear_attn.in_proj_ba", 48, 5120, 96),
    ("linear_attn.out_proj", 48, 6144, 5120),
    ("self_attn.qkv_proj", 16, 5120, 14336),
    ("self_attn.o_proj", 16, 6144, 5120),
    ("lm_head", 1, 5120, 248320),
]
ITERS = 30


def make_packed(K: int, N: int):
    gs = INT2_F16_GROUP_SIZE
    g = torch.Generator(device="cpu").manual_seed(N + K)
    codes = torch.randint(-1, 2, (K // gs, gs, N), generator=g, dtype=torch.int8)
    scales = (torch.rand(K // gs, 1, N, generator=g) * 0.05 + 0.005).to(torch.float16)
    w = (codes.to(torch.float16) * scales).reshape(K, N)
    c, s = quantize_to_ternary_f16(w, gs)
    return pack_ternary_to_int2(c).to("xpu").contiguous(), s.to("xpu").contiguous()


def main() -> None:
    if not torch.xpu.is_available():
        raise SystemExit("XPU not available")
    gemm = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run

    total_us = 0.0
    total_bytes = 0
    print(f"{'layer':28} {'n/tok':>6} {'K':>6} {'N':>7} {'us/call':>9} "
          f"{'us/token':>9} {'GB/s':>7}")
    for name, count, K, N in LAYERS:
        w, s = make_packed(K, N)
        a = (torch.randn(1, K, device="xpu") * 0.5).to(torch.float16)
        for _ in range(5):
            gemm(a, w, s, None)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            gemm(a, w, s, None)
        torch.xpu.synchronize()
        us = (time.perf_counter() - t0) / ITERS * 1e6
        # bytes: packed weights (2 bit/elem) + fp16 scales + act + out
        b = K * N // 4 + (K // 128) * N * 2 + K * 2 + N * 2
        total_us += us * count
        total_bytes += b * count
        print(f"{name:28} {count:>6} {K:>6} {N:>7} {us:>8.1f} "
              f"{us * count:>8.1f} {b / us / 1e3:>7.1f}")

    print("-" * 80)
    print(f"total GEMM time per token : {total_us:>8.1f} us")
    print(f"total bytes per token     : {total_bytes / 1e9:>8.3f} GB")
    print(f"aggregate bandwidth       : {total_bytes / total_us / 1e3:>8.1f} GB/s")
    ms = total_us / 1e3
    print(f"=> GEMM-only ceiling      : {1e3 / ms:>8.1f} tok/s")


if __name__ == "__main__":
    main()
