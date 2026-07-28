"""Compare the two int2 x fp16 kernels at decode shapes (M=1): the upcvt
(GEMV-tuned) kernel currently used for decode vs the DPAS / int8-XMX kernel
currently reserved for prefill.

Reports both speed and accuracy against an fp32 reference, since the DPAS
kernel converts activations to int8.

    python tests/bench_decode_upcvt_vs_dpas.py [--m 1]
"""

from __future__ import annotations

import argparse
import time

import torch

from xetla_vllm_plugin import (
    INT2_F16_GROUP_SIZE,
    pack_ternary_to_int2,
    quantize_to_ternary_f16,
)

import xetla_pt_ext  # noqa: F401

# (name, count per token, K, N) for Bonsai-27B
SHAPES = [
    ("mlp.gate_up_proj", 64, 5120, 34816),
    ("mlp.down_proj", 64, 17408, 5120),
    ("linear_attn.in_proj_qkvz", 48, 5120, 16384),
    ("linear_attn.out_proj", 48, 6144, 5120),
    ("self_attn.qkv_proj", 16, 5120, 14336),
    ("self_attn.o_proj", 16, 6144, 5120),
    ("lm_head", 1, 5120, 248320),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--iters", type=int, default=30)
    return p.parse_args()


def make_packed(K: int, N: int):
    gs = INT2_F16_GROUP_SIZE
    g = torch.Generator(device="cpu").manual_seed(N + K)
    codes = torch.randint(-1, 2, (K // gs, gs, N), generator=g, dtype=torch.int8)
    scales = (torch.rand(K // gs, 1, N, generator=g) * 0.05 + 0.005).to(torch.float16)
    w = (codes.to(torch.float16) * scales).reshape(K, N)
    c, s = quantize_to_ternary_f16(w, gs)
    return (pack_ternary_to_int2(c).to("xpu").contiguous(),
            s.to("xpu").contiguous(), w.to("xpu"))


def timeit(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> None:
    a = parse_args()
    if not torch.xpu.is_available():
        raise SystemExit("XPU not available")
    upcvt = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
    dpas = torch.ops.xetla_int2.int2_fp16_dpas_gemm_run

    print(f"M = {a.m}\n")
    print(f"{'layer':26} {'N':>7} {'upcvt us':>9} {'dpas us':>9} {'speedup':>8} "
          f"{'upcvt err':>10} {'dpas err':>10}")
    tot_u = tot_d = 0.0
    for name, count, K, N in SHAPES:
        w, s, w_ref = make_packed(K, N)
        x = (torch.randn(a.m, K, device="xpu") * 0.5).to(torch.float16)
        ref = x.float() @ w_ref.float()

        if N % 256 != 0:
            print(f"{name:26} {N:>7}   (N not a multiple of 256: DPAS n/a)")
            continue

        t_u = timeit(lambda: upcvt(x, w, s, None), a.iters)
        t_d = timeit(lambda: dpas(x, w, s, None), a.iters)
        o_u = upcvt(x, w, s, None).float()
        o_d = dpas(x, w, s, None).float()
        den = ref.abs().mean().clamp_min(1e-6)
        e_u = ((o_u - ref).abs().mean() / den).item()
        e_d = ((o_d - ref).abs().mean() / den).item()
        tot_u += t_u * count
        tot_d += t_d * count
        print(f"{name:26} {N:>7} {t_u:>8.1f} {t_d:>8.1f} {t_u / t_d:>7.2f}x "
              f"{e_u:>10.2e} {e_d:>10.2e}")

    print("-" * 88)
    print(f"per-token total (DPAS-capable layers): upcvt {tot_u:.0f} us  "
          f"dpas {tot_d:.0f} us  -> {tot_u / tot_d:.2f}x")


if __name__ == "__main__":
    main()
