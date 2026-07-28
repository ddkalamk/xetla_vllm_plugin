"""Micro-benchmark: is it faster to fuse the GDN `in_proj_qkvz` (N=16384) and
`in_proj_ba` (N=96) projections into a single N=16480 int2 GEMM?

Both read the same activation, and the N=96 GEMM is latency-bound (~21 us for
~140 KB), so fusing should save most of that -- but the fused output has to be
split back into two contiguous tensors, which costs a copy that grows with M.

    python tests/bench_fused_in_proj.py
"""

from __future__ import annotations

import torch

from xetla_vllm_plugin import (
    INT2_F16_GROUP_SIZE,
    pack_ternary_to_int2,
    quantize_to_ternary_f16,
)

import xetla_pt_ext  # noqa: F401

K = 5120
N_QKVZ = 16384
N_BA = 96
ITERS = 50


def make_packed(K: int, N: int):
    gs = INT2_F16_GROUP_SIZE
    g = torch.Generator(device="cpu").manual_seed(N)
    codes = torch.randint(-1, 2, (K // gs, gs, N), generator=g, dtype=torch.int8)
    scales = (torch.rand(K // gs, 1, N, generator=g) * 0.05 + 0.005).to(torch.float16)
    w = (codes.to(torch.float16) * scales).reshape(K, N)
    c, s = quantize_to_ternary_f16(w, gs)
    return pack_ternary_to_int2(c).to("xpu").contiguous(), s.to("xpu").contiguous()


def timeit(fn, iters: int = ITERS) -> float:
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def main() -> None:
    if not torch.xpu.is_available():
        raise SystemExit("XPU not available")

    w_q, s_q = make_packed(K, N_QKVZ)
    w_b, s_b = make_packed(K, N_BA)
    w_f = torch.cat([w_q, w_b], dim=1).contiguous()
    s_f = torch.cat([s_q, s_b], dim=1).contiguous()

    gemm = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
    dpas = torch.ops.xetla_int2.int2_fp16_dpas_gemm_run

    print(f"{'M':>6} {'separate':>12} {'fused':>10} {'fused+split':>13} "
          f"{'speedup':>9}")
    for m in (1, 2, 8, 64, 512, 2048):
        a = (torch.randn(m, K, device="xpu") * 0.5).to(torch.float16)

        def sep():
            # N=16384 is a multiple of 256 -> prefill uses the DPAS kernel;
            # N=96 is not, so it always goes through upcvt.
            if m > 1:
                o1 = dpas(a, w_q, s_q, None)
            else:
                o1 = gemm(a, w_q, s_q, None)
            o2 = gemm(a, w_b, s_b, None)
            return o1, o2

        def fused_only():
            return gemm(a, w_f, s_f, None)

        def fused_split():
            o = gemm(a, w_f, s_f, None)
            return o[:, :N_QKVZ].contiguous(), o[:, N_QKVZ:].contiguous()

        t_sep = timeit(sep)
        t_fus = timeit(fused_only)
        t_fs = timeit(fused_split)
        print(f"{m:>6} {t_sep:>10.1f}us {t_fus:>8.1f}us {t_fs:>11.1f}us "
              f"{t_sep / t_fs:>8.2f}x")

    # correctness of the fused split
    a = (torch.randn(4, K, device="xpu") * 0.5).to(torch.float16)
    o = gemm(a, w_f, s_f, None)
    r1 = gemm(a, w_q, s_q, None)
    r2 = gemm(a, w_b, s_b, None)
    e1 = (o[:, :N_QKVZ].float() - r1.float()).abs().max().item()
    e2 = (o[:, N_QKVZ:].float() - r2.float()).abs().max().item()
    print(f"\nsplit correctness: qkvz max_err={e1:.3e} ba max_err={e2:.3e}")


if __name__ == "__main__":
    main()
