"""Which int8 path, if any, beats fp16 for the lm_head GEMV.

    ZE_AFFINITY_MASK=0 python scripts/bench_lm_head_int8.py [--k 4096 --n 151936]

Decode hits lm_head with M=1, so it is purely a weight-streaming problem:
fp16 reads K*N*2 bytes, int8 reads K*N. The question is whether any available
int8 kernel actually converts that halving into time, or whether it falls off
a tuned path the way fp8 _scaled_mm does (0.90x, i.e. slower than fp16).
"""
from __future__ import annotations

import argparse
import time

import torch


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--n", type=int, default=151936)
    p.add_argument("--m", type=int, default=1)
    a = p.parse_args()
    M, K, N = a.m, a.k, a.n
    dev = "xpu"

    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    w = torch.randn(K, N, device=dev, dtype=torch.float16) * 0.02

    ref = (x.float() @ w.float())
    print(f"shape M={M} K={K} N={N}")
    print(f"{'path':<26}{'us':>10}{'GB/s':>9}{'vs fp16':>9}{'max_rel_err':>13}")
    print("-" * 67)

    t16 = bench(lambda: torch.mm(x, w))
    print(f"{'fp16 torch.mm':<26}{t16*1e6:>10.1f}{K*N*2/t16/1e9:>9.1f}"
          f"{1.0:>9.2f}{0.0:>13.1e}")

    # per-output-column symmetric int8, the layout a packer would emit
    scale = (w.abs().amax(dim=0) / 127.0).clamp_min(1e-8).to(torch.float16)
    w8 = torch.round(w.float() / scale.float()).clamp(-127, 127).to(torch.int8)

    def rel(out):
        return ((out.float() - ref).abs().max()
                / ref.abs().max().clamp_min(1e-6)).item()

    # 1. dequantize then mm: reads int8 but materialises fp16, so it should be
    #    slower than fp16 alone. Included to show the cost of not fusing.
    def deq_mm():
        return torch.mm(x, w8.to(torch.float16) * scale)
    try:
        t = bench(deq_mm, iters=5)
        print(f"{'int8 dequant + mm':<26}{t*1e6:>10.1f}{K*N/t/1e9:>9.1f}"
              f"{t16/t:>9.2f}{rel(deq_mm()):>13.1e}")
    except Exception as exc:
        print(f"{'int8 dequant + mm':<26} FAILED {str(exc).splitlines()[0][:38]}")

    # 2. torch._int_mm: int8 x int8 -> int32, the closest thing to a native path
    try:
        x8 = torch.round(x.float() / (x.abs().max() / 127)).clamp(-127, 127).to(torch.int8)
        xs = (x.abs().max() / 127).to(torch.float16)
        out = torch._int_mm(x8, w8)
        t = bench(lambda: torch._int_mm(x8, w8))
        o = torch._int_mm(x8, w8).float() * xs.float() * scale.float()
        print(f"{'torch._int_mm':<26}{t*1e6:>10.1f}{K*N/t/1e9:>9.1f}"
              f"{t16/t:>9.2f}{(o - ref).abs().max().item()/ref.abs().max().item():>13.1e}")
    except Exception as exc:
        print(f"{'torch._int_mm':<26} FAILED {str(exc).splitlines()[0][:38]}")

    # 3. oneDNN qlinear_pointwise, the tuned int8 entry point
    try:
        op = torch.ops.onednn.qlinear_pointwise
        xq = torch.quantize_per_tensor(x.float(), 0.01, 0, torch.qint8)
        t = bench(lambda: op(xq, w8, scale, None, None, None, 1.0, 0,
                             torch.float16, "none", [], ""))
        print(f"{'onednn qlinear_pointwise':<26}{t*1e6:>10.1f}{K*N/t/1e9:>9.1f}"
              f"{t16/t:>9.2f}")
    except Exception as exc:
        print(f"{'onednn qlinear_pointwise':<26} FAILED {str(exc).splitlines()[0][:38]}")

    # 4. the existing int2 kernel, for scale: it is the tuned M=1 path we have
    try:
        import xetla_pt_ext  # noqa: F401
        gs = 128
        qw = torch.randint(0, 255, (K // 16, N), dtype=torch.int32, device=dev)
        sc = torch.rand(K // gs, N, device=dev, dtype=torch.float16) + 0.5
        g = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
        t = bench(lambda: g(x, qw, sc, None))
        print(f"{'int2 upcvt (reference)':<26}{t*1e6:>10.1f}"
              f"{(K*N//4 + (K//gs)*N*2)/t/1e9:>9.1f}{t16/t:>9.2f}")
    except Exception as exc:
        print(f"{'int2 upcvt (reference)':<26} FAILED {str(exc).splitlines()[0][:38]}")


if __name__ == "__main__":
    main()
