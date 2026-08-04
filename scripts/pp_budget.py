"""Per-token time budget for one pipeline stage, measured at real 235B shapes.

    ZE_AFFINITY_MASK=3 python scripts/pp_budget.py [--pp 4]

Pure pipeline parallel keeps every GEMM full width and only hands a [1, hidden]
tensor to the next stage, so the interesting question is what fraction of a
token is actually spent in the quantized GEMMs. Everything not accounted for
here is attention, norms, routing and dispatch -- i.e. what XPU graphs would
target.
"""
from __future__ import annotations

import argparse
import time

import torch
import xetla_pt_ext  # noqa: F401

GS = 128
LAYERS = 94
HIDDEN = 4096
TOP_K = 8
N_EXPERTS = 128
MOE_INTER = 1536


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters


def dense(m, k, n):
    a = torch.randn(m, k, device="xpu", dtype=torch.float16)
    qw = torch.randint(0, 255, (k // 16, n), dtype=torch.int32, device="xpu")
    sc = torch.rand(k // GS, n, device="xpu", dtype=torch.float16) + 0.5
    op = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
    return bench(lambda: op(a, qw, sc, None))


def moe(e, g, k, n):
    """Batched expert GEMV: only the g selected experts are read."""
    qw = torch.randint(0, 255, (e, k // 16, n), dtype=torch.int32, device="xpu")
    sc = (torch.randn(e, k // GS, n) * 0.05).half().to("xpu").contiguous()
    a = torch.randn(g, k, device="xpu", dtype=torch.float16)
    ids = torch.randperm(e)[:g].to(torch.int32).to("xpu")
    op = torch.ops.xetla_int2.int2_fp16_moe_gemv_run
    try:
        op(a, qw, sc, ids)
    except Exception as exc:
        return None, str(exc).split("\n")[0][:60]
    return bench(lambda: op(a, qw, sc, ids)), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pp", type=int, default=4)
    p.add_argument("--measured-tok-s", type=float, default=17.49)
    a = p.parse_args()

    qkv = dense(1, HIDDEN, 9216)
    o = dense(1, 8192, HIDDEN)
    w13, err13 = moe(N_EXPERTS, TOP_K, HIDDEN, 2 * MOE_INTER)
    w2, err2 = moe(N_EXPERTS, TOP_K, MOE_INTER, HIDDEN)

    print(f"qkv_proj          {qkv * 1e6:8.1f} us")
    print(f"o_proj            {o * 1e6:8.1f} us")
    if err13 or err2:
        print(f"moe batched       unavailable: {err13 or err2}")
        moe_layer = 0.0
    else:
        print(f"moe w13 (batched) {w13 * 1e6:8.1f} us   {TOP_K} of {N_EXPERTS} experts")
        print(f"moe w2  (batched) {w2 * 1e6:8.1f} us")
        moe_layer = w13 + w2

    per_layer = qkv + o + moe_layer
    gemm_tok = per_layer * LAYERS
    budget = 1.0 / a.measured_tok_s

    print()
    print(f"per layer         {per_layer * 1e6:8.1f} us")
    print(f"gemm per token    {gemm_tok * 1e3:8.2f} ms   ({LAYERS} layers)")
    print(f"measured token    {budget * 1e3:8.2f} ms   ({a.measured_tok_s} tok/s)")
    print(f"gemm share        {100 * gemm_tok / budget:8.1f} %")
    print(f"everything else   {(budget - gemm_tok) * 1e3:8.2f} ms   "
          f"({100 * (1 - gemm_tok / budget):.1f} %)")
    print()
    print(f"pp={a.pp}: stage handoff is one [1,{HIDDEN}] fp16 tensor "
          f"({a.pp - 1} hops/token, ~30 us each = "
          f"{(a.pp - 1) * 30e-6 * 1e3:.2f} ms)")


if __name__ == "__main__":
    main()
