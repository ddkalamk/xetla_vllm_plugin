"""Correctness and bandwidth for the batched MoE expert GEMV.

Checks torch.ops.xetla_int2.int2_fp16_moe_gemv_run against the per-expert loop
built on the tuned single-shape kernel, on the two Qwen3-30B-A3B expert shapes.
"""

import time

import torch
import xetla_pt_ext  # noqa: F401

DEV = "xpu"
GS = 128
gemm = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
moe = torch.ops.xetla_int2.int2_fp16_moe_gemv_run


def make_experts(e, k, n, seed=0):
    """Random ternary codes {0, 1, 3} packed 16-along-K per int32, plus scales."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 3, (e, k, n), generator=g, dtype=torch.int64)
    codes[codes == 2] = 3
    words = torch.zeros(e, k // 16, n, dtype=torch.int64)
    for j in range(16):
        words |= codes[:, j::16, :] << (2 * j)
    qw = words.to(torch.int32).to(DEV).contiguous()
    sc = (torch.randn(e, k // GS, n, generator=g) * 0.05).half().to(DEV).contiguous()
    return qw, sc


def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def check(name, e, g, k, n, broadcast):
    qw, sc = make_experts(e, k, n)
    ids = torch.randperm(e)[:g].to(torch.int32).to(DEV)
    rows = 1 if broadcast else g
    a = (torch.randn(rows, k) * 0.5).half().to(DEV)

    out = moe(a, qw, sc, ids)

    idl = ids.tolist()
    ref = torch.stack([
        gemm(a[0:1] if broadcast else a[i : i + 1], qw[eid], sc[eid], None)[0]
        for i, eid in enumerate(idl)
    ])

    err = (out.float() - ref.float()).abs()
    rel = (err / ref.float().abs().clamp_min(1e-3)).mean().item()

    nbytes = g * (k * n * 2 // 8 + (k // GS) * n * 2)
    t_moe = bench(lambda: moe(a, qw, sc, ids))
    t_loop = bench(
        lambda: [
            gemm(a[0:1] if broadcast else a[i : i + 1], qw[eid], sc[eid], None)
            for i, eid in enumerate(idl)
        ]
    )
    ok = "OK " if rel < 5e-3 else "BAD"
    print(
        f"{ok} {name:<26} mean_rel {rel:.2e}   "
        f"loop {t_loop:6.1f}us ({nbytes / t_loop / 1073.741824:5.1f} GiB/s)   "
        f"batched {t_moe:6.1f}us ({nbytes / t_moe / 1073.741824:5.1f} GiB/s)   "
        f"{t_loop / t_moe:.2f}x"
    )
    return rel < 5e-3


def main():
    ok = True
    # Qwen3-30B-A3B: hidden 2048, moe_intermediate 768, 128 experts, top-8.
    ok &= check("w13 K=2048 N=1536 bcast", 128, 8, 2048, 1536, True)
    ok &= check("w2  K=768  N=2048", 128, 8, 768, 2048, False)
    # A few odd shapes to exercise the COLS / LS picking.
    ok &= check("K=1024 N=512", 16, 4, 1024, 512, True)
    ok &= check("K=384  N=1536", 16, 8, 384, 1536, False)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
