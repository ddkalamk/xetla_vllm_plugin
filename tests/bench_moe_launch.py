"""Is the MoE expert loop launch-bound, and what would a grouped kernel buy?

Compares the per-expert GEMV loop we run today against the single fused GEMV
that a grouped kernel would issue for the same arithmetic:

  w13   8 x (K=2048, N=1536)   vs   1 x (K=2048, N=12288)
  w2    8 x (K=768,  N=2048)   vs   1 x (K=6144, N=2048)

The w2 concatenation is exact, not just a size-alike: stacking the selected
experts along K and pre-scaling each activation slice by its routing weight
makes one GEMV compute the whole weighted expert sum.
"""

import time

import torch
import xetla_pt_ext  # noqa: F401  (registers torch.ops.xetla_int2)

DEV = "xpu"
GS = 128
TOPK = 8
LAYERS = 48
gemm = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run


def make(k, n):
    qw = torch.randint(0, 2**31, (k // 16, n), dtype=torch.int32, device=DEV)
    sc = torch.randn(k // GS, n, dtype=torch.float16, device=DEV) * 0.02
    return qw, sc


def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def bytes_moved(k, n):
    return k * n * 2 // 8 + (k // GS) * n * 2


def main():
    torch.manual_seed(0)
    H, I = 2048, 768

    x = torch.randn(1, H, dtype=torch.float16, device=DEV)
    w13 = [make(H, 2 * I) for _ in range(TOPK)]
    w2 = [make(I, H) for _ in range(TOPK)]
    act = torch.randn(1, I, dtype=torch.float16, device=DEV)

    # Grouped equivalents.
    g13_q, g13_s = make(H, TOPK * 2 * I)
    g2_q, g2_s = make(TOPK * I, H)
    act_cat = torch.randn(1, TOPK * I, dtype=torch.float16, device=DEV)

    print(f"{'case':<34}{'us':>9}{'GiB/s':>9}")

    def row(name, us, nbytes):
        print(f"{name:<34}{us:>9.1f}{nbytes / us / 1073.741824:>9.1f}")

    one13 = bench(lambda: gemm(x, w13[0][0], w13[0][1], None))
    row("1x w13 (K=2048,N=1536)", one13, bytes_moved(H, 2 * I))

    loop13 = bench(lambda: [gemm(x, q, s, None) for q, s in w13])
    row("8x w13 loop", loop13, TOPK * bytes_moved(H, 2 * I))

    grp13 = bench(lambda: gemm(x, g13_q, g13_s, None))
    row("1x w13 grouped (N=12288)", grp13, bytes_moved(H, TOPK * 2 * I))

    loop2 = bench(lambda: [gemm(act, q, s, None) for q, s in w2])
    row("8x w2 loop", loop2, TOPK * bytes_moved(I, H))

    grp2 = bench(lambda: gemm(act_cat, g2_q, g2_s, None))
    row("1x w2 grouped (K=6144)", grp2, bytes_moved(TOPK * I, H))

    # Full layer as apply() runs it today, including the elementwise tail.
    def layer_now():
        acc = torch.zeros_like(x)
        for i in range(TOPK):
            h = gemm(x, w13[i][0], w13[i][1], None)
            gate, up = h.chunk(2, dim=-1)
            a = torch.nn.functional.silu(gate) * up
            y = gemm(a.contiguous(), w2[i][0], w2[i][1], None)
            acc += y * 0.125
        return acc

    def layer_grouped():
        h = gemm(x, g13_q, g13_s, None).view(TOPK, 2 * I)
        a = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
        return gemm(a.reshape(1, TOPK * I).contiguous(), g2_q, g2_s, None)

    now = bench(layer_now, iters=100)
    grp = bench(layer_grouped, iters=100)
    moe_bytes = TOPK * (bytes_moved(H, 2 * I) + bytes_moved(I, H))
    row("full layer: per-expert loop", now, moe_bytes)
    row("full layer: grouped", grp, moe_bytes)

    print()
    print(f"launch floor (1 tiny GEMV)   : {one13:.1f} us")
    print(f"per-token MoE, loop          : {now * LAYERS / 1000:.1f} ms")
    print(f"per-token MoE, grouped       : {grp * LAYERS / 1000:.1f} ms")
    print(f"speedup                      : {now / grp:.2f}x")
    print(f"tok/s ceiling from MoE alone : {1000 / (grp * LAYERS / 1000):.0f}")


if __name__ == "__main__":
    main()
