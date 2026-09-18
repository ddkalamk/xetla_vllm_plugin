"""XPU check + sweep of the int2 upcvt GEMM's M-tiled prefill tiers against
the legacy M=1-tile fallback (XETLA_INT2_PREFILL_CFG=-1). The env var is read
once per process, so each cfg runs in a subprocess.

    python tests/test_int2_prefill_mtile.py           # correctness + sweep
"""
import os
import subprocess
import sys
import time

CFGS = [-1, 0, 1, 2, 3, 4, 5]
SHAPES = [(5120, 16384, "in_proj_qkvz"), (5120, 34816, "gate_up"),
          (17408, 5120, "down"), (6144, 5120, "out_proj"),
          (5120, 14336, "qkv_proj"), (5120, 248320, "lm_head")]
MS = [2, 4, 8, 16, 63, 128, 512]


def worker(cfg: int):
    import torch
    import xetla_pt_ext  # noqa: F401
    dev = "xpu"
    torch.manual_seed(0)
    op = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
    out = {}
    for K, N, name in SHAPES:
        codes = torch.randint(0, 3, (K // 16, N, 16), device=dev, dtype=torch.int32)
        codes = torch.where(codes == 2, torch.full_like(codes, 3), codes)  # {0,1,3}
        W = (codes << (torch.arange(16, device=dev, dtype=torch.int32) * 2)).sum(-1).to(torch.int32)
        S = (torch.rand(K // 128, N, device=dev) * 0.02).half()
        for M in MS:
            if N == 248320 and M > 63:
                continue
            x = torch.randn(M, K, device=dev).half()
            y = op(x, W, S, None)
            torch.xpu.synchronize()
            for _ in range(3):
                op(x, W, S, None)
            torch.xpu.synchronize()
            n = 20 if M <= 63 else 5
            t0 = time.perf_counter()
            for _ in range(n):
                op(x, W, S, None)
            torch.xpu.synchronize()
            us = (time.perf_counter() - t0) / n * 1e6
            # checksum for cross-process comparison + reference for cfg -1
            out[(name, M)] = (us, y.float().cpu())
    torch.save(out, f"/tmp/int2_mtile_cfg{cfg}.pt")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
        return
    import torch
    here = os.path.abspath(__file__)
    res = {}
    for cfg in CFGS:
        env = dict(os.environ, XETLA_INT2_PREFILL_CFG=str(cfg))
        r = subprocess.run([sys.executable, here, "--worker", str(cfg)], env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"cfg {cfg} FAILED:\n{r.stderr[-2000:]}")
            continue
        res[cfg] = torch.load(f"/tmp/int2_mtile_cfg{cfg}.pt")
    ref = res.get(-1)
    for K, N, name in SHAPES:
        print(f"\n{name} K={K} N={N}   (us per GEMM; rel.err vs legacy GEMV tier)")
        print("  M     " + "".join(f"cfg{c:>3}          " for c in res))
        for M in MS:
            if (name, M) not in ref:
                continue
            row = f"  {M:<5d} "
            for c, d in res.items():
                if (name, M) not in d:
                    row += f"{'-':>16s}"
                    continue
                us, y = d[(name, M)]
                err = (y - ref[(name, M)][1]).abs().max().item() / (ref[(name, M)][1].abs().max().item() + 1e-9)
                row += f"{us:8.1f} {err:7.1e}"
            print(row)


if __name__ == "__main__":
    main()
