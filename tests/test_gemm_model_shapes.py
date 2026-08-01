"""Check the int2 fp16 GEMM against a dense reference on real model shapes.

The CAT-Q Qwen3-32B produced garbage on GPU while its dense CPU reference was
fine and the sidecar unpacked bit-exact, which leaves the kernel. Its shapes hit
different dispatch tiers than anything tuned so far, so sweep the ones that
matter at both decode (M=1) and prefill (M>1).
"""

import argparse

import torch
import xetla_pt_ext  # noqa: F401

DEV = "xpu"
GS = 128

# (name, K, N) for the shapes each model actually issues.
SHAPES = {
    "qwen3-32B": [
        ("qkv_proj", 5120, 10240),
        ("o_proj", 8192, 5120),
        ("gate_up_proj", 5120, 51200),
        ("down_proj", 25600, 5120),
    ],
    "qwen3-8b": [
        ("qkv_proj", 4096, 6144),
        ("o_proj", 4096, 4096),
        ("gate_up_proj", 4096, 24576),
        ("down_proj", 12288, 4096),
    ],
}


def make(k, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 3, (k, n), generator=g, dtype=torch.int64)
    codes[codes == 2] = 3
    words = torch.zeros(k // 16, n, dtype=torch.int64)
    for j in range(16):
        words |= codes[j::16, :] << (2 * j)
    vals = torch.where(codes == 1, 1.0, torch.where(codes == 3, -1.0, 0.0))
    sc = (torch.randn(k // GS, n, generator=g) * 0.05).half()
    dense = (vals * sc.float().repeat_interleave(GS, dim=0)).half()
    return words.to(torch.int32).to(DEV).contiguous(), sc.to(DEV).contiguous(), dense.to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-32B", choices=sorted(SHAPES))
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 8, 64])
    args = ap.parse_args()

    gemm = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
    dpas = torch.ops.xetla_int2.int2_fp16_dpas_gemm_run

    print(f"{'shape':<34}{'M':>5}{'kernel':>8}{'mean_rel':>12}{'max_rel':>12}")
    bad = False
    for name, k, n in SHAPES[args.model]:
        qw, sc, dense = make(k, n)
        for m in args.rows:
            a = (torch.randn(m, k) * 0.5).half().to(DEV)
            ref = (a.float() @ dense.float())
            scale = ref.abs().mean().clamp_min(1e-6)
            for label, fn in (("upcvt", gemm), ("dpas", dpas)):
                if label == "dpas" and m == 1:
                    continue  # decode never takes the DPAS path
                out = fn(a, qw, sc, None).float()
                err = (out - ref).abs()
                mean_rel = (err.mean() / scale).item()
                max_rel = (err.max() / scale).item()
                flag = "" if mean_rel < 2e-2 else "   <-- BAD"
                if mean_rel >= 2e-2:
                    bad = True
                print(f"{name+f' K={k} N={n}':<34}{m:>5}{label:>8}"
                      f"{mean_rel:>12.3e}{max_rel:>12.3e}{flag}")
    print("FAIL" if bad else "PASS")


if __name__ == "__main__":
    main()
