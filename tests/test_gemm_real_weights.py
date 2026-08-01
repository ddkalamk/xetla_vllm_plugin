"""Run the int2 kernel on real packed weights, not synthetic ones.

tests/test_gemm_model_shapes.py passes with random ternary codes and random
scales, yet CAT-Q models produce garbage end to end. This takes the actual
qweight/scale pairs out of a sidecar and the matching dense tensors out of the
export, and compares the kernel against a dense reference on the real
distribution - real sparsity, real scale magnitudes, real per-group structure.
"""

import argparse
import json
import os

import torch
import xetla_pt_ext  # noqa: F401
from safetensors import safe_open

DEV = "xpu"
GS = 128


def unpack(qw, sc):
    """Sidecar -> dense [K, N] fp32, mirroring the kernel's code mapping."""
    K = qw.shape[0] * 16
    out = torch.zeros(K, qw.shape[1], dtype=torch.float32)
    q = qw.to(torch.int64)
    for j in range(16):
        c = (q >> (2 * j)) & 3
        out[j::16, :] = torch.where(c == 1, 1.0, torch.where(c == 3, -1.0, 0.0)).float()
    return out * sc.float().repeat_interleave(GS, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 16])
    args = ap.parse_args()

    gemm = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
    print(f"{'tensor':<44}{'M':>4}{'mean_rel':>12}{'max_rel':>12}{'zeros%':>9}{'scale_absmax':>14}")
    bad = False

    with safe_open(args.sidecar, framework="pt") as f:
        keys = [k[:-len(".qweight")] for k in f.keys() if k.endswith(".qweight")]
        for layer in args.layers:
            for name in [k for k in keys if f".layers.{layer}." in k]:
                qw = f.get_tensor(name + ".qweight")
                sc = f.get_tensor(name + ".scale")
                ref_w = unpack(qw, sc).to(DEV)               # [K, N]
                K, N = ref_w.shape
                zeros = (ref_w == 0).float().mean().item() * 100
                qw_d, sc_d = qw.to(DEV), sc.to(DEV)
                for m in args.rows:
                    a = (torch.randn(m, K) * 0.5).half().to(DEV)
                    out = gemm(a, qw_d, sc_d, None).float()
                    exp = a.float() @ ref_w
                    scale = exp.abs().mean().clamp_min(1e-9)
                    mean_rel = ((out - exp).abs().mean() / scale).item()
                    max_rel = ((out - exp).abs().max() / scale).item()
                    flag = "" if mean_rel < 2e-2 else "  <-- BAD"
                    if mean_rel >= 2e-2:
                        bad = True
                    print(f"{name.replace('model.layers.','L'):<44}{m:>4}"
                          f"{mean_rel:>12.3e}{max_rel:>12.3e}{zeros:>9.1f}"
                          f"{sc.float().abs().max().item():>14.3e}{flag}")
    print("FAIL" if bad else "PASS")


if __name__ == "__main__":
    main()
