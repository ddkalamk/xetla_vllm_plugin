"""fp16 vs bf16 int2 GEMM on real packed weights.

Both paths run the same kernel - the int2 dequant is uint16 bit work, so the
only difference is which 16-bit float the scales and activations live in. The
reference is fp32, computed from the exactly-dequantized weights, so this says
which format actually tracks the ideal result more closely.
"""

import argparse

import torch
import xetla_pt_ext  # noqa: F401
from safetensors import safe_open

DEV = "xpu"
GS = 128


def unpack(qw, sc):
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
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--act-scale", type=float, default=1.0,
                    help="multiply activations, to probe fp16 range limits")
    args = ap.parse_args()

    fp16_gemm = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
    bf16_gemm = torch.ops.xetla_int2.int2_bf16_upcvt_gemm_run

    print(f"{'tensor':<40}{'M':>4}{'fp16 rel':>12}{'bf16 rel':>12}{'winner':>10}")
    fp16_wins = bf16_wins = 0
    with safe_open(args.sidecar, framework="pt") as f:
        names = sorted({k[:-len(".qweight")] for k in f.keys()
                        if k.endswith(".qweight") and f".layers.{args.layer}." in k})
        for name in names:
            qw = f.get_tensor(name + ".qweight")
            sc = f.get_tensor(name + ".scale")
            ref_w = unpack(qw, sc)                       # exact fp32 weights
            K, N = ref_w.shape
            ref_w_d = ref_w.to(DEV)
            qw_d = qw.to(DEV)
            sc_f16 = sc.to(torch.float16).to(DEV)
            sc_bf16 = sc.to(torch.bfloat16).to(DEV)
            for m in args.rows:
                a32 = (torch.randn(m, K) * 0.5 * args.act_scale).to(DEV)
                exp = a32 @ ref_w_d                       # fp32 reference
                den = exp.abs().mean().clamp_min(1e-9)

                o_f = fp16_gemm(a32.half().contiguous(), qw_d, sc_f16, None).float()
                o_b = bf16_gemm(a32.bfloat16().contiguous(), qw_d, sc_bf16, None).float()
                r_f = ((o_f - exp).abs().mean() / den).item()
                r_b = ((o_b - exp).abs().mean() / den).item()
                win = "fp16" if r_f < r_b else "bf16"
                if r_f < r_b:
                    fp16_wins += 1
                else:
                    bf16_wins += 1
                short = name.replace("model.layers.", "L").replace("language_model.", "")
                print(f"{short:<40}{m:>4}{r_f:>12.3e}{r_b:>12.3e}{win:>10}")
    print(f"\nfp16 more accurate in {fp16_wins} cases, bf16 in {bf16_wins}")


if __name__ == "__main__":
    main()
