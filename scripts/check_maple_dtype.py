"""fp16 vs bf16 for the int2 path, on real Maple weights.

    ZE_AFFINITY_MASK=0 python scripts/check_maple_dtype.py

The plugin downcasts activations to fp16 before the int2 gemm. That was the
right call for CAT-Q, whose export is fp16 by construction, but Maple ships
bf16. bf16 has 8 exponent bits against fp16's 5, so the question is whether
Maple's activations carry values fp16 cannot hold, and whether the bf16 kernel
is closer to an fp32 reference.
"""
from __future__ import annotations

import glob

import torch
import xetla_pt_ext  # noqa: F401
from safetensors import safe_open

import sys
sys.path.insert(0, ".")
from xetla_vllm_plugin import (  # noqa: E402
    INT2_F16_GROUP_SIZE as GS,
    pack_ternary_to_int2,
    quantize_to_ternary_f16,
)

SNAP = glob.glob("/data/nfs_home/egeorgan/.cache/huggingface/hub/"
                 "models--deepgrove--maple-preview/snapshots/*")[0]


def load(name_frag, limit=1):
    out = []
    for f in sorted(glob.glob(SNAP + "/*.safetensors")):
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                if name_frag in k:
                    out.append((k, fh.get_tensor(k)))
                    if len(out) >= limit:
                        return out
    return out


def run(name, w_bf16, rms):
    """w is [out, in] as stored; the kernel wants [K=in, N=out]."""
    wkn32 = w_bf16.float().t().contiguous()
    K, N = wkn32.shape
    torch.manual_seed(0)
    x32 = torch.randn(1, K) * rms

    ref = (x32 @ wkn32)                                   # fp32 reference

    codes, scale16 = quantize_to_ternary_f16(wkn32.half(), GS)
    qw = pack_ternary_to_int2(codes).to("xpu")
    s16 = scale16.to("xpu").contiguous()

    x16 = x32.half().to("xpu")
    o16 = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run(x16, qw, s16, None)

    line = f"{name:<26} K={K:<6} N={N:<6} act_rms={rms:<6}"
    e16 = (o16.float().cpu() - ref).abs().max().item() / ref.abs().max().item()
    line += f" fp16 {e16:.3e}"

    try:
        xb = x32.bfloat16().to("xpu")
        sb = scale16.bfloat16().to("xpu").contiguous()
        ob = torch.ops.xetla_int2.int2_bf16_upcvt_gemm_run(xb, qw, sb, None)
        eb = (ob.float().cpu() - ref).abs().max().item() / ref.abs().max().item()
        line += f"   bf16 {eb:.3e}   winner {'bf16' if eb < e16 else 'fp16'}"
    except Exception as exc:
        line += f"   bf16 FAILED: {str(exc).splitlines()[0][:44]}"
    print(line)


def main():
    print(f"fp16 max representable: {torch.finfo(torch.float16).max}")
    print(f"fp16 min normal      : {torch.finfo(torch.float16).tiny}\n")

    for frag in ("layers.0.self_attn.q_proj.weight",
                 "layers.0.self_attn.o_proj.weight",
                 "layers.0.mlp.experts.0.gate_proj.weight",
                 "layers.0.mlp.experts.0.down_proj.weight"):
        got = load(frag)
        if not got:
            print(f"{frag}: not found")
            continue
        k, w = got[0]
        sc = w.float().abs()
        sc = sc[sc > 0]
        print(f"# {k}\n#   alpha min {sc.min():.3e} max {sc.max():.3e} "
              f"(fp16 tiny {torch.finfo(torch.float16).tiny:.1e})")
        for rms in (1.0, 8.0, 64.0):
            run(k.split("model.")[-1], w, rms)
        print()


if __name__ == "__main__":
    main()
