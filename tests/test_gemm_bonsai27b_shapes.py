"""Check the int2 x fp16 xetla GEMM kernels on the exact layer shapes used by
Bonsai-27B (Qwen3.5 hybrid attention), for both the decode (M=1, upcvt) and
prefill (M>1, DPAS when N%256==0) dispatch paths.

    python tests/test_gemm_bonsai27b_shapes.py
"""

from __future__ import annotations

import torch

from xetla_vllm_plugin import (
    INT2_F16_GROUP_SIZE,
    pack_ternary_to_int2,
    quantize_to_ternary_f16,
)

import xetla_pt_ext  # noqa: F401  -- registers torch.ops.xetla_int2.*

# (name, K, N) as seen by the kernel: weight is [K, N]
SHAPES = [
    ("self_attn.qkv_proj", 5120, 14336),
    ("self_attn.o_proj", 6144, 5120),
    ("mlp.gate_up_proj", 5120, 34816),
    ("mlp.down_proj", 17408, 5120),
    ("linear_attn.in_proj_qkvz", 5120, 16384),
    ("linear_attn.in_proj_ba", 5120, 96),
    ("linear_attn.out_proj", 6144, 5120),
    ("lm_head", 5120, 248320),
]


def make_ternary_weight(K: int, N: int, seed: int = 0) -> torch.Tensor:
    gs = INT2_F16_GROUP_SIZE
    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(-1, 2, (K // gs, gs, N), generator=g, dtype=torch.int8)
    scales = (torch.rand(K // gs, 1, N, generator=g) * 0.05 + 0.005).to(torch.float16)
    return (codes.to(torch.float16) * scales).reshape(K, N)


def main() -> None:
    if not torch.xpu.is_available():
        raise SystemExit("XPU not available")
    torch.manual_seed(0)

    failures = []
    for name, K, N in SHAPES:
        w_kn = make_ternary_weight(K, N)
        codes, scale_f16 = quantize_to_ternary_f16(w_kn, INT2_F16_GROUP_SIZE)
        packed = pack_ternary_to_int2(codes).to("xpu").contiguous()
        scale = scale_f16.to("xpu").contiguous()
        w_dev = w_kn.to("xpu")

        for M in (1, 7, 64):
            a = (torch.randn(M, K, device="xpu") * 0.5).to(torch.float16)
            ref = (a.float() @ w_dev.float())

            use_dpas = M > 1 and (N % 256 == 0)
            if use_dpas:
                out = torch.ops.xetla_int2.int2_fp16_dpas_gemm_run(
                    a, packed, scale, None)
            else:
                out = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run(
                    a, packed, scale, None)

            diff = (out.float() - ref).abs()
            denom = ref.abs().mean().clamp_min(1e-6)
            rel = (diff.mean() / denom).item()
            tag = "dpas " if use_dpas else "upcvt"
            ok = rel < 1e-2 and torch.isfinite(out).all()
            status = "ok  " if ok else "FAIL"
            print(f"[{status}] {name:28s} K={K:6d} N={N:6d} M={M:3d} {tag} "
                  f"mean_rel={rel:.3e} max_abs={diff.max().item():.4f}")
            if not ok:
                failures.append((name, K, N, M, rel))

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print("   ", f)
        raise SystemExit(1)
    print("\nALL SHAPES PASS")


if __name__ == "__main__":
    main()
