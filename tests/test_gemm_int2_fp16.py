"""Smoke test for the int2 x fp16 upcvt GEMM kernel and the int2_f16 quantize
helpers in xetla_vllm_plugin.

Run after building the extension:

    source /swtools/intel-gpu/.../intel_gpu_vars.sh
    source /swtools/intel/.../oneapi-vars.sh --force
    source .venv/bin/activate
    python tests/test_gemm_int2_fp16.py
"""

from __future__ import annotations

import torch

from xetla_vllm_plugin import (
    INT2_F16_GROUP_SIZE,
    pack_ternary_to_int2,
    quantize_to_ternary_f16,
)

import xetla_pt_ext  # noqa: F401  -- registers torch.ops.xetla_int2.*


def make_ternary_weight(K: int, N: int, gs: int = INT2_F16_GROUP_SIZE,
                        device: str = "xpu", seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(-1, 2, (K // gs, gs, N), generator=g, dtype=torch.int8)
    scales = (torch.rand(K // gs, 1, N, generator=g) * 2 + 0.25).to(torch.float16)
    w = (codes.to(torch.float16) * scales).reshape(K, N)
    return w.to(device)


def reference(a: torch.Tensor, w_kn: torch.Tensor) -> torch.Tensor:
    return (a.float() @ w_kn.float()).to(torch.float16)


def main() -> None:
    if not torch.xpu.is_available():
        raise SystemExit("XPU not available")

    torch.manual_seed(0)
    M, K, N = 1, 4096, 4096
    a = (torch.randn(M, K, device="xpu") * 2).to(torch.float16)
    w_kn = make_ternary_weight(K, N)

    ref = reference(a, w_kn)

    codes, scale_f16 = quantize_to_ternary_f16(w_kn.cpu(), INT2_F16_GROUP_SIZE)
    packed = pack_ternary_to_int2(codes.to("xpu"))
    scale_dev = scale_f16.to("xpu").contiguous()

    out = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run(a, packed, scale_dev, None)

    assert out.shape == (M, N), out.shape
    diff = (out.float() - ref.float()).abs()
    print(f"M={M} N={N} K={K}  max_abs={diff.max().item():.4f}  "
          f"mean_abs={diff.mean().item():.4f}")
    rel = diff.mean() / ref.float().abs().mean().clamp_min(1e-6)
    print(f"mean_rel = {rel.item():.4e}")
    assert rel.item() < 1e-2, "GEMM output diverges from reference"
    print("PASS")


if __name__ == "__main__":
    main()
