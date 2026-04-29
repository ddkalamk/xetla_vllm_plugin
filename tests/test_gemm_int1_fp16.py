"""Smoke test for the int1 fp16 upcvt GEMM kernel.

Compares the xetla kernel output against a reference fp16 dequant + matmul.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xetla_pt_ext  # noqa: F401  (registers torch.ops.xetla_int2.*)
from xetla_vllm_plugin import (
    INT1_F16_GROUP_SIZE,
    quantize_to_binary_f16,
    pack_int1x32,
)


def reference_int1_gemm(A_h, codes_h, scale_h, gs=INT1_F16_GROUP_SIZE):
    K, N = codes_h.shape
    # codes 0 -> +1, 1 -> -1
    sign = torch.where(codes_h == 0, torch.ones_like(codes_h), -torch.ones_like(codes_h)).to(torch.float32)
    scale_f = scale_h.to(torch.float32)  # [K/gs, N]
    scale_full = scale_f.repeat_interleave(gs, dim=0)  # [K, N]
    W = sign.to(torch.float32) * scale_full
    return (A_h.to(torch.float32) @ W).to(torch.float16)


def main():
    torch.manual_seed(0)
    M, K, N = 8, 4096, 4096
    dev = "xpu:0"

    # Make a binary-quantized fp16 weight: per-128-K group has scale s,
    # values are s * sign uniformly drawn from {-1,+1}.
    K_groups = K // INT1_F16_GROUP_SIZE
    scale_true = (torch.rand(K_groups, N, dtype=torch.float16) * 0.1 + 0.01)
    sign = torch.randint(0, 2, (K, N), dtype=torch.int8) * 2 - 1  # {-1,+1}
    W = (sign.to(torch.float16).view(K_groups, INT1_F16_GROUP_SIZE, N)
         * scale_true.unsqueeze(1)).view(K, N)

    codes, scale_f16 = quantize_to_binary_f16(W, INT1_F16_GROUP_SIZE)
    packed = pack_int1x32(codes)
    print(f"packed: {tuple(packed.shape)} {packed.dtype}, scale: {tuple(scale_f16.shape)} {scale_f16.dtype}")

    A = (torch.randn(M, K, dtype=torch.float16) * 0.1)
    A_d = A.to(dev)
    B_d = packed.to(dev)
    sc_d = scale_f16.to(dev)

    out_d = torch.ops.xetla_int2.int1_fp16_upcvt_gemm_run(A_d, B_d, sc_d, None)
    out_xetla = out_d.to("cpu")

    out_ref = reference_int1_gemm(A, codes, scale_f16)

    diff = (out_xetla.to(torch.float32) - out_ref.to(torch.float32)).abs()
    ref_abs = out_ref.to(torch.float32).abs()
    print(f"out_xetla[0,:8] = {out_xetla[0,:8]}")
    print(f"out_ref  [0,:8] = {out_ref[0,:8]}")
    print(f"max abs diff = {diff.max().item():.4g}")
    print(f"max rel diff = {(diff / (ref_abs + 1e-6)).max().item():.4g}")
    print(f"mean abs diff = {diff.mean().item():.4g}")
    assert diff.max().item() < 1e-1, "Numerical mismatch"
    print("PASS")


if __name__ == "__main__":
    main()
