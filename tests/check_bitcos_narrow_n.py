"""Correctness check for the Bonsai 27B shapes, including the N=96 linear-attention
projection that does not divide the 64-wide prefill tile."""
import torch
import xetla_pt_ext  # noqa: F401  (registers torch.ops.xetla)
from xetla_vllm_plugin import pack_bitcos, bitcos_slices_for

torch.manual_seed(0)
dev = "xpu"
shapes = [(5120, 96), (5120, 34816), (6144, 5120), (17408, 5120), (5120, 14336)]
ms = [1, 7, 64, 1024]

bad = 0
for K, N in shapes:
    codes = torch.randint(-1, 2, (K, N), device=dev, dtype=torch.int8)
    codes[torch.rand(K, N, device=dev) < 0.63] = 0
    sl = bitcos_slices_for(K, N)
    buf, ranks = pack_bitcos(codes, slices=sl)
    scale = torch.rand(K // 128, N, device=dev, dtype=torch.float16) * 0.1 + 0.05
    W = codes.to(torch.float32) * scale.to(torch.float32).repeat_interleave(128, 0)
    for M in ms:
        A = torch.randn(M, K, device=dev, dtype=torch.float16) * 0.1
        out = torch.ops.xetla.bitcos_fp16_upcvt_gemm(A, buf, scale, ranks, None)
        ref = A.float() @ W
        rel = (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        ok = rel < 0.02
        bad += not ok
        print(f"K={K:6d} N={N:6d} M={M:5d} slices={sl}  rel={rel:.5f}  "
              + ("OK" if ok else "FAIL"))

print("FAILURES:", bad)
raise SystemExit(1 if bad else 0)
