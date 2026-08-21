"""int2 upcvt correctness across the 27B shapes, at the M values that select
different kslicing tiers (M==1 decode vs M>1 prefill)."""
import torch
import xetla_pt_ext  # noqa: F401
from xetla_vllm_plugin import pack_int2_vnni16

torch.manual_seed(0)
dev = "xpu"
shapes = [(5120, 96), (5120, 34816), (6144, 5120), (17408, 5120), (5120, 14336)]
bad = 0
for K, N in shapes:
    c = torch.randint(0, 100, (K, N), device=dev, dtype=torch.int8)
    codes = torch.where(c < 63, torch.zeros_like(c),
                        torch.where(c < 81, torch.ones_like(c), -torch.ones_like(c)))
    B = pack_int2_vnni16(codes)
    scale = torch.rand(K // 128, N, device=dev, dtype=torch.float16) * 0.1 + 0.05
    W = codes.to(torch.float32) * scale.to(torch.float32).repeat_interleave(128, 0)
    for M in (1, 7, 1024):
        A = torch.randn(M, K, device=dev, dtype=torch.float16) * 0.1
        out = torch.ops.xetla.int2_fp16_upcvt_gemm(A, B, scale, None)
        ref = A.float() @ W
        rel = (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        ok = rel < 0.02
        bad += not ok
        print(f"K={K:6d} N={N:6d} M={M:5d}  rel={rel:.5f}  " + ("OK" if ok else "FAIL"))
    del B, W
    torch.xpu.empty_cache()
print("FAILURES:", bad)
raise SystemExit(1 if bad else 0)
