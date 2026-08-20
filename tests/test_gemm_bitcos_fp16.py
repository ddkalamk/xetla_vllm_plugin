"""BITCOS packer + kernel check against a dense fp16 reference.

Validates the Python packing against the real kernel, which is the only thing
that proves the three-plane layout and the LS>1 slice ranks agree with the
device code.
"""
import sys

import torch
import xetla_pt_ext  # noqa: F401  (registers xetla_int2 ops)

sys.path.insert(0, "/data/nfs_home/egeorgan/cpu_ternary_vllm/xetla_vllm_plugin")
from xetla_vllm_plugin import BITCOS_F16_GROUP_SIZE, pack_bitcos  # noqa: E402

GS = BITCOS_F16_GROUP_SIZE


def run_case(M, K, N, z, slices, seed=0, pattern="random"):
    g = torch.Generator().manual_seed(seed)
    if pattern == "random":
        r = torch.rand(K, N, generator=g)
        sign = torch.where(torch.rand(K, N, generator=g) < 0.5, -1.0, 1.0)
        codes = torch.where(r < z, torch.zeros(K, N), sign).to(torch.int8)
    elif pattern == "all_zero":
        codes = torch.zeros(K, N, dtype=torch.int8)
    elif pattern == "all_neg":
        codes = -torch.ones(K, N, dtype=torch.int8)
    elif pattern == "all_pos":
        codes = torch.ones(K, N, dtype=torch.int8)

    scale = (torch.rand(K // GS, N, generator=g) * 2.0 + 0.5).to(torch.float16)
    A = (torch.rand(M, K, generator=g) * 2.0 - 1.0).to(torch.float16)

    W = codes.to(torch.float32) * scale.to(torch.float32).repeat_interleave(GS, dim=0)
    ref = (A.to(torch.float32) @ W).to(torch.float16)

    buf, ranks = pack_bitcos(codes, slices=slices)
    dev = "xpu"
    out = torch.ops.xetla_int2.bitcos_fp16_upcvt_gemm_run(
        A.to(dev), buf.to(dev), scale.to(dev),
        ranks.to(dev) if slices > 1 else None, None)
    out = out.cpu().to(torch.float32)

    ref32 = ref.to(torch.float32)
    denom = ref32.abs().clamp(min=1e-3)
    rel = ((out - ref32).abs() / denom)
    bad = int((rel > 2e-2).sum())
    total = rel.numel()
    ok = bad == 0
    bpw = buf.numel() * 32.0 / (K * N)
    print(f"{'PASS' if ok else 'FAIL'}  M={M:<4} K={K:<6} N={N:<6} z={z:<4} "
          f"LS={slices} {pattern:<9} mismatches={bad}/{total} "
          f"maxrel={float(rel.max()):.4f} bits/weight={bpw:.3f}")
    return ok


def main():
    all_ok = True
    # Model-shaped decode cases (Bonsai 8B: K in {4096, 12288}).
    for slices in (1, 4):
        all_ok &= run_case(1, 4096, 4096, 0.4, slices, seed=1)
        all_ok &= run_case(1, 4096, 6144, 0.4, slices, seed=2)
        all_ok &= run_case(1, 4096, 24576, 0.4, slices, seed=3)
        all_ok &= run_case(1, 12288, 4096, 0.4, slices, seed=4)
    # Density endpoints and adversarial sign patterns.
    for pattern in ("all_zero", "all_pos", "all_neg"):
        all_ok &= run_case(1, 4096, 4096, 0.0, 4, seed=5, pattern=pattern)
    for z in (0.0, 0.25, 0.9, 1.0):
        all_ok &= run_case(1, 4096, 4096, z, 4, seed=6)
    # Prefill shapes.
    all_ok &= run_case(8, 4096, 4096, 0.4, 1, seed=7)
    all_ok &= run_case(32, 4096, 6144, 0.4, 1, seed=8)
    print("\nALL PASS" if all_ok else "\nFAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
