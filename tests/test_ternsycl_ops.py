"""TernSYCL ops (torch.ops.ternsycl.*) against an fp32 torch reference and,
when XETLA_REF_SO points at a pre-TernSYCL build of xetla_pt_ext, against the
xetla kernels they replace (same inputs).

    python tests/test_ternsycl_ops.py
"""
import importlib.util
import os
import sys

import torch

import ternsycl_pt_ext  # noqa: F401

dev = "xpu"
torch.manual_seed(0)
ts = torch.ops.ternsycl

xetla = None
if os.environ.get("XETLA_REF_SO"):
    spec = importlib.util.spec_from_file_location("xetla_pt_ext", os.environ["XETLA_REF_SO"])
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    xetla = torch.ops.xetla_int2

SHAPES = [(5120, 34816), (5120, 17408), (17408, 5120), (5120, 16384), (6144, 5120), (5120, 14336),
          (5120, 248320), (256, 48)]
MS = [1, 2, 3, 5, 8, 13, 16, 24, 31, 33, 64, 100]


def pack(k, n):
    """Random ternary codes {0, 1, 3} = {0, +1, -1}: packed [K/16, N] int32, values [K, N] fp32."""
    codes = torch.randint(0, 3, (k, n), dtype=torch.int32, device=dev)
    codes += (codes == 2).int()
    w = torch.zeros(k // 16, n, dtype=torch.int32, device=dev)
    for i in range(16):
        w |= codes[i::16] << (2 * i)
    vals = torch.where(codes == 3, -1, codes).float()
    return w, vals


def rel(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()


fails = 0


def check(name, err, tol):
    global fails
    ok = err <= tol
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name:<58} rel {err:.2e} (tol {tol:.0e})")


for k, n in SHAPES:
    w, vals = pack(k, n)
    for dt in (torch.float16, torch.bfloat16):
        s = (torch.rand(k // 128, n, device=dev) * 0.02 + 0.005).to(dt)
        wf = vals * s.float().repeat_interleave(128, 0)
        tol = 2e-3 if dt == torch.float16 else 1e-2
        for m in MS if n != 248320 else (1, 8, 33, 100):
            a = torch.randn(m, k, device=dev).to(dt)
            ref = a.float() @ wf
            tag = f"K={k} N={n} M={m} {str(dt)[6:]}"
            if dt == torch.float16:
                out = ts.int2_fp16_upcvt_gemm_run(a, w, s, None)
                check(f"upcvt {tag}", rel(out, ref), tol)
                if xetla is not None:
                    check("  vs xetla upcvt", rel(out, xetla.int2_fp16_upcvt_gemm_run(a, w, s, None).float()), tol)
                other = torch.randn(m, n, dtype=dt, device=dev)
                o1 = ts.int2_fp16_upcvt_gemm_postop_run(a, w, s, other, 1)
                check(f"silu*other {tag}", rel(o1, torch.nn.functional.silu(ref) * other.float()), tol * 2)
                o2 = ts.int2_fp16_upcvt_gemm_postop_run(a, w, s, other, 2)
                check(f"+other {tag}", rel(o2, ref + other.float()), tol * 2)
                if m > 1:
                    d = ts.int2_fp16_dpas_gemm_run(a, w, s, None)
                    check(f"dpas {tag}", rel(d, ref), 3e-2)
                    if xetla is not None and n % 256 == 0:
                        # int8 quantization error of its own: compare accuracy, not bits
                        xe = rel(xetla.int2_fp16_dpas_gemm_run(a, w, s, None), ref)
                        check(f"  dpas error vs xetla's ({xe:.2e})", rel(d, ref) - xe, 1e-3)
            else:
                out = ts.int2_bf16_upcvt_gemm_run(a, w, s, None)
                check(f"upcvt {tag}", rel(out, ref), tol)
        del wf
    del w, vals
    torch.xpu.empty_cache()

# Hadamard: against the Sylvester-matrix matmul the plugin falls back to.
h = torch.ones(1, 1)
while h.shape[0] < 1024:
    h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
h /= 32.0
for rows, k in ((1, 5120), (7, 17408), (300, 6144)):
    x = torch.randn(rows, k, dtype=torch.float16)
    sg = torch.where(torch.rand(k) < 0.5, -1.0, 1.0)
    fwd = ((x.float() * sg).reshape(-1, 1024) @ h).reshape(rows, k)
    inv = (x.float().reshape(-1, 1024) @ h).reshape(rows, k) * sg
    xs, s8 = x.to(dev), sg.to(torch.int8).to(dev)
    check(f"hadamard fwd rows={rows} K={k}", rel(ts.hadamard_fwht_run(xs, s8, 1024, False).float().cpu(), fwd), 2e-3)
    check(f"hadamard inv rows={rows} K={k}", rel(ts.hadamard_fwht_run(xs, s8, 1024, True).float().cpu(), inv), 2e-3)
    if xetla is not None:
        s16 = sg.half().to(dev)
        for inverse in (False, True):
            a_ = ts.hadamard_fwht_run(xs, s8, 1024, inverse)
            b_ = xetla.hadamard_fwht_run(xs, s16, 1024, inverse)
            check(f"  vs xetla hadamard inverse={inverse} (bit-exact)", (a_ != b_).sum().item(), 0)

print("FAILED" if fails else "all passed", f"({fails} failures)")
sys.exit(1 if fails else 0)
