"""Guard against a stale extension: NFS attribute caching has served the compute
node an old .so before, silently reverting a kernel fix mid-benchmark.

Prints the binary identity the compute node actually loads, then exercises the
int2 prefill scratch path so XETLA_SCRATCH_DEBUG reports the live bound.
"""
import hashlib
import os
import time

import torch
import xetla_pt_ext  # noqa: F401
from xetla_vllm_plugin import pack_int2_vnni16, pack_bitcos, bitcos_slices_for

so = xetla_pt_ext.__file__
with open(so, "rb") as fh:
    digest = hashlib.md5(fh.read()).hexdigest()
print("loaded   :", so)
print("md5      :", digest)
print("mtime    :", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(so))))
print("size     :", os.path.getsize(so))

K, N, M = 5120, 248320, 1024  # the 27B lm_head at profiling batch size
dev = "xpu"
c = torch.randint(0, 100, (K, N), device=dev, dtype=torch.int8)
codes = torch.where(c < 63, torch.zeros_like(c),
                    torch.where(c < 81, torch.ones_like(c), -torch.ones_like(c)))
del c
scale = torch.rand(K // 128, N, device=dev, dtype=torch.float16) * 0.1 + 0.05
A = torch.randn(M, K, device=dev, dtype=torch.float16) * 0.1

for name in ("int2", "bitcos"):
    if name == "int2":
        B, ranks = pack_int2_vnni16(codes), None
    else:
        B, ranks = pack_bitcos(codes, slices=bitcos_slices_for(K, N))
    free_before = torch.xpu.mem_get_info()[0]
    if name == "int2":
        torch.ops.xetla.int2_fp16_upcvt_gemm(A, B, scale, None)
    else:
        torch.ops.xetla.bitcos_fp16_upcvt_gemm(A, B, scale, ranks, None)
    torch.xpu.synchronize()
    free_after = torch.xpu.mem_get_info()[0]
    print(f"{name:7s} M={M} N={N}: device free {free_before/2**30:.2f} -> "
          f"{free_after/2**30:.2f} GiB (delta {(free_before-free_after)/2**30:.2f} GiB)")
    del B, ranks
    torch.xpu.empty_cache()

