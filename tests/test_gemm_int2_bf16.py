import torch
import time
import os
import xetla_pt_ext


os.environ["SYCL_PROGRAM_COMPILE_OPTIONS"] = "-vc-codegen -vc-disable-indvars-opt -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' -doubleGRF"

def pack_int2_vnni16(t):
    d0, d1 = t.shape
    assert d0 % 16 == 0, "Dim 0 must be multiple of 16"
    t1 = t.view([d0//16, 16, d1]).permute([0, 2, 1]).contiguous()
    t1 = t1 & 0x3
    shifts = torch.arange(16, dtype=torch.int32) * 2
    packed = (t1.to(torch.int32) << shifts).sum(dim=-1).to(torch.int32)
    return packed

def quantize_bf16_to_int2(t):
    #print("t:", t)
    t0 = torch.abs(t)
    # scale = torch.amax(torch.abs(t), dim=0, keepdim=True).to(torch.float32)
    scale = t.abs().amax(dim=0, keepdim=True).to(torch.float32) / 2.0
    t1 = t / scale
    #print("t1:", t1)
    #print("scale:", scale)
    t1 = torch.clamp(t1, -1, 1)
    t1 = t1.to(torch.int8)
    #print("t1:", t1)
    return t1, scale

def dequantize_int2_to_bf16(t, scale):
    t1 = (t.to(torch.float) * scale).to(torch.bfloat16)
    return t1

def qdq_int8(t):
    scale = 127.0 / t.abs().amax(dim=1, keepdim=True).to(torch.float32)
    t1 = t * scale
    t1 = torch.clamp(t1, -127, 127)
    t1 = t1.to(torch.int8)
    t1 = (t1.to(torch.float) / scale).to(torch.bfloat16)
    return t1

M, N, K = 1024, 4096, 4096
M, N, K = 1, 28672, 4096
#M, N, K = 1, 4096, 14336
# M, N, K = 1, 32, 32
# M, N, K = 1024, 8192, 8192

a = torch.rand([M, K], dtype=torch.bfloat16) - 0.5
a = qdq_int8(a) # * 10.0

b_ref = torch.rand([K, N], dtype=torch.bfloat16) - 0.5
b1, scale = quantize_bf16_to_int2(b_ref)

b_ref = dequantize_int2_to_bf16(b1, scale)
b = pack_int2_vnni16(b1)
# b[:,:] = 0x55555555
#a[:,:]= 1

a = a.to("xpu")
b = b.to("xpu")
scale = scale.to("xpu")
# c = a.new_empty([M, N], dtype=torch.int32)

c = xetla_pt_ext.int2_bf16_fused_gemm_run(a, b, scale, None, None)

c_ref = a.mm(b_ref.xpu())
#print(c)
#print(c_ref)
allclose = c_ref.allclose(c)
print(f"allclose = {allclose}")
if not allclose:
    print(c)
    print(c.amax())
    print(c_ref)
    print(c - c_ref)
    print((c - c_ref).abs().max())
# exit(0)
iters = 10
with torch.no_grad():
    c = xetla_pt_ext.int2_bf16_fused_gemm_run(a, b, scale, c, None)
    a_set = [a.clone() for _ in range(iters)]
    b_set = [b.clone() for _ in range(iters)]
    scale_b_set = [scale.clone() for _ in range(iters)]
    c_set = [c.clone() for _ in range(iters)]
    for i in range(iters):
        print("Iter ", i)
        c = xetla_pt_ext.int2_bf16_fused_gemm_run(a_set[i], b_set[i], scale_b_set[i], c_set[i], None)
    torch.xpu.synchronize()
    t0 = time.time()
    for i in range(iters):
        c = xetla_pt_ext.int2_bf16_fused_gemm_run(a_set[i], b_set[i], scale_b_set[i], c_set[i], None)
    torch.xpu.synchronize()
    t1 = time.time()

    tflops = 2*M*N*K/1.e12
    t = (t1 - t0) / iters

    print(f"MNK = {M} {N} {K}")
    print(f"FLOPS = {tflops/t:.2f} TF/s  Avg Time: {t*1000.0:.3f} ms")

    
