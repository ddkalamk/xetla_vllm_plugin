import argparse
import torch
import time
import os
import xetla_pt_ext


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test INT2 fused GEMM with configurable shapes and dtype"
    )
    parser.add_argument("-m", "--m", type=int, default=1, help="GEMM M dimension")
    parser.add_argument("-n", "--n", type=int, default=28672, help="GEMM N dimension")
    parser.add_argument("-k", "--k", type=int, default=4096, help="GEMM K dimension")
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="dtype for original weight and activations",
    )
    parser.add_argument(
        "-s",
        "--scale-dtype",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="dtype for the per-group quantization scale",
    )
    parser.add_argument(
        "-g",
        "--group-size",
        type=int,
        default=0,
        help="quantization group size along K (0 = channel-wise)",
    )
    return parser.parse_args()


_DTYPE_MAP = {
    "float": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


# os.environ[
#    "SYCL_PROGRAM_COMPILE_OPTIONS"
# ] = "-vc-codegen -vc-disable-indvars-opt -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' -doubleGRF"


def pack_int2_vnni16(t):
    d0, d1 = t.shape
    assert d0 % 16 == 0, "Dim 0 must be multiple of 16"
    t1 = t.view([d0 // 16, 16, d1]).permute([0, 2, 1]).contiguous()
    t1 = t1 & 0x3
    shifts = torch.arange(16, dtype=torch.int32) * 2
    packed = (t1.to(torch.int32) << shifts).sum(dim=-1).to(torch.int32)
    return packed


def quantize_to_int2(t, group_size=None, scale_dtype=torch.float32):
    if group_size is None:
        scale = t.abs().amax(dim=0, keepdim=True).to(scale_dtype)
        t1 = t / scale
        t1 = torch.clamp(t1, -2, 1)
        t1 = t1.to(torch.int8)
        # t1 = torch.randint(-1, 2, t1.shape, device=t1.device, dtype=torch.int8)
        # print(f"Qint2: {t1.shape} {t1.amax()} {t1.amin()} {scale.shape}")
        return t1, scale
    else:
        d0, d1 = t.shape
        assert (
            d0 % group_size == 0
        ), f"Dim 0 ({d0}) must be divisible by group_size ({group_size})"
        num_groups = d0 // group_size
        t_grouped = t.view(num_groups, group_size, d1)
        scale = t_grouped.abs().amax(dim=1, keepdim=True).to(scale_dtype)
        t1 = (t_grouped / scale).clamp(-2, 1).to(torch.int8).view(d0, d1)
        return t1, scale.squeeze(1)  # Return scale with shape [num_groups, d1]


def dequantize_int2_to_fp(t, scale, dtype=torch.bfloat16):
    if scale.shape[0] == 1:
        t1 = (t.to(torch.float) * scale.to(torch.float)).to(dtype)
    else:
        d0, d1 = t.shape
        group_size = d0 // scale.shape[0]
        t_grouped = t.view(-1, group_size, d1)
        scale_grouped = scale.view(-1, 1, d1)
        t1 = (
            (t_grouped.to(torch.float) * scale_grouped.to(torch.float))
            .to(dtype)
            .view(d0, d1)
        )
    return t1


def qdq_int8(t, scale_dtype=torch.float32):
    scale = 127.0 / t.abs().amax(dim=1, keepdim=True).to(scale_dtype).to(torch.float32)
    t1 = t * scale
    t1 = torch.clamp(t1, -127, 127)
    t1 = t1.to(torch.int8)
    t1 = (t1.to(torch.float) / scale).to(t.dtype)
    return t1


args = parse_args()
M, N, K = args.m, args.n, args.k
dtype = _DTYPE_MAP[args.dtype]
scale_dtype = _DTYPE_MAP[args.scale_dtype]
# gemm_run = getattr(xetla_pt_ext, f"int2_{args.dtype}_fused_gemm_run")
gemm_run = xetla_pt_ext.int2_woq_fused_gemm_run

a = torch.rand([M, K], dtype=dtype) - 0.5
a = qdq_int8(a)  # * 10.0

b_ref = torch.rand([K, N], dtype=dtype) - 0.5
group_size = args.group_size if args.group_size > 0 else None
b1, scale = quantize_to_int2(b_ref, group_size, scale_dtype=scale_dtype)

b_ref = dequantize_int2_to_fp(b1, scale, dtype=dtype)
b = pack_int2_vnni16(b1)
# b[:,:] = 0x55555555
# a[:,:]= 1

a = a.to("xpu")
b = b.to("xpu")
scale = scale.to("xpu")
# c = a.new_empty([M, N], dtype=torch.int32)

c = gemm_run(a, b, scale, None, None)

c_ref = a.mm(b_ref.xpu())
# print(c)
# print(c_ref)
allclose = c_ref.allclose(c)
print(f"allclose = {allclose}")
if not allclose:
    diff = (c - c_ref).abs()
    rel = diff / c_ref.abs().clamp(min=1e-6)
    torch.set_printoptions(precision=6, sci_mode=False, linewidth=160)
    print(f"c shape={tuple(c.shape)} dtype={c.dtype} device={c.device}")
    print(f"c_ref shape={tuple(c_ref.shape)} dtype={c_ref.dtype} device={c_ref.device}")
    print(
        f"c       : min={c.min().item():.6g}  max={c.max().item():.6g}  "
        f"mean={c.float().mean().item():.6g}  abs_mean={c.float().abs().mean().item():.6g}"
    )
    print(
        f"c_ref   : min={c_ref.min().item():.6g}  max={c_ref.max().item():.6g}  "
        f"mean={c_ref.float().mean().item():.6g}  abs_mean={c_ref.float().abs().mean().item():.6g}"
    )
    print(
        f"abs diff: max={diff.max().item():.6g}  mean={diff.float().mean().item():.6g}  "
        f"median={diff.float().median().item():.6g}"
    )
    print(f"rel diff: max={rel.max().item():.6g}  mean={rel.float().mean().item():.6g}")
    # Mismatch counts at standard tolerances
    for atol, rtol in [(1e-3, 1e-3), (1e-2, 1e-2), (5e-2, 5e-2)]:
        mism = (diff > (atol + rtol * c_ref.abs())).sum().item()
        print(
            f"mismatched (atol={atol}, rtol={rtol}): {mism} / {c.numel()} "
            f"({100.0 * mism / c.numel():.4f}%)"
        )
    # Location and value of worst element
    flat_idx = diff.argmax().item()
    idx = torch.unravel_index(torch.tensor(flat_idx), c.shape)
    idx_tuple = tuple(int(i) for i in idx)
    print(
        f"worst @ {idx_tuple}: c={c[idx_tuple].item():.6g}  "
        f"c_ref={c_ref[idx_tuple].item():.6g}  "
        f"abs_diff={diff[idx_tuple].item():.6g}"
    )
    print("c (first row):", c.flatten()[:16].tolist())
    print("c_ref (first row):", c_ref.flatten()[:16].tolist())
    print("diff (first row):", (c - c_ref).flatten()[:16].tolist())
# exit(0)
iters = 10
with torch.no_grad():
    c = gemm_run(a, b, scale, c, None)
    a_set = [a.clone() for _ in range(iters)]
    b_set = [b.clone() for _ in range(iters)]
    scale_b_set = [scale.clone() for _ in range(iters)]
    c_set = [c.clone() for _ in range(iters)]
    for i in range(iters):
        print("Iter ", i)
        c = gemm_run(a_set[i], b_set[i], scale_b_set[i], c_set[i], None)
    torch.xpu.synchronize()
    t0 = time.time()
    for i in range(iters):
        c = gemm_run(a_set[i], b_set[i], scale_b_set[i], c_set[i], None)
    torch.xpu.synchronize()
    t1 = time.time()

    tflops = 2 * M * N * K / 1.0e12
    t = (t1 - t0) / iters

    print(
        f"MNK = {M} {N} {K}  dtype = {args.dtype}  scale_dtype = {args.scale_dtype}  group_size = {args.group_size}"
    )
    print(f"FLOPS = {tflops/t:.2f} TF/s  Avg Time: {t*1000.0:.3f} ms")
