#!/usr/bin/env python3
"""Check the XPU flash-attention kernel for the Bonsai-27B / Qwen3.5 full
attention geometry (24 q heads, 4 kv heads, head_size=256) against a plain
SDPA reference, sweeping the paged KV block size.

vLLM picks a very large KV block (832 tokens) for this hybrid model because the
attention page must match the GDN state page; FlashAttention is known to
propagate NaNs for large block sizes
(https://github.com/Dao-AILab/flash-attention/issues/1974), which is exactly
what shows up as garbage output.  This isolates that.

    python scripts/diag_flash_attn_xpu.py
"""
from __future__ import annotations

import argparse

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen", type=int, default=32)
    p.add_argument("--num-q-heads", type=int, default=24)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--head-size", type=int, default=256)
    p.add_argument("--block-sizes", type=int, nargs="*",
                   default=[16, 64, 128, 256, 832])
    p.add_argument("--dtype", default="float16")
    p.add_argument("--fill", default="zeros",
                   choices=["zeros", "nan", "garbage"],
                   help="What the unused tail of each KV block holds.")
    p.add_argument("--vllm-layout", action="store_true",
                   help="Allocate the KV cache the way vLLM does for this "
                        "model: one [num_blocks, 2, block, heads, dim] buffer "
                        "viewed as (2, num_blocks, ...) via permute, i.e. "
                        "non-contiguous k/v caches.")
    return p.parse_args()


def reference(q, k, v, scale):
    # q: [T, Hq, D]  k/v: [T, Hkv, D] -> causal SDPA with GQA
    t, hq, d = q.shape
    hkv = k.shape[1]
    rep = hq // hkv
    kk = k.repeat_interleave(rep, dim=1)
    vv = v.repeat_interleave(rep, dim=1)
    qh = q.transpose(0, 1).float()      # [Hq, T, D]
    kh = kk.transpose(0, 1).float()
    vh = vv.transpose(0, 1).float()
    out = torch.nn.functional.scaled_dot_product_attention(
        qh.unsqueeze(0), kh.unsqueeze(0), vh.unsqueeze(0),
        is_causal=True, scale=scale)
    return out.squeeze(0).transpose(0, 1)   # [T, Hq, D]


def main():
    a = parse_args()
    if not torch.xpu.is_available():
        raise SystemExit("XPU not available")
    from vllm_xpu_kernels import flash_attn_varlen_func

    dt = getattr(torch, a.dtype)
    dev = "xpu"
    t, hq, hkv, d = a.seqlen, a.num_q_heads, a.num_kv_heads, a.head_size
    scale = d ** -0.5
    torch.manual_seed(0)

    q = (torch.randn(t, hq, d, device=dev) * 0.5).to(dt)
    k = (torch.randn(t, hkv, d, device=dev) * 0.5).to(dt)
    v = (torch.randn(t, hkv, d, device=dev) * 0.5).to(dt)
    ref = reference(q, k, v, scale)

    cu = torch.tensor([0, t], device=dev, dtype=torch.int32)
    for bs in a.block_sizes:
        n_blocks = (t + bs - 1) // bs
        if a.vllm_layout:
            # [num_blocks, 2, block, heads, dim] -> (2, num_blocks, ...)
            pool = torch.zeros(n_blocks, 2, bs, hkv, d, device=dev, dtype=dt)
            if a.fill == "nan":
                pool.fill_(float("nan"))
            elif a.fill == "garbage":
                pool.normal_(0, 1e4)
            kv = pool.permute(1, 0, 2, 3, 4)
            kc, vc = kv[0], kv[1]
        elif a.fill == "zeros":
            kc = torch.zeros(n_blocks, bs, hkv, d, device=dev, dtype=dt)
            vc = torch.zeros(n_blocks, bs, hkv, d, device=dev, dtype=dt)
        elif a.fill == "nan":
            kc = torch.full((n_blocks, bs, hkv, d), float("nan"),
                            device=dev, dtype=dt)
            vc = torch.full((n_blocks, bs, hkv, d), float("nan"),
                            device=dev, dtype=dt)
        else:  # garbage: large finite values, like uninitialised memory
            kc = (torch.randn(n_blocks, bs, hkv, d, device=dev) * 1e4).to(dt)
            vc = (torch.randn(n_blocks, bs, hkv, d, device=dev) * 1e4).to(dt)
        for i in range(t):
            kc[i // bs, i % bs] = k[i]
            vc[i // bs, i % bs] = v[i]
        block_table = torch.arange(n_blocks, device=dev,
                                   dtype=torch.int32).view(1, n_blocks)
        seqused_k = torch.tensor([t], device=dev, dtype=torch.int32)
        try:
            out = flash_attn_varlen_func(
                q=q, k=kc, v=vc,
                cu_seqlens_q=cu, max_seqlen_q=t,
                seqused_k=seqused_k, max_seqlen_k=t,
                softmax_scale=scale, causal=True,
                block_table=block_table,
            )
            if isinstance(out, tuple):
                out = out[0]
            nan = int(torch.isnan(out.float()).sum().item())
            err = (out.float() - ref).abs().max().item()
            mx = out.float().abs().max().item()
            status = "ok  " if (nan == 0 and err < 5e-2) else "BAD "
            print(f"[{status}] block_size={bs:5d} nan={nan:7d} "
                  f"max_abs={mx:.4g} max_err={err:.4g}")
        except Exception as e:  # noqa: BLE001
            print(f"[err ] block_size={bs:5d} {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
