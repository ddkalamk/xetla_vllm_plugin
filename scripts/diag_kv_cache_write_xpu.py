#!/usr/bin/env python3
"""Check vLLM's paged KV-cache write kernel (`reshape_and_cache_flash`) on XPU
for large block sizes.

Hybrid models like Bonsai-27B force a very large KV block (832 tokens) so the
attention page matches the GDN recurrent-state page.  This writes known K/V
values through the cache-update kernel and reads them back, so a block-size
dependent bug in the write path shows up directly.

    python scripts/diag_kv_cache_write_xpu.py
"""
from __future__ import annotations

import argparse

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen", type=int, default=40)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--head-size", type=int, default=256)
    p.add_argument("--block-sizes", type=int, nargs="*",
                   default=[16, 64, 128, 256, 832])
    p.add_argument("--dtype", default="float16")
    return p.parse_args()


def main():
    a = parse_args()
    if not torch.xpu.is_available():
        raise SystemExit("XPU not available")

    from vllm import _custom_ops as ops

    dt = getattr(torch, a.dtype)
    dev = "xpu"
    t, h, d = a.seqlen, a.num_kv_heads, a.head_size
    torch.manual_seed(0)
    k = (torch.randn(t, h, d, device=dev)).to(dt)
    v = (torch.randn(t, h, d, device=dev)).to(dt)

    for bs in a.block_sizes:
        n_blocks = max(2, (t + bs - 1) // bs + 1)
        key_cache = torch.zeros(n_blocks, bs, h, d, device=dev, dtype=dt)
        val_cache = torch.zeros(n_blocks, bs, h, d, device=dev, dtype=dt)
        # write tokens 0..t-1 into block 1 (offset by a whole block, so a
        # wrong block stride shows up)
        slot_mapping = (torch.arange(t, device=dev, dtype=torch.int64) + bs)
        try:
            ops.reshape_and_cache_flash(k, v, key_cache, val_cache,
                                        slot_mapping, "auto",
                                        torch.tensor(1.0, device=dev),
                                        torch.tensor(1.0, device=dev))
            torch.xpu.synchronize()
            got_k = key_cache.view(-1, h, d)[bs:bs + t]
            got_v = val_cache.view(-1, h, d)[bs:bs + t]
            errk = (got_k.float() - k.float()).abs().max().item()
            errv = (got_v.float() - v.float()).abs().max().item()
            ok = errk == 0 and errv == 0
            print(f"[{'ok  ' if ok else 'BAD '}] block_size={bs:5d} "
                  f"max_err_k={errk:.4g} max_err_v={errv:.4g}")
        except Exception as e:  # noqa: BLE001
            print(f"[err ] block_size={bs:5d} {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
