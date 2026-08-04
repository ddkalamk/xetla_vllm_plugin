"""Measure collective cost at the sizes decode actually uses.

    ZE_AFFINITY_MASK=4,5,6,7 torchrun --nproc-per-node 4 scripts/comm_analysis.py

Tensor-parallel decode does two all-reduces per layer on a [tokens, hidden]
fp16 tensor, so at batch 1 the payload is tiny (8 KB for hidden 4096) and the
cost is pure latency, not bandwidth. This reports per-call latency across the
sizes a real forward pass issues, plus the implied per-token cost for a model
with a given layer count.
"""
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

HIDDEN = 4096
LAYERS = 94  # Qwen3-235B-A22B
ALLREDUCE_PER_LAYER = 2  # after attention o_proj and after the MoE


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    dist.init_process_group("xccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.xpu.set_device(rank)

    if rank == 0:
        print(f"world size {world}, hidden {HIDDEN}, fp16")
        print(f"{'tokens':>8}{'bytes':>10}{'us/call':>10}{'GB/s':>9}"
              f"{'ms/token (94 layers x2)':>26}")
        print("-" * 63)

    for tokens in (1, 2, 8, 32, 128, 512, 2048):
        x = torch.randn(tokens, HIDDEN, device=f"xpu:{rank}", dtype=torch.float16)
        s = bench(lambda: dist.all_reduce(x))
        nbytes = x.numel() * 2
        # bus bandwidth for ring all-reduce: 2(n-1)/n * size
        gbs = 2 * (world - 1) / world * nbytes / s / 1e9
        per_tok = s * LAYERS * ALLREDUCE_PER_LAYER * 1e3
        if rank == 0:
            print(f"{tokens:>8}{nbytes:>10}{s * 1e6:>10.1f}{gbs:>9.2f}"
                  f"{per_tok:>26.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
