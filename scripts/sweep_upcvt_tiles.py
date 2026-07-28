#!/usr/bin/env python3
"""Sweep int2 x fp16 upcvt GEMV tile configs for a model's decode shapes.

Drives ``xetla/int2_fp16_upcvt_dpas_fast_test`` (which rotates through many
distinct device buffer sets so the working set exceeds GPU cache, giving
honest DRAM-bandwidth numbers) once per (shape, wg_n, KS, LS) combination and
ranks the results.

Build the driver first:

    cd xetla/int2_fp16_upcvt_dpas_fast_test && make -j

Then:

    python scripts/sweep_upcvt_tiles.py --model 27b
    python scripts/sweep_upcvt_tiles.py --shapes 5120x34816 --wgn 32 --ls 1,2,4,8
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# M=1 decode shapes, as (K, N)
SHAPES = {
    "27b": [
        ("mlp.gate_up_proj", 5120, 34816),
        ("mlp.down_proj", 17408, 5120),
        ("linear_attn.in_proj_qkvz", 5120, 16384),
        ("linear_attn.out_proj", 6144, 5120),
        ("self_attn.qkv_proj", 5120, 14336),
        ("lm_head", 5120, 248320),
    ],
    "8b": [
        ("qkv_proj", 4096, 6144),
        ("gate_up_proj", 4096, 24576),
        ("down_proj", 12288, 4096),
        ("lm_head", 4096, 151680),
    ],
}

DEFAULT_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "xetla", "int2_fp16_upcvt_dpas_fast_test", "build", "int2_fp16_upcvt_test")

# (wg_n, KS, LS) combinations instantiated in the driver
CONFIGS = [
    (32, 1, 1), (32, 1, 2), (32, 1, 4), (32, 1, 8),
    (32, 2, 4), (32, 4, 2),
    (64, 1, 1), (64, 1, 2), (64, 1, 4), (64, 1, 8),
    (64, 2, 4),
    (128, 1, 1), (128, 1, 2), (128, 1, 4),
    (128, 2, 2),
    (256, 1, 1), (256, 1, 2),
]

DEV_RE = re.compile(r"Avg dev\s+time:\s*([0-9.]+)\s*ms\s*\(([0-9.]+)\s*GFLOPS,"
                    r"\s*([0-9.]+)\s*GiB/s\)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", default=DEFAULT_BIN)
    p.add_argument("--model", default="27b", choices=sorted(SHAPES))
    p.add_argument("--shapes", default=None,
                   help="Comma-separated KxN overriding the model preset.")
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--min-footprint-gb", type=float, default=4.0,
                   help="Total device footprint, to defeat cache.")
    p.add_argument("--wgn", default=None, help="Restrict wg_n values (csv).")
    p.add_argument("--ks", default=None, help="Restrict KS values (csv).")
    p.add_argument("--ls", default=None, help="Restrict LS values (csv).")
    p.add_argument("--validate", action="store_true",
                   help="Keep the gold check (slow).")
    return p.parse_args()


def run_one(a, K: int, N: int, wgn: int, ks: int, ls: int):
    cmd = [a.bin, "--m", str(a.m), "--k", str(K), "--n", str(N),
           "--iters", str(a.iters),
           "--min-footprint-gb", str(a.min_footprint_gb),
           "--wgn", str(wgn), "--ks", str(ks), "--ls", str(ls)]
    if not a.validate:
        cmd.append("--no-validate")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=600).stdout
    except subprocess.TimeoutExpired:
        return None
    if "is not instantiated" in out:
        return None
    m = DEV_RE.search(out)
    if not m:
        return None
    return float(m.group(1)) * 1e3, float(m.group(3))   # us, GiB/s


def main():
    a = parse_args()
    if not os.path.exists(a.bin):
        sys.exit(f"driver not built: {a.bin}\n"
                 "cd xetla/int2_fp16_upcvt_dpas_fast_test && make -j")

    if a.shapes:
        shapes = []
        for s in a.shapes.split(","):
            k, n = s.lower().split("x")
            shapes.append((f"{k}x{n}", int(k), int(n)))
    else:
        shapes = SHAPES[a.model]

    def keep(v, flt):
        return flt is None or v in {int(x) for x in flt.split(",")}

    configs = [c for c in CONFIGS
               if keep(c[0], a.wgn) and keep(c[1], a.ks) and keep(c[2], a.ls)]

    print(f"driver   : {a.bin}")
    print(f"M={a.m}  iters={a.iters}  footprint>={a.min_footprint_gb} GB  "
          f"{len(configs)} configs x {len(shapes)} shapes\n")

    best_overall: dict[tuple[int, int, int], list[float]] = {}
    for name, K, N in shapes:
        print(f"=== {name}  K={K} N={N} ===")
        rows = []
        for wgn, ks, ls in configs:
            r = run_one(a, K, N, wgn, ks, ls)
            if r is None:
                continue
            us, gibs = r
            rows.append((gibs, us, wgn, ks, ls))
            best_overall.setdefault((wgn, ks, ls), []).append(gibs)
        rows.sort(reverse=True)
        for gibs, us, wgn, ks, ls in rows[:6]:
            print(f"  ({wgn:>3},{ks},{ls})  {gibs:>7.1f} GiB/s  {us:>9.1f} us")
        if rows:
            g, u, wgn, ks, ls = rows[0]
            print(f"  -> best ({wgn},{ks},{ls}) {g:.1f} GiB/s\n")

    print("=== average GiB/s across shapes (configs valid for all) ===")
    n_shapes = len(shapes)
    avg = [(sum(v) / len(v), k) for k, v in best_overall.items()
           if len(v) == n_shapes]
    for score, (wgn, ks, ls) in sorted(avg, reverse=True)[:8]:
        print(f"  ({wgn:>3},{ks},{ls})  {score:>7.1f} GiB/s")


if __name__ == "__main__":
    main()
