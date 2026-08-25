"""Per-shape decode GEMV bandwidth for int2 vs BITCOS, on the real sidecar.

Uses the packed weights of an actual checkpoint rather than synthetic codes, so
each shape is measured at its own zero density and its own byte count. Distinct
layers of the same shape are rotated over until the working set exceeds
--min-footprint-gb, because a single weight buffer for these shapes fits in
cache and inverts the ranking.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import xetla_pt_ext  # noqa: F401  (loads the kernels)
import xetla_vllm_plugin  # noqa: F401  (registers torch.ops.xetla)
from safetensors import safe_open

GB = 1 << 30


def module_groups(path):
    """Map module-kind -> [(name, K, N)], ordered, from a sidecar."""
    f = safe_open(path, "pt")
    groups = {}
    for key in f.keys():
        if not key.endswith(".scale"):
            continue
        name = key[: -len(".scale")]
        kg, n = f.get_slice(key).get_shape()
        if "embed_tokens" in name:
            continue  # a lookup table, not a GEMM operand
        kind = "lm_head" if "lm_head" in name else name.split(".")[-1]
        groups.setdefault(kind, []).append((name, kg * 128, n))
    return groups


def load_sets(path, names, method, dev, min_bytes):
    """Load whole modules until the packed working set exceeds min_bytes."""
    f = safe_open(path, "pt")
    sets, total = [], 0
    for name in names:
        w = f.get_tensor(f"{name}.qweight").to(dev)
        s = f.get_tensor(f"{name}.scale").to(dev)
        r = None
        if method == "bitcos":
            try:
                r = f.get_tensor(f"{name}.slice_ranks").to(dev)
            except Exception:
                r = None
        total += w.numel() * w.element_size() + s.numel() * s.element_size()
        sets.append((w, s, r))
        if total >= min_bytes:
            break
    return sets, total


def time_rotating(call, nsets, iters, warmup=20):
    for i in range(warmup):
        call(i % nsets)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        call(i % nsets)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True,
                   help="sidecar prefix; <prefix>.xetla-<method>.safetensors")
    p.add_argument("--min-footprint-gb", type=float, default=2.0)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()

    dev = "xpu"
    min_bytes = int(a.min_footprint_gb * GB)
    paths = {m: f"{a.prefix}.xetla-{'int2_f16' if m == 'int2' else 'bitcos_f16'}"
                f".safetensors" for m in ("int2", "bitcos")}
    groups = module_groups(paths["int2"])

    print(f"{'shape':<16} {'K':>7} {'N':>7} {'n':>4} {'sets i/b':>9} "
          f"{'int2 us':>9} {'int2 GB/s':>10} {'bitcos us':>10} {'bc GB/s':>9} "
          f"{'bytes':>7} {'speedup':>8}")
    tot = {"int2": 0.0, "bitcos": 0.0}
    wtot = {"int2": 0.0, "bitcos": 0.0}
    wbytes = {"int2": 0, "bitcos": 0}
    for kind, mods in sorted(groups.items(), key=lambda kv: -kv[1][0][2]):
        names = [n for n, _, _ in mods]
        K, N = mods[0][1], mods[0][2]
        A = (torch.rand(1, K, device=dev, dtype=torch.float16) * 2 - 1)
        res = {}
        for method in ("int2", "bitcos"):
            sets, footprint = load_sets(paths[method], names, method, dev, min_bytes)
            if method == "int2":
                def call(i, _s=sets):
                    w, s, _ = _s[i]
                    torch.ops.xetla.int2_fp16_upcvt_gemm(A, w, s, None)
            else:
                def call(i, _s=sets):
                    w, s, r = _s[i]
                    torch.ops.xetla.bitcos_fp16_upcvt_gemm(A, w, s, r, None)
            secs = statistics.median(
                time_rotating(call, len(sets), a.iters) for _ in range(a.reps))
            w, s, _ = sets[0]
            per_call = (w.numel() * w.element_size() + s.numel() * s.element_size()
                        + A.numel() * 2 + N * 2)
            res[method] = (secs, per_call, len(sets), footprint)
            del sets
            torch.xpu.empty_cache()
        (ti, bi, nsets, _), (tb, bb, nsetb, _) = res["int2"], res["bitcos"]
        count = len(mods)  # modules of this kind, i.e. calls per decoded token
        tot["int2"] += ti
        tot["bitcos"] += tb
        wtot["int2"] += ti * count
        wtot["bitcos"] += tb * count
        wbytes["int2"] += bi * count
        wbytes["bitcos"] += bb * count
        print(f"{kind:<16} {K:>7} {N:>7} {count:>4} {f'{nsets}/{nsetb}':>9} "
              f"{ti*1e6:>9.1f} {bi/ti/GB:>10.1f} {tb*1e6:>10.1f} {bb/tb/GB:>9.1f} "
              f"{bb/bi:>7.3f} {ti/tb:>7.2f}x")
    print(f"\n{'unweighted sum':<16} {'':>7} {'':>7} {'':>4} {'':>9} "
          f"{tot['int2']*1e6:>9.1f} {'':>10} {tot['bitcos']*1e6:>10.1f} {'':>9} "
          f"{'':>7} {tot['int2']/tot['bitcos']:>7.2f}x")
    print(f"{'per token':<16} {'':>7} {'':>7} {'':>4} {'':>9} "
          f"{wtot['int2']*1e3:>9.2f} {'':>10} {wtot['bitcos']*1e3:>10.2f} {'':>9} "
          f"{wbytes['bitcos']/wbytes['int2']:>7.3f} "
          f"{wtot['int2']/wtot['bitcos']:>7.2f}x   (ms/token, GEMV only)")
    for m in ("int2", "bitcos"):
        print(f"  {m:<7} {wbytes[m]/GB:6.2f} GB/token at "
              f"{wbytes[m]/wtot[m]/GB:6.1f} GB/s effective")


if __name__ == "__main__":
    main()
