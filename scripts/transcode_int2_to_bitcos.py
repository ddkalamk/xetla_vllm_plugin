#!/usr/bin/env python3
"""Rewrite an int2_f16 prequant sidecar in the BITCOS layout.

Both formats encode the same ternary values, so the conversion is exact and
needs no GPU and no model load: unpack the int2 codes, repack them as a
presence bitmap plus a compacted sign stream, and carry the fp16 group scales
over untouched.

Doing it this way also inherits the fused module names (qkv_proj, gate_up_proj,
in_proj_qkvz, ...) that only vLLM knows how to construct, which is why this is
preferable to re-deriving a sidecar from the dense checkpoint.

The input embedding is left in its int2 row-major layout: it is a lookup table
rather than a GEMM operand, so the weight format is irrelevant there and
sharing it keeps the two sidecars comparable in size.

    python scripts/transcode_int2_to_bitcos.py --in <int2 sidecar> --out <path>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xetla_vllm_plugin import pack_bitcos  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    return p.parse_args()


def unpack_int2_vnni16(packed: torch.Tensor) -> torch.Tensor:
    """[K/16, N] int32 -> [K, N] int8 codes in {-1, 0, +1}.

    Inverse of pack_int2_vnni16: word (i, n) holds 16 consecutive k values for
    column n, two bits each, k = i*16 + c at bit 2c.
    """
    rows, n = packed.shape
    shifts = torch.arange(16, dtype=torch.int32) * 2
    codes = (packed.unsqueeze(-1) >> shifts) & 0x3       # [K/16, N, 16]
    codes = torch.where(codes == 3, codes - 4, codes)     # 3 is -1
    return codes.permute(0, 2, 1).reshape(rows * 16, n).to(torch.int8)


def main():
    a = parse_args()
    out: dict[str, torch.Tensor] = {}
    layers_meta: dict[str, dict] = {}

    with safe_open(a.src, framework="pt") as f:
        meta = json.loads((f.metadata() or {}).get("xetla_meta", "{}"))
        src_layers = meta.get("layers", {})
        prefixes = sorted({k.rsplit(".", 1)[0] for k in f.keys()})
        for i, prefix in enumerate(prefixes, 1):
            kind = src_layers.get(prefix, {}).get("kind", "linear")
            qw = f.get_tensor(f"{prefix}.qweight")
            sc = f.get_tensor(f"{prefix}.scale")

            if kind == "embedding":
                out[f"{prefix}.qweight"] = qw
                out[f"{prefix}.scale"] = sc
            else:
                codes = unpack_int2_vnni16(qw)
                buf, ranks = pack_bitcos(codes)
                out[f"{prefix}.qweight"] = buf
                out[f"{prefix}.scale"] = sc
                if ranks.numel():
                    out[f"{prefix}.slice_ranks"] = ranks
                del codes
            layers_meta[prefix] = {
                "kind": kind,
                "qweight_shape": list(out[f"{prefix}.qweight"].shape),
                "scale_shape": list(sc.shape),
            }
            if i % 25 == 0 or i == len(prefixes):
                print(f"  {i}/{len(prefixes)} {prefix}", flush=True)

    os.makedirs(os.path.dirname(a.dst) or ".", exist_ok=True)
    save_file(out, a.dst, metadata={
        "xetla_format_version": "1",
        "xetla_method": "bitcos_f16",
        "xetla_meta": json.dumps({"layers": layers_meta}),
    })
    src_gb = os.path.getsize(a.src) / 1e9
    dst_gb = os.path.getsize(a.dst) / 1e9
    print(f"\nint2   {src_gb:6.2f} GB")
    print(f"bitcos {dst_gb:6.2f} GB  ({dst_gb / src_gb:.3f} of int2)")


if __name__ == "__main__":
    main()
