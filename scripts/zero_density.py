#!/usr/bin/env python3
"""Zero density of an int2_f16 sidecar (fraction of ternary codes that are 0),
overall and per module kind. BITCOS packs zeros as one bit with no sign, so
this is what sets its size and decode traffic.

    python scripts/zero_density.py <sidecar.safetensors> [more sidecars...]
"""
import json
import re
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

SHIFTS = torch.arange(16, dtype=torch.int32) * 2


def code_counts(qw: torch.Tensor) -> tuple[int, int, int]:
    """(zeros, plus, minus) over all 2-bit fields of an int32 word tensor."""
    z = p = m = 0
    for chunk in qw.reshape(-1).split(1 << 24):
        codes = (chunk.unsqueeze(-1) >> SHIFTS) & 0x3
        z += int((codes == 0).sum())
        p += int((codes == 1).sum())
        m += int((codes == 3).sum())
    return z, p, m


def main():
    for path in sys.argv[1:]:
        with safe_open(path, framework="pt") as f:
            meta = json.loads((f.metadata() or {}).get("xetla_meta", "{}"))
            layers = meta.get("layers", {})
            tot = defaultdict(lambda: [0, 0, 0])
            for prefix, info in layers.items():
                qw = f.get_tensor(f"{prefix}.qweight")
                if info.get("kind") == "embedding":
                    kind = "embed_tokens"
                elif prefix == "lm_head":
                    kind = "lm_head"
                else:
                    kind = re.sub(r"layers\.\d+\.", "", prefix.split("model.")[-1])
                z, p, m = code_counts(qw)
                for i, v in enumerate((z, p, m)):
                    tot[kind][i] += v
                    tot["ALL linear (excl. embed)" if kind != "embed_tokens" else "ALL embed"][i] += v
        print(f"\n{path}")
        print(f"{'module':42s} {'params':>12s} {'zero':>7s} {'+1':>7s} {'-1':>7s}")
        for kind in sorted(tot, key=lambda k: (k.startswith("ALL"), k)):
            z, p, m = tot[kind]
            n = z + p + m
            print(f"{kind:42s} {n:12d} {z / n:7.3f} {p / n:7.3f} {m / n:7.3f}")


if __name__ == "__main__":
    main()
