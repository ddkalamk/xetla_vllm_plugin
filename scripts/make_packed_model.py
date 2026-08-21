#!/usr/bin/env python3
"""Build a compact model directory for the xetla packed paths.

vLLM re-reads the whole fp16 checkpoint on every load even when a prequant
sidecar already supplies every linear weight, which for Bonsai 8B means 15.25
GiB of I/O and 15.25 GiB of page cache per run. On a unified-memory part such as
Lunar Lake that cache is charged against the GPU budget, so the engine can end
up with no room for a KV cache at all.

The linear weights can simply be left out: vLLM only requires that every
parameter be *accounted* for, and it already exempts modules whose quant_method
defines process_weights_after_loading, which is exactly the xetla linear and
embedding methods. So this writes a directory holding only the tensors the
sidecar does not replace (embedding, norms), which for Bonsai 8B is ~1.25 GiB.

    python scripts/make_packed_model.py \
        --model  models/Ternary-Bonsai-8B-unpacked \
        --sidecar models/....xetla-bitcos_f16.safetensors \
        --out    models/Ternary-Bonsai-8B-packed

The result is format independent: the same directory serves int2_f16 and
bitcos_f16, since only the sidecar differs. Point XETLA_PREQUANT_PATH at
whichever sidecar you want.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

# A fused vLLM module maps to several checkpoint tensors.
FUSED = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}
# Small files a model directory needs besides the weights.
SIDE_FILES = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
    "merges.txt", "added_tokens.json", "chat_template.jinja",
    "preprocessor_config.json",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="source HF model directory")
    p.add_argument("--sidecar", required=True,
                   help="xetla prequant sidecar whose layers to drop")
    p.add_argument("--out", required=True)
    p.add_argument("--link-sidecar", action="store_true",
                   help="also symlink the sidecar into the output directory")
    return p.parse_args()


def covered_tensor_names(sidecar: str) -> set[str]:
    """Checkpoint tensor names the sidecar makes redundant."""
    with safe_open(sidecar, framework="pt") as f:
        modules = {k.rsplit(".", 1)[0] for k in f.keys()}
    names: set[str] = set()
    for mod in modules:
        parent, _, leaf = mod.rpartition(".")
        for real in FUSED.get(leaf, (leaf,)):
            base = f"{parent}.{real}" if parent else real
            names.add(f"{base}.weight")
            names.add(f"{base}.bias")
    return names


def main():
    a = parse_args()
    drop = covered_tensor_names(a.sidecar)

    keep: dict = {}
    dropped = kept_bytes = dropped_bytes = 0
    for path in sorted(glob.glob(os.path.join(a.model, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name in drop:
                    dropped += 1
                    dropped_bytes += t.numel() * t.element_size()
                    continue
                keep[name] = t
                kept_bytes += t.numel() * t.element_size()

    os.makedirs(a.out, exist_ok=True)
    save_file(keep, os.path.join(a.out, "model.safetensors"),
              metadata={"format": "pt"})
    for fn in SIDE_FILES:
        src = os.path.join(a.model, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(a.out, fn))
    if a.link_sidecar:
        dst = os.path.join(a.out, os.path.basename(a.sidecar))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(a.sidecar), dst)

    print(f"kept    {len(keep):4d} tensors  {kept_bytes / 2**30:6.2f} GiB")
    print(f"dropped {dropped:4d} tensors  {dropped_bytes / 2**30:6.2f} GiB "
          f"(supplied by the sidecar)")
    print(f"wrote   {a.out}")


if __name__ == "__main__":
    main()
