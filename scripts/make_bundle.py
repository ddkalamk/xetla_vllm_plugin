#!/usr/bin/env python3
"""Build a shippable Bonsai int2 bundle: HF metadata + residual weights + sidecar.

The packed sidecar replaces every ternary linear (plus embeddings and lm_head),
which is ~50 GiB of the 51 GiB checkpoint. Everything else the engine still
needs -- config, tokenizer, chat template, norms, conv1d/gates, and the fp16
vision tower -- is under 1 GiB. Shipping those two pieces instead of the full
checkpoint takes the 27B from ~51 GiB to ~7.6 GiB.

    python scripts/make_bundle.py \
        --model prism-ml/Ternary-Bonsai-27B-unpacked \
        --sidecar Ternary-Bonsai-27B.xetla-int2_f16.safetensors \
        --out bonsai27b-int2-bundle

Layout produced:

    <out>/model/                            <- point DEMO_MODEL / --model here
    <out>/model.xetla-int2_f16.safetensors  <- found automatically by serve.sh

On the target node:

    export XETLA_QUANT_METHOD=int2_f16
    export XETLA_PREQUANT_PATH=<out>/model.xetla-int2_f16.safetensors
    python scripts/bench_model.py --model <out>/model
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys

# Leaf module names the packer consumes; their .weight tensors live in the
# sidecar and must NOT be shipped again. Kept in sync with pack_bonsai_hf.py.
FUSED = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
         "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"}
SINGLE = {"o_proj", "down_proj", "out_proj"}
EMBEDDINGS = {"embed_tokens"}
LEAF = FUSED | SINGLE | EMBEDDINGS | {"lm_head"}

# The vision tower is not ternary, so it is never packed and always ships.
SKIP_SUBSTR = ("visual.", "vision_tower.", "mmproj")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF checkpoint dir (or repo id, resolved via the cache)")
    p.add_argument("--sidecar", required=True, help="packed .safetensors sidecar")
    p.add_argument("--out", required=True, help="bundle directory to create")
    p.add_argument("--method", default="int2_f16",
                   help="names the sidecar <model>.xetla-<method>.safetensors")
    p.add_argument("--copy-sidecar", action="store_true",
                   help="copy the sidecar instead of hard-linking it")
    return p.parse_args()


def resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"could not resolve {model}: {exc}")


def is_consumed(name: str) -> bool:
    """True when the sidecar already carries this tensor."""
    if any(s in name for s in SKIP_SUBSTR):
        return False
    if not name.endswith(".weight"):
        return False
    leaf = name[: -len(".weight")].split(".")[-1]
    return leaf in LEAF


def main() -> None:
    a = parse_args()
    src = resolve_model_dir(a.model)
    if not os.path.isfile(a.sidecar):
        sys.exit(f"sidecar not found: {a.sidecar}")

    out_model = os.path.join(a.out, "model")
    os.makedirs(out_model, exist_ok=True)

    index_path = os.path.join(src, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        sys.exit(f"no safetensors index in {src}")
    weight_map = json.load(open(index_path))["weight_map"]

    residual = [k for k in weight_map if not is_consumed(k)]
    dropped = len(weight_map) - len(residual)
    print(f"[bundle] {len(weight_map)} tensors: keeping {len(residual)}, "
          f"dropping {dropped} carried by the sidecar")

    # --- copy the small metadata files -------------------------------------
    copied = []
    for entry in sorted(os.listdir(src)):
        if entry.endswith(".safetensors") or entry == "model.safetensors.index.json":
            continue
        srcp = os.path.join(src, entry)
        if os.path.isfile(srcp):
            shutil.copy2(srcp, os.path.join(out_model, entry))
            copied.append(entry)
    print(f"[bundle] metadata: {', '.join(copied)}")

    # --- gather the residual tensors ---------------------------------------
    from safetensors import safe_open
    from safetensors.torch import save_file

    by_shard: dict[str, list[str]] = {}
    for name in residual:
        by_shard.setdefault(weight_map[name], []).append(name)

    tensors = {}
    total = 0
    for shard, names in sorted(by_shard.items()):
        with safe_open(os.path.join(src, shard), framework="pt") as fh:
            for name in names:
                t = fh.get_tensor(name)
                tensors[name] = t
                total += t.numel() * t.element_size()
        print(f"[bundle]   {shard}: +{len(names)} tensors "
              f"({total / 1024**3:.2f} GiB so far)")

    shard_name = "model-residual.safetensors"
    save_file(tensors, os.path.join(out_model, shard_name),
              metadata={"format": "pt"})
    json.dump(
        {"metadata": {"total_size": total},
         "weight_map": {k: shard_name for k in tensors}},
        open(os.path.join(out_model, "model.safetensors.index.json"), "w"),
        indent=1,
    )
    print(f"[bundle] residual weights: {total / 1024**3:.2f} GiB -> {shard_name}")

    # --- place the sidecar so serve.sh finds it automatically --------------
    dst_sidecar = os.path.join(a.out, f"model.xetla-{a.method}.safetensors")
    if os.path.exists(dst_sidecar):
        os.remove(dst_sidecar)
    linked = False
    if not a.copy_sidecar:
        try:
            os.link(a.sidecar, dst_sidecar)
            linked = True
        except OSError:
            pass
    if not linked:
        shutil.copy2(a.sidecar, dst_sidecar)
    size = os.path.getsize(dst_sidecar) / 1024**3
    print(f"[bundle] sidecar: {size:.2f} GiB "
          f"({'hard-linked' if linked else 'copied'}) -> {dst_sidecar}")

    print(f"\n[bundle] done: {a.out}")
    print(f"[bundle]   model dir : {out_model}")
    print(f"[bundle]   sidecar   : {dst_sidecar}")


if __name__ == "__main__":
    main()
