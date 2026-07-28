#!/usr/bin/env python3
"""Build a small checkpoint out of the first N decoder layers of a large HF
model, so the dense (fp16) and xetla-packed paths can be compared on a GPU that
cannot hold the full model.

    python scripts/make_subset_checkpoint.py --model <hf dir> --out <dir> --layers 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--only-layer", type=int, default=None,
                   help="Keep just this decoder layer and renumber it to 0 "
                        "(useful to isolate one layer type).")
    return p.parse_args()


def main():
    a = parse_args()
    src = a.model
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(src, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]

    keep: dict[str, str] = {}
    rename: dict[str, str] = {}
    for name, shard in weight_map.items():
        m = re.search(r"\.layers\.(\d+)\.", name)
        if a.only_layer is not None:
            if m and int(m.group(1)) != a.only_layer:
                continue
            if m:
                rename[name] = name.replace(f".layers.{a.only_layer}.",
                                            ".layers.0.")
        elif m and int(m.group(1)) >= a.layers:
            continue
        keep[name] = shard
    print(f"[subset] keeping {len(keep)}/{len(weight_map)} tensors", flush=True)

    tensors: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for name, shard in keep.items():
        by_shard.setdefault(shard, []).append(name)
    for shard, names in by_shard.items():
        with safe_open(os.path.join(src, shard), framework="pt") as fh:
            for n in names:
                tensors[rename.get(n, n)] = fh.get_tensor(n)

    out_file = "model.safetensors"
    save_file(tensors, os.path.join(a.out, out_file),
              metadata={"format": "pt"})

    with open(os.path.join(src, "config.json")) as fh:
        cfg = json.load(fh)
    tcfg = cfg.get("text_config", cfg)
    if a.only_layer is not None:
        layer_type = tcfg.get("layer_types", ["full_attention"])[a.only_layer]
        tcfg["num_hidden_layers"] = 1
        tcfg["layer_types"] = [layer_type]
        tcfg["full_attention_interval"] = 1
        print(f"[subset] single layer {a.only_layer} ({layer_type})")
    else:
        tcfg["num_hidden_layers"] = a.layers
        if "layer_types" in tcfg:
            tcfg["layer_types"] = tcfg["layer_types"][: a.layers]
    tcfg["mtp_num_hidden_layers"] = 0
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    for extra in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "generation_config.json",
                  "chat_template.jinja", "preprocessor_config.json",
                  "processor_config.json", "video_preprocessor_config.json",
                  "added_tokens.json", "special_tokens_map.json"):
        srcf = os.path.join(src, extra)
        if os.path.exists(srcf):
            shutil.copy(srcf, os.path.join(a.out, extra))

    size_gb = os.path.getsize(os.path.join(a.out, out_file)) / 1e9
    print(f"[subset] wrote {a.out} ({size_gb:.2f} GB, {a.layers} layers)")


if __name__ == "__main__":
    main()
