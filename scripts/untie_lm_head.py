#!/usr/bin/env python3
"""Materialise lm_head from a tied embedding matrix.

Bonsai 1.7B/4B set tie_word_embeddings=True and ship no lm_head.weight, so the
head is the embedding matrix. pack_bonsai_hf.py can then only either pack both
(the embedding layout reaches the int2 GEMM and trips "B.size(0) must be k/16")
or neither, which silently leaves the head dense.

Writing lm_head.weight as an explicit copy and clearing the tie lets the head be
packed ternary while the embedding lookup stays dense. The shared matrix is
exactly ternary at group size 128, so this costs no accuracy.

  python scripts/untie_lm_head.py <src-model-dir> <dst-model-dir>
"""
import glob
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)

sd = {}
for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
    with safe_open(f, framework="pt") as h:
        for k in h.keys():
            sd[k] = h.get_tensor(k)

if "lm_head.weight" in sd:
    sys.exit("already untied: lm_head.weight is present")

# clone, otherwise safetensors refuses to save two names sharing storage
sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone().contiguous()
sd = {k: v.contiguous() for k, v in sd.items()}
save_file(sd, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})

for fn in os.listdir(src):
    if fn.endswith((".json", ".txt", ".model")) and "safetensors.index" not in fn:
        shutil.copy(os.path.join(src, fn), dst)

cfg_path = os.path.join(dst, "config.json")
cfg = json.load(open(cfg_path))
cfg["tie_word_embeddings"] = False
json.dump(cfg, open(cfg_path, "w"), indent=2)

print(f"{dst}: {len(sd)} tensors, tie_word_embeddings=False")
