#!/usr/bin/env python3
"""HuggingFace-transformers reference logprobs on CPU.

Used to check whether vLLM's XPU path reproduces the reference implementation
for a (possibly truncated) checkpoint.

    python scripts/ref_hf_logprobs.py --model <hf dir> --prompt "..."
"""
from __future__ import annotations

import argparse
import json

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default=(
        "The capital of France is Paris, and the capital of Germany is "
        "Berlin. Photosynthesis converts light energy into chemical energy."))
    p.add_argument("--dtype", default="float32")
    p.add_argument("--out", default=None, help="Write logprobs to this JSON.")
    return p.parse_args()


def main():
    a = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = AutoConfig.from_pretrained(a.model)
    print(f"[ref] config: {cfg.__class__.__name__}", flush=True)

    dtype = getattr(torch, a.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=dtype, device_map="cpu")
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(a.model, dtype=dtype,
                                          device_map="cpu")
    model.eval()

    ids = tok(a.prompt, return_tensors="pt").input_ids
    print(f"[ref] {ids.shape[1]} tokens", flush=True)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    vals = []
    for i in range(1, ids.shape[1]):
        vals.append(round(logprobs[0, i - 1, ids[0, i]].item(), 4))
    print(f"[ref] prompt_logprobs={vals}", flush=True)
    if a.out:
        json.dump(vals, open(a.out, "w"))


if __name__ == "__main__":
    main()
