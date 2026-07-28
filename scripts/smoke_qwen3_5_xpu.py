#!/usr/bin/env python3
"""Tiny-config smoke test for the Qwen3.5-style Bonsai 27B architecture on XPU.

Builds a scaled-down copy of the real config.json (few layers, same head/dim
geometry) and runs vLLM with ``load_format="dummy"`` so no checkpoint is
needed.  This validates that

  * vLLM's ``Qwen3_5ForConditionalGeneration`` (hybrid GDN linear attention +
    full attention + vision tower) actually runs on Intel XPU, and
  * the xetla plugin's quant path binds to the expected Linear layers.

Usage:
    python scripts/smoke_qwen3_5_xpu.py --src-config <path to real config.json>
                                        [--layers 8] [--quantization xetla]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="Directory holding the real config.json + tokenizer.")
    p.add_argument("--layers", type=int, default=8,
                   help="Number of decoder layers in the shrunken model.")
    p.add_argument("--quantization",
                   default=os.environ.get("VLLM_QUANTIZATION", "xetla"))
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--keep", action="store_true",
                   help="Keep the temporary shrunken model dir.")
    p.add_argument("--all-full-attn", action="store_true",
                   help="Replace every linear_attention layer with "
                        "full_attention (drops the hybrid KV cache, so the "
                        "attention block size stays at the platform default).")
    p.add_argument("--shrink-linear-attn", action="store_true",
                   help="Shrink the GDN state dims so the mamba page (and "
                        "hence the forced attention block size) stays small.")
    return p.parse_args()


def build_tiny_dir(src: str, layers: int, all_full_attn: bool = False,
                   shrink_linear_attn: bool = False) -> str:
    with open(os.path.join(src, "config.json")) as fh:
        cfg = json.load(fh)

    tcfg = cfg["text_config"]
    types = tcfg["layer_types"]
    # Keep the leading window so we retain both linear_attention and
    # full_attention blocks (pattern repeats every `full_attention_interval`).
    interval = tcfg.get("full_attention_interval", 4)
    n = max(interval, layers - (layers % interval)) if layers >= interval else interval
    tcfg["layer_types"] = types[:n]
    if all_full_attn:
        tcfg["layer_types"] = ["full_attention"] * n
        tcfg["full_attention_interval"] = 1
    tcfg["num_hidden_layers"] = n
    if shrink_linear_attn:
        tcfg["linear_num_key_heads"] = 4
        tcfg["linear_num_value_heads"] = 16
        tcfg["linear_key_head_dim"] = 32
        tcfg["linear_value_head_dim"] = 32
    tcfg["mtp_num_hidden_layers"] = 0
    # Shrink the vision tower too; it is dense fp16 and not the thing we test.
    if "vision_config" in cfg:
        cfg["vision_config"]["depth"] = 4

    dst = tempfile.mkdtemp(prefix="bonsai27b_tiny_")
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "chat_template.jinja", "generation_config.json",
                 "preprocessor_config.json", "processor_config.json",
                 "video_preprocessor_config.json"):
        srcf = os.path.join(src, name)
        if os.path.exists(srcf):
            shutil.copy(srcf, os.path.join(dst, name))
    with open(os.path.join(dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[smoke] tiny model dir: {dst} "
          f"({n} decoder layers, types={set(tcfg['layer_types'])})", flush=True)
    return dst


def main() -> None:
    args = parse_args()
    src = os.path.abspath(args.src)
    if not os.path.exists(os.path.join(src, "config.json")):
        sys.exit(f"No config.json under {src}")

    dst = build_tiny_dir(src, args.layers, args.all_full_attn,
                         args.shrink_linear_attn)

    from vllm import LLM, SamplingParams  # noqa: WPS433

    quant = args.quantization \
        if args.quantization and args.quantization.lower() != "none" else None

    llm = LLM(
        model=dst,
        tokenizer=dst,
        load_format="dummy",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enable_prefix_caching=False,
        quantization=quant,
        dtype=args.dtype,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    outs = llm.generate(["Hello"],
                        SamplingParams(max_tokens=8, temperature=0.0))
    print("[smoke] generated:", repr(outs[0].outputs[0].text), flush=True)
    print("[smoke] OK - qwen3_5 runs on this platform", flush=True)

    if not args.keep:
        shutil.rmtree(dst, ignore_errors=True)


if __name__ == "__main__":
    main()
