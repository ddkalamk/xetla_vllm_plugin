#!/usr/bin/env python3
"""Run a one-shot vLLM model load on XPU and dump the xetla-quantized
(qweight, scale) pairs to a single safetensors sidecar.

The sidecar can later be loaded directly via ``XETLA_PREQUANT_PATH=...``,
which lets ``process_weights_after_loading`` skip the slow GGUF dequant +
per-layer CPU re-quant step on subsequent runs.

Usage:
    python scripts/prequantize_gguf.py \\
        --model Ternary-Bonsai-8B-F16.gguf \\
        --out   Ternary-Bonsai-8B.xetla-int2.safetensors

Environment is the same as ``scripts/chat.sh`` (``XETLA_QUANT_METHOD``,
``VLLM_QUANTIZATION``, etc.).
"""
from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Path to the source .gguf (or HF folder).")
    p.add_argument("--out", required=True,
                   help="Destination .safetensors sidecar path.")
    p.add_argument("--tokenizer",
                   default=os.environ.get("BONSAI_TOKENIZER", "Qwen/Qwen3-8B"))
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--quantization",
                   default=os.environ.get("VLLM_QUANTIZATION", "xetla"))
    p.add_argument("--dtype", default="float16")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.model.endswith(".gguf") and not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}")

    out = os.path.abspath(args.out)
    if not out.endswith(".safetensors"):
        sys.exit("--out must end in .safetensors")

    # Tell the plugin to capture every layer it quantizes and flush to disk
    # right after model load. This must be set BEFORE importing vllm so the
    # plugin's register() picks it up.
    os.environ["XETLA_PREQUANT_DUMP_PATH"] = out
    # And make sure we are NOT also loading from a sidecar (would short-
    # circuit and leave the dump buffer empty).
    os.environ.pop("XETLA_PREQUANT_PATH", None)

    print(f"[prequantize] model     : {args.model}", flush=True)
    print(f"[prequantize] out       : {out}", flush=True)
    print(f"[prequantize] quant     : {args.quantization}", flush=True)
    print(f"[prequantize] method    : "
          f"{os.environ.get('XETLA_QUANT_METHOD', 'int2')}", flush=True)

    # Import after env is set.
    from vllm import LLM, SamplingParams  # noqa: WPS433

    quant = args.quantization \
        if args.quantization and args.quantization.lower() != "none" else None

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enable_prefix_caching=False,
        quantization=quant,
        dtype=args.dtype,
        enforce_eager=True,  # skip graph capture, we only need weights on dev
    )
    # One trivial generation so any lazy init runs before exit.
    list(llm.generate(["hi"], SamplingParams(max_tokens=1, temperature=0.0)))

    if os.path.exists(out):
        size_mb = os.path.getsize(out) / 1e6
        print(f"[prequantize] DONE  ({size_mb:.1f} MB)  ->  {out}",
              flush=True)
    else:
        sys.exit("[prequantize] ERROR: sidecar file was not produced. "
                "Did the plugin register() run?")


if __name__ == "__main__":
    main()
