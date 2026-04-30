# SPDX-License-Identifier: Apache-2.0
"""Record a vLLM chat demo as an asciinema cast file.

The model is loaded silently first; the cast recording starts at the moment we
begin "typing" the prompt and stops after the final tok/s line is printed.

Usage:
    python scripts/demo_record.py --label "int2 (xetla)" --out demo_int2.cast \
        [--quantization xetla] [--model PATH] [--prompt "..."]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# Quiet vLLM/transformers BEFORE importing them.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.disable(logging.CRITICAL)


DEFAULT_MODEL = os.environ.get(
    "BONSAI_GGUF",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Ternary-Bonsai-8B-F16.gguf",
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=os.environ.get("BONSAI_TOKENIZER", "Qwen/Qwen3-8B"))
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--quantization", default=os.environ.get("VLLM_QUANTIZATION", "xetla"),
                   help="'xetla' for int2 path, 'none' for fp16 baseline.")
    p.add_argument("--label", required=True,
                   help="Banner shown at the top of the recording.")
    p.add_argument("--prompt", default="Tell me what is photosynthesis")
    p.add_argument("--out", required=True, help="Output .cast file.")
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--height", type=int, default=30)
    p.add_argument("--type-cps", type=float, default=18.0,
                   help="Characters per second when 'typing' the prompt.")
    p.add_argument("--enforce-eager", action="store_true", default=False)
    p.add_argument("--gpu-memory-utilization", type=float,
                   default=float(os.environ.get("DEMO_GPU_MEM_UTIL", "0.9")),
                   help="vLLM gpu_memory_utilization. Lower on iGPUs.")
    return p.parse_args()


class CastWriter:
    """Minimal asciinema v2 cast writer."""

    def __init__(self, path: str, width: int, height: int, title: str):
        self.f = open(path, "w", buffering=1)
        header = {
            "version": 2,
            "width": width,
            "height": height,
            "timestamp": int(time.time()),
            "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
            "title": title,
        }
        self.f.write(json.dumps(header) + "\n")
        self.t0 = time.perf_counter()

    def write(self, text: str) -> None:
        if not text:
            return
        ts = time.perf_counter() - self.t0
        self.f.write(json.dumps([round(ts, 4), "o", text]) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def main() -> None:
    args = parse_args()

    if args.model.endswith(".gguf") and not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}")

    # --- Auto-detect integrated XPUs and cap gpu_memory_utilization. --------
    # Mirrors scripts/chat.sh. Only kicks in if the user did not pass
    # --gpu-memory-utilization (default 0.9) and DEMO_GPU_MEM_UTIL is unset.
    if (args.gpu_memory_utilization == 0.9
            and not os.environ.get("DEMO_GPU_MEM_UTIL")):
        try:
            import torch  # noqa: WPS433
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                name = torch.xpu.get_device_properties(0).name.lower()
                total = torch.xpu.get_device_properties(0).total_memory
                sys_total = 0
                with open("/proc/meminfo") as fh:
                    for line in fh:
                        if line.startswith("MemTotal:"):
                            sys_total = int(line.split()[1]) * 1024
                            break
                keywords = ("lunar", "lnl", "meteor", "mtl", "arrow", "arl",
                            "iris", "arc(tm) graphics")
                integrated = (
                    any(k in name for k in keywords)
                    or (sys_total and abs(total - sys_total) / sys_total < 0.15)
                )
                if integrated:
                    free_b, total_b = torch.xpu.mem_get_info(0)
                    frac = max(0.30, min(0.85, 0.85 * free_b / total_b))
                    args.gpu_memory_utilization = round(frac, 2)
                    print(
                        f"[demo] integrated XPU detected -> "
                        f"gpu_memory_utilization={args.gpu_memory_utilization}",
                        file=sys.stderr, flush=True,
                    )
        except Exception as e:  # noqa: BLE001
            print(f"[demo] iGPU autodetect skipped: {e}",
                  file=sys.stderr, flush=True)

    # --- Load the model silently (NOT recorded). ----------------------------
    print(f"[demo] Loading {args.label} ...", file=sys.stderr, flush=True)

    quant = args.quantization if args.quantization and args.quantization.lower() != "none" else None

    # Suppress stdout (vLLM is chatty) but keep stderr visible so init
    # failures actually surface to the user.
    saved_out = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        from vllm import LLM, SamplingParams  # noqa: WPS433
        llm = LLM(
            model=args.model,
            tokenizer=args.tokenizer,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=False,
            quantization=quant,
            dtype="float16",
            enforce_eager=args.enforce_eager,
        )
        sp = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        # Warmup: a tiny generation to make the cast not include a first-step
        # compile spike.
        list(llm.generate(["hi"], SamplingParams(max_tokens=4, temperature=0.0)))
    finally:
        os.dup2(saved_out, 1)
        os.close(devnull)
        os.close(saved_out)

    print(f"[demo] Loaded; recording cast -> {args.out}", file=sys.stderr, flush=True)

    # --- Start recording. ---------------------------------------------------
    cast = CastWriter(args.out, args.width, args.height, title=args.label)

    # ANSI helpers.
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    GREEN = "\x1b[32m"
    CYAN = "\x1b[36m"
    YELLOW = "\x1b[33m"
    RESET = "\x1b[0m"

    # Banner line.
    cast.write(f"{BOLD}{CYAN}vLLM Bonsai-8B chat demo  --  {args.label}{RESET}\r\n")
    cast.write(f"{DIM}prompt: \"{args.prompt}\"{RESET}\r\n\r\n")

    # Type the prompt char-by-char.
    cast.write(f"{GREEN}you> {RESET}")
    delay = 1.0 / max(args.type_cps, 1.0)
    for ch in args.prompt:
        cast.write(ch)
        time.sleep(delay)
    cast.write("\r\n")

    # --- Run inference; stream tokens to the cast. --------------------------
    tokenizer = llm.get_tokenizer()
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    engine = llm.llm_engine
    req_id = f"demo-{time.time_ns()}"
    engine.add_request(req_id, prompt_text, sp)

    cast.write(f"{YELLOW}bot> {RESET}")
    prev_len = 0
    t0 = time.perf_counter()
    first_tok_t: float | None = None
    n_tokens = 0
    final_text = ""
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out.request_id != req_id:
                continue
            o0 = out.outputs[0]
            text = o0.text
            if len(text) > prev_len:
                if first_tok_t is None:
                    first_tok_t = time.perf_counter()
                cast.write(text[prev_len:])
                prev_len = len(text)
            if out.finished:
                final_text = text
                n_tokens = len(o0.token_ids)
    t_end = time.perf_counter()
    elapsed = t_end - t0

    # Decode throughput: inter-token rate after the first token (excludes
    # prefill/TTFT).
    if first_tok_t is not None and n_tokens > 1:
        decode_elapsed = t_end - first_tok_t
        decode_tps = (n_tokens - 1) / decode_elapsed if decode_elapsed > 0 else 0.0
    else:
        decode_elapsed = elapsed
        decode_tps = 0.0

    # Stat line.
    cast.write("\r\n\r\n")
    cast.write(
        f"{BOLD}{GREEN}[{args.label}]{RESET} "
        f"{n_tokens} tokens in {decode_elapsed:.2f}s  "
        f"{BOLD}{decode_tps:.2f} tok/s{RESET} (decode)\r\n"
    )

    # Hold on the final frame for a beat.
    time.sleep(1.5)
    cast.close()
    print(
        f"[demo] Done: {n_tokens} tokens, {decode_tps:.2f} tok/s decode -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
