# SPDX-License-Identifier: Apache-2.0
"""Tiny interactive REPL chat for a local GGUF via vLLM.

Usage:
    python scripts/chat.py [--model PATH] [--tokenizer HF_REPO] [--max-tokens N]

Type your message and press Enter. Empty line submits the previous prompt.
Commands: /reset clears history, /system <text> sets system prompt, /exit quits.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from vllm import LLM, SamplingParams

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
    # GGUF often lacks a usable HF tokenizer; Qwen3-8B base works for this model.
    p.add_argument("--tokenizer", default=os.environ.get("BONSAI_TOKENIZER", "Qwen/Qwen3-8B"))
    p.add_argument("--max-model-len", type=int,
                   default=int(os.environ.get("CHAT_MAX_MODEL_LEN", "2048")))
    p.add_argument("--max-tokens", type=int,
                   default=int(os.environ.get("CHAT_MAX_TOKENS", "1024")))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument(
        "--quantization",
        default=os.environ.get("VLLM_QUANTIZATION", "xetla"),
        help="vLLM quantization scheme. Use 'xetla' to enable the int2 plugin "
             "(default), or 'none' to disable.",
    )
    p.add_argument("--dtype", default="float16")
    p.add_argument(
        "--no-stream",
        action="store_true",
        default=os.environ.get("CHAT_NO_STREAM", "0") not in ("0", "", "false", "False"),
        help="Disable token-by-token streaming; print the full reply at the end.",
    )
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        default=os.environ.get("CHAT_ENFORCE_EAGER", "0") not in ("0", "", "false", "False"),
        help="Disable torch.compile / CUDA graphs. Useful when compile crashes.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}")

    print(f"Loading {args.model} (tokenizer={args.tokenizer}) ...", flush=True)
    quant = args.quantization if args.quantization and args.quantization.lower() != "none" else None
    print(f"  quantization={quant} XETLA_QUANT_METHOD={os.environ.get('XETLA_QUANT_METHOD')}", flush=True)
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enable_prefix_caching=False,
        quantization=quant,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    history: list[dict] = [{"role": "system", "content": args.system}]
    print("Ready. /exit to quit, /reset to clear, /system <txt> to set system prompt.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history = [{"role": "system", "content": args.system}]
            print("(history cleared)")
            continue
        if user.startswith("/system "):
            args.system = user[len("/system "):].strip()
            history = [{"role": "system", "content": args.system}]
            print("(system prompt set)")
            continue

        history.append({"role": "user", "content": user})

        # Render chat template to a single prompt string so we can stream via
        # the underlying LLMEngine (LLM.chat() is non-streaming).
        tokenizer = llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )

        engine = llm.llm_engine
        req_id = f"chat-{time.time_ns()}"
        engine.add_request(req_id, prompt, sp)

        print("bot> ", end="", flush=True)
        prev_len = 0
        n_tokens = 0
        reply = ""
        t0 = time.perf_counter()
        try:
            while engine.has_unfinished_requests():
                step_outputs = engine.step()
                for out in step_outputs:
                    if out.request_id != req_id:
                        continue
                    o0 = out.outputs[0]
                    text = o0.text
                    if not args.no_stream and len(text) > prev_len:
                        delta = text[prev_len:]
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        prev_len = len(text)
                    if out.finished:
                        reply = text
                        n_tokens = len(o0.token_ids)
                        if args.no_stream:
                            sys.stdout.write(text)
                            sys.stdout.flush()
        except KeyboardInterrupt:
            engine.abort_request(req_id)
            print("\n(aborted)")
            history.pop()  # drop the user turn we just added
            continue

        elapsed = time.perf_counter() - t0
        tps = n_tokens / elapsed if elapsed > 0 else float("nan")
        print(
            f"\n[stats] {n_tokens} tokens in {elapsed:.2f}s = {tps:.2f} tok/s\n",
            flush=True,
        )
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
