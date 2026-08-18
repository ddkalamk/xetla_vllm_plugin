"""Report load-time and decode performance for a xetla-quantized model.

    XETLA_QUANT_METHOD=int2_f16 XETLA_PREQUANT_PATH=... \
        python scripts/bench_model.py --model <hf dir or gguf> [--text-only]
"""
from __future__ import annotations

import argparse
import os
import time

# vLLM's AOT torch.compile artifacts are not keyed on every engine setting we
# vary here (context length, multimodal on/off, eager). Reusing a mismatched
# one either crashes inside the compiled graph ("'NoneType' object has no
# attribute 'size'") or silently produces degenerate output -- which would
# quietly corrupt a benchmark. Export VLLM_DISABLE_COMPILE_CACHE=0 to opt in.
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--quantization", default="xetla")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--no-warmup", action="store_true",
                   help="report cold TTFT, including jit and first-touch cost")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--pipeline-parallel-size", type=int, default=1)
    p.add_argument("--full", action="store_true",
                   help="print whole generations plus a repetition report")
    p.add_argument("--cudagraph-sizes", default=None,
                   help="comma-separated batch sizes to capture graphs for")
    p.add_argument("--prompt", action="append", default=None,
                   help="repeat to benchmark several prompts off one load")
    p.add_argument("--prompt-file", action="append", default=None,
                   help="read a prompt verbatim from a file, for text that "
                        "shell quoting would mangle")
    return p.parse_args()


def main():
    a = parse_args()
    import torch
    from vllm import LLM, SamplingParams

    quant = a.quantization if a.quantization.lower() != "none" else None
    extra = {}
    if a.text_only:
        extra["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if a.cudagraph_sizes:
        sizes = [int(s) for s in a.cudagraph_sizes.split(",")]
        extra["compilation_config"] = {"cudagraph_capture_sizes": sizes}

    t0 = time.perf_counter()
    llm = LLM(model=a.model, tokenizer=a.tokenizer or a.model,
              max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization,
              trust_remote_code=True, enable_prefix_caching=False,
              quantization=quant, dtype=a.dtype,
              tensor_parallel_size=a.tensor_parallel_size,
              pipeline_parallel_size=a.pipeline_parallel_size,
              enforce_eager=a.enforce_eager, **extra)
    load_s = time.perf_counter() - t0

    free_b, total_b = torch.xpu.mem_get_info(0)
    tok = llm.get_tokenizer()
    prompts = list(a.prompt or [])
    for path in (a.prompt_file or []):
        with open(path, encoding="utf-8") as fh:
            prompts.append(fh.read().strip())
    if not prompts:
        prompts = ["Tell me what is photosynthesis"]

    def _template(raw):
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            return raw

    templated = [_template(p) for p in prompts]

    # Warm on the real prompts, not a one-token stand-in: the first prefill at
    # a realistic length is what triggers triton jit and first-touch of the
    # attention/graph buckets, and that lands entirely in the first TTFT
    # (2.2 s against 66 ms warmed).
    if not a.no_warmup:
        w0 = time.perf_counter()
        llm.generate(templated, SamplingParams(max_tokens=8, temperature=0.0))
        warm_s = time.perf_counter() - w0
    else:
        warm_s = float("nan")
    engine = llm.llm_engine

    print("\n" + "=" * 70)
    print(f"model                : {a.model}")
    print(f"quantization         : {quant} ({os.environ.get('XETLA_QUANT_METHOD', '-')})"
          f"{'  [sidecar]' if os.environ.get('XETLA_PREQUANT_PATH') else ''}")
    print(f"tensor parallel      : {a.tensor_parallel_size}")
    print(f"pipeline parallel    : {a.pipeline_parallel_size}")
    print(f"engine load          : {load_s:.1f} s")
    print(f"warmup               : {warm_s:.1f} s"
          f"{' (skipped)' if a.no_warmup else ''}")
    print(f"device memory in use : {(total_b - free_b) / 2**30:.2f} GiB "
          f"of {total_b / 2**30:.2f} GiB")

    for idx, (raw, prompt) in enumerate(zip(prompts, templated)):
        sp = SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature,
                            top_p=a.top_p, top_k=a.top_k,
                            repetition_penalty=a.repetition_penalty)
        req = f"bench-{idx}-{time.time_ns()}"
        engine.add_request(req, prompt, sp)
        t0 = time.perf_counter()
        first_tok = None
        n = 0
        text = ""
        finish = ""
        while engine.has_unfinished_requests():
            for out in engine.step():
                if out.request_id != req:
                    continue
                o = out.outputs[0]
                if first_tok is None and o.token_ids:
                    first_tok = time.perf_counter()
                if out.finished:
                    n = len(o.token_ids)
                    text = o.text
                    finish = o.finish_reason or ""
        t_end = time.perf_counter()
        ttft = (first_tok - t0) if first_tok else float("nan")
        decode_s = t_end - first_tok if first_tok else (t_end - t0)
        print("-" * 70)
        print(f"prompt               : {raw}")
        print(f"TTFT (prefill)       : {ttft * 1000:.0f} ms")
        print(f"decode               : {n} tokens in {decode_s:.2f} s = "
              f"{n / decode_s if decode_s else 0:.2f} tok/s   [{finish}]")
        if a.full:
            print(f"repetition           : {_repetition_report(text)}")
            print(text.strip())
        else:
            print(text.strip()[:700])
    return


def _repetition_report(text):
    """Distinguish a decoding loop from output that is merely long."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    uniq = len(set(lines))
    words = text.split()
    report = [f"{len(lines)} lines, {uniq} unique"]
    if words:
        report.append(f"{len(words)} words, {len(set(words))} unique")
    # longest sentence repeated back to back
    for size in (40, 20, 10):
        if len(words) < size * 2:
            continue
        for i in range(len(words) - size * 2 + 1):
            a_ = words[i:i + size]
            if a_ == words[i + size:i + size * 2]:
                report.append(f"LOOP: {size}-word block repeats at word {i}")
                return "; ".join(report)
    return "; ".join(report)



if __name__ == "__main__":
    main()
