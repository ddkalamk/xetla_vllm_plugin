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
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--cudagraph-sizes", default=None,
                   help="comma-separated batch sizes to capture graphs for")
    p.add_argument("--prompt", action="append", default=None,
                   help="repeat to benchmark several prompts off one load")
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
              enforce_eager=a.enforce_eager, **extra)
    load_s = time.perf_counter() - t0

    free_b, total_b = torch.xpu.mem_get_info(0)
    tok = llm.get_tokenizer()
    prompts = a.prompt or ["Tell me what is photosynthesis"]

    llm.generate(["hi"], SamplingParams(max_tokens=4, temperature=0.0))
    engine = llm.llm_engine

    print("\n" + "=" * 70)
    print(f"model                : {a.model}")
    print(f"quantization         : {quant} ({os.environ.get('XETLA_QUANT_METHOD', '-')})"
          f"{'  [sidecar]' if os.environ.get('XETLA_PREQUANT_PATH') else ''}")
    print(f"tensor parallel      : {a.tensor_parallel_size}")
    print(f"engine load          : {load_s:.1f} s")
    print(f"device memory in use : {(total_b - free_b) / 2**30:.2f} GiB "
          f"of {total_b / 2**30:.2f} GiB")

    for idx, raw in enumerate(prompts):
        try:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = raw

        sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.0)
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
        print(text.strip()[:700])
    return



if __name__ == "__main__":
    main()
