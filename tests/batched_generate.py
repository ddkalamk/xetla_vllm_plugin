"""Batched generation smoke test for the ternsycl path (several prompts in one
llm.generate call, i.e. multi-sequence prefill and batched decode).

    python tests/batched_generate.py --model <packed dir> --n 8 [--max-tokens 32]
"""
import argparse
import os
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    p.add_argument("--kv-cache-memory-bytes", type=int, default=None,
                   help="pin the KV cache (LNL: profiling over-reserves)")
    p.add_argument("--cudagraph-sizes", default="1,2,4,8,16")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--ignore-eos", action="store_true",
                   help="fixed-length outputs, for throughput comparisons")
    p.add_argument("--warmup", action="store_true",
                   help="run the batch once untimed first")
    a = p.parse_args()

    from vllm import LLM, SamplingParams

    sizes = [int(s) for s in a.cudagraph_sizes.split(",")]
    llm = LLM(model=a.model, quantization="ternsycl", dtype="bfloat16",
              trust_remote_code=True, enable_prefix_caching=False,
              max_model_len=a.max_model_len,
              max_num_batched_tokens=a.max_num_batched_tokens,
              max_num_seqs=max(sizes),
              gpu_memory_utilization=a.gpu_memory_utilization,
              kv_cache_memory_bytes=a.kv_cache_memory_bytes,
              limit_mm_per_prompt={"image": 0, "video": 0},
              enforce_eager=a.enforce_eager,
              compilation_config={"cudagraph_capture_sizes": sizes,
                                  "inductor_compile_config": {
                                      "combo_kernels": False,
                                      "benchmark_combo_kernel": False}})
    tok = llm.get_tokenizer()
    qs = ["What is the capital of France?", "Name three primary colors.",
          "How many legs does a spider have?", "What is 12 times 12?",
          "Who wrote Hamlet?", "What gas do plants absorb from the air?",
          "Translate 'thank you' to Spanish.", "What is the boiling point of water in Celsius?",
          "Which planet is known as the Red Planet?", "What is the square root of 81?",
          "Name the largest ocean on Earth.", "What year did World War II end?",
          "What is H2O commonly called?", "How many continents are there?",
          "What is the fastest land animal?", "Who painted the Mona Lisa?"]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
               for q in qs[: a.n]]
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.0, ignore_eos=a.ignore_eos)
    if a.warmup:
        llm.generate(prompts, sp)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"\n=== {len(outs)} prompts, {ntok} tokens in {dt:.2f} s = {ntok / dt:.1f} tok/s aggregate")
    for q, o in zip(qs, outs):
        print(f"Q: {q}\nA: {o.outputs[0].text.strip()[:160]!r}\n")


if __name__ == "__main__":
    main()
