"""WikiText-2 perplexity through vLLM + ternsycl, protocol of the TernaryQuench
model card: first 20480 test tokens, 40 independent 512-token chunks, scored
on all 511 predicted positions and on the last 256 of each chunk.

    TERNSYCL_PREQUANT_PATH=<sidecar> TERNSYCL_QUANT_METHOD=int2_f16 \
        python scripts/ppl_wikitext2.py <packed model dir>
"""
import math
import sys


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model = sys.argv[1]
    text = "".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = AutoTokenizer.from_pretrained(model)(text, add_special_tokens=False)["input_ids"][:40 * 512]
    chunks = [ids[i:i + 512] for i in range(0, len(ids), 512)]

    llm = LLM(model=model, tokenizer=model, quantization="ternsycl",
              dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.78,
              trust_remote_code=True, enable_prefix_caching=False, max_num_seqs=8,
              limit_mm_per_prompt={"image": 0, "video": 0},
              compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8],
                                  "inductor_compile_config": {
                                      "combo_kernels": False,
                                      "benchmark_combo_kernel": False}})
    outs = llm.generate([TokensPrompt(prompt_token_ids=c) for c in chunks],
                        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0))

    nll_all, nll_half = [], []
    for c, o in zip(chunks, outs):
        lp = [o.prompt_logprobs[p][c[p]].logprob for p in range(1, 512)]
        nll_all += [-x for x in lp]
        nll_half += [-x for x in lp[-256:]]
    print(f"positions {len(nll_all)} / {len(nll_half)}")
    print(f"PPL all positions : {math.exp(sum(nll_all) / len(nll_all)):.4f}")
    print(f"PPL second half   : {math.exp(sum(nll_half) / len(nll_half)):.4f}")


if __name__ == "__main__":
    main()
