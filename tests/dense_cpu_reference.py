"""Dense CPU reference for the CAT-Q MoE model.

Answers one question: does the unquantized model loop the way our int2 build
does? Runs on CPU so it does not contend with the GPU demo. Only ~3.3 B of the
30 B params are active per token, so this is slow but not hopeless.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = (
    "/data/nfs_home/egeorgan/FRESH/xetla_vllm_plugin/BitTern/projects/"
    "cat-q/configs/qwen3-moe-30B-A3B/export"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--prompt", default="What is 2+2?")
    ap.add_argument("--system", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat = []
    if args.system:
        chat.append({"role": "system", "content": args.system})
    chat.append({"role": "user", "content": args.prompt})
    text = tok.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()
    print(f"[ref] load {time.perf_counter() - t0:.1f}s", flush=True)

    ids = tok(text, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             top_k=None)
    gen = out[0][ids["input_ids"].shape[1]:]
    dt = time.perf_counter() - t0

    stopped = gen[-1].item() in (tok.eos_token_id, 151645, 151643)
    print(f"[ref] {len(gen)} tokens in {dt:.1f}s "
          f"({len(gen) / dt:.2f} tok/s), hit_eos={stopped}", flush=True)
    print("=== REFERENCE OUTPUT ===", flush=True)
    print(tok.decode(gen, skip_special_tokens=False), flush=True)
    print("=== END ===", flush=True)


if __name__ == "__main__":
    main()
