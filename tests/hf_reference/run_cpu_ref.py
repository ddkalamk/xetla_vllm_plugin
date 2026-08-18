"""Dense bf16 CPU reference for Maple, using the upstream HF modeling code.

Shares nothing with our vllm model file or the int2 kernels, so a disagreement
localises the bug. fa3.py in this directory swaps flash-attn for SDPA.
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = "/data/nfs_home/egeorgan/FRESH/maple_ref"
PROMPT = "/data/nfs_home/egeorgan/FRESH/maple_proof_prompt.txt"

tok = AutoTokenizer.from_pretrained(D, trust_remote_code=True)
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(
    D, trust_remote_code=True, dtype=torch.bfloat16, device_map="cpu"
)
m.eval()
print(f"loaded in {time.time() - t0:.0f}s", flush=True)

prompt = open(PROMPT, encoding="utf-8").read().strip()
text = tok.apply_chat_template(
    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
)
ids = tok(text, return_tensors="pt")
print(f"prompt tokens: {ids['input_ids'].shape[1]}", flush=True)

torch.manual_seed(0)
t0 = time.time()
with torch.no_grad():
    out = m.generate(
        **ids, max_new_tokens=2000, do_sample=True,
        temperature=1.0, top_p=0.95, top_k=20,
    )
n = out.shape[1] - ids["input_ids"].shape[1]
print(f"generated {n} tokens in {time.time() - t0:.0f}s", flush=True)
print("=== CPU bf16 REFERENCE OUTPUT ===", flush=True)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True), flush=True)
print("REF_DONE", flush=True)
