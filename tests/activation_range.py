"""How much fp16 headroom do the activations actually have?

CAT-Q exports are bf16 natively (use_bfloat16: true, base Qwen3 is bfloat16) and
we convert them to fp16, which trades range for mantissa: fp16 tops out at 65504
where bf16 carries fp32's exponent. Qwen models are known for large activations,
so run the model in fp32 on CPU and record the largest magnitude flowing through
each layer. Anything approaching 65504 would saturate to inf in fp16.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FP16_MAX = 65504.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Explain de novo genome assembly in detail, with examples.")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                   tokenize=False, add_generation_prompt=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()

    stats = []

    def hook(name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                stats.append((name, t.abs().max().item()))
        return fn

    handles = []
    for name, mod in model.named_modules():
        if any(name.endswith(s) for s in ("mlp", "self_attn", "mlp.down_proj",
                                          "mlp.gate_up_proj", "mlp.up_proj")) \
                or name.endswith("model.norm"):
            handles.append(mod.register_forward_hook(hook(name)))

    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        model(**ids)
    for h in handles:
        h.remove()

    stats.sort(key=lambda kv: -kv[1])
    print(f"{'module':<50}{'max|act|':>14}{'fp16 headroom':>16}")
    for name, v in stats[:args.top]:
        head = FP16_MAX / v if v > 0 else float("inf")
        flag = "  <-- OVERFLOWS fp16" if v > FP16_MAX else ("  <-- tight" if head < 4 else "")
        print(f"{name:<50}{v:>14.1f}{head:>16.1f}x{flag}")
    worst = stats[0][1] if stats else 0.0
    print(f"\nlargest activation seen : {worst:.1f}")
    print(f"fp16 max                : {FP16_MAX:.0f}")
    print(f"headroom                : {FP16_MAX / max(worst, 1e-9):.1f}x")
    print("VERDICT:", "fp16 OVERFLOWS - bf16 required" if worst > FP16_MAX
          else "fits in fp16 with room to spare")


if __name__ == "__main__":
    main()
