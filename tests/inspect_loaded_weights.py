"""Inspect the weights vLLM actually ended up with after the plugin ran.

Every piece checks out in isolation - sidecar unpacks bit-exact, kernel hits
1.7e-4 on real weights, dense fp16 runs fine - yet CAT-Q models emit one token
forever, which is what constant logits look like. So look at what is actually
resident: anything all-zero, NaN, or still on meta.
"""

import argparse

import torch


def find_model(llm):
    obj = llm.llm_engine
    for path in (
        ("model_executor", "driver_worker", "model_runner", "model"),
        ("engine_core", "engine_core", "model_executor", "driver_worker", "model_runner", "model"),
    ):
        cur = obj
        try:
            for p in path:
                cur = getattr(cur, p)
            return cur
        except AttributeError:
            continue
    raise RuntimeError("could not reach the model through llm_engine")


def describe(t):
    if t is None:
        return "None"
    if t.is_meta:
        return "ON META (never materialized)"
    # Sample big tables rather than materializing an fp32 copy of a 151936x4096
    # embedding next to a model that already fills the card.
    flat = t.detach().reshape(-1)
    view = flat[:: max(1, flat.numel() // (1 << 20))]
    f = view.float()
    return (f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} "
            f"absmax={f.abs().max().item():.4e} mean_abs={f.abs().mean().item():.4e} "
            f"zeros={100 * (f == 0).float().mean().item():.1f}% "
            f"nan={bool(torch.isnan(f).any())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--quantization", default="xetla")
    ap.add_argument("--sidecar", default=None)
    args = ap.parse_args()

    from vllm import LLM
    quant = None if args.quantization.lower() == "none" else args.quantization
    llm = LLM(model=args.model, tokenizer=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.85, trust_remote_code=True,
              enable_prefix_caching=False, quantization=quant, dtype="float16",
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
    model = find_model(llm)

    side = {}
    if args.sidecar:
        from safetensors import safe_open
        with safe_open(args.sidecar, framework="pt") as f:
            for k in f.keys():
                side[k] = f.get_tensor(k)
        print(f"\n### comparing runtime tensors against {args.sidecar}")
        for name, mod in model.named_modules():
            for attr, suffix in (("weight", "qweight"), ("scale", "scale")):
                key = f"{name}.{suffix}"
                t = getattr(mod, attr, None)
                if key not in side or not isinstance(t, torch.Tensor) or t.is_meta:
                    continue
                ref = side[key].to(t.device)
                if ref.shape != t.shape:
                    print(f"  SHAPE MISMATCH {key}: runtime {tuple(t.shape)} vs sidecar {tuple(ref.shape)}")
                    continue
                same = bool((t == ref).all()) if t.dtype == ref.dtype else False
                if not same:
                    diff = (t != ref).float().mean().item() * 100
                    print(f"  MISMATCH {key}: {diff:.2f}% of elements differ  <-- CLOBBERED")
                elif name.endswith(("layers.0.self_attn.qkv_proj", "layers.0.mlp.down_proj")):
                    print(f"  ok {key} matches sidecar exactly")

    print("=" * 100)
    # Norms are never quantized, so they should have survived untouched. A
    # zeroed norm collapses the hidden state and makes every token identical,
    # which no GEMM-level check would catch.
    print("### norm / non-quantized weights")
    for name, mod in model.named_modules():
        hit = any(s in name for s in ("layernorm", "q_norm", "k_norm")) or name.endswith("model.norm")
        if not hit:
            continue
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor):
            continue
        if not (name.endswith("model.norm") or ".layers.0." in name or ".layers.1." in name):
            continue
        print(f"  {name:<52} {describe(w)}")

    print("=" * 100)
    interesting = ("embed_tokens", "lm_head", "layers.0.self_attn.qkv_proj",
                   "layers.0.mlp.down_proj", "layers.0.self_attn.o_proj")
    for name, mod in model.named_modules():
        if not any(name.endswith(s) or name == s for s in interesting):
            continue
        print(f"\n--- {name}  ({type(mod).__name__}, "
              f"quant_method={type(getattr(mod, 'quant_method', None)).__name__}) ---")
        for pname, p in list(mod.named_parameters(recurse=False)):
            print(f"    param {pname:<16} {describe(p)}")
        for bname in ("qweight", "scale", "xetla_qweight", "xetla_scale"):
            b = getattr(mod, bname, None)
            if isinstance(b, torch.Tensor):
                print(f"    attr  {bname:<16} {describe(b)}")
        print(f"    xetla_quantized={getattr(mod, 'xetla_quantized', False)}")
    print("=" * 100)
    print("INSPECT_DONE")


if __name__ == "__main__":
    main()
