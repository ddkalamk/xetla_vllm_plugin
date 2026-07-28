#!/usr/bin/env python3
"""Locate where activations blow up / go NaN in a vLLM model on XPU.

Registers forward hooks on every leaf module and reports the first modules
whose output contains NaN/Inf (or a suspiciously large magnitude), which is
what you need when a quantized model emits garbage tokens.

    python scripts/debug_activations.py --model <hf dir> [--quantization xetla]
"""
from __future__ import annotations

import argparse
import os

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--prompt", default="What is photosynthesis?")
    p.add_argument("--quantization", default="xetla")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-report", type=int, default=25)
    p.add_argument("--load-format", default=None,
                   help="e.g. 'dummy' to run the stack with random weights.")
    return p.parse_args()


def get_model(llm):
    """Dig the nn.Module out of an in-process vLLM v1 engine."""
    obj = llm.llm_engine
    for path in (
        ("engine_core", "engine_core", "model_executor", "driver_worker",
         "worker", "model_runner", "model"),
        ("engine_core", "engine_core", "model_executor", "driver_worker",
         "model_runner", "model"),
    ):
        cur = obj
        try:
            for attr in path:
                cur = getattr(cur, attr)
            if isinstance(cur, torch.nn.Module):
                return cur
        except AttributeError:
            continue
    raise SystemExit("could not locate the model module")


def main():
    a = parse_args()
    from vllm import LLM, SamplingParams

    quant = a.quantization if a.quantization.lower() != "none" else None
    extra: dict = {}
    if os.environ.get("TEST_SSM_F32", "0") == "1":
        extra["mamba_ssm_cache_dtype"] = "float32"
    if os.environ.get("TEST_ATTN"):
        extra["attention_config"] = {"backend": os.environ["TEST_ATTN"]}
    if a.load_format:
        extra["load_format"] = a.load_format
    llm = LLM(model=a.model, tokenizer=a.tokenizer or a.model,
              max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization,
              trust_remote_code=True, enable_prefix_caching=False,
              quantization=quant, dtype=a.dtype, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "video": 0}, **extra)

    model = get_model(llm)
    stats: list[tuple[str, float, int, int]] = []
    rows: dict[str, str] = {}

    def hook(name):
        def fn(_mod, _inp, out):
            t = out
            if isinstance(t, (tuple, list)):
                t = next((x for x in t if isinstance(x, torch.Tensor)), None)
            if not isinstance(t, torch.Tensor) or not t.is_floating_point():
                return
            f = t.detach().float()
            n_nan = int(torch.isnan(f).sum().item())
            n_inf = int(torch.isinf(f).sum().item())
            finite = f[torch.isfinite(f)]
            mx = float(finite.abs().max().item()) if finite.numel() else float("nan")
            stats.append((name, mx, n_nan, n_inf))
            if (n_nan or n_inf) and name not in rows:
                flat = f.reshape(f.shape[0], -1) if f.dim() >= 2 else f.view(1, -1)
                bad_rows = torch.nonzero(
                    (~torch.isfinite(flat)).any(dim=1)).flatten().tolist()
                rows[name] = (f"shape={tuple(f.shape)} "
                              f"bad_rows={bad_rows[:8]}"
                              f"{'...' if len(bad_rows) > 8 else ''} "
                              f"({len(bad_rows)}/{flat.shape[0]})")
        return fn

    handles = []
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0 or name.endswith(("_proj", "norm")):
            handles.append(mod.register_forward_hook(hook(name)))
    print(f"[dbg] hooked {len(handles)} modules", flush=True)

    llm.generate([a.prompt], SamplingParams(max_tokens=1, temperature=0.0))
    for h in handles:
        h.remove()

    # KV-cache sanity: after a prefill the attention layers' KV cache must
    # contain the written keys/values. All-zero (or NaN) caches mean the
    # write and the read disagree about the paged layout.
    print("\n[dbg] KV cache after prefill:")
    for name, mod in model.named_modules():
        kv = getattr(mod, "kv_cache", None)
        if kv is None:
            continue
        try:
            t = kv[0] if isinstance(kv, (list, tuple)) else kv
            while isinstance(t, (list, tuple)):
                t = t[0]
            if not isinstance(t, torch.Tensor) or not t.is_floating_point():
                continue
            print(f"[dbg]   {name:60s} shape={tuple(t.shape)} "
                  f"stride={t.stride()} dtype={t.dtype} "
                  f"storage_elems={t.untyped_storage().nbytes() // t.element_size()}",
                  flush=True)
            f = t.detach().float()
            nz = int((f != 0).sum().item())
            nan = int(torch.isnan(f).sum().item())
            print(f"[dbg]   {name:60s} numel={f.numel()} "
                  f"nonzero={nz} nan={nan}")
        except Exception as e:  # noqa: BLE001
            print(f"[dbg]   {name:60s} <{type(e).__name__}: {e}>")

    bad = [s for s in stats if s[2] or s[3]]
    print(f"\n[dbg] {len(stats)} module outputs recorded, "
          f"{len(bad)} with NaN/Inf")
    if bad:
        print("[dbg] first NaN/Inf producers:")
        for name, mx, n_nan, n_inf in bad[: a.max_report]:
            print(f"[dbg]   {name:70s} max={mx:.4g} nan={n_nan} inf={n_inf}")
        print("[dbg] NaN row layout:")
        for name, info in list(rows.items())[: a.max_report]:
            print(f"[dbg]   {name:70s} {info}")
    print("\n[dbg] magnitude trace (first occurrences):")
    seen = set()
    for name, mx, n_nan, n_inf in stats:
        if name in seen:
            continue
        seen.add(name)
        flag = "  <== NaN/Inf" if (n_nan or n_inf) else ""
        print(f"[dbg]   {name:70s} max={mx:.6g}{flag}")


if __name__ == "__main__":
    main()
