#!/usr/bin/env python3
"""Offline packer: HF (unpacked) ternary checkpoint -> xetla int2 sidecar.

The Bonsai releases ship an "unpacked" FP16/BF16 safetensors checkpoint in
which the language-model weights are *already* ternary: every value is
``s_g * {-1, 0, +1}`` with one scale per group of 128 elements along the input
dimension (see the Bonsai 27B whitepaper, sec. 4.1).  This script recovers that
representation losslessly and stores it in the packed layout the xetla int2
kernels consume:

    qweight : int32 [K/16, N]   (16 K-rows packed per int32, codes {0,+1,-1})
    scale   : fp16  [K/128, N]

The result is a single safetensors "sidecar" keyed by *vLLM module prefixes*
(fused ``qkv_proj`` / ``gate_up_proj`` / ``in_proj_qkvz`` / ``in_proj_ba``
included), which the plugin loads through ``XETLA_PREQUANT_PATH``.  Because the
plugin then allocates the dense fp16 weights on the ``meta`` device, the 27B
model never materialises its ~54 GB fp16 form -- packing offline is what makes
it fit on a single GPU.

Layers that are *not* ternary (the HQQ-4bit vision tower, gates, norms, ...)
are detected automatically and left out of the sidecar; the plugin keeps those
dense.

Usage:
    python scripts/pack_bonsai_hf.py \
        --model prism-ml/Ternary-Bonsai-27B-unpacked \
        --out   Ternary-Bonsai-27B.xetla-int2_f16.safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

GROUP_SIZE = 128
PACK_K = 16  # K-rows per int32 word (vnni16 layout)

# leaf module name -> (fused module name, position in the fused output)
FUSE_MAP: dict[str, tuple[str, int]] = {
    # attention (QKVParallelLinear)
    "q_proj": ("qkv_proj", 0),
    "k_proj": ("qkv_proj", 1),
    "v_proj": ("qkv_proj", 2),
    # SwiGLU MLP (MergedColumnParallelLinear)
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    # Gated DeltaNet linear attention (Qwen3.5 / Bonsai-27B)
    "in_proj_qkv": ("in_proj_qkvz", 0),
    "in_proj_z": ("in_proj_qkvz", 1),
    "in_proj_b": ("in_proj_ba", 0),
    "in_proj_a": ("in_proj_ba", 1),
}

# leaf modules that map 1:1 onto a vLLM linear layer
SINGLE = {"o_proj", "down_proj", "out_proj"}

# Embedding tables. Bonsai stores these ternary too (whitepaper sec. 4.3), but
# they are looked up row-wise rather than fed to a GEMM, so they get a
# row-major packed layout instead of the vnni16 one.
EMBEDDINGS = {"embed_tokens"}

# never even look at these (vision tower is HQQ-4bit, not ternary)
SKIP_SUBSTR = ("visual.", "vision_tower.", "mmproj")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF repo id or local directory of the unpacked model.")
    p.add_argument("--out", required=True,
                   help="Destination .safetensors sidecar.")
    p.add_argument("--method", default="int2_f16", choices=["int2_f16"],
                   help="Sidecar quant method tag.")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Max allowed deviation from an exact ternary code.")
    p.add_argument("--no-lm-head", action="store_true",
                   help="Leave lm_head dense instead of packing it.")
    p.add_argument("--no-embeddings", action="store_true",
                   help="Leave embed_tokens dense instead of packing it.")
    p.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads (0 = leave default).")
    p.add_argument("--limit-layers", type=int, default=0,
                   help="Only process the first N decoder layers (debug).")
    p.add_argument("--inspect", action="store_true",
                   help="Only report which tensors are ternary; write nothing.")
    return p.parse_args()


def resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download  # noqa: WPS433
    print(f"[pack] downloading {model} from the Hub ...", flush=True)
    return snapshot_download(model)


def build_shard_index(model_dir: str) -> dict[str, str]:
    """Return {tensor_name -> absolute shard path}."""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as fh:
            weight_map = json.load(fh)["weight_map"]
        return {k: os.path.join(model_dir, v) for k, v in weight_map.items()}

    single = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(single):
        sys.exit(f"No safetensors checkpoint found under {model_dir}")
    from safetensors import safe_open  # noqa: WPS433
    with safe_open(single, framework="pt") as fh:
        return {k: single for k in fh.keys()}


def vllm_prefix_map(model_dir: str) -> tuple[str, str]:
    """(checkpoint prefix, vLLM prefix) for the language model.

    vLLM nests the decoder differently from HF for the multimodal Qwen3.5
    models: HF stores ``model.language_model.layers.N``, vLLM registers the
    module as ``language_model.model.layers.N``.
    """
    cfg_path = os.path.join(model_dir, "config.json")
    arch = ""
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            arch = (json.load(fh).get("architectures") or [""])[0]
    if "ForConditionalGeneration" in arch:
        return "model.language_model.", "language_model.model."
    return "model.", "model."


def quantize_ternary(w_nk: torch.Tensor, tol: float):
    """w_nk: [N, K] weight as stored in the checkpoint.

    Returns (packed [K/16, N] int32, scale [K/128, N] fp16, max_dev) or
    (None, None, max_dev) when the tensor is not ternary within `tol`.
    """
    n, k = w_nk.shape
    if k % GROUP_SIZE != 0:
        return None, None, float("inf")

    w = w_nk.to(torch.float32).t().contiguous()          # [K, N]
    g = w.view(k // GROUP_SIZE, GROUP_SIZE, n)
    scale = g.abs().amax(dim=1)                          # [K/gs, N]
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    r = g / safe.unsqueeze(1)
    codes = torch.round(r)
    max_dev = (r - codes).abs().max().item()
    if max_dev > tol or codes.abs().max().item() > 1:
        return None, None, max_dev

    codes = codes.to(torch.int8).view(k, n)
    # int2 encoding: 0 -> 0, +1 -> 1, -1 -> 3 (two's complement int2).
    c = (codes & 0x3).to(torch.int32).view(k // PACK_K, PACK_K, n)
    c = c.permute(0, 2, 1).contiguous()                  # [K/16, N, 16]
    shifts = torch.arange(PACK_K, dtype=torch.int32) * 2
    packed = (c << shifts).sum(dim=-1).to(torch.int32)   # [K/16, N]

    scale16 = scale.to(torch.float16)
    if torch.isinf(scale16).any():
        raise ValueError("scale overflows fp16")
    return packed, scale16, max_dev


def quantize_ternary_embedding(w_vh: torch.Tensor, tol: float):
    """Pack an embedding table [vocab, hidden] for row-wise lookup.

    Groups run along `hidden` (the same axis a GEMM would call K), but the
    result stays row-major so that fetching one token is a contiguous read:

        qweight : int32 [vocab, hidden/16]
        scale   : fp16  [vocab, hidden/128]

    Returns (None, None, max_dev) when the table is not ternary.
    """
    v, h = w_vh.shape
    if h % GROUP_SIZE != 0:
        return None, None, float("inf")

    w = w_vh.to(torch.float32)
    g = w.view(v, h // GROUP_SIZE, GROUP_SIZE)
    scale = g.abs().amax(dim=2)                       # [vocab, h/gs]
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    r = g / safe.unsqueeze(2)
    codes = torch.round(r)
    max_dev = (r - codes).abs().max().item()
    if max_dev > tol or codes.abs().max().item() > 1:
        return None, None, max_dev

    codes = codes.to(torch.int8).view(v, h)
    c = (codes & 0x3).to(torch.int32).view(v, h // PACK_K, PACK_K)
    shifts = torch.arange(PACK_K, dtype=torch.int32) * 2
    packed = (c << shifts).sum(dim=-1).to(torch.int32)  # [vocab, h/16]

    scale16 = scale.to(torch.float16)
    if torch.isinf(scale16).any():
        raise ValueError("scale overflows fp16")
    return packed, scale16, max_dev


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    model_dir = resolve_model_dir(args.model)
    shard_of = build_shard_index(model_dir)
    src_prefix, dst_prefix = vllm_prefix_map(model_dir)
    print(f"[pack] model dir : {model_dir}")
    print(f"[pack] prefix map: '{src_prefix}' -> '{dst_prefix}'")

    # ---- group checkpoint tensors into vLLM modules ----------------------
    # groups: vllm_module_prefix -> list of (position, checkpoint tensor name)
    groups: dict[str, list[tuple[int, str]]] = {}
    embeddings: dict[str, str] = {}
    for name in shard_of:
        if not name.endswith(".weight"):
            continue
        if any(s in name for s in SKIP_SUBSTR):
            continue
        body = name[: -len(".weight")]
        parent, _, leaf = body.rpartition(".")

        if name == "lm_head.weight":
            if args.no_lm_head:
                continue
            groups.setdefault("lm_head", []).append((0, name))
            continue

        if not body.startswith(src_prefix):
            continue
        rel_parent = parent[len(src_prefix):]

        if leaf in FUSE_MAP:
            fused, pos = FUSE_MAP[leaf]
            target = f"{dst_prefix}{rel_parent}.{fused}"
            groups.setdefault(target, []).append((pos, name))
        elif leaf in SINGLE:
            target = f"{dst_prefix}{rel_parent}.{leaf}"
            groups.setdefault(target, []).append((0, name))
        elif leaf in EMBEDDINGS and not args.no_embeddings:
            target = f"{dst_prefix}{rel_parent}.{leaf}" if rel_parent \
                else f"{dst_prefix}{leaf}"
            embeddings[target] = name

    if args.limit_layers:
        keep = {f"layers.{i}." for i in range(args.limit_layers)}
        groups = {k: v for k, v in groups.items()
                  if k == "lm_head" or any(s in k for s in keep)}

    def sort_key(prefix: str):
        parts = prefix.split(".")
        idx = -1
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                idx = int(parts[i + 1])
                break
        return (idx, prefix)

    ordered = sorted(groups, key=sort_key)
    print(f"[pack] {len(ordered)} candidate modules", flush=True)

    # ---- pack -------------------------------------------------------------
    from safetensors import safe_open  # noqa: WPS433

    tensors: dict[str, torch.Tensor] = {}
    layers_meta: dict[str, dict] = {}
    skipped: list[tuple[str, float]] = []
    open_shards: dict[str, object] = {}
    t0 = time.perf_counter()
    packed_bytes = 0
    dense_bytes = 0

    def get_tensor(name: str) -> torch.Tensor:
        path = shard_of[name]
        fh = open_shards.get(path)
        if fh is None:
            fh = safe_open(path, framework="pt")
            fh.__enter__()
            open_shards[path] = fh
        return fh.get_tensor(name)

    for i, prefix in enumerate(ordered):
        members = sorted(groups[prefix])
        q_parts, s_parts = [], []
        bad = None
        for _, name in members:
            w = get_tensor(name)
            if w.dim() != 2:
                bad = (name, float("inf"))
                break
            packed, scale, dev = quantize_ternary(w, args.tol)
            if packed is None:
                bad = (name, dev)
                break
            q_parts.append(packed)
            s_parts.append(scale)
            dense_bytes += w.numel() * 2
        if bad is not None:
            skipped.append((prefix, bad[1]))
            print(f"[pack] {i + 1}/{len(ordered)} SKIP  {prefix} "
                  f"(not ternary, max_dev={bad[1]:.3g})", flush=True)
            continue

        qw = q_parts[0] if len(q_parts) == 1 else torch.cat(q_parts, dim=1)
        sc = s_parts[0] if len(s_parts) == 1 else torch.cat(s_parts, dim=1)
        if not args.inspect:
            tensors[f"{prefix}.qweight"] = qw.contiguous()
            tensors[f"{prefix}.scale"] = sc.contiguous()
        layers_meta[prefix] = {
            "kind": "lm_head" if prefix == "lm_head" else "linear",
            "qweight_shape": list(qw.shape),
            "scale_shape": list(sc.shape),
        }
        packed_bytes += qw.numel() * 4 + sc.numel() * 2
        if (i + 1) % 25 == 0 or i + 1 == len(ordered):
            el = time.perf_counter() - t0
            print(f"[pack] {i + 1}/{len(ordered)} packed "
                  f"({packed_bytes / 1e9:.2f} GB out / "
                  f"{dense_bytes / 1e9:.1f} GB in, {el:.0f}s)", flush=True)

    for fh in open_shards.values():
        fh.__exit__(None, None, None)

    # ---- embeddings (row-major layout, packed for lookup) -----------------
    for prefix, name in sorted(embeddings.items()):
        path = shard_of[name]
        with safe_open(path, framework="pt") as fh:
            w = fh.get_tensor(name)
        packed, scale, dev = quantize_ternary_embedding(w, args.tol)
        if packed is None:
            skipped.append((prefix, dev))
            print(f"[pack] SKIP  {prefix} (embedding not ternary, "
                  f"max_dev={dev:.3g})", flush=True)
            continue
        if not args.inspect:
            tensors[f"{prefix}.qweight"] = packed.contiguous()
            tensors[f"{prefix}.scale"] = scale.contiguous()
        layers_meta[prefix] = {
            "kind": "embedding",
            "qweight_shape": list(packed.shape),
            "scale_shape": list(scale.shape),
        }
        dense_mb = w.numel() * 2 / 1e6
        packed_mb = (packed.numel() * 4 + scale.numel() * 2) / 1e6
        print(f"[pack] embedding {prefix}: {tuple(w.shape)} "
              f"{dense_mb:.0f} MB -> {packed_mb:.0f} MB", flush=True)

    print(f"[pack] packed {len(layers_meta)} modules, "
          f"skipped {len(skipped)} non-ternary modules")
    for prefix, dev in skipped[:20]:
        print(f"[pack]   skipped: {prefix} (max_dev={dev:.3g})")
    if len(skipped) > 20:
        print(f"[pack]   ... and {len(skipped) - 20} more")

    if args.inspect:
        print("[pack] --inspect: nothing written")
        return
    if not tensors:
        sys.exit("[pack] ERROR: no ternary modules found - wrong model?")

    from safetensors.torch import save_file  # noqa: WPS433
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    meta = {
        "xetla_format_version": "1",
        "xetla_method": args.method,
        "xetla_meta": json.dumps({
            "layers": layers_meta,
            "group_size": GROUP_SIZE,
            "source_model": args.model,
        }),
    }
    save_file(tensors, out, metadata=meta)
    size_gb = os.path.getsize(out) / 1e9
    print(f"[pack] wrote {out} ({size_gb:.2f} GB, "
          f"{len(layers_meta)} modules, method={args.method})")
    print(f"[pack] run with: XETLA_PREQUANT_PATH={out} "
          f"XETLA_QUANT_METHOD={args.method}")


if __name__ == "__main__":
    main()
