#!/usr/bin/env python3
"""Offline packer: Bonsai 2 (Hadamard-folded) PQ2_0 GGUF -> xetla int2 sidecar.

Bonsai 2 ships only as GGUF (PQ2_0 / PTQ1_0 / F16) from the PrismML llama.cpp
fork. Its weights are ternary g128 like Bonsai 1, but stored in a *rotated
basis*: every folded matrix expects its input to be passed through a fixed sign
flip followed by a blockwise (1024) normalised Walsh-Hadamard transform, and the
token embedding is stored rotated too (inverse transform after lookup). The
GGUF carries this contract in ``prism.hadamard.*`` metadata; this script moves
it into the xetla sidecar so the plugin can apply the same transform on XPU.

What comes out:

  <out>.safetensors               int2_f16 sidecar keyed by vLLM module prefix
     <prefix>.qweight  int32 [K/16, N]  (vnni16: 16 K-rows per word, -1 -> 3)
     <prefix>.scale    fp16  [K/128, N]
     hadamard.signs.<K> fp16 [K]       sign vectors, one per folded width
     metadata.xetla_meta.hadamard      block size + which prefixes are folded
  <packed-dir>/                    compact HF-style model dir for vLLM
     model-residual.safetensors    norms, A_log, dt_bias, conv1d, in_proj_a/b,
                                   plus the vision tower from the mmproj GGUF
     config.json, tokenizer*, chat_template.jinja, generation_config.json,
     preprocessor/processor configs

Two llama.cpp conventions are undone on the way to HF/vLLM layout:

  * GDN value heads: the converter reorders V-side rows from HF "grouped"
    (all V heads of one K head adjacent) to llama.cpp "tiled" order. Rows of
    in_proj_qkv (V part), in_proj_z, in_proj_a/b, A_log, dt_bias and the
    conv1d V channels are permuted back. out_proj is *not* touched: a folded
    weight cannot be column-permuted, so the fork keeps it grouped
    (``prism.hadamard.gdn_v_grouped``) and permutes the activation instead,
    which is already the order vLLM produces.
  * ``ssm_a`` is stored as A = -exp(A_log); HF wants A_log.
  * every ``*norm.weight`` except ``ssm_norm`` is stored as w+1 (ggml has no
    (1+w) RMSNorm); HF/vLLM want w.

Usage:
    python scripts/pack_bonsai2_gguf.py \
        --gguf    models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
        --mmproj  models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
        --ref-dir models/Ternary-Bonsai-2-27B-mlx-ref \
        --out     models/Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors \
        --packed  models/Ternary-Bonsai-2-27B-packed

``--ref-dir`` holds the model's tokenizer.json / tokenizer_config.json /
chat_template.jinja / config.json as published in the MLX repo
(prism-ml/Ternary-Bonsai-2-27B-mlx-2bit; only those small files are needed).
Reading PQ2_0 needs the PrismML llama.cpp fork's gguf-py (``--gguf-py`` or
``$PRISM_GGUF_PY``; stock gguf does not know type id 142).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# the fork's gguf-py knows the Prism types (PQ2_0 = 142, PTQ1_0 = 143)
FORK_GGUF_PY_CANDIDATES = (
    os.environ.get("PRISM_GGUF_PY", ""),
    os.path.join(HERE, "..", "third_party", "llama.cpp-prism", "gguf-py"),
    os.path.join(HERE, "..", "..", "third_party", "llama.cpp-prism-v7", "gguf-py"),
)

GROUP_SIZE = 128
PACK_K = 16
PQ2_0_BLOCK_BYTES = 2 + GROUP_SIZE // 4

# GGUF stem -> (vLLM fused module, position). Positions follow vLLM's
# stacked_params_mapping for Qwen3.5 (q,k,v | gate,up | qkv,z | b,a).
FUSE_MAP = {
    "attn_q": ("self_attn.qkv_proj", 0),
    "attn_k": ("self_attn.qkv_proj", 1),
    "attn_v": ("self_attn.qkv_proj", 2),
    "ffn_gate": ("mlp.gate_up_proj", 0),
    "ffn_up": ("mlp.gate_up_proj", 1),
    "attn_qkv": ("linear_attn.in_proj_qkvz", 0),
    "attn_gate": ("linear_attn.in_proj_qkvz", 1),
    "ssm_beta": ("linear_attn.in_proj_ba", 0),
    "ssm_alpha": ("linear_attn.in_proj_ba", 1),
}
SINGLE_MAP = {
    "attn_output": "self_attn.o_proj",
    "ffn_down": "mlp.down_proj",
    "ssm_out": "linear_attn.out_proj",
}
# GGUF stem -> HF leaf for the dense residual tensors
RESIDUAL_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
}
HF_LM_PREFIX = "model.language_model."       # checkpoint (HF) naming
VLLM_LM_PREFIX = "language_model.model."     # vLLM module naming
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                   "chat_template.jinja", "special_tokens_map.json",
                   "merges.txt", "vocab.json")
# processor side files vLLM's Qwen3-VL processing path expects; taken from
# --ref-dir / --template when present, else written from these defaults
# (the Qwen3.5 image/video processors; sizes are the transformers defaults).
PROCESSOR_DEFAULTS = {
    "processor_config.json": {"processor_class": "Qwen3VLProcessor"},
    "preprocessor_config.json": {
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast",
    },
    "video_preprocessor_config.json": {
        "size": {"longest_edge": 25165824, "shortest_edge": 4096},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "video_processor_type": "Qwen3VLVideoProcessor",
    },
}
# mmproj (llama.cpp clip) tensor -> HF Qwen3-VL vision tower tensor
MMPROJ_BLOCK_MAP = {
    "attn_qkv": "attn.qkv", "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1", "ffn_down": "mlp.linear_fc2",
    "ln1": "norm1", "ln2": "norm2",
}
MMPROJ_GLOBAL_MAP = {
    "v.post_ln": "merger.norm", "mm.0": "merger.linear_fc1",
    "mm.2": "merger.linear_fc2", "v.position_embd.weight": "pos_embed.weight",
    "v.patch_embd.bias": "patch_embed.proj.bias",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, help="PQ2_0 GGUF of Bonsai 2")
    p.add_argument("--out", required=True, help="sidecar .safetensors to write")
    p.add_argument("--packed", required=True, help="model directory to write")
    p.add_argument("--mmproj", default=None,
                   help="Bonsai 2 vision projector GGUF (mmproj BF16/F32); "
                        "source of model.visual.* (required unless --template)")
    p.add_argument("--ref-dir", "--tokenizer-dir", dest="ref_dir", default=None,
                   help="dir with tokenizer.json, tokenizer_config.json, "
                        "chat_template.jinja and config.json (MLX repo files)")
    p.add_argument("--template", default=None,
                   help="optional existing Qwen3.5-27B packed dir; its config.json "
                        "and vision tower are used when --mmproj/--ref-dir are "
                        "not given")
    p.add_argument("--gguf-py", default=None,
                   help="path to the PrismML llama.cpp fork's gguf-py")
    p.add_argument("--limit-layers", type=int, default=0,
                   help="only pack the first N blocks (debug)")
    p.add_argument("--inspect", action="store_true", help="report only")
    a = p.parse_args()
    if not a.mmproj and not a.template:
        p.error("need --mmproj (or a --template dir to borrow a vision tower from)")
    if not a.ref_dir and not a.template:
        p.error("need --ref-dir (tokenizer + config.json) or --template")
    return a


def find_gguf_py(explicit: str | None) -> str:
    for cand in (explicit or "",) + FORK_GGUF_PY_CANDIDATES:
        if cand and os.path.isfile(os.path.join(cand, "gguf", "constants.py")):
            with open(os.path.join(cand, "gguf", "constants.py")) as fh:
                if "PQ2_0" in fh.read():
                    return os.path.abspath(cand)
    sys.exit("need the PrismML llama.cpp fork's gguf-py (knows PQ2_0): pass "
             "--gguf-py <fork>/gguf-py or set PRISM_GGUF_PY; e.g.\n  git clone "
             "--depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp "
             "third_party/llama.cpp-prism")


def gguf_tensor_f32(t) -> torch.Tensor:
    """Any F32/F16/BF16 GGUF tensor as a float32 torch tensor in HF (row-major,
    reversed-ne) shape."""
    shape = tuple(int(x) for x in reversed(t.shape))
    arr = np.asarray(t.data)
    if t.tensor_type.name == "BF16":
        arr = arr.view(np.uint16).reshape(shape)
        x = torch.from_numpy(arr.astype(np.int32)).to(torch.int32)
        return (x << 16).view(torch.float32)          # bf16 -> f32 bit shift
    if t.tensor_type.name in ("F32", "F16"):
        return torch.from_numpy(arr.copy()).to(torch.float32).reshape(shape)
    sys.exit(f"{t.name}: unsupported dense type {t.tensor_type.name}")


def vision_tower_from_mmproj(path: str, GGUFReader) -> dict[str, torch.Tensor]:
    """llama.cpp clip mmproj -> HF `model.visual.*` tensors (bf16)."""
    r = GGUFReader(path)
    fields = {k: v.contents() for k, v in r.fields.items()}
    if fields.get("general.architecture") != "clip" or \
            fields.get("clip.projector_type") != "qwen3vl_merger":
        sys.exit(f"{path}: not a qwen3vl_merger clip mmproj")
    out: dict[str, torch.Tensor] = {}
    patch = {}
    for t in r.tensors:
        name = t.name
        if name.startswith("v.blk."):
            _, _, layer, rest = name.split(".", 3)
            stem, kind = rest.rsplit(".", 1)          # e.g. attn_qkv, weight
            if stem not in MMPROJ_BLOCK_MAP or kind not in ("weight", "bias"):
                sys.exit(f"{path}: unmapped tensor {name}")
            hf = f"model.visual.blocks.{layer}.{MMPROJ_BLOCK_MAP[stem]}.{kind}"
            out[hf] = gguf_tensor_f32(t)
        elif name.startswith("v.patch_embd.weight"):
            # Conv3d [C, 3, T=2, 16, 16] is split per temporal slice:
            # v.patch_embd.weight (t=0) and v.patch_embd.weight.1 (t=1)
            idx = int(name.rsplit(".", 1)[1]) if name[-1].isdigit() else 0
            patch[idx] = gguf_tensor_f32(t)              # [C, 3, 16, 16]
        else:
            stem, kind = name.rsplit(".", 1)
            if name in MMPROJ_GLOBAL_MAP:               # bias / pos_embed
                out["model.visual." + MMPROJ_GLOBAL_MAP[name]] = gguf_tensor_f32(t)
            elif stem in MMPROJ_GLOBAL_MAP and kind in ("weight", "bias"):
                out[f"model.visual.{MMPROJ_GLOBAL_MAP[stem]}.{kind}"] = gguf_tensor_f32(t)
            else:
                sys.exit(f"{path}: unmapped tensor {name}")
    if sorted(patch) != [0, 1]:
        sys.exit(f"{path}: expected two temporal patch-embedding slices")
    out["model.visual.patch_embed.proj.weight"] = torch.stack([patch[0], patch[1]], dim=2)
    return {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}


def synth_config(ref_cfg: dict) -> dict:
    """HF Qwen3_5ForConditionalGeneration config from the MLX repo's config.json
    (same text/vision configs, minus the MLX pack/runtime keys)."""
    text = {k: v for k, v in ref_cfg["text_config"].items()
            if not k.startswith("quantization")}
    text.setdefault("model_type", "qwen3_5_text")
    text.setdefault("dtype", "bfloat16")
    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": text,
        "vision_config": dict(ref_cfg["vision_config"]),
        "tie_word_embeddings": bool(ref_cfg.get("tie_word_embeddings", False)),
        "transformers_version": "4.57.1",
    }
    for k in ("image_token_id", "video_token_id", "vision_start_token_id",
              "vision_end_token_id"):
        if k in ref_cfg:
            cfg[k] = ref_cfg[k]
    return cfg


# ---- PQ2_0 -> xetla int2 --------------------------------------------------
def pq2_0_words_scales(t) -> tuple[np.ndarray, np.ndarray]:
    """Return (words uint32 [N, K/16], scale fp16 [N, K/128]) for a PQ2_0
    tensor, with codes already remapped from ggml {0,1,2} = {-1,0,+1} to the
    xetla two's-complement {3,0,1}."""
    n, k = (int(x) for x in reversed(t.shape))
    raw = np.ascontiguousarray(t.data).reshape(-1)
    blocks = n * k // GROUP_SIZE
    if raw.nbytes != blocks * PQ2_0_BLOCK_BYTES:
        raise ValueError(f"{t.name}: PQ2_0 byte length mismatch")
    data = raw.view(np.uint8).reshape(blocks, PQ2_0_BLOCK_BYTES)
    scale = data[:, :2].copy().view("<f2").reshape(n, k // GROUP_SIZE)
    if not np.isfinite(scale.astype(np.float32)).all():
        raise ValueError(f"{t.name}: non-finite scale")
    w = data[:, 2:].copy().view("<u4").reshape(n, k // PACK_K)
    # per 2-bit lane: q -> (q - 1) mod 4.  lo' = ~lo, hi' = hi ^ ~lo
    lanes = np.uint32(0x55555555)
    nlo = (~w) & lanes
    nhi = ((w >> np.uint32(1)) ^ (~w)) & lanes
    words = (nlo | (nhi << np.uint32(1))).astype("<u4")
    return words, scale


def to_kn_layout(words: np.ndarray, scale: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K/16] words / [N, K/128] scales -> xetla [K/16, N] int32, [K/128, N] fp16."""
    qw = torch.from_numpy(words.view(np.int32)).t().contiguous()
    sc = torch.from_numpy(scale.view(np.float16)).t().contiguous()
    return qw, sc


def _selftest_remap() -> None:
    """Check the lane-wise code remap against the ggml dequant formula."""
    rng = np.random.default_rng(0)
    q = rng.integers(0, 3, size=(4, 2 * GROUP_SIZE), dtype=np.uint8)  # {0,1,2}
    d = rng.random((4, 2), dtype=np.float32).astype(np.float16)
    blocks = []
    for row in range(4):
        for b in range(2):
            qs = np.zeros(GROUP_SIZE // 4, dtype=np.uint8)
            for j in range(GROUP_SIZE):
                qs[j // 4] |= q[row, b * GROUP_SIZE + j] << ((j % 4) * 2)
            blocks.append(np.concatenate([d[row, b:b + 1].view(np.uint8), qs]))

    class T:  # minimal stand-in for a gguf ReaderTensor
        name, shape, data = "selftest", (2 * GROUP_SIZE, 4), np.concatenate(blocks)

    words, scale = pq2_0_words_scales(T)
    codes = np.stack([(words >> np.uint32(2 * i)) & 3 for i in range(PACK_K)], -1)
    codes = codes.reshape(4, -1).astype(np.int8)
    codes = np.where(codes == 3, -1, codes)
    assert (codes == q.astype(np.int8) - 1).all(), "int2 code remap broken"
    assert (scale == d).all(), "scale extraction broken"


# ---- GDN head order ---------------------------------------------------------
def vperm(nv: int, nk: int, unit: int) -> np.ndarray:
    """Index vector mapping llama.cpp tiled V order back to HF grouped order."""
    return (np.arange(nv * unit).reshape(nv // nk, nk, unit)
            .transpose(1, 0, 2).reshape(-1))


def main() -> None:
    a = parse_args()
    _selftest_remap()
    sys.path.insert(0, find_gguf_py(a.gguf_py))
    from gguf import GGUFReader  # noqa: WPS433

    t0 = time.perf_counter()
    r = GGUFReader(a.gguf)
    fields = {k: v.contents() for k, v in r.fields.items()}
    if fields.get("general.architecture") != "qwen35":
        sys.exit(f"unsupported architecture {fields.get('general.architecture')!r}")
    g = lambda key: fields["qwen35." + key]  # noqa: E731

    # ---- Hadamard contract ------------------------------------------------
    if int(fields.get("prism.hadamard.version", 0)) != 1:
        sys.exit("GGUF has no prism.hadamard.version == 1 contract")
    if fields["prism.hadamard.transform"] != "normalized-sylvester-walsh-hadamard" \
            or fields["prism.hadamard.axis"] != "input-last-dimension" \
            or fields["prism.hadamard.sign_mode"] != "explicit":
        sys.exit("unexpected prism.hadamard transform/axis/sign_mode")
    block = int(fields["prism.hadamard.block_size"])
    folded = set(fields["prism.hadamard.weight_names"])
    inverse = set(fields.get("prism.hadamard.inverse_weight_names", []))
    if inverse - {"token_embd.weight"}:
        sys.exit(f"unexpected inverse-lookup tensors: {inverse}")
    if not fields.get("prism.hadamard.gdn_v_grouped", False):
        sys.exit("ssm_out is in tiled V order; this packer assumes gdn_v_grouped")
    signs: dict[int, torch.Tensor] = {}
    off = 0
    vals = fields["prism.hadamard.sign_values"]
    for w in fields["prism.hadamard.sign_widths"]:
        w = int(w)
        v = np.asarray(vals[off:off + w], dtype=np.float32)
        if len(v) != w or not np.isin(v, [-1, 1]).all() or w % block:
            sys.exit(f"bad sign vector for width {w}")
        signs[w] = torch.from_numpy(v).to(torch.float16)
        off += w
    if off != len(vals):
        sys.exit("trailing sign values")
    print(f"[pack] hadamard: H{block}, {len(folded)} folded, {len(inverse)} inverse, "
          f"sign widths {sorted(signs)}")

    nv, nk = int(g("ssm.time_step_rank")), int(g("ssm.group_count"))
    hd = int(g("ssm.inner_size")) // nv
    hk = int(g("ssm.state_size"))
    qk_rows = 2 * nk * hk
    n_layers = int(g("block_count"))
    perm_hd, perm_1 = vperm(nv, nk, hd), vperm(nv, nk, 1)
    print(f"[pack] GDN nv={nv} nk={nk} hd={hd} hk={hk}; layers={n_layers}")

    tensors = {t.name: t for t in r.tensors}

    # ---- group GGUF tensors into vLLM modules ------------------------------
    groups: dict[str, list[tuple[int, str]]] = {}
    residual_src: list[tuple[str, str]] = []   # (gguf name, hf name)
    embed_name = None
    for name in tensors:
        if name == "output.weight":
            groups.setdefault("lm_head", []).append((0, name))
        elif name == "token_embd.weight":
            embed_name = name
        elif name == "output_norm.weight":
            residual_src.append((name, f"{HF_LM_PREFIX}norm.weight"))
        elif name.startswith("blk."):
            _, layer, stem = name.split(".", 2)
            if a.limit_layers and int(layer) >= a.limit_layers:
                continue
            leaf = stem[:-len(".weight")] if stem.endswith(".weight") else stem
            hf_layer = f"{HF_LM_PREFIX}layers.{layer}."
            vl_layer = f"{VLLM_LM_PREFIX}layers.{layer}."
            if tensors[name].tensor_type.name == "PQ2_0":
                if leaf in FUSE_MAP:
                    mod, pos = FUSE_MAP[leaf]
                    groups.setdefault(vl_layer + mod, []).append((pos, name))
                elif leaf in SINGLE_MAP:
                    groups.setdefault(vl_layer + SINGLE_MAP[leaf], []).append((0, name))
                else:
                    sys.exit(f"unmapped quantized tensor {name}")
            elif stem in RESIDUAL_MAP:
                residual_src.append((name, hf_layer + RESIDUAL_MAP[stem]))
            else:
                sys.exit(f"unmapped tensor {name} ({tensors[name].tensor_type.name})")
        else:
            sys.exit(f"unmapped tensor {name}")

    def layer_idx(prefix: str) -> int:
        parts = prefix.split(".")
        return int(parts[parts.index("layers") + 1]) if "layers" in parts else -1

    ordered = sorted(groups, key=lambda p: (layer_idx(p), p))
    print(f"[pack] {len(ordered)} quantized modules, {len(residual_src)} residual tensors")

    # ---- pack the quantized modules ---------------------------------------
    out_tensors: dict[str, torch.Tensor] = {}
    layers_meta: dict[str, dict] = {}
    had_layers: dict[str, dict] = {}
    packed_bytes = 0
    for i, prefix in enumerate(ordered):
        members = sorted(groups[prefix])
        q_parts, s_parts, widths, fold_flags = [], [], [], []
        for _, name in members:
            t = tensors[name]
            words, scale = pq2_0_words_scales(t)
            stem = name.split(".", 2)[2] if name.startswith("blk.") else name
            leaf = stem[:-len(".weight")]
            # undo the converter's grouped -> tiled V-row reorder (N axis)
            if leaf == "attn_qkv":
                idx = np.concatenate([np.arange(qk_rows), qk_rows + perm_hd])
                words, scale = words[idx], scale[idx]
            elif leaf == "attn_gate":
                words, scale = words[perm_hd], scale[perm_hd]
            elif leaf in ("ssm_alpha", "ssm_beta"):
                words, scale = words[perm_1], scale[perm_1]
            qw, sc = to_kn_layout(words, scale)
            q_parts.append(qw)
            s_parts.append(sc)
            widths.append(qw.shape[0] * PACK_K)
            fold_flags.append(name in folded)
        if len(set(widths)) != 1 or len(set(fold_flags)) != 1:
            sys.exit(f"{prefix}: members disagree on K or fold status "
                     f"({widths}, {fold_flags})")
        qw = q_parts[0] if len(q_parts) == 1 else torch.cat(q_parts, dim=1)
        sc = s_parts[0] if len(s_parts) == 1 else torch.cat(s_parts, dim=1)
        if not a.inspect:
            out_tensors[f"{prefix}.qweight"] = qw.contiguous()
            out_tensors[f"{prefix}.scale"] = sc.contiguous()
        layers_meta[prefix] = {
            "kind": "lm_head" if prefix == "lm_head" else "linear",
            "qweight_shape": list(qw.shape),
            "scale_shape": list(sc.shape),
        }
        if fold_flags[0]:
            k = widths[0]
            if k not in signs:
                sys.exit(f"{prefix}: folded with K={k} but no sign vector")
            had_layers[prefix] = {"width": k}
        packed_bytes += qw.numel() * 4 + sc.numel() * 2
        if (i + 1) % 50 == 0 or i + 1 == len(ordered):
            print(f"[pack] {i + 1}/{len(ordered)} {prefix} K={widths[0]} "
                  f"N={qw.shape[1]} folded={fold_flags[0]} "
                  f"({packed_bytes / 1e9:.2f} GB, {time.perf_counter() - t0:.0f}s)",
                  flush=True)

    # ---- embedding: row-major packed lookup table --------------------------
    residual: dict[str, torch.Tensor] = {}
    had_inverse: dict[str, dict] = {}
    if embed_name is not None:
        words, scale = pq2_0_words_scales(tensors[embed_name])   # [vocab, h/16]
        prefix = f"{VLLM_LM_PREFIX}embed_tokens"
        qw = torch.from_numpy(words.view(np.int32)).contiguous()
        sc = torch.from_numpy(scale.view(np.float16)).contiguous()
        if not a.inspect:
            out_tensors[f"{prefix}.qweight"] = qw
            out_tensors[f"{prefix}.scale"] = sc
        layers_meta[prefix] = {"kind": "embedding",
                               "qweight_shape": list(qw.shape),
                               "scale_shape": list(sc.shape)}
        if embed_name in inverse:
            had_inverse[prefix] = {"width": qw.shape[1] * PACK_K}
        print(f"[pack] embedding {tuple(qw.shape)} words, inverse-hadamard="
              f"{embed_name in inverse}")

    # ---- residual (dense) tensors ------------------------------------------
    for name, hf in residual_src:
        t = tensors[name]
        x = gguf_tensor_f32(t)
        stem = name.split(".", 2)[2] if name.startswith("blk.") else name
        if stem == "ssm_a":
            if not (x < 0).all():
                sys.exit(f"{name}: expected negative A = -exp(A_log)")
            x = torch.log(-x)[torch.from_numpy(perm_1)]
        elif stem == "ssm_dt.bias":
            x = x[torch.from_numpy(perm_1)]
        elif stem in ("ssm_alpha.weight", "ssm_beta.weight"):
            x = x[torch.from_numpy(perm_1)]                # rows = heads
        elif stem == "ssm_conv1d.weight":
            # ggml [C, k] -> HF [C, 1, k]; V channels back to grouped order
            idx = np.concatenate([np.arange(qk_rows), qk_rows + perm_hd])
            x = x[torch.from_numpy(idx)].unsqueeze(1)
        elif stem.endswith("norm.weight") and stem != "ssm_norm.weight":
            # Qwen3.5 norms are x*(1+w) (GemmaRMSNorm in vLLM); the converter
            # stores w+1 for plain ggml rms_norm. linear_attn.norm is x*w.
            x = x - 1.0
        residual[hf] = x.to(torch.bfloat16).contiguous()

    print(f"[pack] {len(layers_meta)} sidecar modules ({packed_bytes / 1e9:.2f} GB), "
          f"{len(had_layers)} hadamard-folded, {len(had_inverse)} inverse; "
          f"{len(residual)} residual tensors")
    if a.inspect:
        return

    # ---- write sidecar -----------------------------------------------------
    from safetensors.torch import save_file  # noqa: WPS433
    for w, s in signs.items():
        out_tensors[f"hadamard.signs.{w}"] = s.contiguous()
    meta = {
        "xetla_format_version": "1",
        "xetla_method": "int2_f16",
        "xetla_meta": json.dumps({
            "layers": layers_meta,
            "group_size": GROUP_SIZE,
            "source_model": os.path.abspath(a.gguf),
            "hadamard": {
                "block_size": block,
                "transform": "normalized-sylvester-walsh-hadamard",
                "sign_widths": sorted(signs),
                "layers": had_layers,
                "inverse_layers": had_inverse,
            },
        }),
    }
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_file(out_tensors, out, metadata=meta)
    print(f"[pack] wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB, "
          f"{time.perf_counter() - t0:.0f}s)", flush=True)
    del out_tensors

    # ---- packed model directory --------------------------------------------
    os.makedirs(a.packed, exist_ok=True)
    side_dir = a.ref_dir or a.template
    for fn in TOKENIZER_FILES:
        src = os.path.join(side_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(a.packed, fn))
    for fn, default in PROCESSOR_DEFAULTS.items():
        src = next((os.path.join(d, fn) for d in (a.ref_dir, a.template)
                    if d and os.path.exists(os.path.join(d, fn))), None)
        if src:
            shutil.copy2(src, os.path.join(a.packed, fn))
        else:
            with open(os.path.join(a.packed, fn), "w") as fh:
                json.dump(default, fh, indent=4)

    # vision tower: vLLM's Qwen3_5ForConditionalGeneration needs every
    # parameter accounted for. Bonsai 2's own tower ships as the mmproj GGUF;
    # a --template tower is only a stand-in for text-only runs.
    n_vis = 0
    if a.mmproj:
        vis = vision_tower_from_mmproj(a.mmproj, GGUFReader)
        residual.update(vis)
        n_vis = len(vis)
        vis_src = os.path.abspath(a.mmproj)
    else:
        from safetensors import safe_open  # noqa: WPS433
        import glob  # noqa: WPS433
        for path in sorted(glob.glob(os.path.join(a.template, "*.safetensors"))):
            with safe_open(path, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("model.visual."):
                        residual[k] = f.get_tensor(k)
                        n_vis += 1
        vis_src = os.path.abspath(a.template)

    # config.json: synthesised from the published (MLX repo) config, or the
    # template's; token ids from the GGUF win.
    if a.ref_dir and os.path.exists(os.path.join(a.ref_dir, "config.json")):
        with open(os.path.join(a.ref_dir, "config.json")) as fh:
            cfg = synth_config(json.load(fh))
    else:
        with open(os.path.join(a.template, "config.json")) as fh:
            cfg = json.load(fh)
    eos = int(fields.get("tokenizer.ggml.eos_token_id", cfg["text_config"]["eos_token_id"]))
    bos = int(fields.get("tokenizer.ggml.bos_token_id", cfg["text_config"]["bos_token_id"]))
    pad = int(fields.get("tokenizer.ggml.padding_token_id", bos))
    cfg["text_config"]["eos_token_id"] = eos
    cfg["text_config"]["bos_token_id"] = bos
    cfg["_bonsai2_source_gguf"] = os.path.basename(a.gguf)
    cfg["_bonsai2_vision_tower_from"] = vis_src
    with open(os.path.join(a.packed, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=4)
    gen = {"bos_token_id": bos, "pad_token_id": pad, "do_sample": True,
           "eos_token_id": sorted({eos, pad}),
           "temperature": float(fields.get("general.sampling.temp", 1.0)),
           "top_k": int(fields.get("general.sampling.top_k", 20)),
           "top_p": round(float(fields.get("general.sampling.top_p", 0.95)), 3)}
    with open(os.path.join(a.packed, "generation_config.json"), "w") as fh:
        json.dump(gen, fh, indent=4)

    res_path = os.path.join(a.packed, "model-residual.safetensors")
    save_file(residual, res_path, metadata={"format": "pt"})
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size()
                                            for t in residual.values())},
             "weight_map": {k: "model-residual.safetensors" for k in residual}}
    with open(os.path.join(a.packed, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    print(f"[pack] wrote {a.packed} ({len(residual)} tensors incl. {n_vis} vision, "
          f"{os.path.getsize(res_path) / 1e9:.2f} GB; eos={eos} bos={bos})")


if __name__ == "__main__":
    main()
