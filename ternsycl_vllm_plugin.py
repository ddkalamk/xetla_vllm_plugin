

#from vllm import ModelRegistry
from typing import Any, Optional

import torch
import os
import time
from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
            QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                           UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
timing_enabled = int(os.environ.get("TERNSYCL_TIMINGS", "0")) > 0
quantize_lm_heads = int(os.environ.get("TERNSYCL_QUANTIZE_LM_HEADS", "1")) > 0
# The int2 x fp16 DPAS prefill kernel converts activations to int8 (XMX), which
# costs ~1.3% relative error per GEMM. That compounds across layers and is
# enough to destroy a model outright: the CAT-Q Qwen3 8B and 32B emit a single
# repeated token with it on and are coherent with it off, while Bonsai happens
# to survive it. It buys almost nothing anyway (8B TTFT 75 ms with, 80 ms
# without), so it is off unless TERNSYCL_DISABLE_DPAS=0 asks for it.
disable_dpas = int(os.environ.get("TERNSYCL_DISABLE_DPAS", "1")) > 0


# ---- Pre-quantized sidecar (Option B) ---------------------------------------
# When TERNSYCL_PREQUANT_PATH points at a .safetensors file produced by
# scripts/prequantize_gguf.py, process_weights_after_loading() will load each
# layer's (qweight, scale) directly from disk and skip the GGUF dequant +
# CPU re-quant pipeline. When TERNSYCL_PREQUANT_DUMP_PATH is set, the plugin
# captures every quantized layer it processes and flushes them to that path
# right after model load.
_ternsycl_prequant_load_path = os.environ.get("TERNSYCL_PREQUANT_PATH", "") or None
_ternsycl_prequant_dump_path = os.environ.get("TERNSYCL_PREQUANT_DUMP_PATH", "") or None
_ternsycl_prequant_cache: dict = {}   # lazy-loaded {key -> torch.Tensor}
_ternsycl_prequant_meta: dict = {}    # parsed JSON from sidecar header
_ternsycl_prequant_dump_buf: dict = {}  # {prefix: {qweight, scale, kind, dpas}}


def _ternsycl_prequant_load_index() -> None:
    """Open the sidecar safetensors file lazily and populate the meta dict."""
    if not _ternsycl_prequant_load_path or _ternsycl_prequant_meta:
        return
    try:
        from safetensors import safe_open  # noqa: WPS433
        import json  # noqa: WPS433
        with safe_open(_ternsycl_prequant_load_path, framework="pt") as f:
            md = f.metadata() or {}
            _ternsycl_prequant_meta.update({
                "format_version": md.get("ternsycl_format_version", "1"),
                "method": md.get("ternsycl_method", ""),
                "keys": set(f.keys()),
                "extra": json.loads(md.get("ternsycl_meta", "{}") or "{}"),
            })
        print(f"[ternsycl] sidecar loaded: {_ternsycl_prequant_load_path} "
              f"({len(_ternsycl_prequant_meta['keys'])} tensors, "
              f"method={_ternsycl_prequant_meta['method']})", flush=True)
    except Exception as e:
        print(f"[ternsycl] WARN: could not open prequant sidecar "
              f"{_ternsycl_prequant_load_path}: {e}", flush=True)
        _ternsycl_prequant_meta["keys"] = set()


def _ternsycl_prequant_lookup(prefix: str, method: str = "") -> Optional[str]:
    """Resolve `prefix` against the sidecar index and return the matching
    sidecar key prefix, or None when the layer is not in the sidecar.

    When the model is loaded as a speculative draft, vLLM prefixes all layer
    names with 'draft_model.' (e.g. 'draft_model.model.layers.0.self_attn.qkv_proj').
    The sidecar was generated from the standalone model and therefore uses the
    unprefixed names.  We strip leading path components one at a time until we
    find a match, so both loading modes use the same sidecar file.
    """
    if not _ternsycl_prequant_load_path or not prefix:
        return None
    _ternsycl_prequant_load_index()
    if method and _ternsycl_prequant_meta.get("method") and \
            _ternsycl_prequant_meta["method"] != method:
        # Sidecar was produced for a different quant method.
        return None
    keys = _ternsycl_prequant_meta.get("keys", set())

    # Try the prefix as-is, then strip leading components until a match.
    candidates = [prefix]
    parts = prefix.split(".")
    for i in range(1, len(parts)):
        candidates.append(".".join(parts[i:]))

    for cand in candidates:
        if f"{cand}.qweight" in keys and f"{cand}.scale" in keys:
            return cand
    return None


def _ternsycl_target_device(layer: torch.nn.Module) -> torch.device:
    """Device the packed weights should land on.

    ``layer.weight`` may be a zero-storage *meta* placeholder (see
    ``TernsyclLinearMethod.create_weights``), in which case we fall back to the
    current accelerator device.
    """
    dev = layer.weight.data.device
    if dev.type != "meta":
        return dev
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device(f"xpu:{torch.xpu.current_device()}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _ternsycl_shard_packed(layer: torch.nn.Module, qw: torch.Tensor,
                        sc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut full packed tensors down to this rank's shard.

    The sidecar stores whole tensors, qweight [K/16, N] and scale [K/128, N], so
    an input-sharded layer takes a slice of rows and an output-sharded one a
    slice of columns. Fused layers are concatenations of independently sharded
    blocks - qkv splits q, k and v separately, gate_up splits gate and up - so
    each block is sliced on its own and the pieces rejoined.
    """
    tp = getattr(layer, "tp_size", 1) or 1
    if tp == 1:
        return qw, sc
    rank = getattr(layer, "tp_rank", 0)

    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear,
        RowParallelLinear)

    if isinstance(layer, RowParallelLinear):
        # K is split. qweight packs 16 K-rows per int32 and scale one row per
        # 128, so both divide cleanly only if K/tp stays a multiple of 128.
        if qw.shape[0] % tp or sc.shape[0] % tp:
            raise ValueError(
                f"cannot split K={qw.shape[0] * 16} across tp={tp} "
                f"and keep the 128-element scale groups intact")
        kq, ks = qw.shape[0] // tp, sc.shape[0] // tp
        return (qw[rank * kq:(rank + 1) * kq].contiguous(),
                sc[rank * ks:(rank + 1) * ks].contiguous())

    if isinstance(layer, QKVParallelLinear):
        hs = layer.head_size
        # When there are fewer kv heads than ranks vLLM replicates them, so the
        # kv block index is not the rank.
        if layer.total_num_kv_heads >= tp:
            kv_idx, kv_take = rank, layer.num_kv_heads * hs
        else:
            kv_idx, kv_take = rank // (tp // layer.total_num_kv_heads), hs
        blocks = [layer.total_num_heads * hs,
                  layer.total_num_kv_heads * hs,
                  layer.total_num_kv_heads * hs]
        takes = [layer.num_heads * hs, kv_take, kv_take]
        idxs = [rank, kv_idx, kv_idx]
    elif isinstance(layer, MergedColumnParallelLinear):
        blocks = list(layer.output_sizes)
        takes = [b // tp for b in blocks]
        idxs = [rank] * len(blocks)
    elif isinstance(layer, ColumnParallelLinear):
        blocks = [qw.shape[1]]
        takes = [qw.shape[1] // tp]
        idxs = [rank]
    else:
        return qw, sc

    qs, ss, off = [], [], 0
    for block, take, idx in zip(blocks, takes, idxs):
        start = off + idx * take
        qs.append(qw[:, start:start + take])
        ss.append(sc[:, start:start + take])
        off += block
    return (torch.cat(qs, dim=1).contiguous(),
            torch.cat(ss, dim=1).contiguous())


def _ternsycl_prequant_try_load(layer: torch.nn.Module, prefix: str,
                             method: str, kind: str) -> bool:
    """If a sidecar entry exists for `prefix`, populate the layer in-place
    and return True. `kind` is 'linear' or 'lm_head'.
    """
    lookup_prefix = _ternsycl_prequant_lookup(prefix, method)
    if lookup_prefix is None:
        return False
    try:
        from safetensors import safe_open  # noqa: WPS433
        dev = _ternsycl_target_device(layer)
        qkey = f"{lookup_prefix}.qweight"
        skey = f"{lookup_prefix}.scale"
        with safe_open(_ternsycl_prequant_load_path, framework="pt") as f:
            qw = f.get_tensor(qkey)
            sc = f.get_tensor(skey)
        qw, sc = _ternsycl_shard_packed(layer, qw, sc)
        layer.weight = torch.nn.Parameter(
            qw.to(dev).contiguous(), requires_grad=False)
        layer.scale = torch.nn.Parameter(
            sc.to(dev).contiguous(), requires_grad=False)
        layer.ternsycl_quantized = True
        # Re-derive dispatch capability locally (no need to store).
        layer._ternsycl_dpas_capable = (not disable_dpas) and (qw.shape[1] & 15) == 0
        _ternsycl_pre_convert_bias(layer)
        _ternsycl_hadamard_attach(layer, lookup_prefix, dev, qw.shape[0] * 16)
        return True
    except Exception as e:
        print(f"[ternsycl] WARN: sidecar load failed for {prefix} (lookup={lookup_prefix}): {e}",
              flush=True)
        return False


def _ternsycl_prequant_dump_record(prefix: str, layer: torch.nn.Module,
                                method: str, kind: str) -> None:
    """Capture the just-quantized weights for later flush to disk."""
    if not _ternsycl_prequant_dump_path or not prefix:
        return
    try:
        qw = layer.weight.data.detach().to("cpu").contiguous()
        sc = layer.scale.data.detach().to("cpu").contiguous()
        rec = {
            "qweight": qw,
            "scale": sc,
            "method": method,
            "kind": kind,
        }
        _ternsycl_prequant_dump_buf[prefix] = rec
    except Exception as e:
        print(f"[ternsycl] WARN: dump capture failed for {prefix}: {e}",
              flush=True)


def _ternsycl_prequant_flush_dump() -> None:
    """Write the accumulated buffer out as a single safetensors file."""
    if not _ternsycl_prequant_dump_path or not _ternsycl_prequant_dump_buf:
        return
    try:
        from safetensors.torch import save_file  # noqa: WPS433
        import json  # noqa: WPS433
        tensors: dict = {}
        layers_meta: dict = {}
        method_seen = ""
        for prefix, rec in _ternsycl_prequant_dump_buf.items():
            tensors[f"{prefix}.qweight"] = rec["qweight"]
            tensors[f"{prefix}.scale"] = rec["scale"]
            layers_meta[prefix] = {
                "kind": rec["kind"],
                "qweight_shape": list(rec["qweight"].shape),
                "scale_shape": list(rec["scale"].shape),
            }
            method_seen = rec["method"]
        meta = {
            "ternsycl_format_version": "1",
            "ternsycl_method": method_seen,
            "ternsycl_meta": json.dumps({"layers": layers_meta}),
        }
        os.makedirs(os.path.dirname(_ternsycl_prequant_dump_path) or ".",
                    exist_ok=True)
        save_file(tensors, _ternsycl_prequant_dump_path, metadata=meta)
        n = len(_ternsycl_prequant_dump_buf)
        size_mb = os.path.getsize(_ternsycl_prequant_dump_path) / 1e6
        print(f"[ternsycl] sidecar written: {_ternsycl_prequant_dump_path} "
              f"({n} layers, {size_mb:.1f} MB, method={method_seen})",
              flush=True)
        _ternsycl_prequant_dump_buf.clear()
    except Exception as e:
        print(f"[ternsycl] ERROR: sidecar flush failed: {e}", flush=True)


def _ternsycl_is_compiling() -> bool:
    """Return True when running under torch.compile / Dynamo tracing.

    The custom_op wrappers are needed in that case so the FX graph stays
    closed; in eager and XPU-graph capture we can call the kernel directly
    and shave the dispatcher frame off every call.
    """
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:  # older torch
        return False


def _ternsycl_pre_convert_bias(layer: torch.nn.Module) -> None:
    """B8: pre-convert the layer's bias to fp16 once at load time so the
    per-call ``bias.to(torch.float16)`` becomes a no-op (same dtype, returns
    self).  Safe for layers whose forward consumes bias in fp16 only.
    """
    b = getattr(layer, "bias", None)
    if b is None:
        return
    if isinstance(b, torch.nn.Parameter):
        if b.data.dtype != torch.float16:
            b.data = b.data.to(torch.float16)
    elif isinstance(b, torch.Tensor) and b.dtype != torch.float16:
        layer.bias = b.to(torch.float16)


# ---- Hadamard rotated basis (Bonsai 2) --------------------------------------
# Bonsai 2 stores every folded matrix in a rotated input basis: the runtime has
# to multiply the activation by a fixed +-1 sign vector and then apply a
# blockwise normalised Walsh-Hadamard transform (block 1024) before the GEMM,
# and undo the same rotation on token embeddings after the lookup. The packer
# (scripts/pack_bonsai2_gguf.py) carries the GGUF's prism.hadamard contract in
# the sidecar metadata plus one `hadamard.signs.<K>` tensor per width.
#
# The transform runs as one fused TernSYCL kernel (ternsycl/hadamard)
# when the extension provides it; TERNSYCL_HADAMARD_IMPL=matmul falls back to a
# matmul with the (symmetric) H_block/sqrt(block) matrix, which is what the
# PrismML llama.cpp fork does and serves as the reference here.
_ternsycl_hadamard_mats: dict = {}
_ternsycl_hadamard_dtype = (torch.float16
                         if os.environ.get("TERNSYCL_HADAMARD_DTYPE", "fp32") == "fp16"
                         else torch.float32)
_ternsycl_hadamard_impl = os.environ.get("TERNSYCL_HADAMARD_IMPL", "fused").lower()


def _ternsycl_hadamard_fused_available() -> bool:
    if _ternsycl_hadamard_impl != "fused":
        return False
    return hasattr(torch.ops.ternsycl, "hadamard_fwht_run")


def _ternsycl_hadamard_matrix(block: int, device) -> torch.Tensor:
    key = (block, str(device))
    h = _ternsycl_hadamard_mats.get(key)
    if h is None:
        h = torch.ones(1, 1, dtype=torch.float32)
        while h.shape[0] < block:          # Sylvester construction
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h = (h / float(block) ** 0.5).to(device=device, dtype=_ternsycl_hadamard_dtype)
        _ternsycl_hadamard_mats[key] = h
    return h


def _ternsycl_hadamard_attach(layer: torch.nn.Module, lookup_prefix: str,
                           dev: torch.device, k_local: int = 0) -> None:
    """Pin the layer's sign vector when the sidecar marks it as folded.

    `k_local` is the layer's input width after tensor-parallel slicing; when it
    is smaller than the stored width the matching K-range of the sign vector
    is taken (RowParallelLinear splits K contiguously per rank).
    """
    layer.ternsycl_hadamard = None
    layer.ternsycl_hadamard_inverse = None
    had = (_ternsycl_prequant_meta.get("extra") or {}).get("hadamard") or {}
    if not had:
        return
    fwd = (had.get("layers") or {}).get(lookup_prefix)
    inv = (had.get("inverse_layers") or {}).get(lookup_prefix)
    rec = fwd or inv
    if rec is None:
        return
    width, block = int(rec["width"]), int(had["block_size"])
    from safetensors import safe_open  # noqa: WPS433
    with safe_open(_ternsycl_prequant_load_path, framework="pt") as f:
        signs = f.get_tensor(f"hadamard.signs.{width}")
    if k_local and k_local != width:
        if width % k_local or k_local % block:
            raise ValueError(f"[ternsycl] hadamard: cannot slice K={width} to "
                             f"{k_local} on block {block}")
        rank = getattr(layer, "tp_rank", 0)
        signs = signs[rank * k_local:(rank + 1) * k_local]
    signs = signs.to(device=dev, dtype=_ternsycl_hadamard_dtype).contiguous()
    fused = _ternsycl_hadamard_fused_available() and block == 1024
    if fused:
        signs = signs.to(torch.int8).contiguous()
    else:
        _ternsycl_hadamard_matrix(block, dev)   # materialise before graph capture
    layer.ternsycl_hadamard_fused = fused
    if fwd:
        layer.ternsycl_hadamard = (signs, block)
    else:
        layer.ternsycl_hadamard_inverse = (signs, block)


def _ternsycl_hadamard_fwd(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """x -> H_block (signs * x), blockwise along the last dim. Keeps x.dtype."""
    had = getattr(layer, "ternsycl_hadamard", None)
    if had is None:
        return x
    signs, block = had
    if getattr(layer, "ternsycl_hadamard_fused", False):
        x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
        if not x16.is_contiguous():
            x16 = x16.contiguous()
        if _ternsycl_is_compiling():
            y = ternsycl_hadamard_fwht(x16, signs, block, False)
        else:
            y = torch.ops.ternsycl.hadamard_fwht_run(x16, signs, block, False)
        return y if y.dtype == x.dtype else y.to(x.dtype)
    h = _ternsycl_hadamard_matrix(block, x.device)
    y = (x.to(_ternsycl_hadamard_dtype) * signs).reshape(-1, block) @ h
    return y.reshape(x.shape).to(x.dtype)


def _ternsycl_hadamard_inv(layer: torch.nn.Module, e: torch.Tensor) -> torch.Tensor:
    """Inverse for a rotated embedding row: signs * (H_block e)."""
    had = getattr(layer, "ternsycl_hadamard_inverse", None)
    if had is None:
        return e
    signs, block = had
    if getattr(layer, "ternsycl_hadamard_fused", False):
        e16 = e if e.dtype == torch.float16 else e.to(torch.float16)
        if not e16.is_contiguous():
            e16 = e16.contiguous()
        if _ternsycl_is_compiling():
            y = ternsycl_hadamard_fwht(e16, signs, block, True)
        else:
            y = torch.ops.ternsycl.hadamard_fwht_run(e16, signs, block, True)
        return y if y.dtype == e.dtype else y.to(e.dtype)
    h = _ternsycl_hadamard_matrix(block, e.device)
    y = (e.to(_ternsycl_hadamard_dtype).reshape(-1, block) @ h).reshape(e.shape)
    return (y * signs).to(e.dtype)


class Timer:
    """A simple context manager for measuring execution time."""
    def __init__(self, *tensors):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.tensors = tensors

    def __enter__(self):
        """Called when the 'with' statement is entered."""
        if not timing_enabled:
            return self
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when the 'with' statement is exited."""
        if not timing_enabled:
            return
        self.end_time = time.perf_counter()
        self.elapsed_time = (self.end_time - self.start_time) * 1e6  # usecs
        print(f"Execution time: {self.elapsed_time:.4f} usecs, {[list(t.shape) for t in self.tensors]}")

# ---- int2 weights with per-K-group fp16 scales (gs=128), fp16 activations ----
INT2_F16_GROUP_SIZE = 128

@torch.library.custom_op("ternsycl::int2_fp16_upcvt_gemm", mutates_args=())
def ternsycl_int2_fp16_upcvt_gemm(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """int2 weight x fp16 act GEMM with per-K-group fp16 scales (gs=128).

    input  : fp16 [M, K]
    weight : int32 [K/16, N]   (16 K-rows packed per int32, codes {0,+1,-1})
    scale  : fp16  [K/128, N]
    bias   : optional fp16 [N]

    Routes to the int2 x int8 DPAS kernel for prefill (M > 1) when
    TERNSYCL_DISABLE_DPAS=0; otherwise to the upcvt kernel (fp16 DPAS).
    """
    m = input.shape[0]
    n = weight.shape[1]
    # The eager path in TernsyclLinearMethod.apply gates this on disable_dpas via
    # _ternsycl_dpas_capable; this one has to check it too, or TERNSYCL_DISABLE_DPAS
    # silently does nothing whenever the model is compiled.
    use_dpas = (not disable_dpas) and (m > 1) and (n % 16 == 0)
    with Timer(input, weight):
        if use_dpas:
            out = torch.ops.ternsycl.int2_fp16_dpas_gemm_run(
                input, weight, scale, None
            )
        else:
            out = torch.ops.ternsycl.int2_fp16_upcvt_gemm_run(
                input, weight, scale, None
            )
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out

@ternsycl_int2_fp16_upcvt_gemm.register_fake
def _ternsycl_int2_fp16_upcvt_gemm_fake(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return input.new_empty([input.shape[0], weight.shape[1]])


@torch.library.custom_op("ternsycl::int2_fp16_upcvt_postop_gemm", mutates_args=())
def ternsycl_int2_fp16_upcvt_postop_gemm(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    other: torch.Tensor, postop: int,
) -> torch.Tensor:
    """int2 GEMV with the epilogue folded in, as the OpenVINO integration does.

    postop 1 is silu(acc) * other, which is the SwiGLU gate; 2 is acc + other.
    Folding the activation here removes the separate silu and multiply launches
    that otherwise sit between the two MLP projections.
    """
    with Timer(input, weight):
        return torch.ops.ternsycl.int2_fp16_upcvt_gemm_postop_run(
            input, weight, scale, other, postop
        )


@ternsycl_int2_fp16_upcvt_postop_gemm.register_fake
def _ternsycl_int2_fp16_upcvt_postop_gemm_fake(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    other: torch.Tensor, postop: int,
) -> torch.Tensor:
    return input.new_empty([input.shape[0], weight.shape[1]])


@torch.library.custom_op("ternsycl::hadamard_fwht", mutates_args=())
def ternsycl_hadamard_fwht(
    x: torch.Tensor, signs: Optional[torch.Tensor], block: int, inverse: bool,
) -> torch.Tensor:
    """Fused sign flip + blockwise normalised Walsh-Hadamard transform (fp16).

    forward: H(signs*x)/sqrt(block) per block of the last dim; inverse:
    signs*(H x)/sqrt(block). One launch, no Hadamard matrix reads.
    """
    return torch.ops.ternsycl.hadamard_fwht_run(x, signs, block, inverse)


@ternsycl_hadamard_fwht.register_fake
def _ternsycl_hadamard_fwht_fake(
    x: torch.Tensor, signs: Optional[torch.Tensor], block: int, inverse: bool,
) -> torch.Tensor:
    return torch.empty_like(x)


def pack_int2_rowwise(codes):
    """Pack int8 codes in {-1,0,+1} of shape [rows, cols] into int32 words of
    shape [rows, cols/16], 16 consecutive column entries per word.

    This is the layout used for embedding tables: unlike the vnni16 GEMM
    layout, one row stays contiguous so a token lookup is a single linear
    read.
    """
    rows, cols = codes.shape
    assert cols % 16 == 0, f"cols ({cols}) must be multiple of 16"
    c = (codes & 0x3).to(torch.int32).view(rows, cols // 16, 16)
    shifts = (torch.arange(16, dtype=torch.int32, device=codes.device) * 2)
    return (c << shifts).sum(dim=-1).to(torch.int32)


def unpack_int2_rowwise(packed, scale, group_size: int = INT2_F16_GROUP_SIZE):
    """Inverse of `pack_int2_rowwise`, applying the per-group fp16 scales.

    packed : int32 [rows, cols/16]
    scale  : fp16  [rows, cols/group_size]
    returns  fp16  [rows, cols]
    """
    rows = packed.shape[0]
    shifts = (torch.arange(16, dtype=torch.int32, device=packed.device) * 2)
    codes = (packed.unsqueeze(-1) >> shifts) & 0x3
    # int2 two's complement: 3 -> -1
    codes = torch.where(codes == 3, codes - 4, codes)
    vals = codes.reshape(rows, -1).to(torch.float16)
    return vals * scale.repeat_interleave(group_size, dim=1)


def ternsycl_quant_method():
    quant_method = os.environ.get("TERNSYCL_QUANT_METHOD", "").lower()
    # Auto-derive from the sidecar metadata if the user didn't pin a method
    # but TERNSYCL_PREQUANT_PATH is set. Avoids the silent
    # "RuntimeError: A must be bf16" when the wrong default is used.
    if not quant_method and _ternsycl_prequant_load_path:
        try:
            _ternsycl_prequant_load_index()
            sc_method = _ternsycl_prequant_meta.get("method", "")
            if sc_method:
                print(f"[ternsycl] inferring TERNSYCL_QUANT_METHOD={sc_method} "
                      f"from sidecar metadata", flush=True)
                quant_method = sc_method
        except Exception:
            pass
    if not quant_method:
        quant_method = "int2_f16"
    if quant_method not in ["bf16", "int2_f16"]:
        raise ValueError(f"Unsupported ternsycl quantization method: {quant_method}")
    return quant_method

def pack_int2_vnni16(t):
    d0, d1 = t.shape
    assert d0 % 16 == 0, "Dim 0 must be multiple of 16"
    t1 = t.view([d0//16, 16, d1]).permute([0, 2, 1]).contiguous()
    t1 = t1 & 0x3
    shifts = torch.arange(16, dtype=torch.int32) * 2
    shifts = shifts.to(t1.device)
    packed = (t1.to(torch.int32) << shifts).sum(dim=-1).to(torch.int32)
    return packed

def quantize_to_ternary_f16(t, group_size: int = INT2_F16_GROUP_SIZE):
    """Quantize a [K, N] fp16/bfloat16 weight that is already a ternary
    (-s, 0, +s) tensor with shared scale s every `group_size` rows along K.

    Returns:
        codes : int8 [K, N] in {-1, 0, +1}
        scale : fp16 [K // group_size, N]
    """
    K, N = t.shape
    assert K % group_size == 0, f"K ({K}) must be multiple of {group_size}"
    tg = t.float().view(K // group_size, group_size, N)
    scale = tg.abs().amax(dim=1)  # [K/gs, N]
    safe = scale.clone()
    safe[safe == 0] = 1.0
    q = torch.round(tg / safe.unsqueeze(1))
    q = torch.clamp(q, -1, 1).to(torch.int8)
    codes = q.view(K, N)
    return codes, scale.to(torch.float16)

def pack_ternary_to_int2(codes):
    """Pack int8 codes in {-1, 0, +1} into int2 codes {0, 1, 3} and then into
    int32 words (16 K-rows per word) using the same vnni16 layout as the
    bf16 path. Input: [K, N] int8. Output: [K/16, N] int32.
    """
    # int2 encoding: 0 -> 0, +1 -> 1, -1 -> 3 (== two's-complement int2 of -1).
    # codes is signed int8 in {-1,0,1}; (codes & 3) gives {0,1,3}.
    return pack_int2_vnni16((codes & 0x3))

class TernsyclConfig(QuantizationConfig):
    def __init__(self) -> None:
        self.method = ternsycl_quant_method()
        # Required by the GGUF model loader, which calls
        # `vllm_config.quant_config.unquantized_modules.extend(...)` even when
        # the active quant_config is not GGUFConfig (which happens whenever
        # we load a .gguf file but request `quantization=ternsycl` so the ternsycl
        # plugin re-quantizes weights to int2 with fp16 scales).
        self.unquantized_modules: list[str] = []
        super().__init__()

    def __repr__(self) -> str:
        return "TernsyclConfig()"

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "ternsycl"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TernsyclConfig":
        return cls()

    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant, **kwargs) -> Optional[QuantizationMethods]:
        # Allow the user to opt into the ternsycl path even when the source
        # checkpoint advertises a different quantization method (e.g. gguf).
        # When the user explicitly asks for `ternsycl`, claim ownership so
        # `_verify_quantization` does not raise a mismatch error.
        # `**kwargs` absorbs extra arguments added by newer vLLM versions
        # (e.g. `hf_config=` since 0.21.0) -- without it the override is
        # silently skipped and the model loads dense.
        if user_quant == "ternsycl":
            return "ternsycl"
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["LinearMethodBase"]:
        if isinstance(layer, LinearBase):
            return TernsyclLinearMethod(self, prefix=prefix)
        elif isinstance(layer, ParallelLMHead) and quantize_lm_heads:
            return TernsyclEmbeddingMethod(self, True, prefix=prefix)
        elif isinstance(layer, VocabParallelEmbedding) and quantize_lm_heads:
            return TernsyclEmbeddingMethod(self, False, prefix=prefix)
        # Other layer types (notably `Attention`) are not handled by ternsycl;
        # vLLM falls back to the default impl when we return None. Only print
        # once per type when TERNSYCL_DEBUG=1 to avoid spamming one line per
        # transformer block at startup.
        if int(os.environ.get("TERNSYCL_DEBUG", "0")) > 0:
            print(f"TernsyclConfig.get_quant_method: passthrough for {type(layer).__name__} ({prefix})")
        return None

class TernsyclEmbeddingMethod(UnquantizedEmbeddingMethod):
    """ Ternsycl quantized method for embeddings.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: TernsyclConfig, inplace: bool = False,
                 prefix: str = ""):
        self.quant_config = quant_config
        self.inplace = inplace
        self.prefix = prefix
        # self.quant_config.method = "int2"  # Embeddings use int2
        super().__init__()

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        # Same trick as TernsyclLinearMethod: when the sidecar already holds the
        # packed table, allocate the parameter on `meta` so the dense fp16
        # table (2.5 GB for a 248k x 5120 vocab) is never materialized. The
        # shape is preserved so VocabParallelEmbedding's sharded weight_loader
        # still validates, and its copies become no-ops.
        method = self.quant_config.method
        if (not self.inplace and method == "int2_f16"
                and _ternsycl_prequant_lookup(self.prefix, method) is not None):
            from vllm.model_executor.parameter import ModelWeightParameter
            from vllm.model_executor.utils import set_weight_attrs
            weight_loader = extra_weight_attrs.pop("weight_loader")
            weight = ModelWeightParameter(
                data=torch.empty(sum(output_partition_sizes),
                                 input_size_per_partition,
                                 dtype=params_dtype, device="meta"),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)
            set_weight_attrs(weight, extra_weight_attrs)
            layer._ternsycl_meta_placeholder = True
            return
        UnquantizedEmbeddingMethod.create_weights(
            self, layer, input_size_per_partition, output_partition_sizes,
            input_size, output_size, params_dtype, **extra_weight_attrs)

    def embedding(self, layer: torch.nn.Module,
                  input_: torch.Tensor) -> torch.Tensor:
        """Look up rows of a packed ternary embedding table.

        Only the selected rows are unpacked, so this touches
        `len(input_) * hidden / 4` bytes instead of holding a dense fp16
        table resident.
        """
        if not getattr(layer, "ternsycl_embed_packed", False):
            return super().embedding(layer, input_)
        flat = input_.reshape(-1)
        out = unpack_int2_rowwise(layer.weight.data[flat],
                                  layer.scale.data[flat])
        out = _ternsycl_hadamard_inv(layer, out)
        # Hidden states must carry the model dtype: dense (non-ternary)
        # layers downstream, e.g. Bonsai 2's bf16 in_proj_ba, mm against it.
        out = out.to(getattr(layer, "ternsycl_out_dtype", out.dtype))
        return out.view(*input_.shape, -1)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not current_platform.is_xpu():
            return

        method = self.quant_config.method
        # Sidecar load short-circuit (Option B).
        if (self.inplace and method == "int2_f16" and
                _ternsycl_prequant_try_load(layer, self.prefix, method, "lm_head")):
            print(f"[ternsycl] sidecar hit: {self.prefix} lm_head ({method})")
            return

        # Input embedding: the table is ternary in Bonsai checkpoints, but it
        # is looked up rather than multiplied, so it uses the row-major packed
        # layout and is unpacked per token in embedding().
        if not self.inplace and method == "int2_f16":
            lookup = _ternsycl_prequant_lookup(self.prefix, method)
            if lookup is not None:
                try:
                    from safetensors import safe_open  # noqa: WPS433
                    dev = _ternsycl_target_device(layer)
                    layer.ternsycl_out_dtype = layer.weight.dtype
                    with safe_open(_ternsycl_prequant_load_path,
                                   framework="pt") as f:
                        qw = f.get_tensor(f"{lookup}.qweight")
                        sc = f.get_tensor(f"{lookup}.scale")
                    layer.weight = torch.nn.Parameter(
                        qw.to(dev).contiguous(), requires_grad=False)
                    layer.scale = torch.nn.Parameter(
                        sc.to(dev).contiguous(), requires_grad=False)
                    layer.ternsycl_embed_packed = True
                    layer.ternsycl_quantized = True
                    _ternsycl_hadamard_attach(layer, lookup, dev)
                    print(f"[ternsycl] sidecar hit: {self.prefix} embedding "
                          f"({method}), packed {tuple(qw.shape)}, out dtype "
                          f"{layer.ternsycl_out_dtype}"
                          f"{', inverse-hadamard' if layer.ternsycl_hadamard_inverse else ''}",
                          flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[ternsycl] WARN: packed embedding load failed for "
                          f"{self.prefix}: {e}", flush=True)
                return
            if getattr(layer, "_ternsycl_meta_placeholder", False):
                raise RuntimeError(
                    f"[ternsycl] sidecar entry for {self.prefix} vanished "
                    "between create_weights() and "
                    "process_weights_after_loading()")

        if self.quant_config.method == "int2_f16":
            # Only quantize the lm_head (inplace=True). Input embedding
            # lookups still need a dense fp16 table, so leave VocabParallel-
            # Embedding alone.
            if not self.inplace:
                return
            weight = layer.weight.data  # [vocab_size, hidden_size]
            dev = weight.device
            # The lm_head is huge (e.g. 151680x4096). Doing the float()
            # reshape/round on XPU temporarily allocates several GB which
            # easily trips UR_RESULT_ERROR_DEVICE_LOST. Quantize on CPU and
            # ship the small int2 + fp16 scale buffers back to the device.
            wkn = weight.detach().to("cpu", dtype=torch.float16).t().contiguous()
            codes, scale_f16 = quantize_to_ternary_f16(wkn, INT2_F16_GROUP_SIZE)
            packed = pack_ternary_to_int2(codes)
            print(f"Processing lm_head with method int2_f16: weight {tuple(weight.shape)} -> packed {tuple(packed.shape)}, scale {tuple(scale_f16.shape)}")
            layer.weight = torch.nn.Parameter(
                packed.to(dev).contiguous(), requires_grad=False
            )
            layer.scale = torch.nn.Parameter(
                scale_f16.to(dev).contiguous(), requires_grad=False
            )
            layer.ternsycl_quantized = True
            # B7: lm_head N is the (padded) vocab size; check DPAS capability.
            layer._ternsycl_dpas_capable = (not disable_dpas) and (packed.shape[1] & 15) == 0
            _ternsycl_pre_convert_bias(layer)
            _ternsycl_prequant_dump_record(self.prefix, layer, "int2_f16", "lm_head")
    def apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.quant_config.method == "int2_f16" and getattr(layer, "ternsycl_quantized", False):
            x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
            x16 = _ternsycl_hadamard_fwd(layer, x16)
            b16 = bias if bias is None or bias.dtype == torch.float16 else bias.to(torch.float16)
            if not _ternsycl_is_compiling():
                if x16.shape[0] > 1 and getattr(layer, "_ternsycl_dpas_capable", False):
                    out = torch.ops.ternsycl.int2_fp16_dpas_gemm_run(
                        x16, layer.weight, layer.scale, None)
                else:
                    out = torch.ops.ternsycl.int2_fp16_upcvt_gemm_run(
                        x16, layer.weight, layer.scale, None)
                if b16 is not None:
                    out = out + b16
            else:
                out = ternsycl_int2_fp16_upcvt_gemm(x16, layer.weight, layer.scale, b16)
            return out if out.dtype == x.dtype else out.to(x.dtype)
        return super().apply(layer, x, bias)


class TernsyclLinearMethod(LinearMethodBase):
    """Linear method for ternsycl.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: TernsyclConfig, prefix: str = ""):
        self.quant_config = quant_config
        self.prefix = prefix
        super().__init__()
        # dense fallbacks call UnquantizedLinearMethod.apply, which needs its state (_gemm_impl)
        UnquantizedLinearMethod.__init__(self)

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        # When a prequant sidecar already holds the packed weights for this
        # layer we must NOT allocate the dense fp16 tensor: for a 27B model
        # that alone is ~54 GB and never fits on the device. Allocate the
        # parameter on the `meta` device instead -- it keeps the exact shape
        # (so every vLLM weight_loader, including the fused qkv / mamba
        # sharded ones, still validates and "copies" happily) while using
        # zero memory. process_weights_after_loading() then swaps in the real
        # packed tensor from the sidecar.
        method = self.quant_config.method
        if method == "int2_f16" and \
                _ternsycl_prequant_lookup(self.prefix, method) is not None:
            from vllm.model_executor.parameter import ModelWeightParameter
            from vllm.model_executor.utils import set_weight_attrs
            weight_loader = extra_weight_attrs.pop("weight_loader")
            weight = ModelWeightParameter(
                data=torch.empty(sum(output_partition_sizes),
                                 input_size_per_partition,
                                 dtype=params_dtype, device="meta"),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)
            set_weight_attrs(weight, extra_weight_attrs)
            layer._ternsycl_meta_placeholder = True
            return
        # We just reuse UnquantizedLinearMethod to create weights
        UnquantizedLinearMethod.create_weights(self, layer, input_size_per_partition,
                                               output_partition_sizes, input_size,
                                               output_size, params_dtype,
                                               **extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not current_platform.is_xpu():
            return

        method = self.quant_config.method
        # Sidecar load short-circuit (Option B).
        if method == "int2_f16" and \
                _ternsycl_prequant_try_load(layer, self.prefix, method, "linear"):
            print(f"[ternsycl] sidecar hit: {self.prefix} ({method})")
            return

        if getattr(layer, "_ternsycl_meta_placeholder", False):
            # Should never happen: the placeholder is only created when the
            # sidecar has an entry for this prefix.
            raise RuntimeError(
                f"[ternsycl] sidecar entry for {self.prefix} vanished between "
                "create_weights() and process_weights_after_loading()")

        if _ternsycl_prequant_load_path and method == "int2_f16":
            # A sidecar is in use but this layer is not in it. That means the
            # offline packer decided the layer is not ternary/binary (e.g. the
            # vision tower, or a gate projection kept in fp16). Re-quantizing
            # it here would silently destroy accuracy, so keep it dense.
            if int(os.environ.get("TERNSYCL_DEBUG", "0")) > 0:
                print(f"[ternsycl] keeping {self.prefix} dense (not in sidecar)",
                      flush=True)
            return

        print(f"[ternsycl] quantizing {self.prefix} "
              f"[{tuple(layer.weight.shape)}] with method {method}")
        if method == "int2_f16":
            # Bonsai-style ternary fp16 weight: every 128 K-entries share an
            # fp16 scale and values are exactly s*{-1, 0, +1}. Recover that
            # encoding losslessly so we can call the int2 x fp16-scale upcvt
            # GEMM kernel.
            weight = layer.weight.data  # [N_out, K_in], any float dtype
            wkn = weight.t().contiguous().to(torch.float16)  # [K, N]
            codes, scale_f16 = quantize_to_ternary_f16(wkn, INT2_F16_GROUP_SIZE)
            packed = pack_ternary_to_int2(codes.to(weight.device))
            layer.weight.data = packed.contiguous()
            layer.scale = torch.nn.Parameter(
                scale_f16.to(weight.device).contiguous(), requires_grad=False
            )
            # B7: cache the dispatch predicate (depends only on N).
            layer._ternsycl_dpas_capable = (not disable_dpas) and (packed.shape[1] & 15) == 0
            layer.ternsycl_quantized = True
            _ternsycl_pre_convert_bias(layer)
            _ternsycl_prequant_dump_record(self.prefix, layer, method, "linear")
    def apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        method = self.quant_config.method
        if method == "int2_f16" and \
                not getattr(layer, "ternsycl_quantized", False):
            # Layer was deliberately left dense (mixed-precision checkpoint).
            # The ternsycl GEMMs hand the residual stream on in fp16, so a dense
            # bf16 layer (Bonsai 2's in_proj_ba) needs the activation cast.
            w_dtype = layer.weight.dtype
            if x.dtype != w_dtype and w_dtype in (torch.float16, torch.bfloat16):
                b = bias if bias is None or bias.dtype == w_dtype else bias.to(w_dtype)
                return UnquantizedLinearMethod.apply(
                    self, layer, x.to(w_dtype), b).to(x.dtype)
            return UnquantizedLinearMethod.apply(self, layer, x, bias)
        if method == "int2_f16":
            x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
            x16 = _ternsycl_hadamard_fwd(layer, x16)
            # B8: prefer the pre-converted layer.bias (always fp16 already).
            b16 = bias if bias is None or bias.dtype == torch.float16 else bias.to(torch.float16)
            # B7: under eager (no torch.compile tracing), skip the dispatch
            # custom_op and call the right kernel directly using the cached
            # capability flag. Saves ~5us per call from the dispatcher frame.
            if not _ternsycl_is_compiling() and getattr(layer, "_ternsycl_dpas_capable", False) is not None:
                if x16.shape[0] > 1 and layer._ternsycl_dpas_capable:
                    c = torch.ops.ternsycl.int2_fp16_dpas_gemm_run(
                        x16, layer.weight, layer.scale, None)
                else:
                    c = torch.ops.ternsycl.int2_fp16_upcvt_gemm_run(
                        x16, layer.weight, layer.scale, None)
                if b16 is not None:
                    c = c + b16
            else:
                c = ternsycl_int2_fp16_upcvt_gemm(x16, layer.weight, layer.scale, b16)
            return c if c.dtype == x.dtype else c.to(x.dtype)
        return UnquantizedLinearMethod.apply(self, layer, x, bias)


# ---- inline ternsycl GEMM profile shim (TERNSYCL_PROFILE=1) ----------------------
import atexit as _atexit
import collections as _collections
import signal as _signal
import threading as _threading

_xprof_lock = _threading.Lock()
_xprof_stats: dict = {}
_xprof_installed = False
_xprof_armed = False


def _xprof_nbytes(*tensors) -> int:
    total = 0
    for t in tensors:
        if t is not None and hasattr(t, "numel"):
            total += t.numel() * t.element_size()
    return total


def _xprof_capturing() -> bool:
    # Skip host-side sync when an XPU command graph is being recorded;
    # queue.wait() is illegal during capture.
    try:
        return bool(torch.xpu.is_current_stream_capturing())
    except Exception:
        return False


def _xprof_wrap(op, op_name: str):
    def wrapped(A, B, scale_B, *rest, **kw):
        # Inert until armed: the per-call host sync makes the Level Zero driver
        # accumulate ~12 GiB of command-list memory over a profiling prefill,
        # which vLLM charges to the KV budget and which drove the cache
        # negative. Engine init must run unperturbed.
        if not _xprof_armed:
            return op(A, B, scale_B, *rest, **kw)
        m = int(A.shape[0]); k = int(A.shape[1])
        n = int(B.shape[1])
        capturing = _xprof_capturing()
        if not capturing:
            torch.xpu.synchronize()
        t0 = time.perf_counter()
        out = op(A, B, scale_B, *rest, **kw)
        if not capturing:
            torch.xpu.synchronize()
        dt = time.perf_counter() - t0
        nbytes = _xprof_nbytes(A, B, scale_B, out,
                               *(r for r in rest if hasattr(r, "numel")))
        with _xprof_lock:
            key = (op_name, m, n, k)
            s = _xprof_stats.setdefault(key, {"calls": 0, "time_s": 0.0, "bytes": 0})
            s["calls"] += 1
            s["time_s"] += dt
            s["bytes"] += nbytes
        return out
    return wrapped


def xprof_start():
    """Arm the GEMM profiler and drop anything recorded so far."""
    global _xprof_armed
    with _xprof_lock:
        _xprof_stats.clear()
    _xprof_armed = True


def xprof_stop():
    global _xprof_armed
    _xprof_armed = False


def _xprof_print():
    if not _xprof_stats:
        print("\n=== ternsycl profile: no GEMM calls recorded ===\n", flush=True)
        return
    items = sorted(_xprof_stats.items(), key=lambda kv: -kv[1]["time_s"])
    per_op = _collections.defaultdict(lambda: {"calls": 0, "time_s": 0.0, "bytes": 0})
    for (op_name, m, n, k), s in items:
        a = per_op[op_name]
        a["calls"] += s["calls"]; a["time_s"] += s["time_s"]; a["bytes"] += s["bytes"]
    grand_time = sum(s["time_s"] for s in _xprof_stats.values())
    grand_calls = sum(int(s["calls"]) for s in _xprof_stats.values())
    grand_bytes = sum(int(s["bytes"]) for s in _xprof_stats.values())
    GB = 1e9
    print("\n" + "=" * 100, flush=True)
    print("=== ternsycl GEMM profile (per-op aggregate) ===", flush=True)
    print("=" * 100, flush=True)
    print(f"{'op_name':<36} {'calls':>10} {'total_ms':>12} {'avg_us':>10} {'GB/s':>10} {'%time':>8}", flush=True)
    for op_name, a in sorted(per_op.items(), key=lambda kv: -kv[1]["time_s"]):
        avg_us = a["time_s"] / a["calls"] * 1e6
        gbps = a["bytes"] / a["time_s"] / GB if a["time_s"] > 0 else 0.0
        pct = 100 * a["time_s"] / grand_time if grand_time > 0 else 0.0
        print(f"{op_name:<36} {int(a['calls']):>10} {a['time_s']*1e3:>12.2f} {avg_us:>10.1f} {gbps:>10.1f} {pct:>7.2f}%", flush=True)
    print(f"{'TOTAL':<36} {grand_calls:>10} {grand_time*1e3:>12.2f}", flush=True)
    print("\n" + "=" * 100, flush=True)
    print("=== ternsycl GEMM profile (per-shape breakdown) ===", flush=True)
    print("=" * 100, flush=True)
    print(f"{'op_name':<36} {'M':>6} {'N':>8} {'K':>8} {'calls':>8} {'total_ms':>10} {'avg_us':>9} {'GB/s':>8} {'%time':>7}", flush=True)
    for (op_name, m, n, k), s in items:
        avg_us = s["time_s"] / s["calls"] * 1e6
        gbps = s["bytes"] / s["time_s"] / GB if s["time_s"] > 0 else 0.0
        pct = 100 * s["time_s"] / grand_time if grand_time > 0 else 0.0
        print(f"{op_name:<36} {m:>6} {n:>8} {k:>8} {int(s['calls']):>8} {s['time_s']*1e3:>10.2f} {avg_us:>9.1f} {gbps:>8.1f} {pct:>6.2f}%", flush=True)
    print(f"\nGEMM total wall time: {grand_time*1e3:.2f} ms across {grand_calls} calls, "
          f"{grand_bytes/GB:.2f} GB moved, avg eff = {grand_bytes/grand_time/GB:.1f} GB/s", flush=True)
    print("=" * 100, flush=True)


def _install_ternsycl_profile():
    global _xprof_installed
    if _xprof_installed:
        return
    if int(os.environ.get("TERNSYCL_PROFILE", "0")) <= 0:
        return
    candidates = [
        (torch.ops.ternsycl, "int2_fp16_upcvt_gemm_run"),
        (torch.ops.ternsycl, "int2_fp16_dpas_gemm_run"),
    ]
    installed = []
    for ns, name in candidates:
        op = getattr(ns, name, None)
        if op is None:
            continue
        setattr(ns, name, _xprof_wrap(op, name))
        installed.append(name)
    if installed:
        print(f"[ternsycl profile] wrapped: {', '.join(installed)}", flush=True)
        _atexit.register(_xprof_print)
        try:
            _signal.signal(_signal.SIGTERM, lambda *_: (_xprof_print(), os._exit(0)))
        except Exception:
            pass
        _xprof_installed = True
    else:
        print("[ternsycl profile] no ops found to wrap", flush=True)


def _maybe_disable_triton_stride_versioning() -> None:
    """Opt-in workaround for a triton-xpu compiler crash on hybrid models.

    triton-xpu 3.7.0 segfaults inside its Intel-specific
    ``TritonIntelStrideVersioning`` TTIR pass while compiling the FLA chunked
    gated-delta-rule kernel (``fla/ops/chunk_delta_h.py``) that Qwen3.5-style
    models -- e.g. Bonsai-27B -- use for linear-attention prefill.  Every
    autotune config fails, so prefill dies with ``PassManager::run failed``.

    vLLM >= 0.20.2 routes GDN through its XPU path and no longer hits that
    kernel, so this is off by default.  Set
    ``TERNSYCL_TRITON_DISABLE_STRIDE_VERSIONING=1`` to no-op the pass (it is a
    pure optimization) when running on an older vLLM.
    """
    if os.environ.get("TERNSYCL_TRITON_DISABLE_STRIDE_VERSIONING", "0") != "1":
        return
    try:
        from triton._C.libtriton import intel  # noqa: WPS433
    except Exception:
        return
    try:
        if hasattr(intel.passes.ttir, "add_stride_versioning"):
            intel.passes.ttir.add_stride_versioning = lambda pm: None
            print("[ternsycl] disabled triton TritonIntelStrideVersioning pass "
                  "(crashes on FLA gated-delta-rule kernels)", flush=True)
    except Exception as e:
        print(f"[ternsycl] WARN: could not disable stride versioning: {e}",
              flush=True)


def _ternsycl_quantize_lm_head(model: torch.nn.Module) -> None:
    """Quantize embedding tables that vLLM built without a quant_config.

    Several models -- Qwen3.5 / Qwen3-Next (Bonsai-27B) among them -- construct
    ``ParallelLMHead`` and ``VocabParallelEmbedding`` without passing
    ``quant_config`` (the input embedding is built without a ``prefix`` too),
    so ``TernsyclConfig.get_quant_method()`` is never consulted for them and both
    stay dense fp16.  For Bonsai both tables are ternary in the checkpoint
    (whitepaper sec. 4.3) and together they are ~5 GB, so wire them to the
    ternsycl path here, after the weights have been loaded.

    The LM head is the larger win for speed (it is read in full for every
    decoded token); the input embedding is a pure memory win, since only the
    looked-up rows are ever touched.

    Set ``TERNSYCL_QUANTIZE_LM_HEADS=0`` to keep both dense.
    """
    if not quantize_lm_heads or not current_platform.is_xpu():
        return
    method = ternsycl_quant_method()
    if method != "int2_f16":
        return
    try:
        config = TernsyclConfig()
    except Exception:
        return

    for name, module in model.named_modules():
        if not isinstance(module, VocabParallelEmbedding):
            continue
        is_lm_head = isinstance(module, ParallelLMHead)
        if isinstance(getattr(module, "quant_method", None), TernsyclEmbeddingMethod):
            continue  # already handled through get_quant_method()
        if getattr(module, "ternsycl_quantized", False):
            continue
        if _ternsycl_prequant_load_path and \
                _ternsycl_prequant_lookup(name, method) is None:
            # Sidecar in use but it has no packed table: leave it dense rather
            # than silently re-quantizing something that may not be ternary.
            print(f"[ternsycl] {name}: not in sidecar, kept dense", flush=True)
            continue
        if not is_lm_head and not _ternsycl_prequant_load_path:
            # The input embedding is only packed from a sidecar; there is no
            # on-the-fly path for it.
            continue
        try:
            qm = TernsyclEmbeddingMethod(config, inplace=is_lm_head, prefix=name)
            qm.process_weights_after_loading(module)
            if getattr(module, "ternsycl_quantized", False):
                module.quant_method = qm
                kind = "lm_head" if is_lm_head else "embedding"
                print(f"[ternsycl] {kind} {name} quantized ({method}), "
                      f"packed {tuple(module.weight.shape)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[ternsycl] WARN: could not quantize {name}: {e}", flush=True)


# ---- SwiGLU fused into the gate projection's epilogue -----------------------
#
# vLLM merges gate and up into one MergedColumnParallelLinear and applies
# SiluAndMul afterwards, so the activation costs two extra launches and two
# round trips of the intermediate per layer. OpenVINO instead keeps gate and up
# as separate FCs and folds silu*other into the gate's epilogue. We reproduce
# that by splitting the packed gate_up weight once at load time and replacing
# the MLP forward with two GEMVs, the second carrying the activation.
fuse_swiglu = int(os.environ.get("TERNSYCL_FUSE_SWIGLU", "1")) > 0


def _ternsycl_split_gate_up(gu: torch.nn.Module) -> bool:
    """Split a packed gate_up projection into contiguous gate/up halves.

    Only the int2 layout qualifies: it is stored as a 2-D [K/16, N] table
    and has a fused-epilogue kernel.
    """
    sizes = getattr(gu, "output_sizes", None)
    if not sizes or len(sizes) != 2 or sizes[0] != sizes[1]:
        return False
    q = getattr(gu, "weight", None)
    s = getattr(gu, "scale", None)
    if q is None or s is None:
        return False
    q, s = q.data, s.data
    if q.dim() != 2 or s.dim() != 2:
        return False
    if q.dtype != torch.int32 or s.dtype != torch.float16:
        return False
    inter = sizes[0]
    if q.shape[1] != 2 * inter or inter % 4 != 0:
        return False
    gu.ternsycl_gate_q = q[:, :inter].contiguous()
    gu.ternsycl_up_q = q[:, inter:].contiguous()
    gu.ternsycl_gate_s = s[:, :inter].contiguous()
    gu.ternsycl_up_s = s[:, inter:].contiguous()
    # The fused forward never touches the merged tensors again; keeping them
    # would hold a second copy of every gate_up (~3 GiB for the 27B).
    gu.weight.data = q.new_empty(0)
    gu.scale.data = s.new_empty(0)
    return True


def _ternsycl_fused_mlp_forward(self, x):
    gu = self.gate_up_proj
    orig = x.shape
    xf = x.reshape(-1, orig[-1])
    if xf.dtype != torch.float16:
        xf = xf.to(torch.float16)
    # This bypasses TernsyclLinearMethod.apply, so the rotated-basis transform
    # of a Hadamard-folded gate_up has to be applied here.
    xf = _ternsycl_hadamard_fwd(gu, xf).contiguous()
    up = torch.ops.ternsycl.int2_fp16_upcvt_gemm(
        xf, gu.ternsycl_up_q, gu.ternsycl_up_s, None)
    act = torch.ops.ternsycl.int2_fp16_upcvt_postop_gemm(
        xf, gu.ternsycl_gate_q, gu.ternsycl_gate_s, up, 1)
    out, _ = self.down_proj(act)
    # Hand the residual stream back in the model dtype, as apply() does.
    if out.dtype != x.dtype:
        out = out.to(x.dtype)
    return out.reshape(*orig[:-1], out.shape[-1])


def _ternsycl_fuse_swiglu(model: torch.nn.Module) -> None:
    if not fuse_swiglu:
        return
    try:
        from vllm.model_executor.layers.activation import SiluAndMul
    except Exception:
        return
    n = 0
    for mod in model.modules():
        gu = getattr(mod, "gate_up_proj", None)
        act = getattr(mod, "act_fn", None)
        if gu is None or not isinstance(act, SiluAndMul):
            continue
        if getattr(mod, "down_proj", None) is None:
            continue
        # Only the int2 upcvt path has a fused-epilogue kernel; bias on the
        # gate_up projection would need a third post-op slot.
        if not getattr(gu, "ternsycl_quantized", False):
            continue
        if getattr(gu, "bias", None) is not None:
            continue
        try:
            if not _ternsycl_split_gate_up(gu):
                continue
        except Exception as e:
            print(f"[ternsycl] SwiGLU fusion skipped for one block: {e}")
            continue
        mod.forward = _ternsycl_fused_mlp_forward.__get__(mod, type(mod))
        n += 1
    if n:
        print(f"[ternsycl] fused SwiGLU into the gate epilogue for {n} MLP blocks")


def register():
    print("Hello ternsycl plugin!")
    _maybe_disable_triton_stride_versioning()
    # Force-load the TernSYCL extension so its TORCH_LIBRARY block registers
    # `torch.ops.ternsycl.*` in *every* process that loads the plugin (main +
    # each engine worker). Without this the ops are missing in the spawned
    # engine subprocess.
    try:
        import ternsycl_pt_ext  # noqa: F401
    except Exception as e:
        print(f"[ternsycl] WARNING: could not import ternsycl_pt_ext: {e}")

    # Optional GEMM profiling shim: set TERNSYCL_PROFILE=1 to enable.
    try:
        _install_ternsycl_profile()
    except Exception as e:
        print(f"[ternsycl] WARNING: could not install ternsycl_profile: {e}")

    # Sidecar dump hook: wrap the model-level process_weights_after_loading
    # so we can flush the captured (qweight, scale) pairs to a single
    # safetensors file once the whole model has been quantized. Several
    # loaders do `from ...utils import process_weights_after_loading`, so
    # we patch every module that re-exports the binding.
    try:
        import vllm.model_executor.model_loader.utils as _mu  # noqa: WPS433
        _orig_pwal = _mu.process_weights_after_loading

        def _wrapped_pwal(*args, **kwargs):
            _orig_pwal(*args, **kwargs)
            model = args[0] if args else kwargs.get("model")
            if isinstance(model, torch.nn.Module):
                _ternsycl_quantize_lm_head(model)
                _ternsycl_fuse_swiglu(model)
            _ternsycl_prequant_flush_dump()

        _mu.process_weights_after_loading = _wrapped_pwal
        # Patch every loader that imported the symbol by name.
        for _modname in (
            "vllm.model_executor.model_loader.base_loader",
            "vllm.model_executor.model_loader.gguf_loader",
            "vllm.model_executor.model_loader.tensorizer_loader",
            "vllm.model_executor.model_loader",
            "vllm.model_executor.models.utils",
            "vllm.model_executor.models.mllama4",
        ):
            try:
                _m = __import__(_modname, fromlist=["*"])
            except Exception:
                continue
            if hasattr(_m, "process_weights_after_loading"):
                _m.process_weights_after_loading = _wrapped_pwal
        if _ternsycl_prequant_dump_path:
            print(f"[ternsycl] sidecar dump enabled -> {_ternsycl_prequant_dump_path}")
    except Exception as e:
        print(f"[ternsycl] WARN: could not install post-load hook: {e}")

    register_quantization_config("ternsycl")(TernsyclConfig)
    
