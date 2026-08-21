

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
try:
    from vllm._ipex_ops import ipex_ops as ops
    fp8_gemm_w8a16 = torch.ops.torch_ipex.fp8_gemm_w8a16
except:
    from vllm import _custom_ops as ops
    import vllm._xpu_ops
    fp8_gemm_w8a16 = torch.ops._xpu_C.fp8_gemm_w8a16

#os.environ["SYCL_PROGRAM_COMPILE_OPTIONS"] = "-vc-codegen -vc-disable-indvars-opt -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' -doubleGRF"
timing_enabled = int(os.environ.get("XETLA_TIMINGS", "0")) > 0
quantize_lm_heads = int(os.environ.get("XETLA_QUANTIZE_LM_HEADS", "1")) > 0
# The int2 x fp16 DPAS prefill kernel converts activations to int8 (XMX), which
# costs ~1.3% relative error per GEMM. That compounds across layers and is
# enough to destroy a model outright: the CAT-Q Qwen3 8B and 32B emit a single
# repeated token with it on and are coherent with it off, while Bonsai happens
# to survive it. It buys almost nothing anyway (8B TTFT 75 ms with, 80 ms
# without), so it is off unless XETLA_DISABLE_DPAS=0 asks for it.
disable_dpas = int(os.environ.get("XETLA_DISABLE_DPAS", "1")) > 0
# MoE: largest tokens*top_k the batched expert GEMV handles before falling back
# to gathering rows per expert. The batched path re-reads an expert's weights
# once per assignment, so it wins only while that stays under the expert count.
moe_expand_max = int(os.environ.get("XETLA_MOE_EXPAND_MAX", "128"))


def _stream_capturing() -> bool:
    """True while an XPU graph is being captured, when the runtime reports it."""
    try:
        return torch.xpu.is_current_stream_capturing()
    except Exception:
        return False


# ---- Pre-quantized sidecar (Option B) ---------------------------------------
# When XETLA_PREQUANT_PATH points at a .safetensors file produced by
# scripts/prequantize_gguf.py, process_weights_after_loading() will load each
# layer's (qweight, scale) directly from disk and skip the GGUF dequant +
# CPU re-quant pipeline. When XETLA_PREQUANT_DUMP_PATH is set, the plugin
# captures every quantized layer it processes and flushes them to that path
# right after model load.
_xetla_prequant_load_path = os.environ.get("XETLA_PREQUANT_PATH", "") or None
_xetla_prequant_dump_path = os.environ.get("XETLA_PREQUANT_DUMP_PATH", "") or None
_xetla_prequant_cache: dict = {}   # lazy-loaded {key -> torch.Tensor}
_xetla_prequant_meta: dict = {}    # parsed JSON from sidecar header
_xetla_prequant_dump_buf: dict = {}  # {prefix: {qweight, scale, kind, dpas}}


def _xetla_prequant_load_index() -> None:
    """Open the sidecar safetensors file lazily and populate the meta dict."""
    if not _xetla_prequant_load_path or _xetla_prequant_meta:
        return
    try:
        from safetensors import safe_open  # noqa: WPS433
        import json  # noqa: WPS433
        with safe_open(_xetla_prequant_load_path, framework="pt") as f:
            md = f.metadata() or {}
            _xetla_prequant_meta.update({
                "format_version": md.get("xetla_format_version", "1"),
                "method": md.get("xetla_method", ""),
                "keys": set(f.keys()),
                "extra": json.loads(md.get("xetla_meta", "{}") or "{}"),
            })
        print(f"[xetla] sidecar loaded: {_xetla_prequant_load_path} "
              f"({len(_xetla_prequant_meta['keys'])} tensors, "
              f"method={_xetla_prequant_meta['method']})", flush=True)
    except Exception as e:
        print(f"[xetla] WARN: could not open prequant sidecar "
              f"{_xetla_prequant_load_path}: {e}", flush=True)
        _xetla_prequant_meta["keys"] = set()


def _xetla_prequant_lookup(prefix: str, method: str = "") -> Optional[str]:
    """Resolve `prefix` against the sidecar index and return the matching
    sidecar key prefix, or None when the layer is not in the sidecar.

    When the model is loaded as a speculative draft, vLLM prefixes all layer
    names with 'draft_model.' (e.g. 'draft_model.model.layers.0.self_attn.qkv_proj').
    The sidecar was generated from the standalone model and therefore uses the
    unprefixed names.  We strip leading path components one at a time until we
    find a match, so both loading modes use the same sidecar file.
    """
    if not _xetla_prequant_load_path or not prefix:
        return None
    _xetla_prequant_load_index()
    if method and _xetla_prequant_meta.get("method") and \
            _xetla_prequant_meta["method"] != method:
        # Sidecar was produced for a different quant method.
        return None
    keys = _xetla_prequant_meta.get("keys", set())

    # Try the prefix as-is, then strip leading components until a match.
    candidates = [prefix]
    parts = prefix.split(".")
    for i in range(1, len(parts)):
        candidates.append(".".join(parts[i:]))

    for cand in candidates:
        if f"{cand}.qweight" in keys and f"{cand}.scale" in keys:
            return cand
    return None


def _xetla_target_device(layer: torch.nn.Module) -> torch.device:
    """Device the packed weights should land on.

    ``layer.weight`` may be a zero-storage *meta* placeholder (see
    ``XetlaLinearMethod.create_weights``), in which case we fall back to the
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


def _xetla_shard_moe_packed(packed: dict) -> dict:
    """Split stacked expert weights across tensor-parallel ranks.

    vLLM shards MoE experts on the intermediate dimension: every rank keeps all
    experts but only its slice of each one. w13 is [E, K/16, 2I] with gate and
    up concatenated, so both halves are sliced separately; w2 is [E, I/16, K]
    with the intermediate on the input side, so it slices rows.
    """
    try:
        from vllm.distributed import (get_tensor_model_parallel_rank,
                                      get_tensor_model_parallel_world_size)
        tp = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
    except Exception:
        return packed
    if tp <= 1:
        return packed

    q13, s13, q2, s2 = (packed["w13_q"], packed["w13_s"],
                        packed["w2_q"], packed["w2_s"])
    inter = q13.shape[2] // 2
    if inter % tp:
        raise ValueError(f"moe intermediate {inter} not divisible by tp={tp}")
    per = inter // tp
    # The scale groups are 128 wide along K, so w2's row split only lands on a
    # group boundary if the per-rank intermediate is a multiple of 128.
    if per % 128:
        raise ValueError(
            f"moe intermediate {inter} split {tp} ways gives {per} per rank, "
            f"which breaks the 128-element scale groups; use a smaller tp")

    lo, hi = rank * per, (rank + 1) * per
    out = {
        "w13_q": torch.cat([q13[:, :, lo:hi], q13[:, :, inter + lo:inter + hi]], dim=2),
        "w13_s": torch.cat([s13[:, :, lo:hi], s13[:, :, inter + lo:inter + hi]], dim=2),
        "w2_q": q2[:, lo // 16:hi // 16, :],
        "w2_s": s2[:, lo // 128:hi // 128, :],
    }
    return out


def _xetla_shard_packed(layer: torch.nn.Module, qw: torch.Tensor,
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


def _xetla_prequant_try_load(layer: torch.nn.Module, prefix: str,
                             method: str, kind: str) -> bool:
    """If a sidecar entry exists for `prefix`, populate the layer in-place
    and return True. `kind` is 'linear' or 'lm_head'.
    """
    lookup_prefix = _xetla_prequant_lookup(prefix, method)
    if lookup_prefix is None:
        return False
    try:
        from safetensors import safe_open  # noqa: WPS433
        dev = _xetla_target_device(layer)
        qkey = f"{lookup_prefix}.qweight"
        skey = f"{lookup_prefix}.scale"
        rkey = f"{lookup_prefix}.slice_ranks"
        ranks = None
        with safe_open(_xetla_prequant_load_path, framework="pt") as f:
            qw = f.get_tensor(qkey)
            sc = f.get_tensor(skey)
            if rkey in f.keys():
                ranks = f.get_tensor(rkey)
        if method == "bitcos_f16":
            # The BITCOS buffer is a flat three-plane bitstream whose sign
            # plane is data dependent, so it cannot be sliced like a regular
            # packed matrix.
            if (getattr(layer, "tp_size", 1) or 1) > 1:
                raise ValueError(
                    "bitcos_f16 sidecar cannot be sharded; repack per rank")
        else:
            qw, sc = _xetla_shard_packed(layer, qw, sc)
        layer.weight = torch.nn.Parameter(
            qw.to(dev).contiguous(), requires_grad=False)
        layer.scale = torch.nn.Parameter(
            sc.to(dev).contiguous(), requires_grad=False)
        layer.xetla_slice_ranks = (
            ranks.to(dev).contiguous() if ranks is not None and ranks.numel()
            else None)
        layer.xetla_quantized = True
        # Re-derive dispatch capability locally (no need to store).
        if method == "int2_f16":
            layer._xetla_dpas_capable = (not disable_dpas) and (qw.shape[1] & 255) == 0
        else:
            layer._xetla_dpas_capable = False
        _xetla_pre_convert_bias(layer)
        return True
    except Exception as e:
        print(f"[xetla] WARN: sidecar load failed for {prefix} (lookup={lookup_prefix}): {e}",
              flush=True)
        return False


def _xetla_prequant_dump_record(prefix: str, layer: torch.nn.Module,
                                method: str, kind: str) -> None:
    """Capture the just-quantized weights for later flush to disk."""
    if not _xetla_prequant_dump_path or not prefix:
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
        ranks = getattr(layer, "xetla_slice_ranks", None)
        if ranks is not None and ranks.numel():
            rec["slice_ranks"] = ranks.detach().to("cpu").contiguous()
        _xetla_prequant_dump_buf[prefix] = rec
    except Exception as e:
        print(f"[xetla] WARN: dump capture failed for {prefix}: {e}",
              flush=True)


def _xetla_prequant_flush_dump() -> None:
    """Write the accumulated buffer out as a single safetensors file."""
    if not _xetla_prequant_dump_path or not _xetla_prequant_dump_buf:
        return
    try:
        from safetensors.torch import save_file  # noqa: WPS433
        import json  # noqa: WPS433
        tensors: dict = {}
        layers_meta: dict = {}
        method_seen = ""
        for prefix, rec in _xetla_prequant_dump_buf.items():
            tensors[f"{prefix}.qweight"] = rec["qweight"]
            tensors[f"{prefix}.scale"] = rec["scale"]
            if "slice_ranks" in rec:
                tensors[f"{prefix}.slice_ranks"] = rec["slice_ranks"]
            layers_meta[prefix] = {
                "kind": rec["kind"],
                "qweight_shape": list(rec["qweight"].shape),
                "scale_shape": list(rec["scale"].shape),
            }
            method_seen = rec["method"]
        meta = {
            "xetla_format_version": "1",
            "xetla_method": method_seen,
            "xetla_meta": json.dumps({"layers": layers_meta}),
        }
        os.makedirs(os.path.dirname(_xetla_prequant_dump_path) or ".",
                    exist_ok=True)
        save_file(tensors, _xetla_prequant_dump_path, metadata=meta)
        n = len(_xetla_prequant_dump_buf)
        size_mb = os.path.getsize(_xetla_prequant_dump_path) / 1e6
        print(f"[xetla] sidecar written: {_xetla_prequant_dump_path} "
              f"({n} layers, {size_mb:.1f} MB, method={method_seen})",
              flush=True)
        _xetla_prequant_dump_buf.clear()
    except Exception as e:
        print(f"[xetla] ERROR: sidecar flush failed: {e}", flush=True)


def _xetla_is_compiling() -> bool:
    """Return True when running under torch.compile / Dynamo tracing.

    The custom_op wrappers are needed in that case so the FX graph stays
    closed; in eager and XPU-graph capture we can call the kernel directly
    and shave the dispatcher frame off every call.
    """
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:  # older torch
        return False


def _xetla_pre_convert_bias(layer: torch.nn.Module) -> None:
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

@torch.library.custom_op("xetla::int2_bf16_fused_gemm", mutates_args=())
def xetla_int2_bf16_fused_gemm(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Custom operator for fully connected layer with qint2 weight tensors.
    """
    # print(
    #     f"Custom xetla_int2_bf16_fused_gemm called with input shape: {input.shape}, weight shape: {weight.shape}"
    # )
    import xetla_pt_ext
    # out = xetla_pt_ext.int2_bf16_fused_gemm_run(input, weight, scale, bias)
    with Timer(input, weight):
        out = torch.ops.xetla_int2.int2_bf16_fused_gemm_run(input, weight, scale, bias, None)
    return out

@xetla_int2_bf16_fused_gemm.register_fake
def xetla_int2_bf16_fused_gemm_fake_impl(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    # print(
    #     f"Fake xetla_int2_bf16_fused_gemm called with input shape: {input.shape}, weight shape: {weight.shape}, is Fake: {isinstance(input, torch._subclasses.FakeTensor)}"
    # )
    out = input.new_empty([input.shape[0], weight.shape[1]])
    # print(
    #     f"Fake xetla_int2_bf16_fused_gemm returned with shape: {out.shape}, is Fake: {isinstance(out, torch._subclasses.FakeTensor)}"
    # )
    return out


# ---- int2 weights with per-K-group fp16 scales (gs=128), fp16 activations ----
INT2_F16_GROUP_SIZE = 128

@torch.library.custom_op("xetla::int2_fp16_upcvt_gemm", mutates_args=())
def xetla_int2_fp16_upcvt_gemm(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """int2 weight x fp16 act GEMM with per-K-group fp16 scales (gs=128).

    input  : fp16 [M, K]
    weight : int32 [K/16, N]   (16 K-rows packed per int32, codes {0,+1,-1})
    scale  : fp16  [K/128, N]
    bias   : optional fp16 [N]

    Routes to the DPAS / int8-XMX kernel for prefill (M > 1) when N is a
    multiple of 256 (the WGN tile of the DPAS variant); otherwise (and for
    decode M == 1) routes to the upcvt / GEMV-tuned kernel.
    """
    m = input.shape[0]
    n = weight.shape[1]
    # The eager path in XetlaLinearMethod.apply gates this on disable_dpas via
    # _xetla_dpas_capable; this one has to check it too, or XETLA_DISABLE_DPAS
    # silently does nothing whenever the model is compiled.
    use_dpas = (not disable_dpas) and (m > 1) and (n % 256 == 0)
    with Timer(input, weight):
        if use_dpas:
            out = torch.ops.xetla_int2.int2_fp16_dpas_gemm_run(
                input, weight, scale, None
            )
        else:
            out = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run(
                input, weight, scale, None
            )
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out

@xetla_int2_fp16_upcvt_gemm.register_fake
def _xetla_int2_fp16_upcvt_gemm_fake(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return input.new_empty([input.shape[0], weight.shape[1]])


# ---- int1 weights with per-K-group fp16 scales (gs=128), fp16 activations ----
INT1_F16_GROUP_SIZE = 128

@torch.library.custom_op("xetla::int1_fp16_upcvt_gemm", mutates_args=())
def xetla_int1_fp16_upcvt_gemm(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """int1 weight x fp16 act GEMM with per-K-group fp16 scales (gs=128).

    input  : fp16 [M, K]
    weight : int32 [K/32, N]   (32 K-rows packed per int32, codes {0->+1,1->-1})
    scale  : fp16  [K/128, N]
    bias   : optional fp16 [N]
    """
    with Timer(input, weight):
        out = torch.ops.xetla_int2.int1_fp16_upcvt_gemm_run(
            input, weight, scale, None
        )
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out

@xetla_int1_fp16_upcvt_gemm.register_fake
def _xetla_int1_fp16_upcvt_gemm_fake(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return input.new_empty([input.shape[0], weight.shape[1]])


def quantize_to_binary_f16(t, group_size: int = INT1_F16_GROUP_SIZE):
    """Quantize a [K, N] fp16/bfloat16 weight that is already a binary
    (-s, +s) tensor with shared scale s every `group_size` rows along K.

    Returns:
        codes : int8 [K, N] in {0, 1}  (0 -> +1, 1 -> -1, matching int1x32)
        scale : fp16 [K // group_size, N]
    """
    K, N = t.shape
    assert K % group_size == 0, f"K ({K}) must be multiple of {group_size}"
    tg = t.float().view(K // group_size, group_size, N)
    scale = tg.abs().amax(dim=1)  # [K/gs, N]
    # Sign-bit code: positive (incl. 0) -> 0 (=+1), negative -> 1 (=-1).
    codes = (t < 0).to(torch.int8)
    return codes, scale.to(torch.float16)


def pack_int1x32(codes):
    """Pack int8 codes in {0,1} of shape [K, N] into uint32 words of shape
    [K/32, N], where bit i of word at row r is the code for K-row r*32+i.
    """
    K, N = codes.shape
    assert K % 32 == 0, f"K ({K}) must be multiple of 32"
    c = codes.to(torch.int32).view(K // 32, 32, N)
    shifts = torch.arange(32, dtype=torch.int32, device=codes.device).view(1, 32, 1)
    packed = ((c & 1) << shifts).sum(dim=1).to(torch.int32)
    return packed


# ---- BITCOS ternary weights: presence bitmap + compacted sign stream -------
#
# int2 spends 2 bits on every weight and int1 spends 1 bit but cannot express a
# zero. BITCOS stores a presence bit per weight plus a sign bit per *non-zero*,
# so a ternary matrix that is a fraction z zeros costs 2 - z bits per weight.
BITCOS_F16_GROUP_SIZE = 128
# Local-K slices the decode kernel is allowed to use. The prefix ranks cost
# (slices-1) words per output column, which is negligible, and they let a slice
# that does not start at k=0 find where its columns enter the sign stream.
# The kernel infers the slice count from the rank table it is handed, so the
# packer has to commit to it: a long reduction is worth splitting further.
BITCOS_LOCAL_SLICES = 4
BITCOS_LOCAL_SLICES_LONG_K = 8
BITCOS_LONG_K = 8192


def bitcos_slices_for(K: int, N: int) -> int:
    """Slice count the decode kernel wants for this shape.

    Measured on B70 with a >=2 GB rotating footprint at the checkpoint's own
    density; a single cache-resident buffer ranks these the other way round.
    """
    if K >= BITCOS_LONG_K:          # long reduction: more slices to fill it
        return BITCOS_LOCAL_SLICES_LONG_K
    if N <= 4096 or (16384 <= N < 65536):
        return BITCOS_LOCAL_SLICES_LONG_K
    return BITCOS_LOCAL_SLICES


def _to_int32_wrap(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret uint32-valued int64 data as signed int32 (torch has no
    uint32), so the bit patterns survive the trip to the kernel."""
    return torch.where(t >= 2 ** 31, t - 2 ** 32, t).to(torch.int32)


def pack_bitcos(codes, slices: Optional[int] = None,
                col_chunk: int = 2048):
    """Pack int8 ternary codes [K, N] in {-1, 0, +1} into the BITCOS buffer.

    One flat uint32 buffer holds three planes back to back:

        [0,          K*N/32     )  bitmap  : bit c of word kp*N+n marks
                                             k = kp*32 + c present in column n
        [K*N/32,     K*N/32 + N )  offsets : first sign word of each column
        [K*N/32 + N, ...        )  signs   : one bit per non-zero, column
                                             major in increasing k, 1 -> -1

    plus one pad word, because the kernel's sign window always gathers a high
    word past the end of the last run.

    Returns (buf int32 [total + 1], slice_ranks int32 [slices-1, N]).
    """
    K, N = codes.shape
    assert K % 32 == 0, f"K ({K}) must be multiple of 32"
    dev = codes.device
    if slices is None:
        slices = bitcos_slices_for(K, N)

    present = codes != 0
    nnz = present.sum(dim=0, dtype=torch.int64)              # [N]
    words_per_col = (nnz + 31) // 32
    offsets = torch.zeros(N, dtype=torch.int64, device=dev)
    if N > 1:
        offsets[1:] = torch.cumsum(words_per_col, 0)[:-1]
    sign_words = int(words_per_col.sum().item())

    bitmap_words = (K // 32) * N
    buf = torch.zeros(bitmap_words + N + sign_words + 1,
                      dtype=torch.int64, device=dev)
    buf[bitmap_words:bitmap_words + N] = offsets

    bitmap_view = buf[:bitmap_words].view(K // 32, N)
    sign_base = bitmap_words + N
    shifts = torch.arange(32, dtype=torch.int64, device=dev).view(1, 32, 1)

    # Chunk over columns: the intermediate rank/bitmap tensors are [K, chunk],
    # which for a fused gate_up would otherwise be several hundred MB.
    for c0 in range(0, N, col_chunk):
        c1 = min(c0 + col_chunk, N)
        p = present[:, c0:c1]
        bitmap_view[:, c0:c1] = (
            (p.view(K // 32, 32, c1 - c0).to(torch.int64) << shifts).sum(dim=1))

        neg = codes[:, c0:c1] < 0
        if not bool(neg.any()):
            continue
        # Rank of each non-zero within its column, counted from k = 0.
        pi = p.to(torch.int32)
        rank = torch.cumsum(pi, dim=0) - pi
        # Each column's run starts word aligned, so the global bit index is
        # just the column's word offset in bits plus the within-column rank.
        gbit = offsets[c0:c1].view(1, -1) * 32 + rank.to(torch.int64)
        sel = gbit[neg]
        # Every non-zero owns a distinct bit, so add == or here.
        buf.scatter_add_(0, sign_base + (sel >> 5),
                         torch.ones_like(sel) << (sel & 31))

    if slices > 1:
        slice_k = (K + slices - 1) // slices
        rows = [present[:min(s * slice_k, K)].sum(dim=0, dtype=torch.int64)
                for s in range(1, slices)]
        slice_ranks = _to_int32_wrap(torch.stack(rows, 0))
    else:
        slice_ranks = torch.zeros((0, N), dtype=torch.int32, device=dev)

    return _to_int32_wrap(buf), slice_ranks


@torch.library.custom_op("xetla::bitcos_fp16_upcvt_gemm", mutates_args=())
def xetla_bitcos_fp16_upcvt_gemm(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    slice_ranks: Optional[torch.Tensor], bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """BITCOS ternary weight x fp16 act GEMM, per-K-group fp16 scales (gs=128).

    input       : fp16  [M, K]
    weight      : int32 [bitmap | offsets | signs | pad]  (flat)
    scale       : fp16  [K/128, N]
    slice_ranks : int32 [slices-1, N] or None
    bias        : optional fp16 [N]
    """
    with Timer(input, weight):
        out = torch.ops.xetla_int2.bitcos_fp16_upcvt_gemm_run(
            input, weight, scale, slice_ranks, None
        )
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


@xetla_bitcos_fp16_upcvt_gemm.register_fake
def _xetla_bitcos_fp16_upcvt_gemm_fake(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    slice_ranks: Optional[torch.Tensor], bias: Optional[torch.Tensor],
) -> torch.Tensor:
    # weight is a flat multi-plane buffer, so N comes from the scale plane.
    return input.new_empty([input.shape[0], scale.shape[1]])


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


def xetla_quant_method():
    quant_method = os.environ.get("XETLA_QUANT_METHOD", "").lower()
    # Auto-derive from the sidecar metadata if the user didn't pin a method
    # but XETLA_PREQUANT_PATH is set. Avoids the silent
    # "RuntimeError: A must be bf16" when the wrong default is used.
    if not quant_method and _xetla_prequant_load_path:
        try:
            _xetla_prequant_load_index()
            sc_method = _xetla_prequant_meta.get("method", "")
            if sc_method:
                print(f"[xetla] inferring XETLA_QUANT_METHOD={sc_method} "
                      f"from sidecar metadata", flush=True)
                quant_method = sc_method
        except Exception:
            pass
    if not quant_method:
        quant_method = "int2"
    if quant_method not in ["bf16", "int2", "int2_f16", "int1_f16", "bitcos_f16"]:
        raise ValueError(f"Unsupported xetla quantization method: {quant_method}")
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

def quantize_to_int2(t):
    scale = t.abs().amax(dim=0, keepdim=True).to(torch.float32)
    t1 = t / scale
    t1 = torch.clamp(t1, -2, 1)
    t1 = t1.to(torch.int8)
    # t1 = torch.randint(-1, 2, t1.shape, device=t1.device, dtype=torch.int8)
    # print(f"Qint2: {t1.shape} {t1.amax()} {t1.amin()} {scale.shape}")
    return t1, scale

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

def dequantize_int2_to_bf16(t, scale):
    t1 = (t.to(torch.float) * scale).to(torch.bfloat16)
    return t1

class XetlaConfig(QuantizationConfig):
    def __init__(self) -> None:
        self.method = xetla_quant_method()
        # Required by the GGUF model loader, which calls
        # `vllm_config.quant_config.unquantized_modules.extend(...)` even when
        # the active quant_config is not GGUFConfig (which happens whenever
        # we load a .gguf file but request `quantization=xetla` so the xetla
        # plugin re-quantizes weights to int2 with fp16 scales).
        self.unquantized_modules: list[str] = []
        super().__init__()

    def __repr__(self) -> str:
        return "XetlaConfig()"

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "xetla"

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
    def from_config(cls, config: dict[str, Any]) -> "XetlaConfig":
        return cls()

    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant, **kwargs) -> Optional[QuantizationMethods]:
        # Allow the user to opt into the xetla path even when the source
        # checkpoint advertises a different quantization method (e.g. gguf).
        # When the user explicitly asks for `xetla`, claim ownership so
        # `_verify_quantization` does not raise a mismatch error.
        # `**kwargs` absorbs extra arguments added by newer vLLM versions
        # (e.g. `hf_config=` since 0.21.0) -- without it the override is
        # silently skipped and the model loads dense.
        if user_quant == "xetla":
            return "xetla"
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["LinearMethodBase"]:
        if isinstance(layer, LinearBase):
            return XetlaLinearMethod(self, prefix=prefix)
        elif isinstance(layer, ParallelLMHead) and quantize_lm_heads:
            return XetlaEmbeddingMethod(self, True, prefix=prefix)
        elif isinstance(layer, VocabParallelEmbedding) and quantize_lm_heads:
            return XetlaEmbeddingMethod(self, False, prefix=prefix)
        try:
            from vllm.model_executor.layers.fused_moe import FusedMoE
            if isinstance(layer, FusedMoE):
                return XetlaFusedMoEMethod(self, layer.moe_config, prefix=prefix)
        except ImportError:
            pass
        # Other layer types (notably `Attention`) are not handled by xetla;
        # vLLM falls back to the default impl when we return None. Only print
        # once per type when XETLA_DEBUG=1 to avoid spamming one line per
        # transformer block at startup.
        if int(os.environ.get("XETLA_DEBUG", "0")) > 0:
            print(f"XetlaConfig.get_quant_method: passthrough for {type(layer).__name__} ({prefix})")
        return None

class XetlaEmbeddingMethod(UnquantizedEmbeddingMethod):
    """ Xetla quantized method for embeddings.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: XetlaConfig, inplace: bool = False,
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
        # Same trick as XetlaLinearMethod: when the sidecar already holds the
        # packed table, allocate the parameter on `meta` so the dense fp16
        # table (2.5 GB for a 248k x 5120 vocab) is never materialized. The
        # shape is preserved so VocabParallelEmbedding's sharded weight_loader
        # still validates, and its copies become no-ops.
        method = self.quant_config.method
        if (not self.inplace and method == "int2_f16"
                and _xetla_prequant_lookup(self.prefix, method) is not None):
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
            layer._xetla_meta_placeholder = True
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
        if not getattr(layer, "xetla_embed_packed", False):
            return super().embedding(layer, input_)
        flat = input_.reshape(-1)
        out = unpack_int2_rowwise(layer.weight.data[flat],
                                  layer.scale.data[flat])
        return out.view(*input_.shape, -1)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not current_platform.is_xpu():
            return

        method = self.quant_config.method
        # Sidecar load short-circuit (Option B).
        if (self.inplace and method in ("int2_f16", "int1_f16", "bitcos_f16") and
                _xetla_prequant_try_load(layer, self.prefix, method, "lm_head")):
            print(f"[xetla] sidecar hit: {self.prefix} lm_head ({method})")
            return

        # Input embedding: the table is ternary in Bonsai checkpoints, but it
        # is looked up rather than multiplied, so it uses the row-major packed
        # layout and is unpacked per token in embedding(). The lookup is
        # independent of the GEMM format, so bitcos reuses the same table.
        if not self.inplace and method in ("int2_f16", "bitcos_f16"):
            lookup = _xetla_prequant_lookup(self.prefix, method)
            if lookup is not None:
                try:
                    from safetensors import safe_open  # noqa: WPS433
                    dev = _xetla_target_device(layer)
                    with safe_open(_xetla_prequant_load_path,
                                   framework="pt") as f:
                        qw = f.get_tensor(f"{lookup}.qweight")
                        sc = f.get_tensor(f"{lookup}.scale")
                    layer.weight = torch.nn.Parameter(
                        qw.to(dev).contiguous(), requires_grad=False)
                    layer.scale = torch.nn.Parameter(
                        sc.to(dev).contiguous(), requires_grad=False)
                    layer.xetla_embed_packed = True
                    layer.xetla_quantized = True
                    print(f"[xetla] sidecar hit: {self.prefix} embedding "
                          f"({method}), packed {tuple(qw.shape)}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[xetla] WARN: packed embedding load failed for "
                          f"{self.prefix}: {e}", flush=True)
                return
            if getattr(layer, "_xetla_meta_placeholder", False):
                raise RuntimeError(
                    f"[xetla] sidecar entry for {self.prefix} vanished "
                    "between create_weights() and "
                    "process_weights_after_loading()")

        if self.quant_config.method == "int2":
            print(f"Processing weights for layer {layer} with method fp8 (inplace={self.inplace})")
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)
            # Update the layer with the new values.
            if self.inplace:
                layer.weight = torch.nn.Parameter(qweight, requires_grad=False)
            else:
                layer.qweight = torch.nn.Parameter(qweight, requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None
            layer.xetla_quantized = True
        elif self.quant_config.method == "int2_f16":
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
            layer.xetla_quantized = True
            # B7: lm_head N is the (padded) vocab size; check DPAS capability.
            layer._xetla_dpas_capable = (not disable_dpas) and (packed.shape[1] & 255) == 0
            _xetla_pre_convert_bias(layer)
            _xetla_prequant_dump_record(self.prefix, layer, "int2_f16", "lm_head")
        elif self.quant_config.method == "int1_f16":
            if not self.inplace:
                return
            weight = layer.weight.data  # [vocab_size, hidden_size]
            dev = weight.device
            wkn = weight.detach().to("cpu", dtype=torch.float16).t().contiguous()
            codes, scale_f16 = quantize_to_binary_f16(wkn, INT1_F16_GROUP_SIZE)
            packed = pack_int1x32(codes)
            print(f"Processing lm_head with method int1_f16: weight {tuple(weight.shape)} -> packed {tuple(packed.shape)}, scale {tuple(scale_f16.shape)}")
            layer.weight = torch.nn.Parameter(
                packed.to(dev).contiguous(), requires_grad=False
            )
            layer.scale = torch.nn.Parameter(
                scale_f16.to(dev).contiguous(), requires_grad=False
            )
            layer.xetla_quantized = True
            layer._xetla_dpas_capable = False
            _xetla_pre_convert_bias(layer)
            _xetla_prequant_dump_record(self.prefix, layer, "int1_f16", "lm_head")
        elif self.quant_config.method == "bitcos_f16":
            if not self.inplace:
                return
            weight = layer.weight.data  # [vocab_size, hidden_size]
            dev = weight.device
            wkn = weight.detach().to("cpu", dtype=torch.float16).t().contiguous()
            codes, scale_f16 = quantize_to_ternary_f16(wkn, BITCOS_F16_GROUP_SIZE)
            packed, slice_ranks = pack_bitcos(codes)
            print(f"Processing lm_head with method bitcos_f16: weight {tuple(weight.shape)} -> packed {tuple(packed.shape)}, scale {tuple(scale_f16.shape)}")
            layer.weight = torch.nn.Parameter(
                packed.to(dev).contiguous(), requires_grad=False
            )
            layer.scale = torch.nn.Parameter(
                scale_f16.to(dev).contiguous(), requires_grad=False
            )
            layer.xetla_slice_ranks = slice_ranks.to(dev).contiguous()
            layer.xetla_quantized = True
            layer._xetla_dpas_capable = False
            _xetla_pre_convert_bias(layer)
            _xetla_prequant_dump_record(self.prefix, layer, "bitcos_f16", "lm_head")
        else:
            pass

    def apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.quant_config.method == "int2" and getattr(layer, "xetla_quantized", False):
            weight = layer.weight.data if self.inplace else layer.qweight.data
            weight_scale = layer.weight_scale.data
            output = fp8_gemm_w8a16(x, weight.t(), weight_scale, bias)
            # print(f"XetlaEmbeddingMethod apply output shape: {output.shape}, dtype: {output.dtype}")
            return output
        if self.quant_config.method == "int2_f16" and getattr(layer, "xetla_quantized", False):
            x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
            b16 = bias if bias is None or bias.dtype == torch.float16 else bias.to(torch.float16)
            if not _xetla_is_compiling():
                if x16.shape[0] > 1 and getattr(layer, "_xetla_dpas_capable", False):
                    out = torch.ops.xetla_int2.int2_fp16_dpas_gemm_run(
                        x16, layer.weight, layer.scale, None)
                else:
                    out = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run(
                        x16, layer.weight, layer.scale, None)
                if b16 is not None:
                    out = out + b16
            else:
                out = xetla_int2_fp16_upcvt_gemm(x16, layer.weight, layer.scale, b16)
            return out if out.dtype == x.dtype else out.to(x.dtype)
        if self.quant_config.method == "int1_f16" and getattr(layer, "xetla_quantized", False):
            x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
            b16 = bias if bias is None or bias.dtype == torch.float16 else bias.to(torch.float16)
            if not _xetla_is_compiling():
                out = torch.ops.xetla_int2.int1_fp16_upcvt_gemm_run(
                    x16, layer.weight, layer.scale, None)
                if b16 is not None:
                    out = out + b16
            else:
                out = xetla_int1_fp16_upcvt_gemm(x16, layer.weight, layer.scale, b16)
            return out if out.dtype == x.dtype else out.to(x.dtype)
        if self.quant_config.method == "bitcos_f16" and getattr(layer, "xetla_quantized", False):
            x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
            b16 = bias if bias is None or bias.dtype == torch.float16 else bias.to(torch.float16)
            ranks = getattr(layer, "xetla_slice_ranks", None)
            if not _xetla_is_compiling():
                out = torch.ops.xetla_int2.bitcos_fp16_upcvt_gemm_run(
                    x16, layer.weight, layer.scale, ranks, None)
                if b16 is not None:
                    out = out + b16
            else:
                out = xetla_bitcos_fp16_upcvt_gemm(
                    x16, layer.weight, layer.scale, ranks, b16)
            return out if out.dtype == x.dtype else out.to(x.dtype)
        return super().apply(layer, x, bias)


def _fused_moe_method_base():
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    return FusedMoEMethodBase


class XetlaFusedMoEMethod(_fused_moe_method_base()):
    """int2 x fp16 experts for vLLM's FusedMoE layer.

    vLLM stacks the experts into w13_weight [E, 2I, H] and w2_weight [E, H, I];
    the sidecar mirrors that with <prefix>.w13 / <prefix>.w2 packed the same way
    a Linear is, one slice per expert. Only sidecar-backed layers are handled --
    without packed weights we fall back to the dense implementation, because
    ternarizing a checkpoint that is not already ternary destroys it.
    """

    def __init__(self, quant_config, moe, prefix: str = ""):
        super().__init__(moe)
        self.quant_config = quant_config
        self.prefix = prefix
        self.packed = None
        self._fallback = None

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    # -- vLLM plumbing ----------------------------------------------------
    def _dense(self):
        if self._fallback is None:
            from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (  # noqa: E501
                UnquantizedFusedMoEMethod,
            )
            self._fallback = UnquantizedFusedMoEMethod(self.moe)
        return self._fallback

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        method = self.quant_config.method
        w13 = _xetla_prequant_lookup(f"{self.prefix}.w13", method)
        w2 = _xetla_prequant_lookup(f"{self.prefix}.w2", method)
        if method != "int2_f16" or w13 is None or w2 is None:
            return self._dense().create_weights(
                layer, num_experts, hidden_size,
                intermediate_size_per_partition, params_dtype,
                **extra_weight_attrs)

        from vllm.model_executor.utils import set_weight_attrs
        weight_loader = extra_weight_attrs.pop("weight_loader")
        # Meta placeholders keep the dense experts (54 GiB for a 30B MoE) from
        # ever being allocated; the loader's copies become no-ops.
        for name, shape in (
            ("w13_weight", (num_experts, 2 * intermediate_size_per_partition,
                            hidden_size)),
            ("w2_weight", (num_experts, hidden_size,
                           intermediate_size_per_partition)),
        ):
            param = torch.nn.Parameter(
                torch.empty(*shape, dtype=params_dtype, device="meta"),
                requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, {"weight_loader": weight_loader,
                                     **extra_weight_attrs})
        layer._xetla_moe_keys = (w13, w2)
        layer._xetla_meta_placeholder = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        keys = getattr(layer, "_xetla_moe_keys", None)
        if keys is None:
            dense = self._dense()
            if hasattr(dense, "process_weights_after_loading"):
                dense.process_weights_after_loading(layer)
            return
        from safetensors import safe_open

        dev = torch.device(f"xpu:{torch.xpu.current_device()}") \
            if hasattr(torch, "xpu") and torch.xpu.is_available() \
            else torch.device("cpu")
        packed = {}
        with safe_open(_xetla_prequant_load_path, framework="pt") as f:
            for slot, key in zip(("w13", "w2"), keys):
                packed[f"{slot}_q"] = f.get_tensor(f"{key}.qweight")
                packed[f"{slot}_s"] = f.get_tensor(f"{key}.scale")
        packed = _xetla_shard_moe_packed(packed)
        packed = {k: v.to(dev).contiguous() for k, v in packed.items()}
        layer.w13_weight = None
        layer.w2_weight = None
        self.packed = packed
        layer._xetla_moe_packed = packed
        layer.xetla_quantized = True
        if int(os.environ.get("XETLA_DEBUG", "0")) > 0:
            print(f"[xetla] moe {self.prefix}: w13{tuple(packed['w13_q'].shape)} "
                  f"w2{tuple(packed['w2_q'].shape)}", flush=True)

    # -- execution --------------------------------------------------------
    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              topk_weights: torch.Tensor, topk_ids: torch.Tensor,
              shared_experts_input: Optional[torch.Tensor] = None,
              **kwargs) -> torch.Tensor:
        packed = getattr(layer, "_xetla_moe_packed", None)
        if packed is None:
            return self._dense().apply(layer, x, topk_weights, topk_ids,
                                       shared_experts_input, **kwargs)

        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        gemm = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run
        w13_q, w13_s = packed["w13_q"], packed["w13_s"]
        w2_q, w2_s = packed["w2_q"], packed["w2_s"]
        tokens, top_k = x.shape[0], topk_ids.shape[1]

        # The batched kernel takes the expert ids on device, so this whole path
        # is two GEMV launches with no host sync - which is both why decode is
        # fast (the per-expert loop was launch-bound at ~145 GiB/s) and why it
        # can be captured into an XPU graph.
        if x.dtype == torch.float16 and tokens * top_k <= moe_expand_max:
            moe_gemv = torch.ops.xetla_int2.int2_fp16_moe_gemv_run
            sel = topk_ids.reshape(-1).to(torch.int32).contiguous()
            # One row per (token, expert) pair, except for decode where the
            # kernel broadcasts the single row across experts instead.
            a = x.contiguous() if tokens == 1 \
                else x.repeat_interleave(top_k, dim=0).contiguous()
            h = moe_gemv(a, w13_q, w13_s, sel)
            inter = h.shape[1] // 2
            act = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
            y = moe_gemv(act.contiguous(), w2_q, w2_s, sel)
            y = y * topk_weights.reshape(-1, 1).to(y.dtype)
            return y.view(tokens, top_k, -1).sum(1).reshape(orig_shape)

        if _stream_capturing():
            raise RuntimeError(
                f"XPU graph capture at batch size {tokens} exceeds the batched "
                f"MoE path (tokens*top_k {tokens * top_k} > "
                f"XETLA_MOE_EXPAND_MAX {moe_expand_max}); the gathered path "
                f"below syncs to host and cannot be captured. Cap capture to "
                f"{max(1, moe_expand_max // top_k)} tokens, e.g. "
                f"compilation_config={{'cudagraph_capture_sizes': [1, 2, 4, 8, "
                f"{max(1, moe_expand_max // top_k)}]}}.")

        # Prefill: too many rows to re-read weights per assignment, so gather
        # per expert instead. One host transfer per layer covers all of them;
        # doing the bookkeeping on device costs a sync per expert. This path
        # cannot be captured into a graph.
        out = torch.zeros_like(x)
        ids = topk_ids.cpu()
        weights = topk_weights.to(x.dtype)
        buckets: dict[int, list[tuple[int, int]]] = {}
        for row, experts in enumerate(ids.tolist()):
            for slot, expert in enumerate(experts):
                buckets.setdefault(expert, []).append((row, slot))

        for expert, entries in buckets.items():
            rows = torch.tensor([r for r, _ in entries], device=x.device)
            slots = torch.tensor([s for _, s in entries], device=x.device)
            xe = x.index_select(0, rows).contiguous()
            h = gemm(xe, w13_q[expert], w13_s[expert], None)
            gate, up = h.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            y = gemm(act.contiguous(), w2_q[expert], w2_s[expert], None)
            out.index_add_(0, rows, y * weights[rows, slots].unsqueeze(-1))
        return out.reshape(orig_shape)


class XetlaLinearMethod(LinearMethodBase):
    """Linear method for xetla.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: XetlaConfig, prefix: str = ""):
        self.quant_config = quant_config
        self.prefix = prefix
        super().__init__()

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
        if method in ("int2_f16", "int1_f16", "bitcos_f16") and \
                _xetla_prequant_lookup(self.prefix, method) is not None:
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
            layer._xetla_meta_placeholder = True
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
        if method in ("int2_f16", "int1_f16", "bitcos_f16") and \
                _xetla_prequant_try_load(layer, self.prefix, method, "linear"):
            print(f"[xetla] sidecar hit: {self.prefix} ({method})")
            return

        if getattr(layer, "_xetla_meta_placeholder", False):
            # Should never happen: the placeholder is only created when the
            # sidecar has an entry for this prefix.
            raise RuntimeError(
                f"[xetla] sidecar entry for {self.prefix} vanished between "
                "create_weights() and process_weights_after_loading()")

        if _xetla_prequant_load_path and method in ("int2_f16", "int1_f16",
                                                    "bitcos_f16"):
            # A sidecar is in use but this layer is not in it. That means the
            # offline packer decided the layer is not ternary/binary (e.g. the
            # vision tower, or a gate projection kept in fp16). Re-quantizing
            # it here would silently destroy accuracy, so keep it dense.
            if int(os.environ.get("XETLA_DEBUG", "0")) > 0:
                print(f"[xetla] keeping {self.prefix} dense (not in sidecar)",
                      flush=True)
            return

        print(f"[xetla] quantizing {self.prefix} "
              f"[{tuple(layer.weight.shape)}] with method {method}")
        if method == "int2":
            weight = layer.weight.data
            weight_int2, layer.scale = quantize_to_int2(weight.t())
            layer.weight.data = pack_int2_vnni16(weight_int2)
        elif method == "int2_f16":
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
            layer._xetla_dpas_capable = (not disable_dpas) and (packed.shape[1] & 255) == 0
            layer.xetla_quantized = True
            _xetla_pre_convert_bias(layer)
            _xetla_prequant_dump_record(self.prefix, layer, method, "linear")
        elif method == "int1_f16":
            # Bonsai-8B-unpacked-style binary fp16 weight: every 128 K-entries
            # share an fp16 scale and values are s*{-1, +1}. Re-quantize with
            # the sign-bit and per-128-K absmax to feed the int1 upcvt GEMM.
            weight = layer.weight.data  # [N_out, K_in], any float dtype
            dev = weight.device
            # Quantize on CPU to avoid a peak XPU allocation of K*N*4B.
            wkn = weight.detach().to("cpu", dtype=torch.float16).t().contiguous()
            codes, scale_f16 = quantize_to_binary_f16(wkn, INT1_F16_GROUP_SIZE)
            packed = pack_int1x32(codes)
            layer.weight.data = packed.to(dev).contiguous()
            layer.scale = torch.nn.Parameter(
                scale_f16.to(dev).contiguous(), requires_grad=False
            )
            layer._xetla_dpas_capable = False
            layer.xetla_quantized = True
            _xetla_pre_convert_bias(layer)
            _xetla_prequant_dump_record(self.prefix, layer, method, "linear")
        elif method == "bitcos_f16":
            # Ternary fp16 weight stored as presence bitmap + compacted signs,
            # so zeros cost one bit and carry no sign at all.
            weight = layer.weight.data  # [N_out, K_in], any float dtype
            dev = weight.device
            # Pack on CPU: the intermediate rank tensors are K*N int32.
            wkn = weight.detach().to("cpu", dtype=torch.float16).t().contiguous()
            codes, scale_f16 = quantize_to_ternary_f16(wkn, BITCOS_F16_GROUP_SIZE)
            packed, slice_ranks = pack_bitcos(codes)
            layer.weight.data = packed.to(dev).contiguous()
            layer.scale = torch.nn.Parameter(
                scale_f16.to(dev).contiguous(), requires_grad=False
            )
            layer.xetla_slice_ranks = slice_ranks.to(dev).contiguous()
            # BITCOS has no DPAS variant; the unpack is the whole kernel.
            layer._xetla_dpas_capable = False
            layer.xetla_quantized = True
            _xetla_pre_convert_bias(layer)
            _xetla_prequant_dump_record(self.prefix, layer, method, "linear")
        else:
            pass

    def apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        method = self.quant_config.method
        if method == "int2":
            return xetla_int2_bf16_fused_gemm(x, layer.weight, layer.scale, bias)
        if method in ("int2_f16", "int1_f16", "bitcos_f16") and \
                not getattr(layer, "xetla_quantized", False):
            # Layer was deliberately left dense (mixed-precision checkpoint).
            return UnquantizedLinearMethod.apply(self, layer, x, bias)
        if method == "int2_f16":
            x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
            # B8: prefer the pre-converted layer.bias (always fp16 already).
            b16 = bias if bias is None or bias.dtype == torch.float16 else bias.to(torch.float16)
            # B7: under eager (no torch.compile tracing), skip the dispatch
            # custom_op and call the right kernel directly using the cached
            # capability flag. Saves ~5us per call from the dispatcher frame.
            if not _xetla_is_compiling() and getattr(layer, "_xetla_dpas_capable", False) is not None:
                if x16.shape[0] > 1 and layer._xetla_dpas_capable:
                    c = torch.ops.xetla_int2.int2_fp16_dpas_gemm_run(
                        x16, layer.weight, layer.scale, None)
                else:
                    c = torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run(
                        x16, layer.weight, layer.scale, None)
                if b16 is not None:
                    c = c + b16
            else:
                c = xetla_int2_fp16_upcvt_gemm(x16, layer.weight, layer.scale, b16)
            return c if c.dtype == x.dtype else c.to(x.dtype)
        if method == "int1_f16":
            x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
            b16 = bias if bias is None or bias.dtype == torch.float16 else bias.to(torch.float16)
            if not _xetla_is_compiling():
                c = torch.ops.xetla_int2.int1_fp16_upcvt_gemm_run(
                    x16, layer.weight, layer.scale, None)
                if b16 is not None:
                    c = c + b16
            else:
                c = xetla_int1_fp16_upcvt_gemm(x16, layer.weight, layer.scale, b16)
            return c if c.dtype == x.dtype else c.to(x.dtype)
        if method == "bitcos_f16":
            x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
            b16 = bias if bias is None or bias.dtype == torch.float16 else bias.to(torch.float16)
            ranks = getattr(layer, "xetla_slice_ranks", None)
            if not _xetla_is_compiling():
                c = torch.ops.xetla_int2.bitcos_fp16_upcvt_gemm_run(
                    x16, layer.weight, layer.scale, ranks, None)
                if b16 is not None:
                    c = c + b16
            else:
                c = xetla_bitcos_fp16_upcvt_gemm(
                    x16, layer.weight, layer.scale, ranks, b16)
            return c if c.dtype == x.dtype else c.to(x.dtype)
        return UnquantizedLinearMethod.apply(self, layer, x, bias)


# ---- inline xetla GEMM profile shim (XETLA_PROFILE=1) ----------------------
import atexit as _atexit
import collections as _collections
import signal as _signal
import threading as _threading

_xprof_lock = _threading.Lock()
_xprof_stats: dict = {}
_xprof_installed = False


def _xprof_nbytes(*tensors) -> int:
    # BITCOS is data dependent, so traffic is taken from the buffers themselves
    # rather than from a per-format formula.
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
        m = int(A.shape[0]); k = int(A.shape[1])
        # BITCOS packs every plane into one flat buffer, so N lives on scale.
        n = int(B.shape[1]) if B.dim() > 1 else int(scale_B.shape[1])
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


def _xprof_print():
    if not _xprof_stats:
        print("\n=== xetla profile: no GEMM calls recorded ===\n", flush=True)
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
    print("=== xetla GEMM profile (per-op aggregate) ===", flush=True)
    print("=" * 100, flush=True)
    print(f"{'op_name':<36} {'calls':>10} {'total_ms':>12} {'avg_us':>10} {'GB/s':>10} {'%time':>8}", flush=True)
    for op_name, a in sorted(per_op.items(), key=lambda kv: -kv[1]["time_s"]):
        avg_us = a["time_s"] / a["calls"] * 1e6
        gbps = a["bytes"] / a["time_s"] / GB if a["time_s"] > 0 else 0.0
        pct = 100 * a["time_s"] / grand_time if grand_time > 0 else 0.0
        print(f"{op_name:<36} {int(a['calls']):>10} {a['time_s']*1e3:>12.2f} {avg_us:>10.1f} {gbps:>10.1f} {pct:>7.2f}%", flush=True)
    print(f"{'TOTAL':<36} {grand_calls:>10} {grand_time*1e3:>12.2f}", flush=True)
    print("\n" + "=" * 100, flush=True)
    print("=== xetla GEMM profile (per-shape breakdown) ===", flush=True)
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


def _install_xetla_profile():
    global _xprof_installed
    if _xprof_installed:
        return
    if int(os.environ.get("XETLA_PROFILE", "0")) <= 0:
        return
    ns = torch.ops.xetla_int2
    candidates = [
        "int2_fp16_upcvt_gemm_run",
        "int2_fp16_dpas_gemm_run",
        "int1_fp16_upcvt_gemm_run",
        "bitcos_fp16_upcvt_gemm_run",
        "int2_bf16_fused_gemm_run",
    ]
    installed = []
    for name in candidates:
        op = getattr(ns, name, None)
        if op is None:
            continue
        setattr(ns, name, _xprof_wrap(op, name))
        installed.append(name)
    if installed:
        print(f"[xetla profile] wrapped: {', '.join(installed)}", flush=True)
        _atexit.register(_xprof_print)
        try:
            _signal.signal(_signal.SIGTERM, lambda *_: (_xprof_print(), os._exit(0)))
        except Exception:
            pass
        _xprof_installed = True
    else:
        print("[xetla profile] no ops found to wrap", flush=True)


def _maybe_disable_triton_stride_versioning() -> None:
    """Opt-in workaround for a triton-xpu compiler crash on hybrid models.

    triton-xpu 3.7.0 segfaults inside its Intel-specific
    ``TritonIntelStrideVersioning`` TTIR pass while compiling the FLA chunked
    gated-delta-rule kernel (``fla/ops/chunk_delta_h.py``) that Qwen3.5-style
    models -- e.g. Bonsai-27B -- use for linear-attention prefill.  Every
    autotune config fails, so prefill dies with ``PassManager::run failed``.

    vLLM >= 0.20.2 routes GDN through its XPU path and no longer hits that
    kernel, so this is off by default.  Set
    ``XETLA_TRITON_DISABLE_STRIDE_VERSIONING=1`` to no-op the pass (it is a
    pure optimization) when running on an older vLLM.
    """
    if os.environ.get("XETLA_TRITON_DISABLE_STRIDE_VERSIONING", "0") != "1":
        return
    try:
        from triton._C.libtriton import intel  # noqa: WPS433
    except Exception:
        return
    try:
        if hasattr(intel.passes.ttir, "add_stride_versioning"):
            intel.passes.ttir.add_stride_versioning = lambda pm: None
            print("[xetla] disabled triton TritonIntelStrideVersioning pass "
                  "(crashes on FLA gated-delta-rule kernels)", flush=True)
    except Exception as e:
        print(f"[xetla] WARN: could not disable stride versioning: {e}",
              flush=True)


def _xetla_quantize_lm_head(model: torch.nn.Module) -> None:
    """Quantize embedding tables that vLLM built without a quant_config.

    Several models -- Qwen3.5 / Qwen3-Next (Bonsai-27B) among them -- construct
    ``ParallelLMHead`` and ``VocabParallelEmbedding`` without passing
    ``quant_config`` (the input embedding is built without a ``prefix`` too),
    so ``XetlaConfig.get_quant_method()`` is never consulted for them and both
    stay dense fp16.  For Bonsai both tables are ternary in the checkpoint
    (whitepaper sec. 4.3) and together they are ~5 GB, so wire them to the
    xetla path here, after the weights have been loaded.

    The LM head is the larger win for speed (it is read in full for every
    decoded token); the input embedding is a pure memory win, since only the
    looked-up rows are ever touched.

    Set ``XETLA_QUANTIZE_LM_HEADS=0`` to keep both dense.
    """
    if not quantize_lm_heads or not current_platform.is_xpu():
        return
    method = xetla_quant_method()
    if method not in ("int2_f16", "int1_f16", "bitcos_f16"):
        return
    try:
        config = XetlaConfig()
    except Exception:
        return

    for name, module in model.named_modules():
        if not isinstance(module, VocabParallelEmbedding):
            continue
        is_lm_head = isinstance(module, ParallelLMHead)
        if isinstance(getattr(module, "quant_method", None), XetlaEmbeddingMethod):
            continue  # already handled through get_quant_method()
        if getattr(module, "xetla_quantized", False):
            continue
        if _xetla_prequant_load_path and \
                _xetla_prequant_lookup(name, method) is None:
            # Sidecar in use but it has no packed table: leave it dense rather
            # than silently re-quantizing something that may not be ternary.
            print(f"[xetla] {name}: not in sidecar, kept dense", flush=True)
            continue
        if not is_lm_head and not _xetla_prequant_load_path:
            # The input embedding is only packed from a sidecar; there is no
            # on-the-fly path for it.
            continue
        try:
            qm = XetlaEmbeddingMethod(config, inplace=is_lm_head, prefix=name)
            qm.process_weights_after_loading(module)
            if getattr(module, "xetla_quantized", False):
                module.quant_method = qm
                kind = "lm_head" if is_lm_head else "embedding"
                print(f"[xetla] {kind} {name} quantized ({method}), "
                      f"packed {tuple(module.weight.shape)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[xetla] WARN: could not quantize {name}: {e}", flush=True)


def register():
    print("Hello xetla plugin!")
    _maybe_disable_triton_stride_versioning()
    # Force-load the SYCL extension so its TORCH_LIBRARY / TORCH_LIBRARY_FRAGMENT
    # blocks register `torch.ops.xetla_int2.*` in *every* process that loads
    # the plugin (main + each engine worker). Without this the ops are missing
    # in the spawned engine subprocess.
    try:
        import xetla_pt_ext  # noqa: F401
    except Exception as e:
        print(f"[xetla] WARNING: could not import xetla_pt_ext: {e}")

    # Optional GEMM profiling shim: set XETLA_PROFILE=1 to enable.
    try:
        _install_xetla_profile()
    except Exception as e:
        print(f"[xetla] WARNING: could not install xetla_profile: {e}")

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
                _xetla_quantize_lm_head(model)
            _xetla_prequant_flush_dump()

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
        if _xetla_prequant_dump_path:
            print(f"[xetla] sidecar dump enabled -> {_xetla_prequant_dump_path}")
    except Exception as e:
        print(f"[xetla] WARN: could not install post-load hook: {e}")

    register_quantization_config("xetla")(XetlaConfig)
    
