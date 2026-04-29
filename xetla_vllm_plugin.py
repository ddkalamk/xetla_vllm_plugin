

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
    use_dpas = (m > 1) and (n % 256 == 0)
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


def xetla_quant_method():
    quant_method =  os.environ.get("XETLA_QUANT_METHOD", "int2").lower()
    if quant_method not in ["bf16", "int2", "int2_f16"]:
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
            cls, hf_quant_cfg, user_quant) -> Optional[QuantizationMethods]:
        # Allow the user to opt into the xetla path even when the source
        # checkpoint advertises a different quantization method (e.g. gguf).
        # When the user explicitly asks for `xetla`, claim ownership so
        # `_verify_quantization` does not raise a mismatch error.
        if user_quant == "xetla":
            return "xetla"
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["LinearMethodBase"]:
        if isinstance(layer, LinearBase):
            return XetlaLinearMethod(self)
        elif isinstance(layer, ParallelLMHead) and quantize_lm_heads:
            return XetlaEmbeddingMethod(self, True)
        elif isinstance(layer, VocabParallelEmbedding) and quantize_lm_heads:
            return XetlaEmbeddingMethod(self, False)
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

    def __init__(self, quant_config: XetlaConfig, inplace: bool = False):
        self.quant_config = quant_config
        self.inplace = inplace
        # self.quant_config.method = "int2"  # Embeddings use int2
        super().__init__()

    # def create_weights(self, layer: torch.nn.Module,
    #                    input_size_per_partition: int,
    #                    output_partition_sizes: list[int], input_size: int,
    #                    output_size: int, params_dtype: torch.dtype,
    #                    **extra_weight_attrs):
    #     # We just reuse UnquantizedLinearMethod to create weights
    #     UnquantizedEmbeddingMethod.create_weights(self, layer, input_size_per_partition,
    #                                            output_partition_sizes, input_size,
    #                                            output_size, params_dtype,
    #                                            **extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not current_platform.is_xpu():
            return

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
            out = xetla_int2_fp16_upcvt_gemm(
                x16, layer.weight, layer.scale,
                bias.to(torch.float16) if bias is not None else None,
            )
            return out.to(x.dtype)
        return super().apply(layer, x, bias)


class XetlaLinearMethod(LinearMethodBase):
    """Linear method for xetla.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: XetlaConfig):
        self.quant_config = quant_config
        super().__init__()

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        # We just reuse UnquantizedLinearMethod to create weights
        UnquantizedLinearMethod.create_weights(self, layer, input_size_per_partition,
                                               output_partition_sizes, input_size,
                                               output_size, params_dtype,
                                               **extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not current_platform.is_xpu():
            return
       
        print(f"Processing weights for layer {layer} with method {self.quant_config.method}")
        if self.quant_config.method == "int2":
            weight = layer.weight.data
            weight_int2, layer.scale = quantize_to_int2(weight.t())
            layer.weight.data = pack_int2_vnni16(weight_int2)
        elif self.quant_config.method == "int2_f16":
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
        else:
            pass

    def apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        # with Timer(x, layer.weight) as t:
        if True:
            if self.quant_config.method == "int2":
                # import xetla_pt_ext
                # print("Using xetla int2 gemm x: ", x.shape, " weight: ", layer.weight.shape)
                c = xetla_int2_bf16_fused_gemm(x, layer.weight, layer.scale, bias)
                return c
            elif self.quant_config.method == "int2_f16":
                # Activations must be fp16 for this kernel.
                x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
                c = xetla_int2_fp16_upcvt_gemm(
                    x16, layer.weight, layer.scale,
                    bias.to(torch.float16) if bias is not None else None,
                )
                return c.to(x.dtype)
            else:
                return UnquantizedLinearMethod.apply(self, layer, x, bias)


def register():
    print("Hello xetla plugin!")
    # Force-load the SYCL extension so its TORCH_LIBRARY / TORCH_LIBRARY_FRAGMENT
    # blocks register `torch.ops.xetla_int2.*` in *every* process that loads
    # the plugin (main + each engine worker). Without this the ops are missing
    # in the spawned engine subprocess.
    try:
        import xetla_pt_ext  # noqa: F401
    except Exception as e:
        print(f"[xetla] WARNING: could not import xetla_pt_ext: {e}")

    register_quantization_config("xetla")(XetlaConfig)
    
