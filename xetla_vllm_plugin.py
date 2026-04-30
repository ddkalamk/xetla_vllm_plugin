# from vllm import ModelRegistry
from typing import Any, Optional

import torch
from torch import nn
import os
import time
from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)

from vllm.entrypoints.utils import log_non_default_args
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm import _custom_ops as ops
import vllm._xpu_ops

fp8_gemm_w8a16 = torch.ops._xpu_C.fp8_gemm_w8a16

# os.environ["SYCL_PROGRAM_COMPILE_OPTIONS"] = "-vc-codegen -vc-disable-indvars-opt -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' -doubleGRF"
timing_enabled = int(os.environ.get("XETLA_TIMINGS", "0")) > 0
quantize_lm_heads = int(os.environ.get("XETLA_QUANTIZE_LM_HEADS", "1")) > 0
xetla_enabled = int(os.environ.get("ENABLE_XETLA_QUANTIZATION", "0")) > 0

# vllm.model_executor.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod
# vllm.model_executor.layers.linear.UnquantizedLinearMethod
override_quant_methods = [UnquantizedEmbeddingMethod, UnquantizedLinearMethod]


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
        print(
            f"Execution time: {self.elapsed_time:.4f} usecs, {[list(t.shape) for t in self.tensors]}"
        )


@torch.library.custom_op("xetla::int2_woq_fused_gemm", mutates_args=())
def xetla_int2_woq_fused_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Custom operator for fully connected layer with qint2 weight tensors.
    """
    # print(
    #     f"Custom xetla_int2_woq_fused_gemm called with input shape: {input.shape}, weight shape: {weight.shape}"
    # )
    import xetla_pt_ext

    # out = xetla_pt_ext.int2_woq_fused_gemm_run(input, weight, scale, bias)
    with Timer(input, weight):
        out = torch.ops.xetla_int2.int2_woq_fused_gemm_run(
            input, weight, scale, bias, None
        )
    return out


@xetla_int2_woq_fused_gemm.register_fake
def xetla_int2_woq_fused_gemm_fake_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return input.new_empty([input.shape[0], weight.shape[1]], dtype=input.dtype)


def xetla_quant_scheme():
    quant_scheme = os.environ.get("XETLA_QUANT_SCHEME", "int2").lower()
    if quant_scheme not in ["bf16", "int2"]:
        raise ValueError(f"Unsupported xetla quantization scheme: {quant_scheme}")
    return quant_scheme


_SCALE_DTYPE_MAP = {
    "float": torch.float32,
    "float32": torch.float32,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
}


def xetla_group_size():
    # 0 means channel-wise (per-output-channel) quantization
    try:
        gs = int(os.environ.get("XETLA_GROUP_SIZE", "0"))
    except ValueError:
        raise ValueError(
            f"Invalid XETLA_GROUP_SIZE: {os.environ.get('XETLA_GROUP_SIZE')}"
        )
    if gs < 0:
        raise ValueError(f"XETLA_GROUP_SIZE must be >= 0, got {gs}")
    return gs


def xetla_scale_dtype():
    name = os.environ.get("XETLA_SCALE_DTYPE", "float").lower()
    if name not in _SCALE_DTYPE_MAP:
        raise ValueError(
            f"Unsupported XETLA_SCALE_DTYPE: {name}. "
            f"Supported: {sorted(set(_SCALE_DTYPE_MAP))}"
        )
    return _SCALE_DTYPE_MAP[name]


def pack_int2_vnni16(t):
    d0, d1 = t.shape
    assert d0 % 16 == 0, "Dim 0 must be multiple of 16"
    t1 = t.view([d0 // 16, 16, d1]).permute([0, 2, 1]).contiguous()
    t1 = t1 & 0x3
    shifts = torch.arange(16, dtype=torch.int32) * 2
    shifts = shifts.to(t1.device)
    packed = (t1.to(torch.int32) << shifts).sum(dim=-1).to(torch.int32)
    return packed


def quantize_to_int2(t, group_size=None, scale_dtype=torch.float32):
    if not group_size:
        scale = t.abs().amax(dim=0, keepdim=True).to(scale_dtype)
        t1 = t / scale
        t1 = torch.clamp(t1, -2, 1)
        t1 = t1.to(torch.int8)
        # t1 = torch.randint(-1, 2, t1.shape, device=t1.device, dtype=torch.int8)
        # print(f"Qint2: {t1.shape} {t1.amax()} {t1.amin()} {scale.shape}")
        return t1, scale
    else:
        d0, d1 = t.shape
        assert (
            d0 % group_size == 0
        ), f"Dim 0 ({d0}) must be divisible by group_size ({group_size})"
        num_groups = d0 // group_size
        t_grouped = t.view(num_groups, group_size, d1)
        scale = t_grouped.abs().amax(dim=1, keepdim=True).to(scale_dtype)
        t1 = (t_grouped / scale).clamp(-2, 1).to(torch.int8).view(d0, d1)
        return t1, scale.squeeze(1)  # Return scale with shape [num_groups, d1]


def dequantize_int2_to_fp(t, scale, dtype=torch.bfloat16):
    if scale.shape[0] == 1:
        t1 = (t.to(torch.float) * scale).to(dtype)
    else:
        d0, d1 = t.shape
        group_size = d0 // scale.shape[0]
        t_grouped = t.view(-1, group_size, d1)
        scale_grouped = scale.view(-1, 1, d1)
        t1 = (t_grouped.to(torch.float) * scale_grouped).to(dtype).view(d0, d1)
    return t1


class XetlaConfig(QuantizationConfig):
    def __init__(self) -> None:
        self.scheme = xetla_quant_scheme()
        self.group_size = xetla_group_size()
        self.scale_dtype = xetla_scale_dtype()
        super().__init__()

    def __repr__(self) -> str:
        return (
            f"XetlaConfig(scheme={self.scheme}, group_size={self.group_size}, "
            f"scale_dtype={self.scale_dtype})"
        )

    def resolved_scale_dtype(self, input_dtype: torch.dtype) -> torch.dtype:
        # Override default float32 scale to fp16 when input/activation dtype is fp16,
        # unless the user explicitly requested a non-default scale dtype.
        if (
            self.scale_dtype == torch.float32
            and input_dtype == torch.float16
            and "XETLA_SCALE_DTYPE" not in os.environ
        ):
            return torch.float16
        return self.scale_dtype

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
        cls, hf_quant_cfg, user_quant
    ) -> Optional[QuantizationMethods]:
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["LinearMethodBase"]:
        if isinstance(layer, LinearBase):
            return XetlaLinearMethod(self)
        elif isinstance(layer, ParallelLMHead) and quantize_lm_heads:
            return XetlaEmbeddingMethod(self, True)
        elif isinstance(layer, VocabParallelEmbedding) and quantize_lm_heads:
            return XetlaEmbeddingMethod(self, False)
        print(
            f"XetlaConfig.get_quant_method: Unsupported layer type {type(layer)} for layer {prefix}"
        )
        return None


class XetlaEmbeddingMethod(UnquantizedEmbeddingMethod):
    """Xetla quantized method for embeddings.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: XetlaConfig, inplace: bool = False):
        self.quant_config = quant_config
        self.inplace = inplace
        # self.quant_config.scheme = "int2"  # Embeddings use int2
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

        if self.quant_config.scheme == "int2":
            print(
                f"Processing weights for layer {layer} with method fp8 (inplace={self.inplace})"
            )
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)
            # Update the layer with the new values.
            if self.inplace:
                layer.weight = torch.nn.Parameter(qweight, requires_grad=False)
            else:
                layer.qweight = torch.nn.Parameter(qweight, requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None
            layer.xetla_quantized = True
        else:
            pass

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.quant_config.scheme == "int2" and getattr(
            layer, "xetla_quantized", False
        ):
            weight = layer.weight.data if self.inplace else layer.qweight.data
            weight_scale = layer.weight_scale.data
            output = fp8_gemm_w8a16(x, weight.t(), weight_scale, bias)
            # print(f"XetlaEmbeddingMethod apply output shape: {output.shape}, dtype: {output.dtype}")
            return output
        else:
            return super().apply(layer, x, bias)


class XetlaLinearMethod(LinearMethodBase):
    """Linear method for xetla.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: XetlaConfig):
        self.quant_config = quant_config
        super().__init__()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # We just reuse UnquantizedLinearMethod to create weights
        UnquantizedLinearMethod.create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not current_platform.is_xpu():
            return

        print(
            f"Processing weights for layer {layer} with method {self.quant_config.scheme}"
        )
        if self.quant_config.scheme == "int2":
            weight = layer.weight.data
            group_size = self.quant_config.group_size or None
            scale_dtype = self.quant_config.resolved_scale_dtype(weight.dtype)
            weight_int2, layer.scale = quantize_to_int2(
                weight.t(), group_size=group_size, scale_dtype=scale_dtype
            )
            layer.weight.data = pack_int2_vnni16(weight_int2)
        else:
            pass

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # with Timer(x, layer.weight) as t:
        if True:
            if self.quant_config.scheme == "int2":
                # import xetla_pt_ext
                # print("Using xetla int2 gemm x: ", x.shape, " weight: ", layer.weight.shape)
                c = xetla_int2_woq_fused_gemm(x, layer.weight, layer.scale, bias)
                return c
            else:
                return UnquantizedLinearMethod.apply(self, layer, x, bias)


# hack to get command line args for quantization method
def log_non_default_args_xetla(engine_args):
    global xetla_enabled
    log_non_default_args(engine_args)
    print(f"Xetla plugin quantization method: {engine_args.quantization}")
    if engine_args.quantization == "xetla":
        xetla_enabled = True
        os.environ["ENABLE_XETLA_QUANTIZATION"] = "1"


def process_weights_after_loading(model: nn.Module) -> bool:
    global xetla_enabled
    if not xetla_enabled or not current_platform.is_xpu():
        return False
    xetla_config = XetlaConfig()
    ret = False
    for prefix, module in model.named_modules():
        orig_quant_method = getattr(module, "quant_method", None)
        if not (type(orig_quant_method) in override_quant_methods):
            continue
        quant_method = xetla_config.get_quant_method(module, prefix)
        if isinstance(quant_method, QuantizeMethodBase):
            # print(
            #     f"{type(module)} - qmeth: {type(quant_method)}, orig_qmeth: {type(orig_quant_method)}, prefix: {prefix}"
            # )
            quant_method.process_weights_after_loading(module)
            module.quant_method = quant_method  # type: ignore
            ret = True
    return ret


def get_wrapper_load_model_fn(original_fn):
    def wrapper_load_model(self, *args, **kwargs):
        print("GPUModelRunner.load_model called with args: ", args, " kwargs: ", kwargs)
        original_fn(self, *args, **kwargs)
        print(self.model)
        if process_weights_after_loading(self.model):
            self.model_config.quantization = "xetla"

    return wrapper_load_model


def register():
    print("Hello xetla plugin!")

    register_quantization_config("xetla")(XetlaConfig)
    vllm.entrypoints.llm.log_non_default_args = log_non_default_args_xetla
    GPUModelRunner.load_model = get_wrapper_load_model_fn(GPUModelRunner.load_model)
