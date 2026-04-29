# One time Setup
```bash
# Setup vLLM Python uv ENV
bash utils/setup_vllm_xpu.sh
```

# Install oneAPI Deep Learning Essentials
```bash
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/56f7923a-adb8-43f3-8b02-2b60fcac8cab/intel-deep-learning-essentials-2025.3.3.16_offline.sh
bash ./intel-deep-learning-essentials-2025.3.3.16_offline.sh -a --silent --eula accept
```


# Running latency benchmark
```bash
# activate uv env
source .venv/bin/activate

MODEL_NAME="Qwen/Qwen2.5-1.5B"
# example bf16 command line
vllm bench latency --model "$MODEL_NAME"  --batch-size 1 --num-iters 10 --num-iters-warmup 3 --gpu_memory_util=0.7 --input-len 32 --output-len 128 --max-model-len 2048

# use fp8 quantized gemms
vllm bench latency --model "$MODEL_NAME"  --batch-size 1 --num-iters 10 --num-iters-warmup 3 --gpu_memory_util=0.7 --input-len 32 --output-len 128 --max-model-len 2048 -q fp8

# use xetla int2-bf16 gemms
vllm bench latency --model "$MODEL_NAME"  --batch-size 1 --num-iters 10 --num-iters-warmup 3 --gpu_memory_util=0.7 --input-len 32 --output-len 128 --max-model-len 2048 -q xetla

```

---

# int2 × fp16 weight-only quantized GEMMs (`int2_f16` mode)

In addition to the original `int2 × bf16` path, the plugin now supports an
**int2 weights × fp16 activations** kernel with **per-128-K-group fp16 scales**
(no zero-point, no scale-A). This is the variant used by the
`Ternary-Bonsai-8B` ternary GGUF.

The kernel itself lives in the xetla submodule on the
[`feature_int2_woq_f16_act_gs128`](https://github.com/egeor/xetla/tree/feature_int2_woq_f16_act_gs128)
branch (file: `int2_fp16_upcvt_dpas_fast_test/src/main.cpp` plus the headers
under `include/experimental/{group,kernel}/gemm/impl/int2_fp16_upcvt_*`). To
make it available to the build, check out that branch in the submodule:

```bash
git submodule update --init xetla
( cd xetla && git fetch origin feature_int2_woq_f16_act_gs128 \
            && git checkout feature_int2_woq_f16_act_gs128 )
```

Then build the plugin extension:

```bash
source /swtools/intel-gpu/<ver>/intel_gpu_vars.sh
source /swtools/intel/<ver>/oneapi-vars.sh --force
source .venv/bin/activate
python setup.py build_ext --inplace
```

## What was added

* `csrc/int2_fp16_upcvt_kernel.sycl` — SYCL host-side wrapper around the new
  `int2_fp16_upcvt_gemm` kernel. Pre-instantiates a small set of
  `(WGN, KS, LS, kUnaligned)` template variants and exposes one entry point.
  Includes a **shape-keyed dispatch** for the GEMV shapes that arise during
  Bonsai-8B decode, sourced from the autotuner's
  `bonsai_8B_run1/best.csv`:

  | (K, N)            | (WGN, KS, LS) | layer        |
  | ----------------- | ------------- | ------------ |
  | (4096,   6144)    | (32, 1, 4)    | qkv_proj     |
  | (4096,  12288)    | (32, 1, 2)    | (fused)      |
  | (4096,  24576)    | (32, 1, 4)    | gate_up_proj |
  | (4096, 151680)    | (32, 1, 4)    | lm_head      |
  | (12288,  4096)    | (32, 1, 8)    | down_proj    |

  Other shapes fall back to the generic N-tier policy.

* `csrc/int2_kernel_wrapper.cpp` — single TU that registers both
  `xetla_int2::int2_bf16_fused_gemm_run` and the new
  `xetla_int2::int2_fp16_upcvt_gemm_run` ops with `TORCH_LIBRARY`. Op
  registration **must** live in a regular `.cpp` (the static ctor inside a
  `.sycl` TU does not reliably run at `.so` load with oneAPI 2025.3).

* `xetla_vllm_plugin.py`
  * Adds `XETLA_QUANT_METHOD=int2_f16` mode (existing `int2_bf16` is
    untouched). Selected by env var.
  * `quantize_to_ternary_f16()` and `pack_ternary_to_int2()` helpers — pack
    a dense fp16 tensor (with values in `{-s, 0, +s}` per group) into the
    int2x16 layout the kernel expects, with per-128 fp16 scales.
  * `XetlaLinearMethod` and `XetlaEmbeddingMethod` quantize-on-load paths
    for `int2_f16`. lm_head/embedding is quantized **on CPU** to avoid
    `UR_RESULT_ERROR_DEVICE_LOST` on large `[151680, 4096]` matrices.
  * Custom op `xetla_vllm::xetla_int2_fp16_upcvt_gemm` that wraps the
    Torch op, with proper meta-tensor fake registration so dynamic-shape
    tracing (`torch.compile`) works.
  * Optional debug print toggled by `XETLA_DEBUG=1` (silent by default).

## End-to-end Bonsai-8B chat REPL

`scripts/chat.sh` + `scripts/chat.py` give you an interactive REPL that runs
the local `Ternary-Bonsai-8B-F16.gguf` through vLLM with the int2 × fp16
kernels active end-to-end (linear projections **and** lm_head). On Xe2 with
the tunings above we measure ~138 tok/s decode for a single-batch, M=1
prompt.

```bash
# Default: stream tokens, 128 max output, level_zero device 0
./scripts/chat.sh

# Longer responses, no streaming
./scripts/chat.sh --max-tokens 512 --no-stream

# Override device selector
ONEAPI_DEVICE_SELECTOR="opencl:1;level_zero:0" ./scripts/chat.sh
```

Environment variables consumed by `chat.sh`:

| Var                        | Default                       | Purpose                                    |
| -------------------------- | ----------------------------- | ------------------------------------------ |
| `BONSAI_GGUF`              | `<repo>/Ternary-Bonsai-8B-F16.gguf` | Path to the GGUF                     |
| `BONSAI_TOKENIZER`         | `Qwen/Qwen3-8B`               | HF tokenizer (GGUF tokenizer is unusable)  |
| `XETLA_QUANT_METHOD`       | `int2_f16`                    | `int2_f16` or `int2_bf16`                  |
| `VLLM_QUANTIZATION`        | `xetla`                       | Forwarded as `--quantization xetla`        |
| `VLLM_XPU_ENABLE_XPU_GRAPH`| `1`                           | Enable XPU graph capture                   |
| `ONEAPI_DEVICE_SELECTOR`   | `level_zero:0`                | SYCL device selector                       |
| `CHAT_MAX_TOKENS`          | `128`                         | Default for `--max-tokens`                 |
| `CHAT_MAX_MODEL_LEN`       | `2048`                        | Default for `--max-model-len`              |
| `CHAT_NO_STREAM`           | `0`                           | Set to `1` to disable token streaming      |

REPL commands: `/exit`, `/reset` (clear chat history), `/system <text>` (set
system prompt). After each turn a `[stats]` line reports tokens / wallclock /
tok/s.

## vLLM patches required for the GGUF + xetla path

To allow `--quantization xetla` against a `.gguf` file we made two minimal
edits to the vendored `vllm/` copy (branch `xetla_v0.19.0`):

1. `vllm/engine/arg_utils.py` — when the model path ends in `.gguf`, force
   `load_format=gguf` but **don't** clobber a user-supplied `quantization`.
2. `vllm/model_executor/model_loader/weight_utils.py` — short-circuit the
   `snapshot_download(...)` call in `get_quant_config()` when
   `model_config.quantization == "xetla"` (the GGUF is already local).

Both patches are no-ops for non-GGUF / non-xetla models. They're not needed
if you switch to an HF safetensors checkout (e.g.
`prism-ml/Ternary-Bonsai-8B-unpacked`).

## Numerical-correctness smoke test

```bash
python tests/test_gemm_int2_fp16.py
```

Generates a random ternary weight, runs the int2 × fp16 GEMM, compares to a
reference fp16 matmul, and verifies the round-trip of `quantize_to_ternary_f16`
+ `pack_ternary_to_int2`.
