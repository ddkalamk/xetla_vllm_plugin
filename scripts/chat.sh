#!/usr/bin/env bash
# Interactive chat REPL with the local Ternary-Bonsai-8B GGUF on XPU via vLLM.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )

# Activate venv (same as scripts/env.sh xpu)
# shellcheck disable=SC1091
source "${ROOT_DIR}/.venv/bin/activate"

# XPU env (mirror of run_vllm_latency_bench.sh)
export RenderCompressedBuffersEnabled=0
export NEOReadDebugKeys=1
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"

# Use the xetla plugin's int2 weight x fp16 act kernels (per-128 K-group fp16
# scales) for the Ternary-Bonsai GGUF.
export XETLA_QUANT_METHOD="${XETLA_QUANT_METHOD:-int2_f16}"
export VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-xetla}"

MODEL="${BONSAI_GGUF:-${ROOT_DIR}/Ternary-Bonsai-8B-F16.gguf}"

exec python "${SCRIPT_DIR}/chat.py" --model "${MODEL}" "$@"
