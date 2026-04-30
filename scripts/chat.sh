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

# If a pre-quantized xetla sidecar exists next to the model, use it to skip
# the slow GGUF dequant + per-layer re-quant step on every load.
# Convention: <model>.xetla-<method>.safetensors  (e.g.
# Ternary-Bonsai-8B-F16.gguf.xetla-int2_f16.safetensors).
if [[ -z "${XETLA_PREQUANT_PATH:-}" ]]; then
    candidate="${MODEL}.xetla-${XETLA_QUANT_METHOD}.safetensors"
    if [[ -f "${candidate}" ]]; then
        export XETLA_PREQUANT_PATH="${candidate}"
        echo "[chat.sh] using xetla sidecar: ${XETLA_PREQUANT_PATH}"
    fi
fi

# Auto-detect integrated GPUs (LNL/Arrow Lake/MTL) where VRAM is shared with
# system RAM and the default 0.9 utilization is unsafe. Cap to 0.55 unless
# the user explicitly overrode CHAT_GPU_MEM_UTIL.
if [[ -z "${CHAT_GPU_MEM_UTIL:-}" ]]; then
    # On integrated GPUs (LNL/MTL/ARL) the "VRAM" is shared with system RAM
    # and is reported as nearly the full installed RAM.  vLLM's default 0.9
    # is unsafe there.  Detect either by name or by the heuristic that the
    # XPU's "total" matches /proc/meminfo's MemTotal within ~10%.
    auto_util=$(python - <<'PY' 2>/dev/null
import re, torch
try:
    name = torch.xpu.get_device_properties(0).name.lower()
    total = torch.xpu.get_device_properties(0).total_memory
    sys_total = 0
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal:"):
            sys_total = int(line.split()[1]) * 1024
            break
    keywords = ("lunar", "lnl", "meteor", "mtl", "arrow", "arl",
                "iris", "arc(tm) graphics")
    integrated = (
        any(k in name for k in keywords)
        or (sys_total and abs(total - sys_total) / sys_total < 0.15)
    )
    if integrated:
        # Reserve enough free shared memory for the dequant peak.  Use the
        # currently-free fraction, capped at 0.85 and floored at 0.30.
        free_b, total_b = torch.xpu.mem_get_info(0)
        frac = max(0.30, min(0.85, 0.85 * free_b / total_b))
        print(f"{frac:.2f}")
except Exception:
    pass
PY
)
    if [[ -n "${auto_util}" ]]; then
        export CHAT_GPU_MEM_UTIL="${auto_util}"
        echo "[chat.sh] integrated XPU detected -> CHAT_GPU_MEM_UTIL=${CHAT_GPU_MEM_UTIL}"
    fi
fi

GPU_UTIL_ARG=()
if [[ -n "${CHAT_GPU_MEM_UTIL:-}" ]]; then
    GPU_UTIL_ARG=(--gpu-memory-utilization "${CHAT_GPU_MEM_UTIL}")
fi

exec python "${SCRIPT_DIR}/chat.py" --model "${MODEL}" "${GPU_UTIL_ARG[@]}" "$@"
