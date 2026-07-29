#!/usr/bin/env bash
# Start the Bonsai int2 chat studio (FastAPI + single-page UI) on an XPU node.
#
#   cd demo && ./serve.sh                     # 27B, xetla int2, port 8000
#   DEMO_MODEL=/path/to/8B ./serve.sh         # any model the plugin supports
#   DEMO_TEXT_ONLY=1 ./serve.sh               # disable image input
#   JOBID=<slurm jobid> ./serve.sh            # run inside an existing allocation
#
# From a laptop:  ssh -L 8000:<node>:8000 <login-host>   then open
# http://localhost:8000
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# ---- Slurm passthrough: re-exec inside the allocation if asked. -----------
if [[ -n "${JOBID:-}" && -z "${SLURM_JOB_ID:-}" ]]; then
    exec srun --jobid="${JOBID}" --overlap --pty bash -lc \
        "PORT=${PORT} HOST=${HOST} $(printf '%q ' "${BASH_SOURCE[0]}")"
fi

# ---- Toolchain + venv ----------------------------------------------------
# These vendor scripts assume a permissive shell (unset vars, non-zero probes),
# so relax -eu while they run.
GPU_VARS="${INTEL_GPU_VARS:-/swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh}"
ONEAPI="${ONEAPI_VARS:-/swtools/intel/2025.3/oneapi-vars.sh}"
set +eu
[[ -f "${GPU_VARS}" ]] && source "${GPU_VARS}" >/dev/null 2>&1
[[ -f "${ONEAPI}" ]] && source "${ONEAPI}" --force >/dev/null 2>&1
# shellcheck disable=SC1091
source "${ROOT_DIR}/.venv/bin/activate"
set -eu

# ---- Engine / plugin configuration ---------------------------------------
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export XETLA_QUANT_METHOD="${XETLA_QUANT_METHOD:-int2_f16}"

# vLLM's AOT torch.compile cache is not keyed on every engine setting this demo
# varies (context length, multimodal on/off, eager). Loading a mismatched
# artifact either raises "'NoneType' object has no attribute 'size'" inside the
# compiled graph or silently produces degenerate, repeating output - so pay the
# ~60 s recompile instead. Set DEMO_COMPILE_CACHE=1 to opt back in.
if [[ "${DEMO_COMPILE_CACHE:-0}" == "1" ]]; then
    echo "[serve.sh] torch.compile cache ENABLED (DEMO_COMPILE_CACHE=1)"
else
    export VLLM_DISABLE_COMPILE_CACHE=1
fi

BONSAI_27B_DEFAULT=/data/nfs_home/egeorgan/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-27B-unpacked/snapshots/427bc01949f6122fda741199506b0d00f1fc9122
export DEMO_MODEL="${DEMO_MODEL:-${BONSAI_27B:-${BONSAI_27B_DEFAULT}}}"

# Pre-quantized sidecar: <model>.xetla-<method>.safetensors next to the model
# directory, or the well-known 27B one.
if [[ -z "${XETLA_PREQUANT_PATH:-}" ]]; then
    for candidate in \
        "${DEMO_MODEL}.xetla-${XETLA_QUANT_METHOD}.safetensors" \
        "${ROOT_DIR}/../Ternary-Bonsai-27B.xetla-${XETLA_QUANT_METHOD}.safetensors"
    do
        if [[ -f "${candidate}" ]]; then
            export XETLA_PREQUANT_PATH="${candidate}"
            echo "[serve.sh] xetla sidecar: ${XETLA_PREQUANT_PATH}"
            break
        fi
    done
fi

echo "[serve.sh] model   : ${DEMO_MODEL}"
echo "[serve.sh] quant   : ${XETLA_QUANT_METHOD} (${DEMO_QUANT:-xetla})"
echo "[serve.sh] serving : http://$(hostname):${PORT}"

cd "${SCRIPT_DIR}"

# The engine asks for a restart (exit 42) when the KV cache does not fit at the
# current context length; it leaves the next value to try in DEMO_RETRY_FILE.
# A failed vLLM build cannot free its device memory, so a fresh process is the
# only way to retry.
RUN_DIR="${SCRIPT_DIR}/.run"
mkdir -p "${RUN_DIR}"
export DEMO_RETRY_FILE="${RUN_DIR}/retry_max_model_len"
rm -f "${DEMO_RETRY_FILE}"

while true; do
    set +e
    python -m uvicorn server:app --host "${HOST}" --port "${PORT}" --timeout-keep-alive 600
    rc=$?
    set -e
    if [[ "${rc}" -eq 42 && -s "${DEMO_RETRY_FILE}" ]]; then
        DEMO_MAX_MODEL_LEN="$(cat "${DEMO_RETRY_FILE}")"
        export DEMO_MAX_MODEL_LEN
        rm -f "${DEMO_RETRY_FILE}"
        echo "[serve.sh] restarting with DEMO_MAX_MODEL_LEN=${DEMO_MAX_MODEL_LEN}"
        continue
    fi
    exit "${rc}"
done
