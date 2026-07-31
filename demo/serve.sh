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

# XPU graphs are worth ~2.6x decode (62 -> 159 tok/s on the CAT-Q MoE model),
# but so far that is the only model they have been measured on, and capture is
# fragile: it aborts on any host sync or large allocation in the captured
# region. Opt in with VLLM_XPU_ENABLE_XPU_GRAPH=1 once you have checked the
# model you care about. Capture sizes are capped in server.py
# (DEMO_CUDAGRAPH_SIZES) to stay inside the plugin's batched MoE path.
export VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_XPU_ENABLE_XPU_GRAPH:-0}"

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

# Model resolution, in order of preference:
#   1. DEMO_MODEL / BONSAI_27B, if set
#   2. a bundle produced by scripts/make_bundle.py next to the repo
#   3. a plain HF checkout next to the repo
#   4. the HF repo id (resolved through the local cache or downloaded)
# Never hardcode a machine-specific absolute path: on another host it silently
# becomes a bogus "repo id" and transformers fails with a confusing
# HFValidationError instead of "model not found".
if [[ -z "${DEMO_MODEL:-}" && -z "${BONSAI_27B:-}" ]]; then
    for candidate in \
        "${ROOT_DIR}/../bonsai27b-int2-bundle/model" \
        "${ROOT_DIR}/bonsai27b-int2-bundle/model" \
        "${ROOT_DIR}/../Ternary-Bonsai-27B-unpacked"
    do
        if [[ -d "${candidate}" ]]; then
            DEMO_MODEL="${candidate}"
            break
        fi
    done
fi
export DEMO_MODEL="${DEMO_MODEL:-${BONSAI_27B:-prism-ml/Ternary-Bonsai-27B-unpacked}}"

# If it looks like a path, it must exist: fail here with a clear message rather
# than letting transformers treat a missing directory as a Hugging Face repo id.
case "${DEMO_MODEL}" in
    /*|./*|../*)
        if [[ ! -d "${DEMO_MODEL}" ]]; then
            echo "[serve.sh] ERROR: DEMO_MODEL is not a directory on this host:" >&2
            echo "[serve.sh]   ${DEMO_MODEL}" >&2
            echo "[serve.sh] Set DEMO_MODEL to a local checkout or bundle, e.g." >&2
            echo "[serve.sh]   DEMO_MODEL=/path/to/bonsai27b-int2-bundle/model ./serve.sh" >&2
            echo "[serve.sh] or use the repo id: DEMO_MODEL=prism-ml/Ternary-Bonsai-27B-unpacked" >&2
            exit 2
        fi
        ;;
esac

# Pre-quantized sidecar: <model>.xetla-<method>.safetensors next to the model
# directory, or the well-known 27B one.
if [[ -z "${XETLA_PREQUANT_PATH:-}" ]]; then
    for candidate in \
        "${DEMO_MODEL}.xetla-${XETLA_QUANT_METHOD}.safetensors" \
        "${DEMO_MODEL}/../model.xetla-${XETLA_QUANT_METHOD}.safetensors" \
        "${ROOT_DIR}/../Ternary-Bonsai-27B.xetla-${XETLA_QUANT_METHOD}.safetensors" \
        "${ROOT_DIR}/Ternary-Bonsai-27B.xetla-${XETLA_QUANT_METHOD}.safetensors"
    do
        if [[ -f "${candidate}" ]]; then
            export XETLA_PREQUANT_PATH="$(cd "$(dirname "${candidate}")" && pwd)/$(basename "${candidate}")"
            echo "[serve.sh] xetla sidecar: ${XETLA_PREQUANT_PATH}"
            break
        fi
    done
fi
if [[ -z "${XETLA_PREQUANT_PATH:-}" ]]; then
    echo "[serve.sh] WARNING: no int2 sidecar found; the 27B will try to load" >&2
    echo "[serve.sh]          dense (~54 GB) and will not fit. See" >&2
    echo "[serve.sh]          scripts/pack_bonsai_hf.py / scripts/make_bundle.py" >&2
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
