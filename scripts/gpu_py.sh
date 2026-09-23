#!/bin/bash
# Run a python script from this repo inside the GPU allocation with the Bonsai 2
# xetla int2 environment.   bash scripts/gpu_py.sh <jobid> <script> [args...]
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLUG=${PLUG:-$(cd -- "$HERE/.." && pwd)}
MODELS=${MODELS:-$PLUG/../models}
SIDECAR=${SIDECAR:-$MODELS/Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors}
QMETHOD=${QMETHOD:-int2_f16}
JOB=${1:?jobid}; shift
srun --jobid="$JOB" --overlap bash -lc "
source /swtools/intel-gpu/latest/intel_gpu_vars.sh >/dev/null 2>&1
source /swtools/intel/2026.0/oneapi-vars.sh >/dev/null 2>&1
source $PLUG/.venv/bin/activate
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export VLLM_XPU_ENABLE_XPU_GRAPH=${VLLM_XPU_ENABLE_XPU_GRAPH:-1}
export XETLA_PREQUANT_PATH=$SIDECAR XETLA_QUANT_METHOD=$QMETHOD
export XETLA_HADAMARD_DTYPE=${XETLA_HADAMARD_DTYPE:-fp32}
${RUN_ENV:-}
pids=\$(ps -u \$USER -o pid=,comm= | awk '\$2 ~ /^(vllm|VLLM::EngineCor|lm_eval|lm-eval)\$/ {print \$1}')
[[ -n \"\$pids\" ]] && kill -9 \$pids 2>/dev/null
cd $PLUG && python -u $*
"
