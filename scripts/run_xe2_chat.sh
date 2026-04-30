#!/usr/bin/env bash
set -e
source /swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh >/dev/null 2>&1 || true
source /swtools/intel/2025.3/oneapi-vars.sh --force >/dev/null 2>&1 || true
cd ~/fresh_int2_fp16_setup/xetla_vllm_plugin
echo "[xe2] node=$(hostname)"
export ONEAPI_DEVICE_SELECTOR="level_zero:0"
# CCL/OFI workaround for single-rank xccl init on Xe2
export FI_PROVIDER=tcp
export CCL_LOCAL_RANK=0 CCL_LOCAL_SIZE=1 CCL_ATL_TRANSPORT=ofi CCL_PROCESS_LAUNCHER=none
export CCL_KVS_MODE=mpi
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
{ echo "Hello! In one short sentence, what is the capital of France?"; echo "/exit"; } | bash scripts/chat.sh --max-tokens 32 --temperature 0.0 --enforce-eager 2>&1
