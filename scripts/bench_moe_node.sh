#!/bin/bash
# Benchmark the CAT-Q MoE model on the GPU node. Args are passed to bench_model.py.
# No `set -u`: the oneAPI env scripts trip over unbound variables.
source /swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh >/dev/null 2>&1
source /swtools/intel/2025.3/oneapi-vars.sh --force >/dev/null 2>&1
source /data/nfs_home/egeorgan/FRESH/xetla_vllm_plugin/.venv/bin/activate
cd /data/nfs_home/egeorgan/FRESH/xetla_vllm_plugin

export ONEAPI_DEVICE_SELECTOR=level_zero:0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export XETLA_QUANT_METHOD=int2_f16
export XETLA_QUANTIZE_LM_HEADS=0
export XETLA_PREQUANT_PATH=/data/nfs_home/egeorgan/FRESH/CAT-Q-Qwen3-30B-A3B.xetla-int2_f16.safetensors

MODEL=/data/nfs_home/egeorgan/FRESH/xetla_vllm_plugin/BitTern/projects/cat-q/configs/qwen3-moe-30B-A3B/export

python scripts/bench_model.py \
  --model "$MODEL" \
  --max-model-len 2048 --max-tokens 150 \
  --prompt "Explain de novo genome assembly in 60 words" \
  "$@"
echo "BENCH_EXIT=$?"
