#!/bin/bash
###############################################################################
# Bonsai (ternary) decode throughput on Intel Xe2 GPUs: int2 vs bf16.
#
#   bash scripts/run_bonsai_gpu.sh <slurm-jobid> <TAG> [max-model-len]
#
#   TAG is free-form and names the output, e.g. B70 or LNL:
#     bash scripts/run_bonsai_gpu.sh 358811 B70
#
# Writes bonsai_gpu_<TAG>.csv next to the logs in bonsai_logs/, and is
# resumable: rows already recorded as ok/OOM are skipped, so a sweep that
# outlives its allocation can simply be rerun.
#
# ---------------------------------------------------------------------------
# Preparing the models (once; $MODELS defaults to ../models)
#
# The int2 arm loads a packed model plus an int2 sidecar. Build both with:
#
#   # 1.7B and 4B tie lm_head to the embeddings and ship no lm_head.weight,
#   # so the head must be split out first or it stays dense (bf16).
#   for M in 1.7B 4B; do
#     python scripts/untie_lm_head.py \
#       $HUB/models--prism-ml--Ternary-Bonsai-$M-unpacked/snapshots/<hash> \
#       $MODELS/Bonsai-$M-untied
#   done
#
#   # --no-embeddings keeps the embedding lookup dense; the head is still
#   # packed, ternary at group size 128. --tol 0.02 because a handful of
#   # groups sit just above the 1e-3 default.
#   for M in 1.7B 4B; do
#     python scripts/pack_bonsai_hf.py --tol 0.02 --no-embeddings \
#       --model $MODELS/Bonsai-$M-untied \
#       --out   $MODELS/Bonsai-$M-untied.xetla-int2_f16.safetensors
#     python scripts/make_packed_model.py \
#       --model   $MODELS/Bonsai-$M-untied \
#       --sidecar $MODELS/Bonsai-$M-untied.xetla-int2_f16.safetensors \
#       --out     $MODELS/Bonsai-$M-untied-packed
#   done
#
#   # 8B/27B are already untied, so they skip the first step and drop
#   # --no-embeddings only if their embeddings are meant to be packed.
#
# Verify the head really is ternary before trusting a number: the sidecar
# should hold lm_head.qweight/lm_head.scale, and scale.shape[0] * 128 must
# equal hidden_size.
# ---------------------------------------------------------------------------
#
# Four things here are not obvious and each one cost a debugging round:
#
#   1. XPU graph capture. vLLM defaults to ~51 piecewise graphs up to size 512,
#      which needs ~24 GiB and does not fit a 32 GiB B70 next to the weights and
#      KV cache; capture then dies with UR_RESULT_ERROR_OUT_OF_RESOURCES from
#      whatever kernel happens to synchronise next. We benchmark at batch 1, so
#      capturing 1..8 is enough and costs ~2 GiB.
#   2. Page cache. torch.xpu.mem_get_info() reports MemFree, not MemAvailable,
#      so cached weight files are deducted from the KV budget. Must be evicted
#      before every arm, and *.safetensors globbing does not find the HF blobs.
#   3. Stale processes. An aborted run leaves an EngineCore holding the device.
#      Match on comm, not on the command line: every path here contains the
#      string "vllm", so a cmdline match kills the reclaim step itself.
#   4. Reclaim and benchmark must share one srun step, otherwise the steps
#      contend and the reclaim is silently Force Terminated before it runs.
###############################################################################
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLUG=${PLUG:-$(cd -- "$HERE/.." && pwd)}
# where the packed models and sidecars live; override for a different layout
MODELS=${MODELS:-$PLUG/../models}
HUB=${HUB:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
# sourced inside the compute-node step; point these at the local install
GPU_VARS=${GPU_VARS:-/swtools/intel-gpu/latest/intel_gpu_vars.sh}
ONEAPI_VARS=${ONEAPI_VARS:-/swtools/intel/2025.3/oneapi-vars.sh}

# hub ids: let huggingface resolve them, so no snapshot hashes are baked in
declare -A SNAP=(
  [1.7B]=prism-ml/Ternary-Bonsai-1.7B-unpacked
  [4B]=prism-ml/Ternary-Bonsai-4B-unpacked
  [8B]=prism-ml/Ternary-Bonsai-8B-unpacked
  [27B]=prism-ml/Ternary-Bonsai-27B-unpacked
)
# 1.7B/4B ship tied embeddings and carry no lm_head.weight, so packing them
# straight from the hub leaves the head dense. The -untied copies materialise
# lm_head from the shared matrix (exactly ternary at GS=128) so every model here
# runs a ternary head with dense embeddings, matching the CPU runs.
declare -A SIDECAR=(
  [1.7B]=$MODELS/Bonsai-1.7B-untied.xetla-int2_f16.safetensors
  [4B]=$MODELS/Bonsai-4B-untied.xetla-int2_f16.safetensors
  [8B]=$MODELS/Ternary-Bonsai-8B-unpacked.xetla-int2_f16.safetensors
  [27B]=$MODELS/Bonsai-27B.xetla-int2_f16.safetensors
)
declare -A PACKED=(
  [1.7B]=$MODELS/Bonsai-1.7B-untied-packed
  [4B]=$MODELS/Bonsai-4B-untied-packed
  [8B]=$MODELS/Ternary-Bonsai-8B-packed
  [27B]=$MODELS/Bonsai-27B-packed
)

# order matters: the GPU runtime and oneAPI must come before the venv, else
# torch.xpu.device_count() is 0 and vLLM fails with "Device string must not be empty"
ENV_SETUP="
source /swtools/intel-gpu/latest/intel_gpu_vars.sh >/dev/null 2>&1
source /swtools/intel/2025.3/oneapi-vars.sh >/dev/null 2>&1
source $PLUG/.venv/bin/activate
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export VLLM_XPU_ENABLE_XPU_GRAPH=1
"

JOB=${1:?usage: $0 <slurm-jobid> <TAG> [max-model-len]}
TAG=${2:?usage: $0 <slurm-jobid> <TAG> [max-model-len]}
# 288 tokens are generated (32 in, 256 out), so a 512-token context is already
# generous; a larger one only inflates the KV cache vllm insists on reserving
MAXLEN=${3:-512}
CGSIZES=${CGSIZES:-1,2,4,8}
# vllm profiles peak activation memory at max_num_batched_tokens, which defaults
# to 8192. For Bonsai 8B in bf16 that profile alone charged ~12 GiB and drove the
# KV budget to -4.91 GiB even though the weights are only 15.27 GiB of a 22.3 GiB
# budget. The prompts here are 32 tokens, so profiling that wide is pure waste.
MAXBATCHTOK=${MAXBATCHTOK:-512}

OUT=$PLUG/bonsai_gpu_${TAG}.csv
LOGD=$PLUG/bonsai_logs
mkdir -p "$LOGD"
[[ -f "$OUT" ]] || echo "platform,model,wdtype,ttft_ms,tokens_per_s,status" > "$OUT"

for M in 1.7B 4B 8B 27B; do
  for WD in int2 bf16; do
    if grep -qE "^$TAG,$M,$WD,.*,(ok|OOM)$" "$OUT"; then
      echo "    $M $WD -> already recorded"; continue
    fi
    # 27B in 16-bit is ~54 GB of weights, past both the 32 GB B70 and the
    # shared memory of the Arc 140V, so it is recorded rather than attempted
    if [[ "$M" == "27B" && "$WD" == "bf16" ]]; then
      echo "$TAG,27B,bf16,,,OOM" >> "$OUT"
      echo "    27B bf16 -> OOM (~54 GB, not attempted)"; continue
    fi

    LOG="$LOGD/gpu_${TAG}_${M}_${WD}.log"
    if [[ "$WD" == "int2" ]]; then
      QENV="export XETLA_PREQUANT_PATH=${SIDECAR[$M]} XETLA_QUANT_METHOD=int2_f16"
      ARGS="--model ${PACKED[$M]} --tokenizer ${SNAP[$M]} --quantization xetla"
    else
      QENV="unset XETLA_PREQUANT_PATH XETLA_QUANT_METHOD"
      ARGS="--model ${SNAP[$M]} --quantization none"
    fi

    echo ">>> $TAG $M $WD"
    srun --jobid="$JOB" --overlap bash -lc "$ENV_SETUP
      pids=\$(ps -u \$USER -o pid=,comm= | awk '\$2 ~ /^(vllm|VLLM::EngineCor)\$/ {print \$1}')
      [[ -n \"\$pids\" ]] && kill -9 \$pids 2>/dev/null
      sleep 8
      python3 $HERE/evict_page_cache.py $HUB $(dirname "${SIDECAR[$M]}")
      # Cap rather than just subtract: free memory is sampled here but vllm
      # re-prefetches the checkpoints into page cache before it checks, and on
      # unified memory that is deducted from the budget. Asking for nearly the
      # whole device loses that race.
      U=\$(python -c 'import torch;f,t=torch.xpu.mem_get_info();print(f\"{max(0.20,min(0.78,(f-4*2**30)/t)):.2f}\")')
      echo \"[util] \$U\"
      $QENV
      python -u $HERE/bench_model.py $ARGS --dtype bfloat16 \
        --max-model-len $MAXLEN --gpu-memory-utilization \$U \
        --cudagraph-sizes $CGSIZES --max-num-batched-tokens $MAXBATCHTOK \
        --max-tokens 256 --temperature 0.0 \
        --prompt 'Tell me about photosynthesis in 200 words'" > "$LOG" 2>&1

    tps=$(grep -aoE 'decode +: [0-9]+ tokens in [0-9.]+ s = [0-9.]+' "$LOG" | tail -1 | awk '{print $NF}')
    ttft=$(grep -aoE 'TTFT \(prefill\) +: [0-9]+' "$LOG" | tail -1 | awk '{print $NF}')
    if [[ -n "$tps" ]]; then
      st=ok
    else
      st=$(grep -qaiE 'out of memory|OutOfMemory|UR_RESULT_ERROR_OUT_OF|No available memory' "$LOG" && echo OOM || echo fail)
    fi
    grep -vE "^$TAG,$M,$WD," "$OUT" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
    echo "$TAG,$M,$WD,${ttft:-},${tps:-},$st" >> "$OUT"
    echo "    $M $WD -> ${tps:-$st}"
  done
done

column -s, -t "$OUT"
