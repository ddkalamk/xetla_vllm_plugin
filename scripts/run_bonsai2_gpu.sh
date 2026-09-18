#!/bin/bash
###############################################################################
# Bonsai 2 27B (Hadamard-folded ternary) decode on an Intel Xe2 GPU, int2 path.
#
#   bash scripts/run_bonsai2_gpu.sh <slurm-jobid> <TAG> [prompt ...]
#
# Same machinery as run_bonsai_gpu.sh (see the notes there on graph sizes,
# page cache and stale EngineCores), for the one model this covers:
#
#   sidecar : $MODELS/Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors
#   packed  : $MODELS/Ternary-Bonsai-2-27B-packed
#
# both produced by scripts/pack_bonsai2_gguf.py from the PQ2_0 GGUF. The
# sidecar carries the prism.hadamard contract; the plugin applies the sign
# flip + blockwise WHT to activations before every folded GEMM.
#
# Env knobs:
#   XETLA_HADAMARD_DTYPE=fp16|fp32  transform precision (default fp32)
#   UTIL=0.35   pin gpu-memory-utilization (LNL BKM: unified memory, the
#               sampled value over-reserves; 0.35 is what the 27B runs used)
#   KVBYTES=N   pin the KV cache size (bytes). On LNL a fresh torch.compile
#               inflates the profiled usage by >10 GiB and the profiling-based
#               KV budget goes negative; 2 GiB is plenty for these prompts.
#   DETERMINISTIC=1 (default) pass --deterministic-compile: inductor's
#               benchmark-selected combo kernels are off, so the greedy text
#               does not depend on which fusions a fresh compile timed best.
#   MAXTOK, MAXLEN, CGSIZES, MAXBATCHTOK as in run_bonsai_gpu.sh
###############################################################################
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLUG=${PLUG:-$(cd -- "$HERE/.." && pwd)}
MODELS=${MODELS:-$PLUG/../models}
SIDECAR=${SIDECAR:-$MODELS/Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors}
PACKED=${PACKED:-$MODELS/Ternary-Bonsai-2-27B-packed}

JOB=${1:?usage: $0 <slurm-jobid> <TAG> [prompt ...]}
TAG=${2:?usage: $0 <slurm-jobid> <TAG> [prompt ...]}
shift 2
PROMPTS=("$@")
[[ ${#PROMPTS[@]} -gt 0 ]] || PROMPTS=('Tell me about photosynthesis in 200 words')
PROMPT_ARGS=""
for p in "${PROMPTS[@]}"; do PROMPT_ARGS+=" --prompt \"$p\""; done

MAXLEN=${MAXLEN:-512}
MAXTOK=${MAXTOK:-256}
CGSIZES=${CGSIZES:-1,2,4,8}
MAXBATCHTOK=${MAXBATCHTOK:-512}
UTIL=${UTIL:-}
KVBYTES=${KVBYTES:-}
DETERMINISTIC=${DETERMINISTIC:-1}
EXTRA=${EXTRA:-}${KVBYTES:+ --kv-cache-memory-bytes $KVBYTES}
[[ "$DETERMINISTIC" == "1" ]] && EXTRA+=" --deterministic-compile"

ENV_SETUP="
source /swtools/intel-gpu/latest/intel_gpu_vars.sh >/dev/null 2>&1
source /swtools/intel/2025.3/oneapi-vars.sh >/dev/null 2>&1
source $PLUG/.venv/bin/activate
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export VLLM_XPU_ENABLE_XPU_GRAPH=${VLLM_XPU_ENABLE_XPU_GRAPH:-1}
export XETLA_PREQUANT_PATH=$SIDECAR XETLA_QUANT_METHOD=int2_f16
export XETLA_HADAMARD_DTYPE=${XETLA_HADAMARD_DTYPE:-fp32}
${RUN_ENV:-}
"

LOGD=$PLUG/bonsai_logs
mkdir -p "$LOGD"
LOG="$LOGD/gpu_${TAG}_B2-27B_int2.log"
echo ">>> $TAG Bonsai-2-27B int2 (hadamard) -> $LOG"

srun --jobid="$JOB" --overlap bash -lc "$ENV_SETUP
  pids=\$(ps -u \$USER -o pid=,comm= | awk '\$2 ~ /^(vllm|VLLM::EngineCor)\$/ {print \$1}')
  [[ -n \"\$pids\" ]] && kill -9 \$pids 2>/dev/null
  sleep 5
  python3 $HERE/evict_page_cache.py $(dirname "$SIDECAR") $PACKED
  U=$UTIL
  [[ -n \"\$U\" ]] || U=\$(python -c 'import torch;f,t=torch.xpu.mem_get_info();print(f\"{max(0.20,min(0.78,(f-4*2**30)/t)):.2f}\")')
  echo \"[util] \$U\"
  B=/tmp/bonsai2_bench.\$\$.log
  python -u $HERE/bench_model.py --model $PACKED --quantization xetla --dtype bfloat16 \
    --max-model-len $MAXLEN --gpu-memory-utilization \$U \
    --cudagraph-sizes $CGSIZES --max-num-batched-tokens $MAXBATCHTOK \
    --max-tokens $MAXTOK --temperature 0.0 --full $EXTRA $PROMPT_ARGS > \$B 2>&1 &
  BPID=\$!
  ( while kill -0 \$BPID 2>/dev/null; do
      if grep -q 'Model loading took' \$B 2>/dev/null; then
        for i in 1 2 3 4 5 6; do
          python3 $HERE/evict_page_cache.py $(dirname "$SIDECAR") $PACKED >/dev/null 2>&1
          sleep 1
        done
        break
      fi
      sleep 1
    done ) &
  WPID=\$!
  wait \$BPID; RC=\$?
  kill \$WPID 2>/dev/null
  cat \$B; rm -f \$B
  exit \$RC" > "$LOG" 2>&1
RC=$?

grep -aE 'sidecar loaded|inverse-hadamard|TTFT|decode +:|repetition|Error|error' "$LOG" | grep -v "^\[xetla\] sidecar hit" | head -40
echo "--- output ---"
awk '/^prompt +:/{p=1} p' "$LOG" | head -60
echo "rc=$RC log=$LOG"
