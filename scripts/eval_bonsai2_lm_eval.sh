#!/bin/bash
###############################################################################
# Correctness check of Bonsai 2 27B on the xetla int2 path with the standard
# lm-evaluation-harness (vLLM backend), on the GPU node.
#
#   bash scripts/eval_bonsai2_lm_eval.sh <slurm-jobid> <TAG> [lm_eval args...]
#
# Defaults: gsm8k_cot_llama (8-shot, "The final answer is N"), chat template + thinking (reasoning_effort from
# $EFFORT, default medium), greedy, answers taken after </think>. Bonsai 2's
# card reports the math group (GSM8K/MATH-500/AIME) at 96.57 in thinking mode.
#
# Env knobs:
#   LIMIT=N          examples per task (default 200)
#   TASKS="..."      lm_eval task list (default gsm8k_cot_zeroshot)
#   EFFORT=medium|xhigh   reasoning effort passed to the chat template
#   THINK=0          instruct (non-thinking) mode, short answers
#   MAXGEN=N         max generated tokens (default 4096 thinking, 1024 not)
#   MAXLEN=N         max_model_len (default MAXGEN+2048)
#   NSEQ=N           max_num_seqs (default 16)
#   METHOD=int2|bitcos, BITCOS_SFX, MODELS, PACKED, SIDECAR as in run_bonsai2_gpu.sh
###############################################################################
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLUG=${PLUG:-$(cd -- "$HERE/.." && pwd)}
MODELS=${MODELS:-$PLUG/../models}
PACKED=${PACKED:-$MODELS/Ternary-Bonsai-2-27B-packed}
METHOD=${METHOD:-int2}
BITCOS_SFX=${BITCOS_SFX:-.b70}
if [[ "$METHOD" == "bitcos" ]]; then
  SIDECAR=${SIDECAR:-$MODELS/Ternary-Bonsai-2-27B.xetla-bitcos_f16${BITCOS_SFX}.safetensors}
  QMETHOD=bitcos_f16
else
  SIDECAR=${SIDECAR:-$MODELS/Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors}
  QMETHOD=int2_f16
fi
JOB=${1:?usage: $0 <slurm-jobid> <TAG> [lm_eval args...]}
TAG=${2:?usage: $0 <slurm-jobid> <TAG> [lm_eval args...]}
shift 2

LIMIT=${LIMIT:-200}
TASKS=${TASKS:-gsm8k_cot_llama}
THINK=${THINK:-1}
EFFORT=${EFFORT:-medium}
if [[ "$THINK" == "1" ]]; then
  MAXGEN=${MAXGEN:-4096}
  THINK_ARGS="enable_thinking: true
  think_end_token: \"</think>\"
  chat_template_args: {reasoning_effort: $EFFORT}"
else
  MAXGEN=${MAXGEN:-1024}
  THINK_ARGS="enable_thinking: false"
fi
MAXLEN=${MAXLEN:-$((MAXGEN + 2048))}
NSEQ=${NSEQ:-16}
UTIL=${UTIL:-0.78}
OUT=$PLUG/bonsai_logs/lm_eval_${TAG}
mkdir -p "$OUT"

cat > "$OUT/config.yaml" <<EOF
model: vllm
model_args:
  pretrained: $PACKED
  quantization: xetla
  dtype: bfloat16
  trust_remote_code: true
  max_model_len: $MAXLEN
  max_gen_toks: $MAXGEN
  gpu_memory_utilization: $UTIL
  max_num_seqs: $NSEQ
  max_num_batched_tokens: 2048
  enable_prefix_caching: false
  limit_mm_per_prompt: {image: 0, video: 0}
  compilation_config: {cudagraph_capture_sizes: [1, 2, 4, 8, 16], inductor_compile_config: {combo_kernels: false, benchmark_combo_kernel: false}}
  $THINK_ARGS
tasks: [$TASKS]
apply_chat_template: true
fewshot_as_multiturn: true
batch_size: auto
limit: $LIMIT
log_samples: true
output_path: $OUT
gen_kwargs: {temperature: 0}
EOF
echo ">>> $TAG: $TASKS limit=$LIMIT think=$THINK effort=$EFFORT maxgen=$MAXGEN -> $OUT"

srun --jobid="$JOB" --overlap bash -lc "
source /swtools/intel-gpu/latest/intel_gpu_vars.sh >/dev/null 2>&1
source /swtools/intel/2026.0/oneapi-vars.sh >/dev/null 2>&1
source $PLUG/.venv/bin/activate
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export VLLM_XPU_ENABLE_XPU_GRAPH=${VLLM_XPU_ENABLE_XPU_GRAPH:-1}
export XETLA_PREQUANT_PATH=$SIDECAR XETLA_QUANT_METHOD=$QMETHOD
export XETLA_HADAMARD_DTYPE=${XETLA_HADAMARD_DTYPE:-fp32}
export HF_DATASETS_OFFLINE=\${HF_DATASETS_OFFLINE:-0}
${RUN_ENV:-}
pids=\$(ps -u \$USER -o pid=,comm= | awk '\$2 ~ /^(vllm|VLLM::EngineCor|lm_eval|lm-eval)\$/ {print \$1}')
[[ -n \"\$pids\" ]] && kill -9 \$pids 2>/dev/null
cd $OUT
lm_eval run --config $OUT/config.yaml $* 2>&1 | tee $OUT/run.log
"
