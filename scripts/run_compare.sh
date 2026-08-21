#!/usr/bin/env bash
# Alternating int2 vs bitcos decode comparison on one XPU.
#
# Two things make naive runs on Lunar Lake unreliable, both handled here:
#   - the driver does not return all memory when a run aborts, so free memory
#     decays across runs; the utilization is therefore derived from what is
#     actually free right now rather than hardcoded, and applied to both arms.
#   - the XPU driver counts page cache as unavailable, so the model files are
#     evicted before every run.
#
# usage: run_compare.sh <packed-model-dir> <sidecar-prefix> [reps] [util] [maxlen]
#   sidecar paths are <sidecar-prefix>.xetla-<method>.safetensors
set -uo pipefail

MODEL=${1:?packed model dir}
PREFIX=${2:?sidecar path prefix}
REPS=${3:-2}
UTIL=${4:-}
MAXLEN=${5:-1024}
# Order matters on Lunar Lake: the driver leaks memory across runs, so whichever
# arm goes first gets the healthiest device.
METHODS=${METHODS:-"int2_f16 bitcos_f16"}
PROMPT="Tell me about photosynthesis in 200 words"
LOGS=/data/nfs_home/egeorgan/cpu_ternary_vllm/logs
BENCH=$(dirname "$0")/bench_model.py
TAG=$(basename "$PREFIX")
# Stamped per invocation: a rerun must not clobber the log of the attempt that
# failed, which is usually the one worth reading.
STAMP=$(date +%Y%m%d-%H%M%S)
log_for() { echo "$LOGS/cmp_${TAG}_${1}_${2}_${STAMP}.log"; }

cleanup() {
  # Never match this script itself: key on the python entry point only.
  local pids
  pids=$(ps -u "$USER" -o pid=,cmd= | awk '/[b]ench_model\.py|[E]ngineCore/ {print $1}')
  [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null
  python3 - "$MODEL" "$(dirname "$PREFIX")" <<'PY'
import glob, os, sys
paths = set()
for d in sys.argv[1:]:
    paths |= set(glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True))
    paths |= set(glob.glob(os.path.join(d, "*.safetensors")))
for f in paths:
    fd = os.open(f, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
PY
}

free_frac() {
  python3 -c 'import torch; f,t = torch.xpu.mem_get_info(); print(f"{f/t:.3f} {f/2**30:.1f}")'
}

cleanup
read -r frac freegb <<<"$(free_frac)"
if [[ -z "$UTIL" ]]; then
  # Leave a margin below what is free; both arms get the same value.
  UTIL=$(python3 -c "print(f'{max(0.20, float('$frac')*0.75):.2f}')")
fi
echo "free=${freegb} GiB (${frac} of total) -> gpu_memory_utilization=${UTIL}"
echo

for r in $(seq 1 "$REPS"); do
  for m in $METHODS; do
    cleanup
    log=$(log_for "$m" "$r")
    XETLA_QUANT_METHOD=$m \
    XETLA_PREQUANT_PATH=$PREFIX.xetla-$m.safetensors \
    python "$BENCH" --model "$MODEL" --tokenizer "$MODEL" \
      --dtype float16 --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" \
      --max-num-batched-tokens "$MAXLEN" \
      --max-tokens 256 --temperature 0 --text-only --full --cudagraph-sizes 1 \
      --prompt "$PROMPT" > "$log" 2>&1
    printf '%-12s rep%s  ' "$m" "$r"
    grep -E '^decode ' "$log" | tr -s ' ' \
      || grep -m1 -oE 'Available KV cache memory: [-0-9.]+ GiB|less than desired GPU memory utilization|Killed' "$log" \
      || echo "(no result)"
  done
done

echo
echo "--- output text parity ---"
for m in $METHODS; do
  awk '/^prompt +:/{f=1} f' "$(log_for "$m" 1)" 2>/dev/null \
    | grep -vE '^(INFO|WARNING|Processed|Rendering|TTFT|decode|device|engine|warmup|quantization|model |tensor|pipeline|repetition|-{10,}|={10,})' \
    > "/tmp/cmp_${TAG}_$m.txt"
done
if [[ ! -s /tmp/cmp_${TAG}_int2_f16.txt || ! -s /tmp/cmp_${TAG}_bitcos_f16.txt ]]; then
  # An empty side means that arm never produced output; "identical" would lie.
  echo "SKIPPED (an arm produced no output)"
elif cmp -s "/tmp/cmp_${TAG}_int2_f16.txt" "/tmp/cmp_${TAG}_bitcos_f16.txt"; then
  echo "IDENTICAL"
else
  diff "/tmp/cmp_${TAG}_int2_f16.txt" "/tmp/cmp_${TAG}_bitcos_f16.txt" | head
fi
echo "logs: $LOGS/cmp_${TAG}_*_${STAMP}.log"
