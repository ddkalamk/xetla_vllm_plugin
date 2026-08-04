#!/usr/bin/env bash
# Serve or benchmark an int2 CAT-Q model across several B70s.
#
#   ./scripts/run_multi_gpu.sh --model <export_dir> --sidecar <file> --tp 4
#   ./scripts/run_multi_gpu.sh --model ... --sidecar ... --tp 4 --serve
#
# Defaults to a short benchmark; --serve starts the chat demo instead.
#
# Sizing
#   Weights are the sidecar size divided by tp, plus a dense lm_head and
#   embedding (~1.2 GiB each for a 152k vocab) that are not sharded by the
#   sidecar. A B70 has 30.3 GiB, and the KV cache needs whatever is left, so
#   aim for sidecar/tp under about 20 GiB.
#
#     model                  sidecar   tp=2     tp=4     tp=8
#     CAT-Q Qwen3-8B          1.8 GiB   0.9      0.5      0.2
#     CAT-Q Qwen3-32B         7.8 GiB   3.9      2.0      1.0
#     CAT-Q Qwen3-30B-A3B     7.9 GiB   4.0      2.0      1.0
#     CAT-Q Qwen3-235B-A22B    58 GiB    29 x     15       7.3
#
#   The 235B at tp=2 was measured: it dies during load with a level_zero
#   UR_RESULT_ERROR_DEVICE_LOST rather than a clean OOM. tp=4 is the smallest
#   working layout, at 27.5 GiB of 30.3 GiB in use.
#
# Choosing tp
#   Scales are stored per 128 elements along K, so any dimension the shard
#   splits has to stay a multiple of 128 afterwards. For MoE that is the expert
#   intermediate size:
#
#     moe_intermediate 1536 (235B):  tp=2 -> 768  tp=4 -> 384  tp=8 -> 192 (bad)
#     moe_intermediate  768 (30B):   tp=2 -> 384  tp=4 -> 192 (bad)
#
#   The plugin raises rather than silently mis-slicing. Going wider than the
#   limit needs expert parallelism, which keeps whole experts per rank.
#
#   Note tp costs throughput: it adds an all-reduce per layer and shrinks the
#   expert GEMVs, which are already launch-bound. The 30B measured 62 tok/s on
#   one card against 22.7 on two. Use tp for capacity, not speed.
#
#   --graphs is a no-op whenever tp > 1: vLLM logs "XPU Graph doesn't support
#   capture communication ops, disabling cudagraph_mode" and falls back to
#   eager, because the collectives cannot be captured. The 2.6x that graphs buy
#   on a single card is therefore not available under tp today.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )

MODEL="" ; SIDECAR="" ; TP=2 ; SERVE=0 ; MAXLEN=2048 ; MAXTOK=80
PROMPT="What is 2+2?"
GRAPHS="${VLLM_XPU_ENABLE_XPU_GRAPH:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --sidecar) SIDECAR="$2"; shift 2 ;;
        --tp)      TP="$2"; shift 2 ;;
        --serve)   SERVE=1; shift ;;
        --graphs)  GRAPHS=1; shift ;;
        --max-model-len) MAXLEN="$2"; shift 2 ;;
        --max-tokens)    MAXTOK="$2"; shift 2 ;;
        --prompt)  PROMPT="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${MODEL}" || -z "${SIDECAR}" ]]; then
    echo "usage: $0 --model <export_dir> --sidecar <file> --tp N [--serve] [--graphs]" >&2
    exit 2
fi

# No `set -u` while the vendor env scripts run; they trip over unbound vars.
set +eu
source /swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh >/dev/null 2>&1
source /swtools/intel/2025.3/oneapi-vars.sh --force >/dev/null 2>&1
source "${ROOT_DIR}/.venv/bin/activate"
set -eu

# Expose exactly tp devices. ONEAPI_DEVICE_SELECTOR pins a single device and
# would leave the other ranks with nothing, so select by affinity mask instead.
MASK=$(seq -s, 0 $((TP - 1)))
export ZE_AFFINITY_MASK="${MASK}"
unset ONEAPI_DEVICE_SELECTOR || true

export XETLA_QUANT_METHOD="${XETLA_QUANT_METHOD:-int2_f16}"
# CAT-Q keeps lm_head and the embeddings in full precision; quantizing them at
# load time destroys the model.
export XETLA_QUANTIZE_LM_HEADS="${XETLA_QUANTIZE_LM_HEADS:-0}"
export XETLA_PREQUANT_PATH="${SIDECAR}"
export VLLM_XPU_ENABLE_XPU_GRAPH="${GRAPHS}"

echo "[run] model    : ${MODEL}"
echo "[run] sidecar  : ${SIDECAR} ($(du -h "${SIDECAR}" | cut -f1))"
echo "[run] tp       : ${TP}  (ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK})"
echo "[run] xpu graph: ${VLLM_XPU_ENABLE_XPU_GRAPH}"
if [[ "${GRAPHS}" == "1" && "${TP}" -gt 1 ]]; then
    echo "[run] warning  : vllm disables graph capture when tp > 1 (collectives)"
fi

cd "${ROOT_DIR}"
if [[ "${SERVE}" == "1" ]]; then
    DEMO_MODEL="${MODEL}" DEMO_TEXT_ONLY=1 DEMO_MAX_MODEL_LEN="${MAXLEN}" \
    DEMO_TENSOR_PARALLEL_SIZE="${TP}" PORT="${PORT:-8000}" \
        bash demo/serve.sh
else
    python scripts/bench_model.py \
        --model "${MODEL}" --tensor-parallel-size "${TP}" \
        --max-model-len "${MAXLEN}" --max-tokens "${MAXTOK}" --prompt "${PROMPT}"
fi
