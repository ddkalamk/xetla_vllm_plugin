#!/usr/bin/env bash
# Serve or benchmark an int2 CAT-Q model across several B70s.
#
#   ./scripts/run_multi_gpu.sh --model <export_dir> --sidecar <file> --pp 4 --graphs
#   ./scripts/run_multi_gpu.sh --model ... --sidecar ... --pp 4 --graphs --serve
#
# Prefer pipeline parallel with graphs.
#
# Why pipeline wins. At batch 1 an all-reduce costs ~30 us no matter how
# small the payload, so tp pays 2 x layers of pure latency per token.
# Pipeline instead hands one [1, hidden] tensor to the next stage per hop.
# See scripts/perf_analysis.py and scripts/comm_analysis.py.
#
# Graphs need pure pipeline. With tp > 1 the collectives sit inside the
# captured region and vllm refuses to capture. With pure pp there is nothing
# collective to capture, and graphs remove the ~42% of each token that is
# dispatch overhead rather than gemm. This needs the
# two xpu fixes carried in vllm.patch.
#
# Sizing
#   Weights are the sidecar size divided by the number of cards, plus a dense
#   lm_head and embedding (~1.2 GiB each for a 152k vocab). A B70 has 30.3 GiB
#   and the KV cache needs whatever is left.
#
#     model                  sidecar    2 cards  3 cards  4 cards
#     CAT-Q Qwen3-8B          1.8 GiB     0.9      0.6      0.5
#     CAT-Q Qwen3-32B         7.8 GiB     3.9      2.6      2.0
#
# Choosing tp
#   Scales are stored per 128 elements along K, so any dimension tp splits has
#   to stay a multiple of 128 afterwards (e.g. the MLP intermediate size, the
#   K of down_proj). The plugin raises rather than silently mis-slicing.
#   Pipeline parallel has no such constraint since it splits whole layers,
#   which is the other reason to reach for it first.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )

MODEL="" ; SIDECAR="" ; TP=1 ; PP=1 ; SERVE=0 ; MAXLEN=2048 ; MAXTOK=80
PROMPT="What is 2+2?"
GRAPHS="${VLLM_XPU_ENABLE_XPU_GRAPH:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --sidecar) SIDECAR="$2"; shift 2 ;;
        --tp)      TP="$2"; shift 2 ;;
        --pp)      PP="$2"; shift 2 ;;
        --serve)   SERVE=1; shift ;;
        --graphs)  GRAPHS=1; shift ;;
        --max-model-len) MAXLEN="$2"; shift 2 ;;
        --max-tokens)    MAXTOK="$2"; shift 2 ;;
        --prompt)  PROMPT="$2"; shift 2 ;;
        -h|--help) sed -n '2,34p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${MODEL}" || -z "${SIDECAR}" ]]; then
    echo "usage: $0 --model <export_dir> --sidecar <file> [--pp N] [--tp N]" \
         "[--serve] [--graphs]" >&2
    exit 2
fi

# No `set -u` while the vendor env scripts run; they trip over unbound vars.
set +eu
source /swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh >/dev/null 2>&1
source /swtools/intel/2026.0/oneapi-vars.sh --force >/dev/null 2>&1
source "${ROOT_DIR}/.venv/bin/activate"
set -eu

# Expose exactly tp*pp devices. ONEAPI_DEVICE_SELECTOR pins a single device and
# would leave the other ranks with nothing, so select by affinity mask instead.
WORLD=$((TP * PP))
MASK=$(seq -s, 0 $((WORLD - 1)))
export ZE_AFFINITY_MASK="${MASK}"
unset ONEAPI_DEVICE_SELECTOR || true

export TERNSYCL_QUANT_METHOD="${TERNSYCL_QUANT_METHOD:-int2_f16}"
# CAT-Q keeps lm_head and the embeddings in full precision; quantizing them at
# load time destroys the model.
export TERNSYCL_QUANTIZE_LM_HEADS="${TERNSYCL_QUANTIZE_LM_HEADS:-0}"
export TERNSYCL_PREQUANT_PATH="${SIDECAR}"
export VLLM_XPU_ENABLE_XPU_GRAPH="${GRAPHS}"

echo "[run] model    : ${MODEL}"
echo "[run] sidecar  : ${SIDECAR} ($(du -h "${SIDECAR}" | cut -f1))"
echo "[run] pp x tp  : ${PP} x ${TP} = ${WORLD} cards (ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK})"
echo "[run] xpu graph: ${VLLM_XPU_ENABLE_XPU_GRAPH}"
if [[ "${GRAPHS}" == "1" && "${TP}" -gt 1 ]]; then
    echo "[run] warning  : graph capture is disabled when tp > 1 (collectives" \
         "live inside the captured region); use pure --pp for graphs"
fi

cd "${ROOT_DIR}"
if [[ "${SERVE}" == "1" ]]; then
    DEMO_MODEL="${MODEL}" DEMO_TEXT_ONLY=1 DEMO_MAX_MODEL_LEN="${MAXLEN}" \
    DEMO_TENSOR_PARALLEL_SIZE="${TP}" DEMO_PIPELINE_PARALLEL_SIZE="${PP}" \
    PORT="${PORT:-8000}" \
        bash demo/serve.sh
else
    python scripts/bench_model.py \
        --model "${MODEL}" --tensor-parallel-size "${TP}" \
        --pipeline-parallel-size "${PP}" --repetition-penalty 1.1 \
        --max-model-len "${MAXLEN}" --max-tokens "${MAXTOK}" --prompt "${PROMPT}"
fi
