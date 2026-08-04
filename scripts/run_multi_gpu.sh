#!/usr/bin/env bash
# Serve or benchmark an int2 CAT-Q model across several B70s.
#
#   ./scripts/run_multi_gpu.sh --model <export_dir> --sidecar <file> --pp 4 --graphs
#   ./scripts/run_multi_gpu.sh --model ... --sidecar ... --pp 4 --graphs --serve
#
# Prefer pipeline parallel with graphs. Measured on CAT-Q Qwen3-235B-A22B,
# 4x B70, same prompt, output byte-identical across all of them:
#
#     config              cards   mem/card   tok/s
#     tp=4                  4     27.52 GiB  11.75
#     pp=3                  3     27.84 GiB  16.59
#     pp=4                  4     25.98 GiB  17.49
#     pp=2 x tp=2           4     26.48 GiB  19.15
#     pp=4 + --graphs       4     25.96 GiB  37.80   <- 3.2x tp=4
#
# Why pipeline wins here. At batch 1 an all-reduce costs ~30 us no matter how
# small the payload, so tp pays 2 x layers of pure latency per token: 6.1 ms
# for 94 layers. Pipeline instead hands one [1, hidden] tensor to the next
# stage, 3 hops, 0.09 ms. Sharding also does not buy back the difference --
# tp=4 makes the gemms only 1.6x faster, not 4x, because the expert matrices
# are already small enough to be launch bound (expert_w2 at tp=4 reaches 4%
# of achievable bandwidth against 13% unsharded). See scripts/perf_analysis.py
# and scripts/comm_analysis.py.
#
# Graphs need pure pipeline. With tp > 1 the collectives sit inside the
# captured region and vllm refuses to capture. With pure pp there is nothing
# collective to capture, and graphs remove the ~42% of each token that is
# dispatch overhead rather than gemm (scripts/pp_budget.py). This needs the
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
#     CAT-Q Qwen3-30B-A3B     7.9 GiB     4.0      2.6      2.0
#     CAT-Q Qwen3-235B-A22B    58 GiB      29 x     19       15
#
#   The 235B on 2 cards was measured both ways, tp=2 and pp=2: both die during
#   load with level_zero UR_RESULT_ERROR_DEVICE_LOST rather than a clean OOM.
#   Three cards is the floor.
#
# Choosing tp
#   Scales are stored per 128 elements along K, so any dimension tp splits has
#   to stay a multiple of 128 afterwards. For MoE that is the expert
#   intermediate size:
#
#     moe_intermediate 1536 (235B):  tp=2 -> 768  tp=4 -> 384  tp=8 -> 192 (bad)
#     moe_intermediate  768 (30B):   tp=2 -> 384  tp=4 -> 192 (bad)
#
#   The plugin raises rather than silently mis-slicing. Pipeline parallel has
#   no such constraint since it splits whole layers, which is the other reason
#   to reach for it first.
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
        -h|--help) sed -n '2,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
source /swtools/intel/2025.3/oneapi-vars.sh --force >/dev/null 2>&1
source "${ROOT_DIR}/.venv/bin/activate"
set -eu

# Expose exactly tp*pp devices. ONEAPI_DEVICE_SELECTOR pins a single device and
# would leave the other ranks with nothing, so select by affinity mask instead.
WORLD=$((TP * PP))
MASK=$(seq -s, 0 $((WORLD - 1)))
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
