#!/usr/bin/env bash
# Turn a CAT-Q ternary checkpoint into an int2 sidecar this plugin can serve,
# then run it.
#
#   ./scripts/deploy_catq.sh https://huggingface.co/IntelLabsChina/CAT-Q/tree/main/qwen3-1.7b
#
# CAT-Q publishes learned modulation parameters, not weights. Four stages:
#
#   1. download   parameters.pth + config.yaml from the CAT-Q repo
#   2. export     merge them into the base model and save a fake-quantized HF
#                 checkpoint (ternary values stored dense in fp16)
#   3. pack       squeeze the ternary tensors into the int2 sidecar
#   4. run        decode a sample prompt and report tok/s
#
# The export stage runs CAT-Q's own main.py from IntelChina-AI/BitTern, which
# is cloned on first use (it is not vendored: its configs/ dir is where the
# exports land, hundreds of GB, so it has to stay untracked). Point CATQ_DIR
# at an existing checkout to skip the clone.
#
# Stages are skippable so a failed run can resume:
#   --skip-download / --skip-export / --skip-pack / --skip-run
#
# Serving defaults, measured rather than assumed:
#   * XPU graphs (VLLM_XPU_ENABLE_XPU_GRAPH=1) remove kernel dispatch overhead,
#     which is most of a token at batch 1. The win scales with how many ops a
#     layer issues, so it is largest on MoE: 2.6x on Qwen3-30B-A3B, 2.2x on
#     235B-A22B under pp=4, but only 1.20x on the dense 1.7B (301 vs 251 tok/s)
#     where there is less dispatch to remove. Pass --no-graphs to compare.
#   * repetition_penalty 1.1. Without it these checkpoints answer and then
#     repeat the last sentence forever. 1.05 collapsed a long answer into
#     "000000", so do not tune it blind.
#   * XETLA_QUANTIZE_LM_HEADS=0. CAT-Q embeddings and lm_head are not ternary
#     and ternarizing them destroys the model. The packer already leaves them
#     out; this stops the plugin quantizing them on the fly.
#
# Multi-card (only needed when the sidecar does not fit in 30.3 GiB) lives in
# scripts/run_multi_gpu.sh. Prefer pipeline over tensor parallel there: it
# keeps graphs usable and costs far less at batch 1.
#
# Notes
#   * Export is CPU and RAM bound, not GPU bound. A 32B needs ~70 GB resident,
#     so run it somewhere with the memory - the login node, not a compute node
#     that is already serving. The 30B MoE export was OOM-killed on a 94 GB node.
#   * use_bfloat16 is forced off: the plugin's kernels are fp16, and packing a
#     bf16 export loses the low mantissa bits of the scales.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
CATQ_DIR="${CATQ_DIR:-${ROOT_DIR}/BitTern/projects/cat-q}"
SIDECAR_DIR="${SIDECAR_DIR:-${ROOT_DIR}/..}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

do_download=1; do_export=1; do_pack=1; do_run=1; graphs=1
MAXTOK="${MAXTOK:-256}"
PROMPT="${PROMPT:-Tell me about CPU caches}"
URL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download) do_download=0 ;;
        --skip-export)   do_export=0 ;;
        --skip-pack)     do_pack=0 ;;
        --skip-run)      do_run=0 ;;
        --no-graphs)     graphs=0 ;;
        --max-tokens)    MAXTOK="$2"; shift ;;
        --prompt)        PROMPT="$2"; shift ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) URL="$1" ;;
    esac
    shift
done

if [[ -z "${URL}" ]]; then
    echo "usage: $0 <huggingface tree URL> [--skip-download|--skip-export|--skip-pack]" >&2
    echo "  e.g. $0 https://huggingface.co/IntelLabsChina/CAT-Q/tree/main/qwen3-32B" >&2
    exit 2
fi

# https://huggingface.co/<owner>/<repo>/tree/<rev>/<subfolder>
STRIPPED="${URL#https://huggingface.co/}"
REPO_ID="$(echo "${STRIPPED}" | cut -d/ -f1,2)"
SUBFOLDER="$(echo "${STRIPPED}" | sed -E 's#^[^/]+/[^/]+/tree/[^/]+/?##')"
if [[ -z "${REPO_ID}" || -z "${SUBFOLDER}" ]]; then
    echo "could not parse repo/subfolder out of: ${URL}" >&2
    exit 2
fi

CFG_DIR="${CATQ_DIR}/configs/${SUBFOLDER}"
EXPORT_DIR="${CFG_DIR}/export"

echo "[deploy] repo      : ${REPO_ID}"
echo "[deploy] subfolder : ${SUBFOLDER}"
echo "[deploy] config dir: ${CFG_DIR}"

# ---- 0. CAT-Q source ------------------------------------------------------
# The export stage runs CAT-Q's own main.py, which lives in the BitTern repo.
# It is not vendored here (its configs/ dir is where the multi-hundred-GB
# exports land, so it must stay untracked), so fetch it on first use.
BITTERN_REPO="${BITTERN_REPO:-https://github.com/IntelChina-AI/BitTern.git}"
if [[ ! -f "${CATQ_DIR}/main.py" ]]; then
    BITTERN_DIR="$(dirname "$(dirname "${CATQ_DIR}")")"
    echo "[deploy] --- fetching CAT-Q source (${BITTERN_REPO}) ---"
    git clone --depth 1 "${BITTERN_REPO}" "${BITTERN_DIR}"
    [[ -f "${CATQ_DIR}/main.py" ]] || {
        echo "[deploy] ERROR: ${CATQ_DIR}/main.py still missing after clone." >&2
        echo "[deploy]        Set CATQ_DIR to an existing checkout instead." >&2
        exit 6
    }
fi

# CAT-Q's pyproject pins torch==2.4.0 / transformers==4.51.0, which would tear
# out the xpu torch and vllm this plugin is built against. So never install it
# as a package: just add the handful of modules its export path imports, with
# everything already present pinned to the installed version.
MISSING=()
for mod in accelerate lm_eval yaml sentencepiece; do
    "${PYTHON}" -c "import ${mod}" 2>/dev/null || MISSING+=("${mod}")
done
if (( ${#MISSING[@]} )); then
    echo "[deploy] --- installing CAT-Q imports: ${MISSING[*]} ---"
    PIP_PKGS=("${MISSING[@]/#yaml/PyYAML}")
    PIP_PKGS=("${PIP_PKGS[@]/#lm_eval/lm_eval}")
    CONSTRAINTS="$(mktemp)"
    "${PYTHON}" - <<'PY' > "${CONSTRAINTS}"
import importlib.metadata as md
for p in ("torch", "transformers", "vllm", "numpy", "datasets", "tokenizers",
          "huggingface-hub", "safetensors"):
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        pass
PY
    "${PYTHON}" -m pip install -q "${PIP_PKGS[@]}" -c "${CONSTRAINTS}"
    rm -f "${CONSTRAINTS}"
fi

# ---- 1. download ----------------------------------------------------------
if [[ "${do_download}" == "1" ]]; then
    echo "[deploy] --- downloading CAT-Q parameters ---"
    mkdir -p "${CFG_DIR}"
    REPO_ID="${REPO_ID}" SUBFOLDER="${SUBFOLDER}" CFG_DIR="${CFG_DIR}" \
    "${PYTHON}" - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download

repo, sub, dest = os.environ["REPO_ID"], os.environ["SUBFOLDER"], os.environ["CFG_DIR"]
for name in ("config.yaml", "parameters.pth"):
    target = os.path.join(dest, name)
    if os.path.exists(target):
        print(f"[deploy] have {name}, skipping")
        continue
    print(f"[deploy] fetching {sub}/{name}", flush=True)
    src = hf_hub_download(repo_id=repo, filename=f"{sub}/{name}")
    # Copy out of the blob cache so the config dir is self-contained.
    shutil.copyfile(src, target)
    print(f"[deploy]   -> {target} ({os.path.getsize(target) / 2**30:.2f} GiB)")
PY
fi

if [[ ! -f "${CFG_DIR}/config.yaml" ]]; then
    echo "[deploy] ERROR: ${CFG_DIR}/config.yaml missing" >&2
    exit 3
fi

BASE_MODEL="$(sed -nE 's/^model:[[:space:]]*(.*)$/\1/p' "${CFG_DIR}/config.yaml" | tr -d '"'"'"' ')"
MODEL_NAME="${BASE_MODEL##*/}"
SIDECAR="${SIDECAR_DIR}/CAT-Q-${MODEL_NAME}.xetla-int2_f16.safetensors"
echo "[deploy] base model: ${BASE_MODEL}"
echo "[deploy] sidecar   : ${SIDECAR}"

# ---- 2. export ------------------------------------------------------------
# CAT-Q's own export_model.sh reads config.yaml and grabs a GPU. We use a fp16
# variant of the config and skip the GPU lock, since the merge is a CPU job.
if [[ "${do_export}" == "1" ]]; then
    echo "[deploy] --- exporting fake-quantized HF checkpoint ---"
    FP16_CFG="${CFG_DIR}/config_fp16.yaml"
    sed -E 's/^use_bfloat16:.*/use_bfloat16: false/' "${CFG_DIR}/config.yaml" > "${FP16_CFG}"
    grep -E '^(model|wbits|abits|group_size|use_bfloat16):' "${FP16_CFG}" | sed 's/^/[deploy]   /'

    mkdir -p "${EXPORT_DIR}"
    ( cd "${CATQ_DIR}" && "${PYTHON}" main.py \
        --config "${FP16_CFG}" \
        --output_dir "${EXPORT_DIR}" \
        --export_model_path "${EXPORT_DIR}" \
        --checkpoint "${CFG_DIR}/parameters.pth" )
    echo "[deploy] export size: $(du -sh "${EXPORT_DIR}" | cut -f1)"
fi

if [[ ! -f "${EXPORT_DIR}/config.json" ]]; then
    echo "[deploy] ERROR: no exported model at ${EXPORT_DIR}" >&2
    exit 4
fi

# ---- 3. pack --------------------------------------------------------------
if [[ "${do_pack}" == "1" ]]; then
    echo "[deploy] --- packing int2 sidecar ---"
    "${PYTHON}" "${ROOT_DIR}/scripts/pack_bonsai_hf.py" \
        --model "${EXPORT_DIR}" --out "${SIDECAR}"
    ls -lh "${SIDECAR}"
fi

# ---- 4. run ---------------------------------------------------------------
if [[ "${do_run}" == "1" ]]; then
    echo "[deploy] --- decoding ${MAXTOK} tokens ---"
    if [[ ! -f "${SIDECAR}" ]]; then
        echo "[deploy] ERROR: no sidecar at ${SIDECAR}" >&2
        exit 5
    fi

    # The vendor env scripts assume a permissive shell.
    set +eu
    GPU_VARS="${INTEL_GPU_VARS:-/swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh}"
    ONEAPI="${ONEAPI_VARS:-/swtools/intel/2025.3/oneapi-vars.sh}"
    [[ -f "${GPU_VARS}" ]] && source "${GPU_VARS}" >/dev/null 2>&1
    [[ -f "${ONEAPI}" ]] && source "${ONEAPI}" --force >/dev/null 2>&1
    set -eu

    export XETLA_QUANT_METHOD=int2_f16
    export XETLA_QUANTIZE_LM_HEADS=0
    export XETLA_PREQUANT_PATH="${SIDECAR}"
    export VLLM_XPU_ENABLE_XPU_GRAPH="${graphs}"
    export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
    export VLLM_ENABLE_V1_MULTIPROCESSING=0

    echo "[deploy] xpu graphs: ${VLLM_XPU_ENABLE_XPU_GRAPH}"
    "${PYTHON}" "${ROOT_DIR}/scripts/bench_model.py" \
        --model "${EXPORT_DIR}" \
        --max-tokens "${MAXTOK}" --repetition-penalty 1.1 --full \
        --cudagraph-sizes 1,2,4,8 \
        --prompt "${PROMPT}"
fi

cat <<EOF

[deploy] done.

  export : ${EXPORT_DIR}
  sidecar: ${SIDECAR}

Chat with it:

  XETLA_QUANT_METHOD=int2_f16 XETLA_QUANTIZE_LM_HEADS=0 \\
  XETLA_PREQUANT_PATH=${SIDECAR} \\
  DEMO_MODEL=${EXPORT_DIR} DEMO_TEXT_ONLY=1 \\
  VLLM_XPU_ENABLE_XPU_GRAPH=1 PORT=8000 ./demo/serve.sh

A CAT-Q run can converge to a model that repeats forever. The run stage above
prints a repetition report (unique words vs total, and any repeated block) so
that shows up without reading the whole generation.
EOF
