#!/usr/bin/env bash
# Turn a CAT-Q ternary checkpoint into an int2 sidecar this plugin can serve.
#
#   ./scripts/deploy_catq.sh https://huggingface.co/IntelLabsChina/CAT-Q/tree/main/qwen3-32B
#
# CAT-Q publishes learned modulation parameters, not weights. Three stages:
#
#   1. download   parameters.pth + config.yaml from the CAT-Q repo
#   2. export     merge them into the base model and save a fake-quantized HF
#                 checkpoint (ternary values stored dense in fp16)
#   3. pack       squeeze the ternary tensors into the int2 sidecar
#
# Stages are skippable so a failed run can resume:
#   --skip-download / --skip-export / --skip-pack
#
# Notes
#   * Export is CPU and RAM bound, not GPU bound. A 32B needs ~70 GB resident,
#     so run it somewhere with the memory - the login node, not a compute node
#     that is already serving. The 30B MoE export was OOM-killed on a 94 GB node.
#   * use_bfloat16 is forced off: the plugin's kernels are fp16, and packing a
#     bf16 export loses the low mantissa bits of the scales.
#   * Serving a CAT-Q model needs XETLA_QUANTIZE_LM_HEADS=0. Its embeddings and
#     lm_head are not ternary, and ternarizing them destroys the model. The
#     packer already leaves them out of the sidecar; this stops the plugin
#     quantizing them on the fly.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
CATQ_DIR="${CATQ_DIR:-${ROOT_DIR}/BitTern/projects/cat-q}"
SIDECAR_DIR="${SIDECAR_DIR:-${ROOT_DIR}/..}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

do_download=1; do_export=1; do_pack=1
URL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download) do_download=0 ;;
        --skip-export)   do_export=0 ;;
        --skip-pack)     do_pack=0 ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
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

cat <<EOF

[deploy] done.

  export : ${EXPORT_DIR}
  sidecar: ${SIDECAR}

Serve it:

  XETLA_QUANT_METHOD=int2_f16 \\
  XETLA_QUANTIZE_LM_HEADS=0 \\
  XETLA_PREQUANT_PATH=${SIDECAR} \\
  DEMO_MODEL=${EXPORT_DIR} \\
  DEMO_TEXT_ONLY=1 PORT=8000 ./demo/serve.sh

Sanity-check it first - a CAT-Q run can converge to a model that repeats
forever, and that is visible only in a long generation:

  ${PYTHON} tests/dense_cpu_reference.py --model ${EXPORT_DIR} \\
      --prompt "What is 2+2?" --max-new-tokens 60
EOF
