#!/bin/bash
# BONSAI2.md steps 1-3 (build, download, pack) in one go, in an empty folder.
#   bash utils/bonsai2_from_scratch.sh /path/to/empty/folder
# Env: PLUGIN_REPO (default GitHub), PLUGIN_BRANCH, HF_TOKEN not needed (public repos).
# Afterwards run step 4 of BONSAI2.md (needs the GPU).
set -eo pipefail
DEST=${1:?usage: $0 <empty destination folder>}
mkdir -p "$DEST"; DEST=$(cd "$DEST" && pwd)
PLUGIN_REPO=${PLUGIN_REPO:-https://github.com/ddkalamk/xetla_vllm_plugin.git}
PLUGIN_BRANCH=${PLUGIN_BRANCH:-feature/bitcos-int2-integration}
export PATH="$HOME/.local/bin:$PATH"
# the Intel env scripts reference unset vars; keep set -u off around them
source /swtools/intel-gpu/latest/intel_gpu_vars.sh >/dev/null 2>&1 || true
source /swtools/intel/2026.0/oneapi-vars.sh --force >/dev/null 2>&1 || true
set -u
command -v icpx >/dev/null || { echo "icpx not found: source the oneAPI env first"; exit 1; }
icpx --version | head -1

echo "=== step 1: build (vLLM XPU + plugin)"
[[ -d "$DEST/xetla_vllm_plugin/.git" ]] || \
  git clone -b "$PLUGIN_BRANCH" --recurse-submodules "$PLUGIN_REPO" "$DEST/xetla_vllm_plugin"
PLUGIN_BRANCH="$PLUGIN_BRANCH" PLUGIN_REPO="$PLUGIN_REPO" \
  bash "$DEST/xetla_vllm_plugin/utils/setup_fresh.sh" "$DEST"

echo "=== step 2: download"
source "$DEST/xetla_vllm_plugin/.venv/bin/activate"
export MODELS="$DEST/models"; mkdir -p "$MODELS"
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
    Ternary-Bonsai-2-27B-PQ2_0.gguf Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
    --local-dir "$MODELS/Ternary-Bonsai-2-27B-gguf"
hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit \
    config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
    --local-dir "$MODELS/Ternary-Bonsai-2-27B-ref"
[[ -d "$DEST/xetla_vllm_plugin/third_party/llama.cpp-prism" ]] || \
  git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp \
    "$DEST/xetla_vllm_plugin/third_party/llama.cpp-prism"

echo "=== step 3: pack"
cd "$DEST/xetla_vllm_plugin"
python scripts/pack_bonsai2_gguf.py \
    --gguf    "$MODELS/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf" \
    --mmproj  "$MODELS/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-mmproj-BF16.gguf" \
    --ref-dir "$MODELS/Ternary-Bonsai-2-27B-ref" \
    --out     "$MODELS/Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors" \
    --packed  "$MODELS/Ternary-Bonsai-2-27B-packed"
echo "=== done. Next (BONSAI2.md step 4), e.g.:"
echo "  cd $DEST/xetla_vllm_plugin && MODELS=$MODELS MAXTOK=256 MAXLEN=512 bash scripts/run_bonsai2_gpu.sh <slurm-jobid> B70 'Tell me about photosynthesis in 200 words'"
