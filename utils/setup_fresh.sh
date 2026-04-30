#!/usr/bin/env bash
# From-scratch build of xetla_vllm_plugin + vendored vllm + xetla kernels
# in a clean directory using a fresh venv.
#
# Usage:
#   ./utils/setup_fresh.sh /path/to/empty/build/dir
#
# Or if invoked directly inside the destination dir, pass "." to use $PWD.
#
# Requirements (must be loaded BEFORE running this script):
#   - source /swtools/intel-gpu/<ver>/intel_gpu_vars.sh
#   - source /swtools/intel/<oneapi>/oneapi-vars.sh --force
#   - python >= 3.10 in PATH (uv will prefer 3.12)
#   - git, uv (or pip; uv is preferred)
#
# What this script does:
#   1) Clones xetla_vllm_plugin (with the xetla submodule) into <DEST>/xetla_vllm_plugin
#   2) Creates a fresh venv at <DEST>/xetla_vllm_plugin/.venv (Python 3.12 via uv)
#   3) Clones ddkalamk/vllm (xetla_v0.19.0 branch) into <DEST>/xetla_vllm_plugin/vllm
#      and (best-effort) applies the vendored vllm.patch on top
#   4) Installs vllm (XPU target) and triton-xpu into the venv
#   5) Builds the xetla plugin (PyTorch SYCL extension) into the venv

set -euo pipefail

err()  { printf '\033[31m[setup_fresh] %s\033[0m\n' "$*" >&2; }
log()  { printf '\033[36m[setup_fresh] %s\033[0m\n' "$*"; }

# ---- 0. validate environment & dest dir -------------------------------------
DEST="${1:-}"
if [[ -z "$DEST" ]]; then
    err "Usage: $0 <destination-directory>"
    exit 1
fi
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
log "Destination: $DEST"

if ! command -v icpx >/dev/null; then
    err "icpx not in PATH; source intel_gpu_vars.sh and oneapi-vars.sh first."
    exit 1
fi
log "icpx: $(icpx --version 2>&1 | head -1)"

if ! command -v uv >/dev/null; then
    log "uv not found; bootstrapping into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

PLUGIN_REPO="${PLUGIN_REPO:-https://github.com/ddkalamk/xetla_vllm_plugin.git}"
PLUGIN_BRANCH="${PLUGIN_BRANCH:-feature/int2-fp16-bonsai-chat}"

VLLM_REPO="${VLLM_REPO:-https://github.com/ddkalamk/vllm.git}"
VLLM_BRANCH="${VLLM_BRANCH:-xetla_v0.19.0}"

# Optional: override the xetla submodule URL (useful for local file:// builds
# when the plugin's .gitmodules points to a remote that doesn't yet have the
# required commits).
XETLA_SUBMODULE_REPO="${XETLA_SUBMODULE_REPO:-}"

# Fallback branch to check out if the pinned submodule commit is no longer
# fetchable from the remote (the upstream fork sometimes rewrites history).
# Empty => no fallback (fail hard like before).
XETLA_SUBMODULE_FALLBACK_BRANCH="${XETLA_SUBMODULE_FALLBACK_BRANCH:-feature_int2_woq_f16_act_gs128}"

# ---- 1. clone plugin --------------------------------------------------------
PLUGIN_DIR="$DEST/xetla_vllm_plugin"
if [[ ! -d "$PLUGIN_DIR/.git" ]]; then
    log "Cloning $PLUGIN_REPO ($PLUGIN_BRANCH) -> $PLUGIN_DIR"
    git clone -b "$PLUGIN_BRANCH" "$PLUGIN_REPO" "$PLUGIN_DIR"
fi

if [[ -n "$XETLA_SUBMODULE_REPO" ]]; then
    log "Overriding xetla submodule URL -> $XETLA_SUBMODULE_REPO"
    git -C "$PLUGIN_DIR" config -f .gitmodules submodule.xetla.url "$XETLA_SUBMODULE_REPO"
    git -C "$PLUGIN_DIR" submodule sync xetla
fi

log "Initialising xetla submodule"
if ! git -C "$PLUGIN_DIR" -c protocol.file.allow=always submodule update --init --recursive xetla; then
    if [[ -n "$XETLA_SUBMODULE_FALLBACK_BRANCH" ]]; then
        XETLA_URL=$(git -C "$PLUGIN_DIR" config -f .gitmodules submodule.xetla.url)
        err "Pinned xetla submodule commit not fetchable; falling back to branch '$XETLA_SUBMODULE_FALLBACK_BRANCH' from $XETLA_URL"
        rm -rf "$PLUGIN_DIR/xetla" "$PLUGIN_DIR/.git/modules/xetla"
        git clone -b "$XETLA_SUBMODULE_FALLBACK_BRANCH" "$XETLA_URL" "$PLUGIN_DIR/xetla"
        log "xetla now at $(git -C "$PLUGIN_DIR/xetla" rev-parse HEAD) (branch $XETLA_SUBMODULE_FALLBACK_BRANCH)"
    else
        err "xetla submodule init failed and no XETLA_SUBMODULE_FALLBACK_BRANCH set"
        exit 1
    fi
fi

cd "$PLUGIN_DIR"

# ---- 2. fresh venv ----------------------------------------------------------
if [[ ! -d "$PLUGIN_DIR/.venv" ]]; then
    log "Creating venv (.venv) with Python 3.12 via uv"
    uv venv --python 3.12 --seed --managed-python
fi
# shellcheck disable=SC1091
source "$PLUGIN_DIR/.venv/bin/activate"
python -V

# ---- 3. vllm (vendored) -----------------------------------------------------
VLLM_DIR="$PLUGIN_DIR/vllm"
if [[ ! -d "$VLLM_DIR/.git" ]]; then
    # If the plugin already ships a vendored vllm dir from cloning (it can,
    # depending on how the upstream tracks it), wipe and re-clone cleanly.
    rm -rf "$VLLM_DIR"
    log "Cloning $VLLM_REPO ($VLLM_BRANCH) -> $VLLM_DIR"
    git clone -b "$VLLM_BRANCH" "$VLLM_REPO" "$VLLM_DIR"
else
    log "vllm already present; leaving as-is"
fi

# Apply vendored vllm.patch (best-effort): the plugin tree carries vllm.patch
# at vllm/vllm.patch (next to its vllm/ snapshot). Apply it on top of the
# cloned upstream vllm.
PATCH_FILE="$PLUGIN_DIR/vllm/vllm.patch"
if [[ ! -f "$PATCH_FILE" ]]; then
    # Some plugin checkouts may keep it at the repo root.
    [[ -f "$PLUGIN_DIR/vllm.patch" ]] && PATCH_FILE="$PLUGIN_DIR/vllm.patch"
fi
if [[ -f "$PATCH_FILE" ]]; then
    cp "$PATCH_FILE" "$VLLM_DIR/vllm.patch"
    if (cd "$VLLM_DIR" && git apply --check vllm.patch >/dev/null 2>&1); then
        log "Applying vllm.patch on top of vendored vllm"
        (cd "$VLLM_DIR" && git apply vllm.patch)
    elif (cd "$VLLM_DIR" && git apply --reverse --check vllm.patch >/dev/null 2>&1); then
        log "vllm.patch already applied; skipping"
    else
        err "vllm.patch FAILED to apply cleanly to $VLLM_DIR (this will silently disable xetla quant hooks)"
        (cd "$VLLM_DIR" && git apply --check vllm.patch) || true
        exit 1
    fi
else
    err "FATAL: no vllm.patch found in plugin tree (looked in $PLUGIN_DIR/vllm/vllm.patch and $PLUGIN_DIR/vllm.patch); xetla quant hooks would be missing"
    exit 1
fi

# ---- 4. install vllm + triton-xpu in the venv -------------------------------
log "Installing vllm requirements (XPU)"
pip install --upgrade pip
pip install -v -r "$VLLM_DIR/requirements/xpu.txt"

log "Installing vllm (editable, VLLM_TARGET_DEVICE=xpu)"
VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e "$VLLM_DIR" -v

log "Replacing triton with triton-xpu==3.7.0"
pip uninstall -y triton triton-xpu || true
pip install triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu

# ---- 5. build the xetla plugin (PyTorch SYCL extension) ---------------------
log "Building xetla_vllm_plugin (PyTorch SYCL ext)"
cd "$PLUGIN_DIR"
python setup.py install

log "Build complete."
log ""
log "Quick sanity check:"
log "  source $PLUGIN_DIR/.venv/bin/activate"
log "  python -c 'import torch, xetla_vllm_plugin, xetla_pt_ext; print(\"ok\")'"
log ""
log "To run the Bonsai chat demo (after placing the GGUF at $PLUGIN_DIR):"
log "  cd $PLUGIN_DIR && bash scripts/chat.sh"
