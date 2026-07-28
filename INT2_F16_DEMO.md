# Int2×fp16 Bonsai-8B demo — end-to-end guide (from scratch)

This document walks through everything needed to go from a clean Linux
machine with an Intel GPU to a running int2-quantized Bonsai-8B chat through
vLLM + the xetla plugin, including the `demo_record.py` cast capture used in
[`DEMO.md`](DEMO.md).

> **TL;DR:** load the Intel toolchain, run [`utils/setup_fresh.sh`](utils/setup_fresh.sh)
> against an empty directory, drop the GGUF and (optionally) the prequant
> sidecar next to it, then run [`scripts/chat.sh`](scripts/chat.sh).

Tested combo:
- Linux x86_64
- Intel oneAPI 2025.3 + Intel GPU driver 26.05.37020.3
- Python 3.12 (managed by `uv`)
- PyTorch 2.11.0+xpu, triton-xpu 3.7.0
- vLLM fork [`ddkalamk/vllm@xetla_v0.21.0`](https://github.com/ddkalamk/vllm/tree/xetla_v0.21.0)
- Plugin branch `feature/int2-fp16-bonsai-chat`
- GPUs verified: BMG dGPU (pcl-zen4) at full speed (~110 tok/s), BMG-XG3 / Xe2 (pcl-kini04) with workarounds

---

## 1. Prerequisites

### 1.1 OS-level requirements
- Working Intel GPU userspace + level-zero/OpenCL runtimes:
  - Driver bundle reachable at `/swtools/intel-gpu/<ver>/intel_gpu_vars.sh`
- oneAPI base toolkit reachable at `/swtools/intel/<oneapi-ver>/oneapi-vars.sh`
- `git`, `curl`, `bash`, ~30 GB free disk for the venv + vllm source build.

### 1.2 Load the Intel toolchain (every shell)
```bash
source /swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh
source /swtools/intel/2025.3/oneapi-vars.sh --force
icpx --version    # must report 2025.3.0
sycl-ls           # must show the GPU under [level_zero:gpu]
```

### 1.3 Optional: install `uv` (fast Python package manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```
(`setup_fresh.sh` will bootstrap `uv` automatically if it is not on `$PATH`.)

---

## 2. Build everything from scratch

The plugin ships [`utils/setup_fresh.sh`](utils/setup_fresh.sh) which:
1. Clones `xetla_vllm_plugin` (with the `xetla` submodule)
2. Creates a fresh venv at `<dest>/xetla_vllm_plugin/.venv` (Python 3.12 via uv)
3. Clones the vendored vllm fork and applies `vllm/vllm.patch`
4. Installs vllm (XPU target) + replaces triton with `triton-xpu==3.7.0`
5. Builds the xetla plugin (PyTorch SYCL extension)

### 2.1 Run the script

```bash
mkdir -p ~/fresh_int2_fp16_setup
bash /path/to/xetla_vllm_plugin/utils/setup_fresh.sh ~/fresh_int2_fp16_setup
```

Expected wall-clock: vllm dependency install + editable build is the dominant
cost (typically 30–60 min depending on disk and network).

After completion:
```
~/fresh_int2_fp16_setup/
└── xetla_vllm_plugin/
    ├── .venv/                # uv-managed Python 3.12 environment
    ├── vllm/                 # cloned + patched ddkalamk/vllm@xetla_v0.21.0
    ├── xetla/                # int2 kernels (submodule)
    ├── csrc/, scripts/, ...  # plugin source
    └── ...
```

### 2.2 Sanity check

```bash
cd ~/fresh_int2_fp16_setup/xetla_vllm_plugin
source .venv/bin/activate
python -c "import torch, xetla_pt_ext, xetla_vllm_plugin; \
  print('torch', torch.__version__, 'xpu=', torch.xpu.is_available())"
# Expect: torch 2.11.0+xpu xpu= True   (when run on a GPU node)
```

If `xpu=False`, you are on a login node with no GPU — that's fine, just run
the demo from a compute node (see §4).

### 2.3 Building from a local repo (no network)

If you already have the plugin checked out locally and the upstream remotes
are unreachable, point the script at file:// URLs:

```bash
export PLUGIN_REPO="file:///data/nfs_home/egeorgan/xetla_vllm_plugin"
export PLUGIN_BRANCH="feature/int2-fp16-bonsai-chat"
export VLLM_REPO="file:///data/nfs_home/egeorgan/xetla_vllm_plugin/vllm"
export VLLM_BRANCH="xetla_v0.21.0"
export XETLA_SUBMODULE_REPO="file:///data/nfs_home/egeorgan/xetla_vllm_plugin/xetla"
export GIT_ALLOW_PROTOCOL=file:https
bash /data/nfs_home/egeorgan/xetla_vllm_plugin/utils/setup_fresh.sh ~/fresh_int2_fp16_setup
```

---

## 3. Get the model + (optional) sidecar

Two files live next to each other inside `~/fresh_int2_fp16_setup/xetla_vllm_plugin/`:

| File | Purpose | Required? |
|---|---|---|
| `Ternary-Bonsai-8B-F16.gguf` | full fp16 ternary weights (~16 GB) | **yes** |
| `Ternary-Bonsai-8B-F16.gguf.xetla-int2_f16.safetensors` | pre-quantized int2 sidecar (~2 GB) | **strongly recommended** |

Without the sidecar, vLLM's first load runs the full GGUF→int2 requantize
path on the GPU (slow + uses much more memory). With the sidecar, the plugin
streams int2 codes + fp16 scales straight from disk.

```bash
# from the original tree (or wherever you have them)
ln -s /path/to/Ternary-Bonsai-8B-F16.gguf \
      ~/fresh_int2_fp16_setup/xetla_vllm_plugin/Ternary-Bonsai-8B-F16.gguf
ln -s /path/to/Ternary-Bonsai-8B-F16.gguf.xetla-int2_f16.safetensors \
      ~/fresh_int2_fp16_setup/xetla_vllm_plugin/Ternary-Bonsai-8B-F16.gguf.xetla-int2_f16.safetensors
```

`scripts/chat.sh` auto-detects the sidecar by the `<model>.xetla-${XETLA_QUANT_METHOD}.safetensors`
naming convention and exports `XETLA_PREQUANT_PATH` for you.

### 3.1 Re-creating the sidecar from the GGUF

If you only have the GGUF, generate the sidecar once with:

```bash
cd ~/fresh_int2_fp16_setup/xetla_vllm_plugin
source .venv/bin/activate
export XETLA_QUANT_METHOD=int2_f16
python scripts/prequantize_gguf.py \
    --model Ternary-Bonsai-8B-F16.gguf \
    --out   Ternary-Bonsai-8B-F16.gguf.xetla-int2_f16.safetensors
```

---

## 4. Run the chat demo

### 4.1 Standard interactive chat (BMG dGPU / regular Xe / fast Xe2 path)

```bash
# attach to your existing GPU job (or start one)
srun --jobid=<JOBID> --overlap --pty bash -c '
  source /swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh
  source /swtools/intel/2025.3/oneapi-vars.sh --force
  cd ~/fresh_int2_fp16_setup/xetla_vllm_plugin
  bash scripts/chat.sh
'
```

You should see:
```
[chat.sh] using xetla sidecar: .../Ternary-Bonsai-8B-F16.gguf.xetla-int2_f16.safetensors
...
[xetla] sidecar hit: model.layers.0.self_attn.qkv_proj (int2_f16)
... (36 layers) ...
[xetla] sidecar hit: lm_head lm_head (int2_f16)
INFO ... init engine (profile, create kv cache, warmup model) took 24.04 seconds
Ready. /exit to quit, /reset to clear, /system <txt> to set system prompt.
you> 
```
Type a prompt and hit Enter. Expected ~110 tok/s decode on BMG dGPU.

### 4.2 One-shot non-interactive run

```bash
srun --jobid=<JOBID> --overlap bash -c '
  source /swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh
  source /swtools/intel/2025.3/oneapi-vars.sh --force
  cd ~/fresh_int2_fp16_setup/xetla_vllm_plugin
  { echo "What is the capital of France?"; echo "/exit"; } |
      bash scripts/chat.sh --max-tokens 64 --temperature 0.0
'
```

### 4.3 BMG-XG3 / Xe2 workaround path

On the Xe2 node `pcl-kini04` (`bmtxg31` partition) the default xccl init
crashes the engine-core subprocess. Use [`scripts/run_xe2_chat.sh`](scripts/run_xe2_chat.sh)
which sets:
```
FI_PROVIDER=tcp
CCL_LOCAL_RANK=0 CCL_LOCAL_SIZE=1 CCL_ATL_TRANSPORT=ofi CCL_PROCESS_LAUNCHER=none
CCL_KVS_MODE=mpi
SYCL_UR_USE_LEVEL_ZERO_V2=0
VLLM_ENABLE_V1_MULTIPROCESSING=0
```
and runs `chat.sh` in-process. This is **only** needed on the XG3 platform —
on regular BMG / Xe2 dGPUs you do not need any of these.

---

## 5. Record an asciinema demo cast

[`scripts/demo_record.py`](scripts/demo_record.py) loads the model silently
then records a cast file you can later render to GIF (see [`DEMO.md`](DEMO.md)).

The script needs `XETLA_QUANT_METHOD` to know which kernel family to use. The
plugin can auto-derive it from the sidecar metadata, but **only** if you
either set `XETLA_QUANT_METHOD` directly or expose the sidecar via
`XETLA_PREQUANT_PATH`. Otherwise the default is `int2` (the bf16 kernel) and
you'll see `RuntimeError: A must be bf16`.

```bash
srun --jobid=<JOBID> --overlap --pty bash -c '
  source /swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh
  source /swtools/intel/2025.3/oneapi-vars.sh --force
  cd ~/fresh_int2_fp16_setup/xetla_vllm_plugin
  source .venv/bin/activate

  # Either set the method explicitly...
  export XETLA_QUANT_METHOD=int2_f16
  # ...or point at the sidecar (which both auto-derives the method *and*
  # skips the slow GGUF requantize):
  export XETLA_PREQUANT_PATH=$PWD/Ternary-Bonsai-8B-F16.gguf.xetla-int2_f16.safetensors

  python scripts/demo_record.py \
      --label "2-bit inference with vLLM + Xe2 xetla kernels" \
      --quantization xetla \
      --out demo_int2_rerun.cast \
      --model $PWD/Ternary-Bonsai-8B-F16.gguf
'
```

To render to GIF afterwards see [`DEMO.md`](DEMO.md) §1 (asciinema +
[`agg`](https://github.com/asciinema/agg) + ImageMagick).

---

## 6. Common pitfalls

| Symptom | Root cause | Fix |
|---|---|---|
| `RuntimeError: A must be bf16` (during model load) | `XETLA_QUANT_METHOD` defaulted to `int2` (= bf16 kernel) | export `XETLA_QUANT_METHOD=int2_f16` *or* `XETLA_PREQUANT_PATH=<sidecar>` |
| `XPU device count is zero` on import | Running on a login node with no GPU | run inside `srun --jobid=…` on a GPU node |
| Engine core SIGABRT during `parallel_state.py` xccl init | BMG-XG3 (Xe2) single-rank xccl bug | use `scripts/run_xe2_chat.sh` (sets `FI_PROVIDER=tcp` + `CCL_KVS_MODE=mpi` + `SYCL_UR_USE_LEVEL_ZERO_V2=0`) |
| OOM at load on a 16 GB integrated GPU (LNL/MTL/ARL) | default 0.9 GPU memory utilisation is unsafe on shared system RAM | nothing — `chat.sh` auto-detects integrated XPUs and lowers `CHAT_GPU_MEM_UTIL` to ~0.55 |
| Plugin build fails: `unknown type name 'gemm_t'`, `compute_policy_int1_fp16_upcvt_xmx` not found | `xetla` submodule on the wrong commit | `git -C xetla checkout feature/int2-fp16-bonsai-chat && git submodule update --init --recursive` |
| `vllm.patch` does not apply | already merged in the fork (or applied earlier) | `setup_fresh.sh` runs `git apply --check` first and skips cleanly |
| Slow first decode token (~30 s spike) | inductor cudagraph capture | normal on first run; cached afterwards under `~/.cache/vllm/torch_compile_cache/` |
| Decode tok/s ~25 instead of ~110 | running with `--enforce-eager` (no cudagraph) | drop `--enforce-eager` |

---

## 7. Verifying performance

Quick non-interactive smoke test (one prompt, deterministic):

```bash
srun --jobid=<JOBID> --overlap bash -c '
  source /swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh
  source /swtools/intel/2025.3/oneapi-vars.sh --force
  cd ~/fresh_int2_fp16_setup/xetla_vllm_plugin
  { echo "Hello! In one short sentence, what is the capital of France?";
    echo "/exit"; } | bash scripts/chat.sh --max-tokens 32 --temperature 0.0
' 2>&1 | grep -E "tok/s|init engine.*took|Model loading took"
```

Expected lines (BMG dGPU):
```
INFO ... [gpu_model_runner.py:4820] Model loading took 3.06 GiB memory and ~13 seconds
INFO ... [core.py:283] init engine (profile, create kv cache, warmup model) took ~24 seconds
[stats] 32 tokens in 0.29s = 110.41 tok/s
```

---

## 8. Layout of the working tree

| Path | Purpose |
|---|---|
| `csrc/` | xetla SYCL kernel wrappers exposed as Torch ops |
| `xetla_vllm_plugin.py` | vLLM plugin (`XetlaConfig`, `XetlaLinearMethod`, sidecar prequant loader) |
| `xetla/` | xetla kernel headers (submodule) |
| `vllm/` | vendored ddkalamk/vllm@xetla_v0.21.0 + `vllm.patch` |
| `scripts/chat.sh` | turnkey chat wrapper (auto sidecar detection, integrated-XPU mem util) |
| `scripts/chat.py` | Python REPL underneath |
| `scripts/demo_record.py` | asciinema cast generator used for the GIFs in `DEMO.md` |
| `scripts/run_xe2_chat.sh` | BMG-XG3 / Xe2 workaround wrapper |
| `utils/setup_fresh.sh` | from-scratch build orchestrator (this guide) |
| `utils/setup_vllm_xpu.sh` | older standalone vllm+xpu setup (canonical reference) |
| `tests/test_gemm_int2_fp16.py` | torch-level numeric sanity tests for the kernel |

---

## 9. Where to look first when something breaks

1. `tail ~/fresh_int2_fp16_setup.log` — full setup_fresh.sh log
2. `~/.cache/vllm/torch_compile_cache/` — delete to force a recompile if a stale graph is suspected
3. `python -c "from xetla_vllm_plugin import xetla_quant_method; print(xetla_quant_method())"` — confirm the active method
4. `python tests/test_gemm_int2_fp16.py` — run kernel numerics in isolation
