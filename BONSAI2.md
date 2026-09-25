# Bonsai 2 27B on Intel Xe2 GPUs with TernSYCL, from an empty folder

This runs **prism-ml/Ternary-Bonsai-2-27B** (ternary g128, Qwen3.5-27B
architecture, GGUF-only release) with vLLM v0.30 and the TernSYCL int2
kernels on an Arc Pro B70. The expected outputs below are from the
2026-09-24 verification on this cluster.

Other ternary Qwen3.5-architecture GGUFs (plain llama.cpp Q2_0, dense tail
layers, non-ternary embedding/lm_head) go through the same packer and runner:
see [TERNARYQUENCH.md](TERNARYQUENCH.md) for TernaryQuench Qwen3.8-27B.

Bonsai 2 stores its matrices in a **rotated basis**: before every folded GEMM
the activation is multiplied by a fixed ±1 sign vector and passed through a
blockwise (1024) normalised Walsh-Hadamard transform, and the token embedding
is stored rotated and is un-rotated after the lookup (`prism.hadamard.*` GGUF
metadata; Bonsai 2 whitepaper, A.2). Stock runtimes load the file and emit
gibberish. Here the packer carries that contract into the sidecar, and the
plugin runs the transform as one fused TernSYCL kernel in front of the int2
GEMMs.

Result (greedy, photosynthesis prompt, 256 output tokens):

| | B70 TTFT | B70 decode | Arc 140V TTFT | Arc 140V decode | GSM8K (1319, thinking, B70) |
| --- | --- | --- | --- | --- | --- |
| TernSYCL (this branch) | 107-111 ms | 46.4-47.6 tok/s | 657 ms | 8.20-8.33 tok/s | 96.9% (41 min) |
| previous XeTLA kernels | 183 ms | 46.3 tok/s | 993 ms | 8.17 tok/s | 96.7% (90 min) |

---

## 0. Prerequisites

* Intel GPU driver with Level Zero, and oneAPI 2026.0 (`icpx`; it must match
  the SYCL runtime that torch 2.13 xpu ships). On this cluster:
  ```bash
  unset LD_LIBRARY_PATH   # a stale libur_loader breaks torch 2.13's libsycl
  source /swtools/intel-gpu/latest/intel_gpu_vars.sh
  source /swtools/intel/2026.0/oneapi-vars.sh --force
  icpx --version          # Intel(R) oneAPI DPC++/C++ Compiler 2026.0.0
  ```
* `git`, `curl`; `uv` is bootstrapped by the setup script if missing.
* ~40 GB free disk: venv and vLLM build (~15 GB), the two GGUFs (8.1 GB),
  sidecar and packed model directory (8.2 GB), compiler caches.
* A GPU for the run steps (here a SLURM allocation on a B70 node). The build
  and pack steps run anywhere with `icpx`.

```bash
export DEST=/path/to/empty/folder
mkdir -p "$DEST" && cd "$DEST"
```

Steps 1-3 are also scripted as `utils/bonsai2_from_scratch.sh "$DEST"`.

## 1. Build vLLM (XPU) and the plugin

```bash
git clone -b feature/vllm-v0.30-ternsycl --recurse-submodules \
    https://github.com/ddkalamk/xetla_vllm_plugin.git "$DEST/ternsycl_vllm_plugin"
bash "$DEST/ternsycl_vllm_plugin/utils/setup_fresh.sh" "$DEST"
```

`setup_fresh.sh` (re-runnable, skips finished steps):

1. initialises the `ternsycl` submodule
   ([libxsmm/TernSYCL](https://github.com/libxsmm/TernSYCL): kernel headers);
2. creates `ternsycl_vllm_plugin/.venv` (Python 3.12 via uv);
3. clones upstream `vllm-project/vllm` at **v0.30.0** into
   `ternsycl_vllm_plugin/vllm` (no patch needed);
4. installs vLLM for XPU (`requirements/xpu.txt`: torch 2.13 xpu, triton-xpu),
   `VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e vllm`;
5. builds and installs the plugin: `ternsycl_pt_ext` (the `csrc/*.sycl`
   launchers and the TernSYCL kernels, compiled ahead of time for
   `bmg-g31,lnl-m`) and the `vllm.general_plugins` entry point
   `ternsycl_model`.

Sanity check (needs a GPU; on SLURM prefix with `srun --jobid=<id> --overlap`):

```bash
cd "$DEST/ternsycl_vllm_plugin"
source .venv/bin/activate
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
python -c 'import torch, ternsycl_vllm_plugin, ternsycl_pt_ext; print(torch.xpu.get_device_name(0), hasattr(torch.ops.ternsycl, "hadamard_fwht_run"))'
# -> Intel(R) Arc(TM) Pro B70 Graphics True
python tests/test_ternsycl_ops.py
# -> 497 "ok" lines, "all passed (0 failures)"
```

## 2. Download the model files

```bash
source "$DEST/ternsycl_vllm_plugin/.venv/bin/activate"
export MODELS="$DEST/models"; mkdir -p "$MODELS"

# language model (PQ2_0, 7.2 GB) + vision projector (mmproj BF16, 0.93 GB)
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
    Ternary-Bonsai-2-27B-PQ2_0.gguf Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
    --local-dir "$MODELS/Ternary-Bonsai-2-27B-gguf"

# tokenizer + HF-style config (small files from the MLX release of the same model)
hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit \
    config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
    --local-dir "$MODELS/Ternary-Bonsai-2-27B-ref"

# PrismML llama.cpp fork: its gguf-py knows the PQ2_0 type (id 142); stock gguf does not
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp \
    "$DEST/ternsycl_vllm_plugin/third_party/llama.cpp-prism"
```

## 3. Pack the sidecar and the model directory

```bash
cd "$DEST/ternsycl_vllm_plugin"
python scripts/pack_bonsai2_gguf.py \
    --gguf    "$MODELS/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf" \
    --mmproj  "$MODELS/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-mmproj-BF16.gguf" \
    --ref-dir "$MODELS/Ternary-Bonsai-2-27B-ref" \
    --out     "$MODELS/Ternary-Bonsai-2-27B.ternsycl-int2_f16.safetensors" \
    --packed  "$MODELS/Ternary-Bonsai-2-27B-packed"
```

Expected (under a minute, CPU only):

```
[pack] hadamard: H1024, 401 folded, 1 inverse, sign widths [5120, 6144, 17408]
[pack] GDN nv=48 nk=16 hd=128 hk=128; layers=64
[pack] 257 quantized modules, 449 residual tensors
...
[pack] wrote .../Ternary-Bonsai-2-27B.ternsycl-int2_f16.safetensors (7.14 GB, ...)
```

What it does (lossless, no re-quantization):

* PQ2_0 blocks (fp16 scale + 128 two-bit codes) are re-mapped bit-wise into
  the `int2_f16` layout (`qweight int32 [K/16, N]`, `scale fp16 [K/128, N]`),
  fused into vLLM's `qkv_proj` / `gate_up_proj` / `in_proj_qkvz` modules, plus
  `lm_head` and a row-major packed `embed_tokens`.
* The `prism.hadamard` contract goes into the sidecar metadata
  (`ternsycl_meta`), plus one `hadamard.signs.<K>` tensor per folded width.
* llama.cpp conventions are undone for the HF/vLLM layout: tiled -> grouped
  GDN value-head order, `A = -exp(A_log)` -> `A_log`, and the `w+1` norm
  weights (Qwen3.5 norms are `x*(1+w)`).
* The vision tower is converted from the mmproj GGUF.
* `config.json` is synthesised from the MLX release's config
  (`Qwen3_5ForConditionalGeneration`, `mtp_num_hidden_layers=0` because the
  GGUF has no MTP head).

## 4. Run on the B70

With SLURM (the script evicts the model files from the page cache, kills stale
engines of the same user on the node, picks the memory utilization and sets
the XPU-graph options):

```bash
cd "$DEST/ternsycl_vllm_plugin"
JOB=<allocation on a B70 node>          # e.g. sbatch ... --wrap "sleep 14400"
MODELS="$MODELS" MAXTOK=256 MAXLEN=512 \
    bash scripts/run_bonsai2_gpu.sh "$JOB" B70 "Tell me about photosynthesis in 200 words"
```

`RUN_ENV="export ZE_AFFINITY_MASK=<card>"` pins one GPU of a multi-GPU node.
Launch it from a clean login shell: a oneAPI environment already loaded in
the calling shell leaks into the job and breaks `import torch`.

Without SLURM, on a machine with the GPU:

```bash
cd "$DEST/ternsycl_vllm_plugin"
source .venv/bin/activate
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu VLLM_XPU_ENABLE_XPU_GRAPH=1
export TERNSYCL_PREQUANT_PATH="$MODELS/Ternary-Bonsai-2-27B.ternsycl-int2_f16.safetensors"
python scripts/bench_model.py --model "$MODELS/Ternary-Bonsai-2-27B-packed" \
    --quantization ternsycl --dtype bfloat16 --max-model-len 512 \
    --gpu-memory-utilization 0.78 --cudagraph-sizes 1,2,4,8 --max-num-batched-tokens 512 \
    --max-num-seqs 16 --deterministic-compile --max-tokens 256 --temperature 0.0 --full \
    --prompt "Tell me about photosynthesis in 200 words"
```

Expected (B70; the first run includes a ~30 s torch.compile, engine load ~50 s):

```
[ternsycl] sidecar loaded: .../Ternary-Bonsai-2-27B.ternsycl-int2_f16.safetensors (519 tensors, method=int2_f16)
[ternsycl] sidecar hit: language_model.model.embed_tokens embedding (int2_f16), packed (248320, 320), out dtype torch.bfloat16, inverse-hadamard
[ternsycl] fused SwiGLU into the gate epilogue for 64 MLP blocks
...
TTFT (prefill)       : 107 ms
decode               : 256 tokens in 5.38 s = 47.56 tok/s   [length]
repetition           : 5 lines, 5 unique; 187 words, 133 unique
We need to respond to user: "Tell me about photosynthesis in 200 words". ...
"Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy. Using chlorophyll in chloroplasts, they capture sunlight and use it to transform carbon dioxide and water into glucose and oxygen. The overall reaction is: six carbon dioxide molecules plus six water molecules, with light energy, yield one glucose molecule and six oxygen molecules. ...
```

The model thinks first: the output starts with its reasoning, then
`</think>`, then the answer (`--full` prints everything). With
`--deterministic-compile` (on by default in the runner) the text is identical
from run to run.

### Lunar Lake / Arc 140V

The extension is built for `lnl-m` too, and the same runner applies with the
LNL knobs below (the kernel tile tables were tuned on the B70).

```bash
MODELS="$MODELS" UTIL=0.35 KVBYTES=$((2<<30)) MAXTOK=256 MAXLEN=512 \
    bash scripts/run_bonsai2_gpu.sh "$JOB" LNL "Tell me about photosynthesis in 200 words"
```

Measured on the Arc 140V (fresh allocation per run, two alternating runs
each):

| Kernels | TTFT | decode |
| --- | --- | --- |
| TernSYCL (this branch) | 656-657 ms | 8.20-8.33 tok/s |
| previous XeTLA kernels | 992-993 ms | 8.17 tok/s |

The TernSYCL text on the Arc 140V is identical to the B70 text.

* `UTIL=0.35` pins `--gpu-memory-utilization`; the sampled value over-reserves
  on unified memory.
* `KVBYTES` pins the KV-cache size: a fresh torch.compile inflates vLLM's
  memory profile by >10 GiB on LNL, and the KV budget otherwise goes negative.
  2 GiB is plenty for these prompts.
* A killed engine leaves its device memory with the SLURM job: `scancel` and
  allocate again before retrying.

## 5. How a forward pass maps to kernels

| Step | Op | M |
| --- | --- | --- |
| embedding lookup (int2 rows unpacked per token) + inverse Hadamard | `hadamard_fwht_run(inverse=True)` | tokens |
| Hadamard on the input of every folded GEMM | `hadamard_fwht_run` | tokens |
| `in_proj_qkvz`, `out_proj`, attention `qkv_proj` / `o_proj`, `down_proj`, `lm_head` | `int2_fp16_upcvt_gemm_run` | tokens |
| MLP: up, then gate with `silu(gate)·up` in its epilogue | `int2_fp16_upcvt_gemm_run`, `int2_fp16_upcvt_gemm_postop_run` | tokens |

Decode is M = 1 (one GEMV per layer, tile per shape). A prompt runs at
M = prompt length: M-tiled GEMMs (16/32/64-row tiles) that stream each weight
once per row tile. The tile table is in `csrc/upcvt.sycl`.

## 6. Knobs

| Env / flag | Default | Effect |
| --- | --- | --- |
| `TERNSYCL_HADAMARD_IMPL` | `fused` | `matmul`: reference implementation (sign multiply + matmul with H_1024) |
| `TERNSYCL_HADAMARD_DTYPE` | `fp32` | precision of the matmul reference only |
| `TERNSYCL_FUSE_SWIGLU` | `1` | SwiGLU folded into the gate GEMM epilogue |
| `TERNSYCL_DISABLE_DPAS` | `1` | `0` runs prefill on the int2 x int8 DPAS kernel (activations quantized to int8 per row and 128-group); the output stays coherent but is not identical to the fp16 path |
| `TERNSYCL_PROFILE` | `0` | `1` prints a per-shape GEMM time and bandwidth table at exit |
| `--deterministic-compile` (`DETERMINISTIC=1` in the runner) | on | disables inductor's benchmark-selected combo kernels, so the greedy text does not depend on which fusions a fresh compile picked |
| `KVBYTES`, `UTIL`, `MAXLEN`, `MAXTOK`, `CGSIZES`, `MAXSEQS`, `RUN_ENV` | | runner knobs, see the header of `scripts/run_bonsai2_gpu.sh` |

## 7. Verifying a change

* `python tests/test_ternsycl_ops.py` (GPU): every TernSYCL op against an fp32
  reference, all Bonsai 2 27B shapes, M = 1 to 100, fp16 and bf16, both
  epilogues, the int8 DPAS path, Hadamard forward and inverse.
* `python tests/test_hadamard_cpu.py` (no GPU), `python tests/test_hadamard_xpu.py`
  (GPU): Hadamard helpers and the fused kernel against the matmul reference.
* End to end: run step 4 with `TERNSYCL_HADAMARD_IMPL=fused` and `=matmul` and
  compare the text.

### Accuracy with a standard harness

A single greedy prompt catches gross breakage, not a subtle numeric bug. Run
the lm-evaluation-harness against the vLLM engine (same plugin and kernels,
batched):

```bash
.venv/bin/pip install "lm_eval>=0.4.8"
LIMIT=1319 bash scripts/eval_bonsai2_lm_eval.sh "$JOB" gsm8k1319
```

The script writes a config (`quantization=ternsycl`, chat template, thinking
with `reasoning_effort=medium`, answers after `</think>`, greedy,
`max_num_seqs=16`, cudagraph sizes 1..16) and runs `gsm8k_cot_llama`
(8-shot, prompts of ~900 tokens, answers up to 4096 tokens, so both the
M-tiled prefill and batched decode run). Results land in
`bonsai_logs/lm_eval_<TAG>/` with per-sample outputs. Knobs: `LIMIT`, `TASKS`,
`EFFORT=xhigh`, `THINK=0`, `MAXGEN`, `NSEQ`.

Bonsai 2 27B, B70, all 1319 GSM8K test problems:

| Kernels | exact match (strict) | flexible extract | wall time |
| --- | --- | --- | --- |
| TernSYCL (this branch) | **96.89%** (1278/1319, ±0.48) | 96.82% | 41 min |
| previous XeTLA kernels | 96.7% (1276/1319) | | 90 min |

The model card reports the math group (GSM8K/MATH-500/AIME, thinking mode) at
96.57 for Bonsai 2 and 97.06 for the FP16 base, so a score in the high 90s is
what an intact model gives; a kernel or packing bug shows up as a collapse.
