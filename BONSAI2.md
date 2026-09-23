# Bonsai 2 27B (Hadamard rotated basis) on Intel Xe2 GPUs, from an empty folder

This runs **prism-ml/Ternary-Bonsai-2-27B** (ternary g128, Qwen3.8-27B based,
GGUF-only release) with vLLM + the xetla int2 kernels on an Arc Pro B70 (or an
Arc 140V / Lunar Lake). Every step below was executed verbatim in an empty
directory on this cluster (`bitcos_paper/test_latest`, 2026-09-17: build 25
min on a 28-core login node, pack 1 min, B70 run 46.1 tok/s with text
byte-identical to the development tree); the expected outputs are quoted from
that run.

Bonsai 2 differs from Bonsai 1 in one way that matters here: its matrices are
stored in a **rotated basis**. Before every folded GEMM the activation is
multiplied by a fixed ±1 sign vector and passed through a blockwise (1024)
normalised Walsh-Hadamard transform; the token embedding is stored rotated and
is un-rotated after lookup (`prism.hadamard.*` GGUF metadata; see the Bonsai 2
whitepaper, A.2). Stock runtimes load the file and emit gibberish. Here the
transform is one fused SYCL kernel (`csrc/hadamard_fwht_kernel.sycl`) applied
by the plugin in front of the int2 GEMMs; the GGUF -> sidecar packer carries the
contract (block size, sign vectors, which modules are folded) into the sidecar.

Result on this cluster (greedy, 256 output tokens, 63-token prompt):

| GPU | decode | TTFT | vs Bonsai 1 27B (no rotation) |
| --- | --- | --- | --- |
| Arc Pro B70 (Arrow Lake host) | 45.7-46.1 tok/s | ~965 ms | 47.9 tok/s |
| Arc 140V (Lunar Lake, unified memory) | 7.42 tok/s | ~6.6 s | 7.51 tok/s |

The generated text is byte-identical on both GPUs, to the un-fused (matmul)
reference implementation of the transform, and across independent builds
(development tree vs. the from-scratch checkout) as long as
`--deterministic-compile` is on (default in the runner; see Knobs).

---

## 0. Prerequisites

* Intel GPU user-space driver + Level Zero, and oneAPI 2025.3 (icpx). On this
  cluster:
  ```bash
  source /swtools/intel-gpu/latest/intel_gpu_vars.sh
  source /swtools/intel/2025.3/oneapi-vars.sh --force
  icpx --version      # Intel(R) oneAPI DPC++/C++ Compiler 2025.3.0
  ```
* `git`, `curl`; `uv` is bootstrapped by the setup script if missing
  (`~/.local/bin/uv`).
* ~40 GB free disk: venv + vLLM source build (~15 GB), the two GGUFs (8.1 GB),
  sidecar + packed model dir (8.2 GB), inductor/triton caches.
* Access to the GPU for the run steps (here: a SLURM allocation on the B70 or
  LNL node; the build steps run anywhere with icpx).

```bash
export DEST=/path/to/empty/folder          # everything lands under here
mkdir -p "$DEST" && cd "$DEST"
```

Steps 1-3 are also scripted as `utils/bonsai2_from_scratch.sh "$DEST"` (this
is what the verification run used); they are spelled out below.

## 1. Build vLLM (XPU) + the xetla plugin

```bash
git clone -b feature/bitcos-int2-integration --recurse-submodules \
    https://github.com/ddkalamk/xetla_vllm_plugin.git "$DEST/xetla_vllm_plugin"
PLUGIN_BRANCH=feature/bitcos-int2-integration \
    bash "$DEST/xetla_vllm_plugin/utils/setup_fresh.sh" "$DEST"
```

`setup_fresh.sh` (re-runnable, skips finished steps):

1. initialises the `xetla` submodule (kernel headers, branch
   `feature/bitcos-int2-integration` of `egeor/xetla`);
2. creates `xetla_vllm_plugin/.venv` (Python 3.12 via uv);
3. clones upstream `vllm-project/vllm` at **v0.21.0** into
   `xetla_vllm_plugin/vllm` and applies the vendored `vllm.patch` (5 files:
   GGUF/xetla quant hooks, XPU graph on TP=1, and the XPU GDN kernel call
   sliced to `num_actual_tokens` so graph-padded batches of >2 sequences run);
4. `pip install -r requirements/xpu.txt`, then
   `VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e vllm`,
   then `triton-xpu==3.7.0`;
5. `python setup.py install` for the plugin: builds `xetla_pt_ext` (all
   `csrc/*.sycl`, including the fused Hadamard kernel) and registers the
   `vllm.general_plugins` entry point.

Sanity check (needs a GPU; on SLURM prefix with `srun --jobid=<id> --overlap`):

```bash
source "$DEST/xetla_vllm_plugin/.venv/bin/activate"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
python -c 'import torch, xetla_vllm_plugin, xetla_pt_ext; print(torch.xpu.get_device_name(0), hasattr(torch.ops.xetla_int2, "hadamard_fwht_run"))'
# -> Intel(R) Arc(TM) Pro B70 Graphics True
python "$DEST/xetla_vllm_plugin/tests/test_hadamard_xpu.py"
# -> per-shape rel.err ~3e-4 (fp16 rounding), fused 7-22x faster than the matmul path, "OK"
```

## 2. Download the model files

```bash
source "$DEST/xetla_vllm_plugin/.venv/bin/activate"
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
    "$DEST/xetla_vllm_plugin/third_party/llama.cpp-prism"
```

## 3. Pack the sidecar and the model directory

```bash
cd "$DEST/xetla_vllm_plugin"
python scripts/pack_bonsai2_gguf.py \
    --gguf    "$MODELS/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf" \
    --mmproj  "$MODELS/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-mmproj-BF16.gguf" \
    --ref-dir "$MODELS/Ternary-Bonsai-2-27B-ref" \
    --out     "$MODELS/Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors" \
    --packed  "$MODELS/Ternary-Bonsai-2-27B-packed"
```

Expected tail (~1 min, CPU only):

```
[pack] hadamard: H1024, 401 folded, 1 inverse, sign widths [5120, 6144, 17408]
[pack] GDN nv=48 nk=16 hd=128 hk=128; layers=64
[pack] 257 quantized modules, 449 residual tensors
...
[pack] embedding (248320, 320) words, inverse-hadamard=True
[pack] 258 sidecar modules (6.80 GB), 257 hadamard-folded, 1 inverse; 449 residual tensors
[pack] wrote .../Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors (7.14 GB, ...s)
[pack] wrote .../Ternary-Bonsai-2-27B-packed (782 tensors incl. 333 vision, 0.97 GB; eos=248046 bos=248044)
```

What it does (lossless; no re-quantisation):

* PQ2_0 blocks (fp16 scale + 128 two-bit codes) are re-mapped bit-wise into
  the xetla `int2_f16` layout (`qweight int32 [K/16, N]`, `scale fp16
  [K/128, N]`), fused into vLLM's `qkv_proj` / `gate_up_proj` /
  `in_proj_qkvz` modules, with `lm_head` and a row-major packed
  `embed_tokens`.
* The GGUF `prism.hadamard` contract goes into the sidecar metadata plus one
  `hadamard.signs.<K>` tensor per folded width.
* llama.cpp conventions are undone for HF/vLLM layout: tiled -> grouped GDN
  value-head order, `A = -exp(A_log)` -> `A_log`, and the `w+1` norm weights
  (Qwen3.5 norms are `x*(1+w)`).
* The vision tower is converted from the mmproj GGUF (`v.blk.*` -> 
  `model.visual.blocks.*`, `mm.0/mm.2` -> merger, split Conv3d slices re-stacked).
* `config.json` is synthesised from the MLX release's config
  (`Qwen3_5ForConditionalGeneration`; text/vision configs identical to
  Qwen3.5-27B, `mtp_num_hidden_layers=0` because the GGUF has no MTP head).

## 4. Run on the B70

With SLURM (the script does the page-cache eviction, stale-process reclaim,
memory-utilisation choice and XPU-graph settings that the B70/LNL runs need):

```bash
cd "$DEST/xetla_vllm_plugin"
JOB=$(sbatch -p b70 -w pcl-arl01 -t 4:00:00 --parsable --wrap "sleep 14400")
MODELS="$MODELS" MAXTOK=256 MAXLEN=512 \
    bash scripts/run_bonsai2_gpu.sh "$JOB" B70 "Tell me about photosynthesis in 200 words"
```

Without SLURM, on a machine with the GPU:

```bash
cd "$DEST/xetla_vllm_plugin"
source /swtools/intel-gpu/latest/intel_gpu_vars.sh
source /swtools/intel/2025.3/oneapi-vars.sh --force
source .venv/bin/activate
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export XETLA_PREQUANT_PATH="$MODELS/Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors"
export XETLA_QUANT_METHOD=int2_f16
python scripts/bench_model.py --model "$MODELS/Ternary-Bonsai-2-27B-packed" \
    --quantization xetla --dtype bfloat16 --max-model-len 512 \
    --gpu-memory-utilization 0.78 --cudagraph-sizes 1,2,4,8 --max-num-batched-tokens 512 \
    --deterministic-compile --max-tokens 256 --temperature 0.0 --full \
    --prompt "Tell me about photosynthesis in 200 words"
```

Expected (B70; first run includes a ~30 s torch.compile, engine load ~50 s):

```
[xetla] sidecar loaded: .../Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors (519 tensors, method=int2_f16)
[xetla] sidecar hit: language_model.model.embed_tokens embedding (int2_f16), packed (248320, 320), out dtype torch.bfloat16, inverse-hadamard
[xetla] fused SwiGLU into the gate epilogue for 64 MLP blocks
...
TTFT (prefill)       : 179 ms
decode               : 256 tokens in 5.54 s = 46.25 tok/s   [length]
repetition           : 5 lines, 5 unique; 187 words, 133 unique
We need to respond to user: "Tell me about photosynthesis in 200 words". ...
"Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy. Using chlorophyll in chloroplasts, they capture sunlight and use it to transform carbon dioxide and water into glucose and oxygen. The overall reaction is: six carbon dioxide molecules plus six water molecules, powered by light, yield one glucose molecule and six oxygen molecules. ...
```

(The model is a thinking model: the output starts with its reasoning, then
`</think>`, then the answer. `--full` prints everything.)

### Lunar Lake / Arc 140V (unified memory)

Same script, two knobs from the LNL BKM plus one specific to a fresh compile:

```bash
JOB=$(sbatch -p lnl -w pcl-lnl01 -t 3:00:00 --parsable --wrap "sleep 10800")
MODELS="$MODELS" UTIL=0.35 KVBYTES=$((2<<30)) MAXTOK=256 MAXLEN=512 \
    bash scripts/run_bonsai2_gpu.sh "$JOB" LNL "Tell me about photosynthesis in 200 words"
```

* `UTIL=0.35` pins `--gpu-memory-utilization`; the sampled value over-reserves
  on unified memory.
* `KVBYTES` pins the KV-cache size. vLLM otherwise sizes it from a memory
  profile taken right after torch.compile, and on LNL a *fresh* compile
  inflates that profile by >10 GiB, so the budget goes negative
  ("No available memory for the cache blocks"). 2 GiB is plenty for these
  prompts (KV is 64 KiB/token).
* If an engine was killed on LNL, the device memory it held stays with the
  SLURM job cgroup: `scancel` and re-`sbatch` before retrying.

Expected: `decode : 256 tokens in ~33.8 s = 7.57 tok/s`, TTFT ~1.0 s, same text
as the B70.

### Prefill

For M>1 the int2 fp16-upcvt kernel uses an M tile (WGM 8 for M<=8, 16 for
M<=16, else 32; SGM 8, SGN 16, SGK 128, WGN 64), so the weights are streamed
once per row tile rather than once per token. Measured against the old GEMV
tiers on the same binary: B70 TTFT 966 -> 179 ms, LNL 6591 -> 998 ms, decode
unchanged (M=1 path untouched). The plain GEMM is bit-exact with the old path
at every M; the fused-SwiGLU epilogue sums its fp32 k-slices in a different
order (1-ulp differences), which is enough to pick the other of the two known
greedy trajectories for this prompt. `XETLA_INT2_PREFILL_CFG=-1` restores the
old tiers; `python tests/test_int2_prefill_mtile.py` (GPU) sweeps the
alternatives and checks each against the legacy result.

### BITCOS

The int2 sidecar transcodes to BITCOS exactly as for Bonsai 1; the
`hadamard.*` tensors are carried over. BITCOS bakes in a per-device slice
count, so make one file per GPU:

```bash
python scripts/zero_density.py "$MODELS/Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors"   # 0.328
for dev in b70 lnl; do
  XETLA_BITCOS_SLICE_TARGET=$dev python scripts/transcode_int2_to_bitcos.py \
      --in  "$MODELS/Ternary-Bonsai-2-27B.xetla-int2_f16.safetensors" \
      --out "$MODELS/Ternary-Bonsai-2-27B.xetla-bitcos_f16.$dev.safetensors"
done
METHOD=bitcos BITCOS_SFX=.b70 MAXTOK=256 MAXLEN=512 \
    bash scripts/run_bonsai2_gpu.sh "$JOB" B70 "Tell me about photosynthesis in 200 words"
```

Measured: B70 47.8 tok/s, TTFT 151 ms; LNL (`UTIL=0.35 KVBYTES=$((2<<30))
BITCOS_SFX=.lnl`) 8.31 tok/s, TTFT 969 ms. Sidecar 6.19 GB vs 7.14 GB int2;
greedy text identical on the two GPUs.

### MTP speculative decoding

The community MTP head
[ProCreations/Ternary-Bonsai-2-27B-MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP)
(`model_mtp.safetensors`, bf16, one Qwen3.5 decoder layer + `fc` + norms,
~425M params) drafts tokens for the int2 target. It shares the target's
`embed_tokens`/`lm_head`, which are loaded through the xetla sidecar.

```bash
huggingface-cli download ProCreations/Ternary-Bonsai-2-27B-MTP --local-dir "$MODELS/Ternary-Bonsai-2-27B-MTP"
D=$MODELS/Ternary-Bonsai-2-27B-MTP-draft; mkdir -p $D
cp $PACKED/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json,chat_template.jinja} $D/
python - <<EOF   # the packed config says mtp_num_hidden_layers: 0
import json; p="$D/config.json"; c=json.load(open(p))
c.setdefault("text_config", c)["mtp_num_hidden_layers"] = 1; json.dump(c, open(p, "w"), indent=2)
EOF
ln -sf $MODELS/Ternary-Bonsai-2-27B-MTP/model_mtp.safetensors $D/model.safetensors
SPEC=3 MAXTOK=256 bash scripts/run_bonsai2_gpu.sh "$JOB" B70 "Explain photosynthesis in detail."
SPEC=3 bash scripts/eval_bonsai2_lm_eval.sh "$JOB" B70_mtp3       # GSM8K with MTP
```

Needed pieces: XPU GDN attention runs verify rows through vLLM's Triton
recurrent kernels (the SYCL kernel has no per-token state slots) and mixed
batches split between the two (`vllm/_xpu_ops.py`); and a k-sliced small-M
tile (1 < M <= 8) in the int2 kernel, because the plain M tile made the M=4
verify GEMMs 1.2-3.8x slower than the M=1 GEMV (down_proj 202 vs 54 us; now
61). `XETLA_INT2_SMALLM_CFG=0` turns it off, `1..6` sweeps (WGN, LS).

B70, 256 tokens, batch 1 (baseline 46.2 tok/s):

| draft tokens | tok/step | tok/s | speedup |
| --- | --- | --- | --- |
| 1 | 1.79 | 61.2 | 1.32x |
| 2 | 2.42 | 73.9 | 1.60x |
| 3 | 2.88 | 80.2 | 1.73x |

Before the small-M tile the same runs gave 43.3 / 52.9 / 56.9 tok/s. GSM8K
(`gsm8k_cot_llama`, 300 examples, thinking, batch 16): MTP k=3 97.7%
against 98.0% without MTP (one question, within the 0.9% stderr), 1311 s
against 1299 s. At batch 16 MTP brings no throughput gain because the
verify GEMMs are M=64 and no longer bandwidth-bound. Without the small-M tile the MTP text is
identical to the non-speculative run. With it, the tile's k-slices are
summed in a different order (rel. err ~3e-4 against the GEMV tier), so the
greedy trajectory drifts after a few tokens. k=1, 2 and 3 all produce the
same text.

## 5. Knobs

| Env / flag | Default | Effect |
| --- | --- | --- |
| `XETLA_HADAMARD_IMPL` | `fused` | `matmul` = reference implementation (sign multiply + matmul with H_1024). Same text, ~10% slower decode. |
| `XETLA_HADAMARD_DTYPE` | `fp32` | accumulation dtype of the matmul reference only |
| `XETLA_INT2_PREFILL_CFG` | `0` | M>1 tile for the int2 upcvt kernel: `-1` = old per-token GEMV tiers (B70 TTFT 966 ms vs 179 ms), `1..5` alternative tiles (see `csrc/int2_fp16_upcvt_kernel.sycl`). |
| `XETLA_INT2_SMALLM_CFG` | per-shape | 1<M<=8 k-sliced tile (MTP verify): `0` = plain M tile, `1..6` = (WGN, LS) (32,2) (32,4) (32,8) (64,2) (64,4) (64,8) |
| `SPEC=N` (run / eval scripts) | off | MTP speculative decoding with N draft tokens, draft dir `DRAFT` |
| `XETLA_DISABLE_DPAS` | `1` | `0` enables the DPAS (XMX) int2 prefill kernel: activations are quantised to int8 for prefill (B70 TTFT ~391 ms, now slower than the default M-tiled fp16 path). The output stays coherent and on-topic but is *not* bit-identical to the fp16 path. |
| `METHOD=bitcos`, `BITCOS_SFX=.b70\|.lnl` | `int2` | run-script switch to the BITCOS sidecar (section 4, BITCOS) |
| `--deterministic-compile` (`DETERMINISTIC=1` in the script) | on | disables inductor's benchmark-selected combo kernels; without it the greedy text can differ between two fresh compiles (different fusions, different fp rounding). No measurable perf cost. |
| `XETLA_FUSE_SWIGLU` | `1` | SwiGLU folded into the gate GEMM epilogue |
| `KVBYTES`, `UTIL`, `MAXLEN`, `MAXTOK`, `CGSIZES`, `RUN_ENV="export ..."` | | run-script knobs, see its header |

## 6. Verifying a change

* `python tests/test_hadamard_cpu.py` (no GPU): helper math vs a butterfly
  FWHT and ggml's parity construction.
* `python tests/test_hadamard_xpu.py` (GPU): fused kernel vs matmul reference,
  with timings.
* End-to-end: run step 4 twice with `XETLA_HADAMARD_IMPL=fused` and `=matmul`
  and `diff` the printed text; they must match.

### Accuracy with a standard harness

The single greedy prompt above catches gross breakage, not a subtle numeric
bug. For that, run the standard lm-evaluation-harness against the vLLM engine
(same plugin, same kernels, batched):

```bash
.venv/bin/pip install "lm_eval>=0.4.8"        # torch/vLLM pinned via constraints
LIMIT=300 bash scripts/eval_bonsai2_lm_eval.sh "$JOB" gsm8k300
```

The script writes a YAML config (`--model vllm`, `quantization=xetla`, chat
template, thinking on with `reasoning_effort=medium`, answers taken after
`</think>`, greedy, `max_num_seqs=16`, cudagraph sizes 1..16) and runs
`gsm8k_cot_llama` (8-shot, "The final answer is N": prompts of ~900 tokens,
answers from a few hundred to 4096 tokens, so both the M-tiled prefill and the
batched decode path are exercised). Results land in
`bonsai_logs/lm_eval_<TAG>/` with per-sample outputs. Knobs: `LIMIT`, `TASKS`,
`EFFORT=xhigh`, `THINK=0` (instruct mode), `MAXGEN`, `NSEQ`, `METHOD=bitcos`.

Bonsai 2 27B, int2 + fused Hadamard + M-tiled prefill, B70:

| Examples | exact match | wall time |
| --- | --- | --- |
| GSM8K test, first 300 | 98.0% (294/300) | 21 min |
| **GSM8K test, all 1319** | **96.7%** (1276/1319, ±0.5; 1 answer hit the 4096-token budget, 1 extraction miss, the rest model errors) | 90 min |

The model card reports the math group (GSM8K/MATH-500/AIME25/AIME26, EvalScope,
thinking mode, H100) at 96.57 for Bonsai 2 and 97.06 for the FP16 base, so a
GSM8K score in the high 90s is what an intact model gives; a kernel or packing
bug shows up as a collapse (the pre-fix norm-fold bug, for instance, produced
fluent gibberish at 0%).

Batched decode needed one vLLM-side fix (now in `vllm.patch`): the XPU GDN
kernel asserts `rows == num_actual_tokens`, but with >2 sequences the model
runner pads the batch to the graph capture size; the call is now sliced to the
actual tokens. `python tests/batched_generate.py --model <packed> --n 16` is
the quick check (16 prompts in one `generate`, ~170 tok/s aggregate on the B70).

## Files

| | |
| --- | --- |
| `scripts/pack_bonsai2_gguf.py` | GGUF (PQ2_0 + mmproj) -> sidecar + packed model dir |
| `scripts/run_bonsai2_gpu.sh` | SLURM runner used for the numbers above |
| `scripts/eval_bonsai2_lm_eval.sh`, `scripts/gpu_py.sh`, `tests/batched_generate.py` | lm-evaluation-harness run (GSM8K), generic GPU-node python launcher, batched-generation smoke test |
| `scripts/transcode_int2_to_bitcos.py`, `scripts/zero_density.py` | int2 -> BITCOS sidecar (keeps `hadamard.*`); zero density of a sidecar |
| `tests/test_int2_prefill_mtile.py`, `tests/mbench_smallM.py` | M-tiled prefill sweep vs legacy tiers; small-M microbench |
| `scripts/bench_model.py` | the benchmark (`--deterministic-compile`, `--kv-cache-memory-bytes`) |
| `csrc/hadamard_fwht_kernel.sycl` | fused sign flip + blockwise WHT (block 1024), op `xetla_int2.hadamard_fwht_run` |
| `xetla_vllm_plugin.py` | `_xetla_hadamard_*`: attaches sign vectors from the sidecar, applies the transform before folded GEMMs and after the embedding lookup |
| `tests/test_hadamard_{cpu,xpu}.py` | unit checks |
