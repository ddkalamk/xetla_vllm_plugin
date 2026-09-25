# TernaryQuench Qwen3.8-27B (Q2_0 GGUF) on Intel Xe2 GPUs with TernSYCL

Companion to [BONSAI2.md](BONSAI2.md). Same build, same runner, same kernels;
only the download and pack steps differ. The model is
[penkia/TernaryQuench-Qwen3.8-27B-GGUF](https://huggingface.co/penkia/TernaryQuench-Qwen3.8-27B-GGUF)
(`TernaryQuench-Qwen3.8-27B-Hybrid-Q2_0.gguf`, 9,757,875,616 bytes, sha256
`e640991f9842f509ead53d91233de7fd13a80459c1f6b25d51295fd2ba54a14b`): a
ternary CAT-Q quantization of Qwen3.8-27B, which has the same Qwen3.5-27B
architecture as Bonsai 2 27B.

> **Check the sha256.** A differently obtained `TernaryQuench-Qwen3.8-27B-Q2_0.gguf`
> (9,757,875,648 bytes, sha256 `83e0b19d…`, `general.finetune = 1d4bf0f…`) has
> the same layout but different ternary weights: 79-96% of the codes agree,
> scales differ by a median of 2-10%. Layers 62-63, the norms, the embedding
> and the lm_head are identical. It is a different training checkpoint, and it
> is clearly worse (see perplexity and GSM8K below). The release file is the
> one on the Hub.

## How it differs from Bonsai 2 27B

| | Bonsai 2 27B (PQ2_0) | TernaryQuench 27B (Q2_0) |
|---|---|---|
| ternary matrices | all 64 layers, `PQ2_0` (g128, type 142) | layers 0-61 (481 tensors), `Q2_0` (g64, type 42) |
| layers 62-63 | ternary | 15 BF16 matrices (dense) |
| `token_embd` / `output` (lm_head) | ternary | `Q4_1` (not ternary) |
| `in_proj_a/b` (`ssm_alpha/beta`) | BF16 | ternary (layers 0-61) |
| Hadamard rotation (`prism.hadamard.*`) | yes | none, plain weights |
| GDN `ssm_out` columns | grouped V order | llama.cpp tiled V order |
| vision tower | mmproj GGUF | not shipped (text-only release) |

Q2_0 uses the same 2-bit codec as PQ2_0 at group 64. In this file every pair
of g64 blocks either has the same fp16 scale or one half has scale 0 with all
its codes 0. The model was trained at g128, so merging to TernSYCL's g128
`int2_f16` layout loses nothing. The packer checks this for every block and
refuses a tensor where it does not hold.

## Changes on this branch for this model

* `scripts/pack_bonsai2_gguf.py` now also packs plain llama.cpp ternary
  Qwen3.5 GGUFs. It merges Q2_0 losslessly to g128 and makes the Hadamard
  contract optional. Without that contract it also undoes the tiled V order
  on the `ssm_out` K axis. Dense block matrices go to the residual checkpoint
  with the same V-head fixes. Non-ternary embedding and lm_head (Q4_1) are
  dequantized to bf16. `mtp_num_hidden_layers` is set to 0, because the GGUF
  has no MTP head. Bonsai 2 GGUFs pack exactly as before.
* `ternsycl_vllm_plugin.py`: vLLM builds the Qwen3.5 `lm_head` with the quant
  config. The plugin used to ternarize any lm_head missing from the sidecar on
  the fly, which is wrong for a Q4_1 head. With a sidecar in use, an lm_head
  that is not in it now stays dense, the same policy the linear layers follow.
* `utils/setup_fresh.sh`: after `unset LD_LIBRARY_PATH` the script now
  re-sources `intel_gpu_vars.sh`. Without it, `ocloc` cannot find
  `libocloc.so` and the AOT device link of `ternsycl_pt_ext` fails.

## Steps

Prerequisites and step 1 (build) are exactly as in [BONSAI2.md](BONSAI2.md).
Run the scripts from a clean login shell. If you use `env -i`, keep
`https_proxy`/`http_proxy` so pip and the HF downloads still work.

### 2. Model files

```bash
source "$DEST/ternsycl_vllm_plugin/.venv/bin/activate"
export MODELS="$DEST/models"; mkdir -p "$MODELS"
# the GGUF (9.76 GB)
hf download penkia/TernaryQuench-Qwen3.8-27B-GGUF TernaryQuench-Qwen3.8-27B-Hybrid-Q2_0.gguf \
    --local-dir "$MODELS"
# tokenizer + HF config: only these five files from the MLX release. Its
# processor_config.json has a nested layout that transformers 5.17 rejects,
# so leave the processor files out; the packer writes working defaults.
hf download penkia/TernaryQuench-Qwen3.8-27B-MLX \
    config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
    --local-dir "$MODELS/TernaryQuench-Qwen3.8-27B-ref"
# vision tower: the GGUF is text-only, but vLLM's Qwen3_5ForConditionalGeneration
# needs the tower's weights. All of model.visual.* sits in shard 1 of the
# base model (3.97 GB).
hf download Qwen/Qwen3.8-27B model-00001-of-00018.safetensors \
    --local-dir "$MODELS/Qwen3.8-27B-shard1"
# the packer requires the PrismML fork's gguf-py (it probes for PQ2_0; Q2_0 is upstream type 42)
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp \
    "$DEST/ternsycl_vllm_plugin/third_party/llama.cpp-prism"
```

### 3. Pack

```bash
cd "$DEST/ternsycl_vllm_plugin"
python scripts/pack_bonsai2_gguf.py \
    --gguf     "$MODELS/TernaryQuench-Qwen3.8-27B-Hybrid-Q2_0.gguf" \
    --ref-dir  "$MODELS/TernaryQuench-Qwen3.8-27B-ref" \
    --template "$MODELS/Qwen3.8-27B-shard1" \
    --out      "$MODELS/TernaryQuench-Qwen3.8-27B.ternsycl-int2_f16.safetensors" \
    --packed   "$MODELS/TernaryQuench-Qwen3.8-27B-packed"
```

Expected (about 2 min, CPU only):

```
[pack] no prism.hadamard contract: plain (unrotated) weights
[pack] GDN nv=48 nk=16 hd=128 hk=128; layers=64
[pack] 295 quantized modules, 370 residual tensors
...
[pack] token_embd.weight: Q4_1 -> bf16 (248320, 5120) (dense)
[pack] output.weight: Q4_1 -> bf16 (248320, 5120) (dense)
[pack] 295 sidecar modules (6.27 GB), 0 hadamard-folded, 0 inverse; 370 residual tensors
[pack] wrote .../TernaryQuench-Qwen3.8-27B.ternsycl-int2_f16.safetensors (6.27 GB, ...)
[pack] wrote .../TernaryQuench-Qwen3.8-27B-packed (703 tensors incl. 333 vision, 7.52 GB; eos=248046 bos=248044)
```

The layout was checked against the original Qwen3.8-27B weights. Every dense
tensor of layers 62-63 matches bit for bit: V-head order, `ssm_out` columns,
conv1d, `A_log`, and the `w+1` norms. The dequantized ternary matrices of
layer 61 have cosine 0.83-0.87 to the originals, a normal value for ternary.
The same `ssm_out` left in tiled order would score 0.03.

### 4. Run on the B70

Same runner as Bonsai 2 with `SIDECAR` and `PACKED` overridden. The log file
name still reads `B2-27B`.

```bash
cd "$DEST/ternsycl_vllm_plugin"
SIDECAR="$MODELS/TernaryQuench-Qwen3.8-27B.ternsycl-int2_f16.safetensors" \
PACKED="$MODELS/TernaryQuench-Qwen3.8-27B-packed" MAXTOK=256 MAXLEN=512 \
    bash scripts/run_bonsai2_gpu.sh "$JOB" B70_TQ27B "Tell me about photosynthesis in 200 words"
```

Expected (B70, 2026-09-25; this sample text is from the non-release
`83e0b19d…` file, and load messages and speed are the same for the release file):

```
[ternsycl] sidecar loaded: .../TernaryQuench-Qwen3.8-27B.ternsycl-int2_f16.safetensors (590 tensors, method=int2_f16)
[ternsycl] language_model.lm_head: not in sidecar, kept dense
[ternsycl] language_model.model.embed_tokens: not in sidecar, kept dense
[ternsycl] fused SwiGLU into the gate epilogue for 62 MLP blocks
Model loading took 13.05 GiB memory
TTFT (prefill)       : 105 ms
decode               : 256 tokens in 6.49 s = 39.45 tok/s   [length]
The user is asking for an explanation of photosynthesis in 200 words. Let me write a concise, informative summary ...
Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy. ...
```

Decode runs at 39.5 tok/s, against 47.6 for Bonsai 2. The expected cause is
the extra dense bytes read per token: the bf16 lm_head (2.5 GB) and the two
bf16 tail layers (1.5 GB), where Bonsai 2 reads about 1 GB of int2 for
the same modules. There are 295 sidecar hits. Layers 62-63 and the
embedding/lm_head run dense in bf16.

### 5. Perplexity (quick runtime check)

This is the model card's WikiText-2 protocol. It runs in about 3 minutes on
the B70; expect 12.45 / 9.91 for the release file (table below).

```bash
srun --jobid="$JOB" --overlap bash -lc "
  source /swtools/intel-gpu/latest/intel_gpu_vars.sh; source /swtools/intel/2026.0/oneapi-vars.sh
  source $DEST/ternsycl_vllm_plugin/.venv/bin/activate
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu VLLM_XPU_ENABLE_XPU_GRAPH=1 TERNSYCL_QUANT_METHOD=int2_f16
  export TERNSYCL_PREQUANT_PATH=$MODELS/TernaryQuench-Qwen3.8-27B.ternsycl-int2_f16.safetensors
  python $DEST/ternsycl_vllm_plugin/scripts/ppl_wikitext2.py $MODELS/TernaryQuench-Qwen3.8-27B-packed"
```

### 6. GSM8K

```bash
SIDECAR="$MODELS/TernaryQuench-Qwen3.8-27B.ternsycl-int2_f16.safetensors" \
PACKED="$MODELS/TernaryQuench-Qwen3.8-27B-packed" LIMIT=1319 \
    bash scripts/eval_bonsai2_lm_eval.sh "$JOB" tq27b_gsm8k1319
```

Settings are the same as the Bonsai 2 run: `gsm8k_cot_llama`, 8-shot,
chat template, thinking with `reasoning_effort=medium`, greedy, up to 4096
generated tokens.

| B70, all 1319 GSM8K test problems | strict-match | flexible-extract | wall time |
|---|---|---|---|
| TernaryQuench Qwen3.8-27B, Hub release (`e640991f…`) | 60.20% (±1.35) | 60.27% (±1.35) | 46 min |
| TernaryQuench Qwen3.8-27B, non-release file (`83e0b19d…`) | 49.20% (±1.38) | 49.28% (±1.38) | 43 min |
| Bonsai 2 27B (TernSYCL, BONSAI2.md) | 96.89% | 96.82% | 41 min |

The low score comes from the model, not from the runtime:

* The model card publishes no GSM8K score, only WikiText-2 perplexity and
  five zero-shot likelihood tasks. The same vLLM + TernSYCL path, with the
  card's protocol (first 20,480 test tokens, 40 × 512 chunks), gives the
  perplexities below. The release file does slightly better than the card.
  That is the expected direction: the card measured the Q2_K repack, which
  adds 1.55% scale error, while TernSYCL runs the Q2_0 weights losslessly.

  | WikiText-2 perplexity | all positions | second half |
  |---|---|---|
  | TernSYCL on B70, Hub release (`scripts/ppl_wikitext2.py`) | 12.45 | 9.91 |
  | model card (Q2_K, llama.cpp Metal) | 12.86 | 10.10 |
  | TernSYCL on B70, non-release file | 13.15 | 10.65 |
* The failures are reasoning errors. In the release run, 25% of the explicit
  `a op b = c` steps are wrong (`13 * 2 = 24`, `160 + 80 + 20 = 240`), and
  300 of the 524 misses contain at least one such slip. The 41 answers
  without a final line are mostly unfinished "wait, let me reconsider" loops.
  The model keeps the base model's likelihood-task scores (the card's
  five-task mean is 72.3 vs 74.4 for BF16). Multi-step arithmetic is where
  this ternary model loses the most.
