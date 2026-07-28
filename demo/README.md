# Bonsai int2 Chat Studio — launch guide

Step-by-step instructions for bringing up the interactive chat demo: a FastAPI
backend holding one warm vLLM engine on an Intel XPU, plus a single-page UI
that streams tokens and reports the **observed decode throughput after every
prompt**. Supports text and image input.

```
browser  ──►  demo/static/index.html   (single page, SSE)
                    │
                    ▼
              demo/server.py           (FastAPI, port 8000)
                    │
                    ▼
              vLLM engine  +  xetla int2 plugin  ──►  Intel GPU
```

---

## 0. Prerequisites (one time)

| Requirement | Notes |
|---|---|
| Intel GPU (Battlemage/Xe2 class, e.g. B70) | ~30 GiB VRAM for the 27B |
| oneAPI + Intel GPU driver | `/swtools/intel/2025.3`, `/swtools/intel-gpu/26.05.37020.3` on this cluster |
| Python venv at `<repo>/.venv` | torch-xpu, vLLM (vendored, branch `xetla_v0.21.0`), fastapi, uvicorn, pillow |
| xetla plugin built into the venv | provides `torch.ops.xetla_int2.*` |
| Model checkpoint | e.g. `prism-ml/Ternary-Bonsai-27B-unpacked` |
| int2 sidecar | pre-packed weights, see step 2 |

If the venv does not exist yet, build it with `utils/setup_fresh.sh` (fresh
machine) or `utils/setup_vllm_xpu.sh`, then install the plugin:

```bash
source .venv/bin/activate
pip install -q --no-build-isolation --no-deps .
# The install replaces the plugin symlink with a copy — restore it so edits to
# xetla_vllm_plugin.py take effect without reinstalling:
ln -sf "$PWD/xetla_vllm_plugin.py" .venv/lib/python3.12/site-packages/xetla_vllm_plugin.py
```

---

## 1. Get a GPU node

The backend must run where the GPU is. On a Slurm cluster, allocate (or reuse)
an interactive job and note the **job id** and **node name**:

```bash
squeue -u "$USER"                       # reuse an existing allocation
# or allocate one:
salloc -p b70 -t 4:00:00
```

Everything below assumes `JOBID=<your job id>` and that the node is reachable
by name (e.g. `pcl-zen4`).

---

## 2. Pre-pack the weights into an int2 sidecar (one time per model)

The 27B never fits in its dense fp16 form (~54 GB), so the demo loads a
**sidecar** of pre-packed int2 weights; the plugin then allocates the dense
tensors on the `meta` device and the checkpoint copies become no-ops.

```bash
srun --jobid="$JOBID" --overlap bash -lc '
source /swtools/intel-gpu/26.05.37020.3/intel_gpu_vars.sh
source /swtools/intel/2025.3/oneapi-vars.sh --force
source ~/FRESH/xetla_vllm_plugin/.venv/bin/activate
cd ~/FRESH/xetla_vllm_plugin
python scripts/pack_bonsai_hf.py \
    --model prism-ml/Ternary-Bonsai-27B-unpacked \
    --out   ~/FRESH/Ternary-Bonsai-27B.xetla-int2_f16.safetensors'
```

Produces ~7.1 GB from a 51 GB checkpoint (306 modules, including `lm_head` and
`embed_tokens`), round-trip error 0. Skip this step if the sidecar already
exists — `serve.sh` finds it automatically at either of:

* `<model dir>.xetla-int2_f16.safetensors`
* `<repo>/../Ternary-Bonsai-27B.xetla-int2_f16.safetensors`

---

## 3. Start the backend

```bash
cd ~/FRESH/xetla_vllm_plugin/demo
JOBID=<your job id> ./serve.sh
```

`serve.sh` sources the GPU driver + oneAPI vars, activates the venv, locates
the sidecar, and launches uvicorn on `0.0.0.0:8000`.

To keep it running after you close the terminal:

```bash
cd ~/FRESH/xetla_vllm_plugin/demo
setsid nohup srun --jobid=<your job id> --overlap bash -lc \
  'cd ~/FRESH/xetla_vllm_plugin/demo && PORT=8000 ./serve.sh' \
  > ~/FRESH/demo_server.log 2>&1 < /dev/null & disown
tail -f ~/FRESH/demo_server.log
```

Startup takes ~100 s the first time (61 s of it is `torch.compile`; the graph
is cached under `~/.cache/vllm/torch_compile_cache`, so later launches are much
faster). Wait for:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Healthy startup also logs, for Ternary-Bonsai-27B on a B70:

```
Model loading took 7.72 GiB memory        # 6.85 GiB int2 LM + ~0.9 GiB fp16 vision tower
GPU KV cache size: 257,088 tokens
```

---

## 4. Open the UI

### 4a. From your laptop over SSH

```bash
ssh -L 8000:<node>:8000 <login-host>
# then open http://localhost:8000
```

### 4b. Inside VS Code Remote (preview pane / Simple Browser)

VS Code forwards ports from the **login node**, which cannot see the compute
node's port. Run the bundled relay on the login node:

```bash
cd ~/FRESH/xetla_vllm_plugin/demo
setsid nohup python port_relay.py 8765 <node> 8000 > ~/FRESH/demo_relay.log 2>&1 < /dev/null & disown
```

Then open <http://localhost:8765>. Use a port that is not already forwarded
(8000 is often taken by a stale tunnel, which shows up as a blank preview).

---

## 5. Use it

* Type a message, **Enter** to send (Shift+Enter for a newline).
* **Images**: the 🖼️ button, drag-and-drop onto the chat, or paste from the
  clipboard. Up to 4 per conversation; they are downscaled server-side to
  1024 px on the longest side.
* **Thinking mode** checkbox toggles the model's reasoning phase; the
  `<think>` block renders as a collapsible section.
* Sampling controls, system prompt, and **Clear conversation** are in the right
  sidebar, together with an engine panel (quant method, sidecar, KV cache size)
  and running session totals.
* **Stop** aborts an in-flight generation.

### What the tok/s numbers mean

After every prompt a stat strip appears under the reply:

| Stat | Definition |
|---|---|
| **decode tok/s** | `(output_tokens - 1) / (t_end - t_first_token)` — the steady-state generation rate, **excluding** prefill. This is the headline number and matches `scripts/bench_model.py`. |
| TTFT | time to first token (prefill + scheduling) |
| prefill tok/s | `prompt_tokens / TTFT` |
| end-to-end tok/s | `output_tokens / total_time`, i.e. including prefill |
| output / prompt tokens | as counted by the engine; image tokens are included in the prompt count |

The header badge keeps the last turn's decode rate and the sidebar accumulates
a session average using the same definition.

Reference numbers, Ternary-Bonsai-27B int2 on a single B70:

| Turn | decode | TTFT |
|---|---|---|
| text | ~46–48 tok/s | ~285 ms |
| image (448×448, 224 prompt tokens) | ~45 tok/s | ~670 ms warm, ~2.9 s on the first image |

---

## 6. Configuration

All knobs are environment variables read by `demo/server.py`; set them before
`./serve.sh`.

| Variable | Default | Purpose |
|---|---|---|
| `DEMO_MODEL` | Ternary-Bonsai-27B snapshot | HF dir or repo id |
| `DEMO_TOKENIZER` | `DEMO_MODEL` | override if the checkpoint has no tokenizer |
| `DEMO_QUANT` | `xetla` | `none` disables the plugin (dense baseline) |
| `DEMO_DTYPE` | `float16` | |
| `DEMO_MAX_MODEL_LEN` | `8192` | context window |
| `DEMO_GPU_MEM_UTIL` | `0.85` | lower it on integrated GPUs |
| `DEMO_MAX_IMAGES` | `4` | images per conversation |
| `DEMO_MAX_IMAGE_SIDE` | `1024` | server-side downscale; caps image tokens |
| `DEMO_TEXT_ONLY` | `0` | `1` disables the vision tower (saves ~0.9 GiB) |
| `DEMO_ENFORCE_EAGER` | `0` | `1` skips `torch.compile` (faster startup) |
| `DEMO_WARMUP` | `1` | pays the first-token JIT cost at startup |
| `XETLA_QUANT_METHOD` | `int2_f16` | plugin kernel family |
| `XETLA_PREQUANT_PATH` | auto-detected | explicit sidecar path |
| `PORT` / `HOST` | `8000` / `0.0.0.0` | uvicorn bind |

Examples:

```bash
# 8B instead of 27B (text-only model, so no image input)
DEMO_MODEL=prism-ml/Bonsai-8B-unpacked \
XETLA_PREQUANT_PATH=~/FRESH/Bonsai-8B.xetla-int2_f16.safetensors ./serve.sh

# text-only, longer context
DEMO_TEXT_ONLY=1 DEMO_MAX_MODEL_LEN=32768 ./serve.sh

# dense fp16 baseline for comparison
DEMO_QUANT=none ./serve.sh
```

---

## 7. HTTP API

The UI is a thin client over these endpoints, so scripts can use them too.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | the single-page UI |
| `GET` | `/health` | `{"ready": bool, "error": str\|null}` |
| `GET` | `/config` | model, quantization, device, memory, KV cache, image support |
| `POST` | `/chat` | SSE stream: `delta` events, then one `stats` event, then `done` |
| `POST` | `/abort` | stop the in-flight generation |

```bash
curl -sN --noproxy '*' -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three primary colors."}],
       "max_tokens":150,"thinking":false}'
```

```
event: delta
data: {"text": "Red"}
...
event: stats
data: {"decode_tps": 45.57, "ttft_ms": 284.3, "decode_s": 3.27, "total_s": 3.55,
       "output_tokens": 150, "prompt_tokens": 17, "prefill_tps": 59.8,
       "images": 0, "finish_reason": "length", "overall_tps": 42.21}
event: done
data: {}
```

Images are sent as `data:` URLs in `messages[].images`.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Preview pane is blank | The server is on the compute node; VS Code forwards the login node. Start `port_relay.py` (step 4b) and use a free port. |
| `curl` returns `403` on localhost | The cluster `http_proxy` intercepts localhost. Add `--noproxy '*'`. |
| `Unable to confirm allocation for job N` | The Slurm job expired; get a new one with `squeue -u $USER`. |
| Loads slowly / uses ~54 GB and OOMs | The sidecar was not found, so weights loaded dense. Check the `[serve.sh] xetla sidecar:` line and `/config` → `"prequantized": true`. |
| Model badge shows a hex hash | HF cache path; cosmetic only, the UI now resolves the repo name. |
| Startup takes ~100 s every time | `torch.compile` cache is cold or unwritable; check `~/.cache/vllm/torch_compile_cache`. Use `DEMO_ENFORCE_EAGER=1` to skip compilation. |
| Image button disabled | `/config` reports `supports_images: false` — either `DEMO_TEXT_ONLY=1` or the model has no vision tower. |
| Generation looks stuck | One GPU serves one turn at a time; a second request waits for the lock. |

Server log: `~/FRESH/demo_server.log` (or the terminal running `serve.sh`).
