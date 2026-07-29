# SPDX-License-Identifier: Apache-2.0
"""FastAPI backend for the xetla int2 Bonsai chat studio (Intel XPU).

Keeps one warm vLLM engine in-process and streams tokens to a single-page UI,
reporting the observed decode throughput (tok/s) after every prompt.

Endpoints:
  GET  /            -> the single-page chat UI (static/index.html)
  GET  /health      -> readiness probe
  GET  /config      -> model / quantization / device / memory summary
  POST /chat        -> Server-Sent Events: token deltas, then per-turn metrics
  POST /abort       -> stop the in-flight generation

Configuration is entirely environment driven so the same server can host any
model the plugin supports:

  DEMO_MODEL            HF dir or repo id                (required)
  DEMO_TOKENIZER        defaults to DEMO_MODEL
  DEMO_QUANT            xetla | none                     (xetla)
  DEMO_DTYPE            float16
  DEMO_MAX_MODEL_LEN    8192
  DEMO_GPU_MEM_UTIL     0.85
  DEMO_MAX_IMAGES       max images per prompt            (4)
  DEMO_MAX_IMAGE_SIDE   downscale longest side to        (1024)
  DEMO_TEXT_ONLY        1 disables image inputs          (0)
  DEMO_ENFORCE_EAGER    1 disables torch.compile         (0)
  DEMO_WARMUP           1 pays the first-token JIT cost  (1)

The xetla plugin itself is configured as usual via XETLA_QUANT_METHOD /
XETLA_PREQUANT_PATH; see demo/serve.sh.

NOTE: serve.sh sets VLLM_DISABLE_COMPILE_CACHE=1 by default. vLLM's AOT
torch.compile artifacts are keyed in a way that does not capture every engine
setting this demo varies (context length, multimodal on/off, eager), and
reusing a mismatched artifact does not fail loudly: it either raises
"'NoneType' object has no attribute 'size'" inside the compiled graph or,
worse, silently generates degenerate repeating text. Recompiling costs ~60 s
per start; set DEMO_COMPILE_CACHE=1 to opt back in.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).parent / "static"

# Exit code asking serve.sh to relaunch us with a smaller context.
RETRY_EXIT_CODE = 42

# A single decoded image is capped at this many bytes before resizing, so a
# malicious/oversized upload cannot exhaust host memory.
MAX_IMAGE_BYTES = 24 * 1024 * 1024


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Config:
    """Resolved demo configuration, read once at startup."""

    def __init__(self) -> None:
        self.model = os.environ.get("DEMO_MODEL", "")
        self.tokenizer = os.environ.get("DEMO_TOKENIZER") or self.model
        quant = os.environ.get("DEMO_QUANT", "xetla")
        self.quant = None if quant.lower() in ("", "none") else quant
        self.dtype = os.environ.get("DEMO_DTYPE", "float16")
        self.max_model_len = int(os.environ.get("DEMO_MAX_MODEL_LEN", "8192"))
        self.gpu_mem_util = float(os.environ.get("DEMO_GPU_MEM_UTIL", "0.85"))
        self.text_only = _env_bool("DEMO_TEXT_ONLY", False)
        self.max_images = 0 if self.text_only else int(os.environ.get("DEMO_MAX_IMAGES", "4"))
        self.max_image_side = int(os.environ.get("DEMO_MAX_IMAGE_SIDE", "1024"))
        self.enforce_eager = _env_bool("DEMO_ENFORCE_EAGER", False)
        self.warmup = _env_bool("DEMO_WARMUP", True)


CFG = Config()


# ---------------------------------------------------------------------------
# Integrated-GPU memory tuning
# ---------------------------------------------------------------------------
# On integrated GPUs (Lunar Lake, Arrow Lake, Meteor Lake, ...) the "VRAM" is
# system RAM. vLLM budgets the KV cache as
#     requested = total_memory * gpu_memory_utilization
# and refuses to start if mem_get_info() reports less *free* than that. On an
# iGPU that free figure is essentially MemFree, so page cache left over from
# reading a multi-GB checkpoint counts as "used" and can starve the engine
# even though the memory is trivially reclaimable ("No available memory for
# the cache blocks"). So: reclaim the cache first, then size the budget from
# what is really free.
def _meminfo() -> dict[str, float]:
    """/proc/meminfo in GiB."""
    info: dict[str, float] = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0]) / 1024**2
    except OSError:
        pass
    return info


def _is_integrated_gpu(name: str, total_bytes: int) -> bool:
    keywords = ("lunar", "lnl", "meteor", "mtl", "arrow", "arl",
                "iris", "arc(tm) graphics")
    if any(k in name.lower() for k in keywords):
        return True
    sys_total = _meminfo().get("MemTotal", 0.0) * 2**30
    return bool(sys_total) and abs(total_bytes - sys_total) / sys_total < 0.15


def _reclaim_page_cache(target_gib: float) -> None:
    """Fault in a large anonymous mapping to make the kernel drop page cache,
    then release it. Raises MemFree without needing root."""
    import mmap

    size = int(target_gib * 2**30)
    if size <= 0:
        return
    try:
        buf = mmap.mmap(-1, size)
    except (OSError, ValueError) as exc:
        print(f"[demo] page-cache reclaim skipped: {exc}", flush=True)
        return
    try:
        chunk = b"\0" * (64 * 1024 * 1024)
        written = 0
        while written < size:
            written += buf.write(chunk[: min(len(chunk), size - written)])
    except (OSError, ValueError):
        pass
    finally:
        buf.close()


def _tune_integrated_gpu(cfg: Config) -> None:
    """Adjust cfg.gpu_mem_util in place when running on an integrated GPU."""
    if os.environ.get("DEMO_GPU_MEM_UTIL"):
        return  # explicit user choice wins
    try:
        import torch

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return
        props = torch.xpu.get_device_properties(0)
        if not _is_integrated_gpu(props.name, props.total_memory):
            return
    except Exception:  # noqa: BLE001 - detection must never block startup
        return

    info = _meminfo()
    free_gib = info.get("MemFree", 0.0)
    avail_gib = info.get("MemAvailable", 0.0)
    headroom = float(os.environ.get("DEMO_RECLAIM_HEADROOM_GIB", "1.0"))
    if avail_gib and free_gib < avail_gib - headroom:
        target = avail_gib - headroom
        print(f"[demo] integrated GPU: reclaiming page cache "
              f"(MemFree {free_gib:.1f} -> target {target:.1f} GiB) ...", flush=True)
        _reclaim_page_cache(target)

    try:
        import torch

        free_b, total_b = torch.xpu.mem_get_info(0)
    except Exception:  # noqa: BLE001
        return

    # Between this measurement and vLLM's own snapshot, torch/vLLM init costs
    # ~1.5-2 GiB on an iGPU; budget below that or vLLM refuses to start with
    # "Free memory on device ... is less than desired GPU memory utilization".
    reserve_b = float(os.environ.get("DEMO_INIT_RESERVE_GIB", "2.0")) * 2**30
    util = max(0.20, min(0.92, (free_b - reserve_b) / total_b))
    cfg.gpu_mem_util = round(util, 2)
    budget_gib = cfg.gpu_mem_util * total_b / 2**30
    print(f"[demo] integrated GPU detected: free {free_b / 2**30:.1f} / "
          f"{total_b / 2**30:.1f} GiB, reserving {reserve_b / 2**30:.1f} GiB "
          f"-> gpu_memory_utilization={cfg.gpu_mem_util} "
          f"({budget_gib:.1f} GiB budget)", flush=True)

    # Shrink the things that eat that budget before the KV cache gets any:
    # the profiling run scales with max_model_len, torch.compile keeps a
    # multi-GiB workspace, and the vision tower is ~0.9 GiB of weights. On a
    # shared-memory GPU there is rarely room for all three plus a usable KV
    # cache, so trade them away by default. Explicit settings always win.
    notes = []
    if not os.environ.get("DEMO_MAX_MODEL_LEN"):
        cfg.max_model_len = min(cfg.max_model_len, 2048)
        notes.append(f"max_model_len={cfg.max_model_len}")
    if not os.environ.get("DEMO_ENFORCE_EAGER"):
        cfg.enforce_eager = True
        notes.append("enforce_eager")
    if not os.environ.get("DEMO_TEXT_ONLY"):
        cfg.text_only = True
        cfg.max_images = 0
        notes.append("text_only (vision tower costs ~0.9 GiB; "
                     "set DEMO_TEXT_ONLY=0 to keep image input)")
    if notes:
        print("[demo] integrated GPU defaults: " + ", ".join(notes), flush=True)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = ""
    # data: URLs (or bare base64) for user messages, at most CFG.max_images.
    images: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    max_tokens: int = Field(512, ge=1, le=32768)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    thinking: bool = True


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ChatEngine:
    """Wraps one warm vLLM engine; serializes turns (single GPU)."""

    def __init__(self, cfg: Config) -> None:
        if not cfg.model:
            raise RuntimeError("DEMO_MODEL is not set")
        _tune_integrated_gpu(cfg)
        from vllm import LLM

        self.cfg = cfg
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # How long a turn waits for the engine before giving up, so a stuck
        # generation surfaces as an error rather than an endless spinner.
        self.busy_timeout = float(os.environ.get("DEMO_BUSY_TIMEOUT", "900"))

        extra: dict[str, Any] = {}
        if cfg.text_only:
            extra["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        else:
            extra["limit_mm_per_prompt"] = {"image": cfg.max_images, "video": 0}

        t0 = time.perf_counter()
        self.llm = self._build_llm(cfg, extra)
        self.load_seconds = time.perf_counter() - t0
        self.engine = self.llm.llm_engine
        self.tokenizer = self.llm.get_tokenizer()
        self.supports_images = (not cfg.text_only) and _model_has_vision(self.llm)

    @staticmethod
    def _build_llm(cfg: Config, extra: dict[str, Any]):
        """Construct the engine. If the KV cache does not fit, ask the wrapper
        (serve.sh) to restart us with half the context: on memory-constrained
        integrated GPUs that is often the difference between "no cache blocks"
        and a working demo, and a failed vLLM build cannot release its device
        memory in-process, so retrying here would only make things worse."""
        from vllm import LLM

        try:
            return LLM(
                model=cfg.model,
                tokenizer=cfg.tokenizer,
                max_model_len=cfg.max_model_len,
                gpu_memory_utilization=cfg.gpu_mem_util,
                trust_remote_code=True,
                enable_prefix_caching=False,
                quantization=cfg.quant,
                dtype=cfg.dtype,
                enforce_eager=cfg.enforce_eager,
                **extra,
            )
        except ValueError as exc:
            message = str(exc)
            out_of_kv = "KV cache" in message or "cache blocks" in message
            retry_file = os.environ.get("DEMO_RETRY_FILE")
            if not (out_of_kv and retry_file) or cfg.max_model_len <= 512:
                raise
            retry_len = cfg.max_model_len // 2
            print(f"[demo] KV cache does not fit at "
                  f"max_model_len={cfg.max_model_len}; "
                  f"restarting with {retry_len}", flush=True)
            with open(retry_file, "w") as fh:
                fh.write(str(retry_len))
            raise SystemExit(RETRY_EXIT_CODE) from exc

    # -- prompt building ---------------------------------------------------
    def build_prompt(self, req: ChatRequest) -> tuple[Any, int]:
        """Return (vLLM prompt, image count) for the conversation."""
        chat: list[dict[str, Any]] = []
        images: list[Any] = []
        for msg in req.messages:
            if msg.images and msg.role == "user" and self.supports_images:
                parts: list[dict[str, Any]] = []
                for data_url in msg.images[: self.cfg.max_images]:
                    images.append(_decode_image(data_url, self.cfg.max_image_side))
                    parts.append({"type": "image"})
                if msg.content:
                    parts.append({"type": "text", "text": msg.content})
                chat.append({"role": msg.role, "content": parts})
            else:
                chat.append({"role": msg.role, "content": msg.content})

        if len(images) > self.cfg.max_images:
            raise HTTPException(
                status_code=400,
                detail=f"At most {self.cfg.max_images} images per conversation",
            )

        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            text = self.tokenizer.apply_chat_template(
                chat, enable_thinking=req.thinking, **kwargs)
        except TypeError:
            # Template without an enable_thinking parameter.
            text = self.tokenizer.apply_chat_template(chat, **kwargs)

        if not images:
            return text, 0
        return {"prompt": text, "multi_modal_data": {"image": images}}, len(images)

    # -- generation --------------------------------------------------------
    def stream(self, req: ChatRequest) -> Iterator[str]:
        """Stream one turn as SSE.

        The engine loop runs in a worker thread that owns the lock, and the
        HTTP response only drains a queue. That decoupling matters: if the
        client disconnects, Starlette may never resume this generator, so
        holding the lock across the engine loop here would leak it and wedge
        every later request behind a generation nobody is reading (the UI then
        sits on "thinking..." forever).
        """
        prompt, n_images = self.build_prompt(req)
        events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

        worker = threading.Thread(
            target=self._generate_into,
            args=(req, prompt, n_images, events),
            daemon=True,
        )
        worker.start()
        try:
            while True:
                item = events.get()
                if item is None:
                    break
                yield _sse(item[0], item[1])
        except GeneratorExit:
            # Reader went away: ask the worker to stop; it owns the cleanup.
            self._stop.set()
            raise

    def _generate_into(self, req: ChatRequest, prompt: Any, n_images: int,
                       events: "queue.Queue") -> None:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        acquired = False
        try:
            # One GPU, one engine: one turn at a time. Bounded so a wedged
            # generation surfaces as an error instead of an infinite spinner.
            acquired = self._lock.acquire(timeout=self.busy_timeout)
            if not acquired:
                events.put(("error", {"detail": "engine busy, try again"}))
                return

            self._stop.clear()
            req_id = f"chat-{time.time_ns()}"
            self.engine.add_request(req_id, prompt, params)

            t_start = time.perf_counter()
            t_first: float | None = None
            sent = 0
            n_out = 0
            n_prompt = 0
            finish_reason = None
            aborted = False
            while self.engine.has_unfinished_requests():
                if self._stop.is_set() and not aborted:
                    self.engine.abort_request(req_id)
                    aborted = True
                for out in self.engine.step():
                    if out.request_id != req_id:
                        continue
                    completion = out.outputs[0]
                    text = completion.text
                    if len(text) > sent:
                        if t_first is None:
                            t_first = time.perf_counter()
                        events.put(("delta", {"text": text[sent:]}))
                        sent = len(text)
                    n_out = len(completion.token_ids)
                    n_prompt = len(getattr(out, "prompt_token_ids", None) or ()) or n_prompt
                    if out.finished:
                        finish_reason = completion.finish_reason
            t_end = time.perf_counter()

            events.put(("stats", _metrics(
                t_start=t_start, t_first=t_first, t_end=t_end,
                n_prompt=n_prompt, n_out=n_out, n_images=n_images,
                finish_reason="aborted" if aborted else finish_reason,
            )))
            events.put(("done", {}))
        except Exception as exc:  # noqa: BLE001 - report, never wedge the lock
            events.put(("error", {"detail": str(exc)}))
        finally:
            if acquired:
                self._lock.release()
            events.put(None)

    def abort(self) -> None:
        self._stop.set()

    def warmup(self) -> None:
        try:
            for _ in self.stream(ChatRequest(
                    messages=[Message(role="user", content="hi")],
                    max_tokens=8, temperature=0.0, thinking=False)):
                pass
        except Exception:  # noqa: BLE001 - warmup must never block startup
            pass


def _metrics(*, t_start: float, t_first: float | None, t_end: float,
             n_prompt: int, n_out: int, n_images: int,
             finish_reason: str | None) -> dict[str, Any]:
    """Per-turn timings. `decode_tps` is the observed inter-token rate, i.e.
    it excludes prefill: (tokens - 1) / (end - first token)."""
    ttft = (t_first - t_start) if t_first else None
    decode_s = (t_end - t_first) if t_first else None
    decode_tps = ((n_out - 1) / decode_s) if (decode_s and n_out > 1) else None
    prefill_tps = (n_prompt / ttft) if (ttft and n_prompt) else None
    total_s = t_end - t_start
    return {
        "decode_tps": round(decode_tps, 2) if decode_tps else None,
        "ttft_ms": round(ttft * 1e3, 1) if ttft else None,
        "decode_s": round(decode_s, 2) if decode_s else None,
        "total_s": round(total_s, 2),
        "output_tokens": n_out,
        "prompt_tokens": n_prompt,
        "prefill_tps": round(prefill_tps, 1) if prefill_tps else None,
        "images": n_images,
        "finish_reason": finish_reason,
        "overall_tps": round(n_out / total_s, 2) if total_s > 0 and n_out else None,
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _decode_image(data_url: str, max_side: int):
    """Decode a browser data: URL into a downscaled RGB PIL image."""
    from PIL import Image

    payload = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Malformed image data") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # noqa: BLE001 - any decoder failure is a bad upload
        raise HTTPException(status_code=400, detail="Unsupported image format") from exc

    image = image.convert("RGB")
    longest = max(image.size)
    if max_side and longest > max_side:
        scale = max_side / longest
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.LANCZOS,
        )
    return image


def _model_has_vision(llm: Any) -> bool:
    try:
        hf_config = llm.llm_engine.vllm_config.model_config.hf_config
    except Exception:  # noqa: BLE001
        return False
    if getattr(hf_config, "language_model_only", False):
        return False
    return getattr(hf_config, "vision_config", None) is not None


def _device_name() -> str:
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.get_device_properties(0).name
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).name
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _memory_gib() -> dict[str, Any]:
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            free_b, total_b = torch.xpu.mem_get_info(0)
            return {
                "used_gib": round((total_b - free_b) / 2**30, 2),
                "total_gib": round(total_b / 2**30, 2),
            }
    except Exception:  # noqa: BLE001
        pass
    return {}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Bonsai int2 Chat Studio")
_engine: ChatEngine | None = None
_engine_error: str | None = None


@app.on_event("startup")
def _startup() -> None:
    global _engine, _engine_error
    try:
        _engine = ChatEngine(CFG)
    except Exception as exc:  # noqa: BLE001 - surface load failures in the UI
        _engine_error = str(exc)
        raise
    if CFG.warmup:
        _engine.warmup()


def _require_engine() -> ChatEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail=_engine_error or "Engine not ready")
    return _engine


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ready": _engine is not None, "error": _engine_error}


@app.get("/config")
def config() -> dict[str, Any]:
    info: dict[str, Any] = {
        "model": CFG.model,
        "quantization": CFG.quant or "none",
        "quant_method": os.environ.get("XETLA_QUANT_METHOD", "-"),
        "prequantized": bool(os.environ.get("XETLA_PREQUANT_PATH")),
        "dtype": CFG.dtype,
        "max_model_len": CFG.max_model_len,
        "device": _device_name(),
        "ready": _engine is not None,
        "supports_images": bool(_engine and _engine.supports_images),
        "max_images": CFG.max_images,
        **_memory_gib(),
    }
    if _engine is not None:
        info["load_seconds"] = round(_engine.load_seconds, 1)
        info["kv_cache_tokens"] = _kv_cache_tokens(_engine)
    return info


def _kv_cache_tokens(engine: ChatEngine) -> int | None:
    try:
        cache = engine.engine.vllm_config.cache_config
        return int(cache.num_gpu_blocks) * int(cache.block_size)
    except Exception:  # noqa: BLE001
        return None


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    engine = _require_engine()
    return StreamingResponse(
        engine.stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/abort")
def abort() -> dict[str, Any]:
    _require_engine().abort()
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
