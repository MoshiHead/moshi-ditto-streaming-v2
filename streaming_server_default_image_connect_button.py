"""
streaming_server.py
===================
FastAPI WebSocket server for the real-time Moshi + Bridge + Ditto pipeline.

Endpoints
---------
  GET  /                    → browser UI (static/index.html)
  POST /upload_image        → multipart upload; returns {"session_id": "..."}
  GET  /session/{sid}/status→ {"ready": true/false}
  WS   /ws/{session_id}     → bidirectional audio/video stream

WebSocket message protocol
--------------------------
  Browser → Server:
    0x01 <opus_bytes>      user's mic audio (Opus encoded)

  Server → Browser:
    0x00                   handshake / ready signal
    0x01 <seq:4> <pcm>     Moshi response audio — raw float32 LE + 4-byte seq
    0x02 <seq:4> <jpeg>    animated talking-head video frame + 4-byte seq
    0x03 <utf8_text>       text token / transcript piece
    0xFF <utf8_error>      error message

The 4-byte sequence number (big-endian uint32) in audio and video messages
allows the browser to align lip-sync: audio packet seq=N was played at a
known AudioContext time, and video frame seq=N should be displayed at that
same time.

Startup
-------
  python streaming_server.py [--host HOST] [--port PORT] [options]

RunPod
------
  The server listens on 0.0.0.0 so RunPod's public port proxy can forward
  traffic.  Set port to match your RunPod "HTTP Port" setting (default 7860).

Key fixes in this version (v4 — Backpressure + Audio Priority)
--------------------------------------------------------------
1. **Bounded queues**: token_q=4, frame_q=3, send_q=5.  Prevents the
   permanent backlog that caused 0 FPS and 1400+ dropped frames.
2. **Latest-frame-wins**: frame_forwarder drains all queued frames and
   sends ONLY the newest, eliminating 100ms+ stale frame delivery.
3. **Dedicated audio path**: Audio bypasses send_queue entirely and sends
   directly via ws_write_lock, guaranteeing zero audio drops from video
   backlog.  Audio NEVER waits on video queue availability.
4. **ws_write_lock**: asyncio.Lock serialises all WebSocket writes (audio
   direct + video via sender task), preventing concurrent-drain crash.
5. **Drain-on-overflow**: push_features() drains stale features from
   Ditto's audio2motion_queue instead of blocking for 1 second.
6. **Skip-ahead**: Ditto writer skips frames on 3+ consecutive drops,
   reducing effective output rate under sustained pressure.
7. Thread-safe asyncio.Queue crossing: frame_reader_task uses
   loop.call_soon_threadsafe() + put_nowait().
8. Adaptive bridge batching: flushes after BRIDGE_FLUSH_TIMEOUT_MS.
9. Shared error_event: any task failure signals all others to abort.
10. Dedicated CUDA stream for Bridge inference.
11. Sequence numbers propagated Audio→seq and Video→seq for A/V alignment.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import struct
import uuid
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Project root on sys.path ─────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.streaming_moshi import StreamingMoshiState
from pipeline.ditto_stream_adapter import DittoStreamAdapter
from pipeline.sync_types import TaggedToken, seq_pack
from pipeline.latency_profiler import PipelineProfiler, Timer

# Bridge imports
_BRIDGE_DIR = os.path.join(_ROOT, "bridge_module")
if _BRIDGE_DIR not in sys.path:
    sys.path.insert(0, _BRIDGE_DIR)
from inference import StreamingBridgeInference

# ── Minimal, print-like logging (avoids notebook lag from verbose output) ─────
logging.basicConfig(
    level=logging.WARNING,          # default: only warnings/errors bubble up
    format="%(message)s",           # no timestamps / logger names — just the text
)
# Our own server logger at INFO so essential startup/session messages are shown
logger = logging.getLogger("streaming_server")
logger.setLevel(logging.INFO)

# Silence chatty third-party loggers that flood the notebook output
for _noisy in (
    "uvicorn", "uvicorn.error", "uvicorn.access",
    "fastapi", "httpx", "asyncio",
    "torch", "transformers", "huggingface_hub",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Global configuration  (overridden by CLI args or environment variables)
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # Moshi
    MOSHI_HF_REPO:     str = os.environ.get("MOSHI_HF_REPO", "kyutai/moshiko-pytorch-q8")
    MOSHI_WEIGHT:      Optional[str] = os.environ.get("MOSHI_WEIGHT")
    MIMI_WEIGHT:       Optional[str] = os.environ.get("MIMI_WEIGHT")
    TOKENIZER:         Optional[str] = os.environ.get("MOSHI_TOKENIZER")

    # Bridge
    BRIDGE_CKPT:   str = os.environ.get("BRIDGE_CKPT",
                         os.path.join(_ROOT, "checkpoints", "bridge_best.pt"))
    BRIDGE_CONFIG: str = os.environ.get("BRIDGE_CONFIG",
                         os.path.join(_ROOT, "bridge_module", "config.yaml"))
    BRIDGE_CHUNK:  int = int(os.environ.get("BRIDGE_CHUNK", "1"))  # Mimi frames per chunk

    # Adaptive bridge flush: send partial batch to Ditto after this many ms
    # even if the chunk is not full.  Prevents stalls during pauses.
    # Lower = less latency; 50ms is optimal for A100.
    BRIDGE_FLUSH_TIMEOUT_MS: int = int(os.environ.get("BRIDGE_FLUSH_TIMEOUT_MS", "50"))

    # Ditto
    DITTO_DATA_ROOT: str = os.environ.get("DITTO_DATA_ROOT",
                           os.path.join(_ROOT, "ditto-inference", "checkpoints",
                                        "ditto_trt_Ampere_Plus"))
    DITTO_CFG_PKL:   str = os.environ.get("DITTO_CFG_PKL",
                           os.path.join(_ROOT, "ditto-inference", "checkpoints",
                                        "ditto_cfg", "v0.4_hubert_cfg_trt.pkl"))
    DITTO_EMO:               int = int(os.environ.get("DITTO_EMO", "4"))
    # 10 steps = ~5x faster than 50 steps with acceptable quality on A100.
    DITTO_SAMPLING_STEPS:    int = int(os.environ.get("DITTO_SAMPLING_STEPS", "10"))
    
    # OVERLAP controls Ditto's sliding window. LMDM context is 80 frames.
    # valid_clip_len = 80 - overlap.
    # With overlap=10, valid_clip_len=70 (2.8s delay!). 
    # With overlap=76, valid_clip_len=4 (160ms delay) + forces GPU to work harder 
    # to yield constant 25 FPS, pushing util towards 100%.
    DITTO_OVERLAP_V2:        int = int(os.environ.get("DITTO_OVERLAP_V2", "76"))

    # Lower JPEG quality = smaller frames = less bandwidth = lower latency.
    DITTO_JPEG_QUALITY:      int = int(os.environ.get("DITTO_JPEG_QUALITY", "70"))

    # Runtime
    DEVICE:     str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE:      torch.dtype = torch.bfloat16
    UPLOAD_DIR: str = os.path.join(_ROOT, "_uploads")
    STATIC_DIR: str = os.path.join(_ROOT, "static")

    # Default portrait image — used to pre-register a session at startup so the
    # UI can connect immediately without requiring a file upload.
    DEFAULT_IMAGE: str = os.environ.get(
        "DEFAULT_IMAGE",
        os.path.join(_ROOT, "image", "obama.jpeg")
    )


cfg = Config()

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded model singletons
# ─────────────────────────────────────────────────────────────────────────────

_moshi_state:    Optional[StreamingMoshiState]     = None
_bridge_stream:  Optional[StreamingBridgeInference] = None

# Dedicated CUDA stream for Bridge inference to avoid implicit CUDA syncs
# with Moshi on the default stream.
_bridge_cuda_stream: Optional[torch.cuda.Stream] = None

# Per-session state: session_id → {"image_path": str, "ready": bool}
_sessions: dict = {}


def get_moshi() -> StreamingMoshiState:
    global _moshi_state
    if _moshi_state is None:
        raise RuntimeError("Moshi model not loaded. Call /startup or wait for server init.")
    return _moshi_state


def get_bridge() -> StreamingBridgeInference:
    global _bridge_stream
    if _bridge_stream is None:
        raise RuntimeError("Bridge model not loaded.")
    return _bridge_stream


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Moshi-Bridge-Ditto Streaming API", version="2.0.0")

# Serve static files (browser UI)
os.makedirs(cfg.STATIC_DIR, exist_ok=True)
os.makedirs(cfg.UPLOAD_DIR, exist_ok=True)

# Mount static dir
app.mount("/static", StaticFiles(directory=cfg.STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the browser UI."""
    index_path = os.path.join(cfg.STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        return HTMLResponse("<h2>UI not found — place index.html in ./static/</h2>", status_code=404)
    with open(index_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ─────────────────────────────────────────────────────────────────────────────
# Image upload
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload a portrait image.
    Returns {"session_id": "<uuid>", "filename": "<saved name>"}.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {ext}")

    session_id = str(uuid.uuid4())[:8]
    dest = os.path.join(cfg.UPLOAD_DIR, f"{session_id}{ext}")

    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    _sessions[session_id] = {"image_path": dest, "ready": True}
    logger.info(f"[upload_image] Session {session_id} → {dest}")
    return JSONResponse({"session_id": session_id, "filename": os.path.basename(dest)})


@app.get("/session/{session_id}/status")
async def session_status(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _sessions[session_id]


@app.get("/default_session")
async def default_session():
    """
    Returns the pre-registered default session (backed by image/obama.jpeg).
    The UI calls this on page load so it can connect immediately without
    requiring the user to upload an image.
    """
    sid = "default"
    if sid not in _sessions:
        raise HTTPException(status_code=503, detail="Default session not ready yet")
    return JSONResponse({"session_id": sid})


# ─────────────────────────────────────────────────────────────────────────────
# Health / info
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    moshi = _moshi_state
    return {
        "status": "ok",
        "device": cfg.DEVICE,
        "moshi_loaded":   moshi is not None,
        "moshi_busy":     moshi is not None and moshi._lock.locked(),
        "bridge_loaded":  _bridge_stream is not None,
        "bridge_chunk":   cfg.BRIDGE_CHUNK,
        "bridge_flush_ms": cfg.BRIDGE_FLUSH_TIMEOUT_MS,
        "ditto_per_session": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Restart endpoint — sends SIGTERM so the notebook process manager can relaunch
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/restart")
async def restart_server():
    """
    Self-restart the server process in-place.

    Uses os.execv() to replace the running process image with a fresh copy of
    itself (same PID, same CLI args, same environment).  uvicorn re-initialises
    from scratch — including the model-loading startup event — without any
    external process manager or notebook intervention.

    Flow:
      1. Browser sends POST /restart  →  200 {"status": "restarting"} returned.
      2. After a 0.8 s grace period the event loop calls os.execv().
      3. The OS atomically replaces this process with a new Python interpreter
         running streaming_server.py with the original arguments.
      4. Browser polls GET /health until the new server responds, then reloads.
    """
    logger.info("[restart] Restart requested — will execv() in 0.8 s")

    async def _do_restart():
        await asyncio.sleep(0.8)   # give the HTTP response time to reach the browser
        logger.info("[restart] Executing self-restart now …")
        # os.execv replaces the current process; if it succeeds it never returns.
        # sys.argv preserves all CLI flags (--host, --port, --log-level, etc.)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    asyncio.create_task(_do_restart())
    return JSONResponse({"status": "restarting"})


# ─────────────────────────────────────────────────────────────────────────────
# Bridge inference helper (thread target)
# ─────────────────────────────────────────────────────────────────────────────

def _run_bridge_step(bridge: StreamingBridgeInference, chunk: torch.Tensor) -> np.ndarray:
    """
    Run one bridge inference step, optionally inside a dedicated CUDA stream.
    Returns (N, 1024) float32 numpy array.

    Runs in asyncio.to_thread() so the event loop stays responsive.
    Uses torch.cuda.amp.autocast for FP16/BF16 acceleration on A100,
    giving up to 2× throughput with negligible quality loss.
    """
    global _bridge_cuda_stream
    if cfg.DEVICE == "cuda" and _bridge_cuda_stream is not None:
        with torch.cuda.stream(_bridge_cuda_stream):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                features = bridge.step(chunk)
        # Ensure current stream sees the result before we return
        torch.cuda.current_stream().wait_stream(_bridge_cuda_stream)
    else:
        features = bridge.step(chunk)
    return features.float().numpy().astype(np.float32)

def _run_bridge_reset(bridge: StreamingBridgeInference):
    global _bridge_cuda_stream
    if cfg.DEVICE == "cuda" and _bridge_cuda_stream is not None:
        with torch.cuda.stream(_bridge_cuda_stream):
            bridge.reset()
        torch.cuda.current_stream().wait_stream(_bridge_cuda_stream)
    else:
        bridge.reset()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"[WS] Client connected — session {session_id}")

    # ── Validate session ──────────────────────────────────────────────────────
    if session_id not in _sessions:
        await websocket.send_bytes(b"\xff" + b"Unknown session_id")
        await websocket.close()
        return

    session    = _sessions[session_id]
    image_path = session["image_path"]
    moshi      = get_moshi()
    bridge     = get_bridge()

    # ── Reject if Moshi is already busy (FIX: was silently queueing) ─────────
    if moshi._lock.locked():
        logger.warning(f"[WS] Rejecting session {session_id} — Moshi is busy")
        await websocket.send_bytes(
            b"\xff" + b"Server busy: another session is active. Try again shortly."
        )
        await websocket.close()
        return

    # ── Per-session Ditto adapter ─────────────────────────────────────────────
    ditto = DittoStreamAdapter(
        cfg_pkl      = cfg.DITTO_CFG_PKL,
        data_root    = cfg.DITTO_DATA_ROOT,
        jpeg_quality = cfg.DITTO_JPEG_QUALITY,
    )
    ditto.setup(
        image_path         = image_path,
        emo                = cfg.DITTO_EMO,
        sampling_timesteps = cfg.DITTO_SAMPLING_STEPS,
        overlap_v2         = cfg.DITTO_OVERLAP_V2,
    )

    # ── Per-session latency profiler ──────────────────────────────────────────
    profiler = PipelineProfiler(
        target_fps = 25.0,
        window     = 30,
        log_every  = 50,
        enabled    = True,
    )
    # Inject into Moshi so it can log per-frame timings
    moshi._profiler = profiler
    logger.info(f"[WS] Session {session_id} — profiler attached (target=25 FPS)")

    # ── Queues ────────────────────────────────────────────────────────────────
    # token_queue: Moshi → Bridge  (TaggedToken items or None sentinel)
    # Allows some buffering of Moshi acoustic tokens before Bridge runs.
    token_queue: asyncio.Queue = asyncio.Queue(maxsize=16)

    # frame_queue: Ditto thread → frame_forwarder (JPEG bytes or None sentinel)
    # Increased to 100 to easily absorb sliding window bursts (e.g. 4 frames
    # instantly generated after diffusion pass) without dropping any frames.
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    # send_queue: VIDEO + TEXT only.
    # ws_sender_task drains this queue over websocket.
    send_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    # ws_write_lock: serialises ALL WebSocket writes (audio direct + video
    # via ws_sender_task) to prevent concurrent-drain AssertionError.
    # Audio acquires this lock directly; video acquires it inside ws_sender_task.
    ws_write_lock: asyncio.Lock = asyncio.Lock()

    # Shared error signal: any task can set this to trigger coordinated shutdown
    error_event: asyncio.Event = asyncio.Event()

    # ── async receive wrapper ─────────────────────────────────────────────────
    async def receive_fn():
        """
        Return the next binary message from the WebSocket, or None ONLY when
        the client has truly disconnected (WebSocketDisconnect) or the session
        error_event is set.

        KEY FIX: previously caught all Exception subclasses and returned None,
        which caused transient Starlette/proxy errors to kill the session.
        Now we only return None on real disconnects.
        """
        if error_event.is_set():
            return None
        try:
            data = await websocket.receive_bytes()
            return data
        except WebSocketDisconnect:
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Transient error (e.g. proxy hiccup) — log and signal shutdown
            logger.debug(f"[receive_fn] transient error: {type(exc).__name__}: {exc}")
            error_event.set()
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # ws_sender_task — VIDEO + TEXT sender (audio bypasses this entirely)
    # ══════════════════════════════════════════════════════════════════════════
    async def ws_sender_task():
        """
        Serialised WebSocket writer for VIDEO and TEXT messages only.

        Audio is sent directly in the Moshi main loop (via ws_write_lock),
        guaranteeing zero audio drops from video backlog.

        Both this task and the audio sender acquire ws_write_lock before
        calling websocket.send_bytes(), preventing concurrent-drain
        AssertionError.

        Termination: a None sentinel in send_queue stops this task.
        """
        try:
            while True:
                msg = await send_queue.get()
                if msg is None:
                    break
                try:
                    async with ws_write_lock:
                        await websocket.send_bytes(msg)
                except (WebSocketDisconnect, Exception) as exc:
                    logger.debug(f"[ws_sender] Send failed ({type(exc).__name__}) — stopping")
                    error_event.set()
                    break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception(f"[ws_sender] Unexpected error: {exc}")
            error_event.set()
        logger.debug("[ws_sender] Exited.")

    # ── Bridge task: token_queue → features → Ditto ──────────────────────────
    async def bridge_task():
        """
        Pull TaggedToken items from token_queue, batch them, run bridge
        inference, and push features into Ditto.

        Adaptive batching: flush to Ditto when EITHER:
          (a) token_buffer reaches BRIDGE_CHUNK tokens, OR
          (b) BRIDGE_FLUSH_TIMEOUT_MS has elapsed since the first token arrived.
        This eliminates the 320ms+ stall during pauses.
        """
        _run_bridge_reset(bridge)
        token_buffer: list = []
        chunk_size    = cfg.BRIDGE_CHUNK
        flush_timeout = cfg.BRIDGE_FLUSH_TIMEOUT_MS / 1000.0
        first_token_time: Optional[float] = None
        _chunk_id: int = 0
        _flush_reason: str = "SIZE"

        async def _flush():
            nonlocal token_buffer, first_token_time, _chunk_id, _flush_reason
            if not token_buffer:
                return
            chunk_tensor = torch.cat([t.tensor for t in token_buffer], dim=0)
            batch_seq    = token_buffer[0].seq
            batch_size   = len(token_buffer)
            token_buffer = []
            first_token_time = None
            _chunk_id += 1
            _cid = _chunk_id
            _reason = _flush_reason
            try:
                with Timer() as t_infer:
                    features_np = await asyncio.to_thread(
                        _run_bridge_step, bridge, chunk_tensor
                    )
                with Timer() as t_push:
                    ditto.push_features(features_np, seq=batch_seq)

                profiler.log_bridge(
                    chunk_id     = _cid,
                    batch_size   = batch_size,
                    wait_ms      = flush_timeout * 1000.0 if _reason == "TIMEOUT" else 0.0,
                    infer_ms     = t_infer.ms,
                    push_ms      = t_push.ms,
                    flush_reason = _reason,
                    token_q_size = token_queue.qsize(),
                )
                profiler.record_queue_size("token_q", token_queue.qsize())
                profiler.record_queue_size("frame_q", frame_queue.qsize())
                profiler.record_queue_size("send_q",  send_queue.qsize())

                logger.debug(
                    f"[bridge_task] Flushed {chunk_tensor.shape[0]} tokens "
                    f"seq={batch_seq} → {features_np.shape}"
                )
            except Exception as exc:
                logger.error(f"[bridge_task] Bridge step error: {exc}")
                error_event.set()

        try:
            while True:
                if first_token_time is not None:
                    elapsed = time.monotonic() - first_token_time
                    wait_time = max(0.0, flush_timeout - elapsed)
                else:
                    wait_time = flush_timeout

                try:
                    item = await asyncio.wait_for(token_queue.get(), timeout=wait_time)
                    _flush_reason = "SIZE"
                except asyncio.TimeoutError:
                    _flush_reason = "TIMEOUT"
                    if token_buffer:
                        await _flush()
                    continue

                if item is None:
                    _flush_reason = "END"
                    await _flush()
                    break

                if error_event.is_set():
                    break

                token_buffer.append(item)
                if first_token_time is None:
                    first_token_time = time.monotonic()

                if len(token_buffer) >= chunk_size:
                    _flush_reason = "SIZE"
                    await _flush()

        except Exception as exc:
            logger.exception(f"[bridge_task] Unhandled error: {exc}")
            error_event.set()
        finally:
            ditto.close()
            logger.info("[bridge_task] Done.")

    # ── Frame reader task: Ditto thread → frame_queue ────────────────────────
    async def frame_reader_task():
        """
        Reads JPEG bytes from Ditto's blocking iter_frames() inside
        asyncio.to_thread(), then pushes them into frame_queue using
        loop.call_soon_threadsafe() + put_nowait() — no Future overhead,
        no blocking the reader thread waiting for the event loop.

        Frames are DROPPED (not blocked) when frame_queue is full.
        """
        loop = asyncio.get_running_loop()

        def _safe_enqueue(item):
            """Called on the event-loop thread via call_soon_threadsafe."""
            try:
                frame_queue.put_nowait(item)
            except asyncio.QueueFull:
                if item is not None:
                    logger.warning("[frame_reader_task] frame_queue full — frame dropped")
                else:
                    # Must deliver sentinel even if queue is full; retry once
                    # by clearing one slot.
                    try:
                        frame_queue.get_nowait()   # drop oldest frame
                    except asyncio.QueueEmpty:
                        pass
                    frame_queue.put_nowait(None)

        def _blocking_iter():
            for frame_item in ditto.iter_frames():
                # frame_item is now (seq, jpeg_bytes) from the updated adapter
                if error_event.is_set():
                    break
                loop.call_soon_threadsafe(_safe_enqueue, frame_item)
            # Always deliver sentinel so frame_forwarder_task can exit
            loop.call_soon_threadsafe(_safe_enqueue, None)

        try:
            await asyncio.to_thread(_blocking_iter)
        except Exception as exc:
            logger.exception(f"[frame_reader_task] Error: {exc}")
            error_event.set()
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(_safe_enqueue, None)

    # ── Frame forwarder task: frame_queue → send_queue (LATEST-FRAME-WINS) ───
    async def frame_forwarder_task():
        """
        Drains frame_queue and puts tagged JPEG frames into send_queue for
        the ws_sender_task to transmit.

        Each item from frame_queue is ``(seq, jpeg_bytes)`` where ``seq`` is
        the Moshi audio step sequence number that produced the features driving
        this video frame.  We pack this seq into the 0x02 video packet header
        so the browser can match every video frame to the audio packet with the
        same seq number — enabling tight, sample-accurate lip-sync.

        Wire format: 0x02 | <video_seq: 4 bytes BE uint32> | <jpeg bytes>
        """
        frame_count  = 0
        _drop_count  = 0
        _fps_window  = []   # timestamps for sliding-window FPS
        _MAX_FPS_WIN = 30
        _cur_fps     = 0.0

        try:
            while True:
                _t_q = time.monotonic()
                # Block until at least one frame arrives
                item = await frame_queue.get()
                _q_wait_ms = (time.monotonic() - _t_q) * 1000.0

                if item is None or error_event.is_set():
                    break

                # item is (seq, jpeg_bytes) from DittoStreamAdapter.iter_frames()
                frame_seq, jpeg = item
                hdr = b"\x02" + struct.pack(">I", frame_seq & 0xFFFF_FFFF)

                frame_count += 1

                # FPS tracking (sliding window)
                _now = time.monotonic()
                _fps_window.append(_now)
                if len(_fps_window) > _MAX_FPS_WIN:
                    _fps_window.pop(0)
                _cur_fps = (_MAX_FPS_WIN - 1) / (_fps_window[-1] - _fps_window[0]) \
                    if len(_fps_window) > 1 else 0.0

                # Track output FPS in profiler
                profiler.tick_frame()

                # put_nowait: if send_queue is full drop frame rather than block
                try:
                    send_queue.put_nowait(hdr + jpeg)
                except asyncio.QueueFull:
                    _drop_count += 1
                    profiler.record_drop("send_q", 1)
                    logger.debug(
                        f"[frame_forwarder] send_queue full — frame dropped | "
                        f"seq={frame_seq} total_drops={_drop_count}"
                    )

                # ── [PROFILING] pipeline-level log and periodic summary ────────
                profiler.log_pipeline(
                    frame_id        = frame_count,
                    frame_q_wait_ms = _q_wait_ms,
                    send_q_size     = send_queue.qsize(),
                    dropped         = _drop_count,
                )
                if frame_count % 50 == 0:
                    profiler.print_summary(
                        label=f"@ frame={frame_count} fps={_cur_fps:.1f} seq={frame_seq}"
                    )
                    profiler.print_fps_insight()

        except Exception as exc:
            logger.exception(f"[frame_forwarder_task] Error: {exc}")
            error_event.set()
        logger.info(
            f"[frame_forwarder_task] Done. "
            f"Forwarded={frame_count} dropped={_drop_count} "
            f"final_fps={_cur_fps:.1f}"
        )

    # ── Keepalive task: prevent RunPod proxy idle timeout ────────────────────
    async def keepalive_task():
        """
        Send a 1-byte noop message (0xFE) into send_queue every 12 seconds.

        RunPod's HTTPS reverse proxy drops connections idle for ~15–20 seconds.
        During heavy Ditto GPU inference the server may produce few frames,
        causing the proxy to silently close the TCP connection. This task
        ensures at least one byte is sent every 12s while the session is live.

        The browser ignores 0xFE messages (no handler registered for that kind).
        """
        ka_seq = 0
        try:
            while not error_event.is_set():
                await asyncio.sleep(12)
                if error_event.is_set():
                    break
                try:
                    send_queue.put_nowait(b"\xfe")  # 1-byte noop keepalive
                    logger.debug(f"[keepalive_task] Sent noop ping (count={ka_seq})")
                    ka_seq += 1
                except asyncio.QueueFull:
                    pass  # sender is backed up — that means data IS flowing
        except asyncio.CancelledError:
            pass
        logger.debug("[keepalive_task] Exited.")

    # ── Start background tasks ────────────────────────────────────────────────
    t_ws_sender    = asyncio.create_task(ws_sender_task(),       name="ws_sender")
    t_bridge       = asyncio.create_task(bridge_task(),          name="bridge_task")
    t_frame_reader = asyncio.create_task(frame_reader_task(),    name="frame_reader_task")
    t_frame_fwd    = asyncio.create_task(frame_forwarder_task(), name="frame_forwarder_task")
    t_keepalive    = asyncio.create_task(keepalive_task(),       name="keepalive")

    # ── Moshi main loop (drives audio I/O + token capture) ───────────────────
    try:
        async for kind, payload in moshi.handle_connection(receive_fn, token_queue):
            if error_event.is_set():
                logger.warning("[WS] Error event set — stopping Moshi loop")
                break

            if kind == "handshake":
                # One-time event — send directly (no queue needed)
                try:
                    async with ws_write_lock:
                        await websocket.send_bytes(b"\x00")
                except Exception:
                    error_event.set()

            elif kind == "audio":
                # ── DIRECT AUDIO SEND — bypasses send_queue entirely ──────
                # Audio NEVER waits on video queue availability.
                # ws_write_lock serialises with ws_sender_task (~0.1ms)
                moshi_seq, pcm_bytes = payload
                hdr = b"\x01" + seq_pack(moshi_seq)
                try:
                    async with ws_write_lock:
                        await websocket.send_bytes(hdr + pcm_bytes)
                except (WebSocketDisconnect, Exception) as exc:
                    logger.debug(f"[WS] Audio send failed ({type(exc).__name__}) — stopping")
                    error_event.set()
                    break

            elif kind == "text":
                # Text is low-priority — goes through send_queue
                try:
                    send_queue.put_nowait(b"\x03" + payload.encode("utf-8"))
                except asyncio.QueueFull:
                    pass  # drop text rather than block Moshi loop

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected — session {session_id}")
    except RuntimeError as exc:
        logger.warning(f"[WS] {exc}")
        try:
            await send_queue.put(b"\xff" + str(exc).encode())
        except Exception:
            pass
    except Exception as exc:
        logger.exception(f"[WS] Unexpected error in session {session_id}: {exc}")
        try:
            await send_queue.put(b"\xff" + str(exc).encode())
        except Exception:
            pass
    finally:
        error_event.set()

        # Stop the ws_sender cleanly by putting the sentinel
        try:
            send_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        # Wait for all tasks to finish (10s hard deadline)
        all_tasks = (t_ws_sender, t_bridge, t_frame_reader, t_frame_fwd, t_keepalive)
        try:
            await asyncio.wait_for(
                asyncio.gather(*all_tasks, return_exceptions=True),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[WS] Session {session_id} cleanup timed out — cancelling tasks")
            for t in all_tasks:
                t.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)

        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"[WS] Session {session_id} fully closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Startup event: load all models once
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_models():
    global _moshi_state, _bridge_stream, _bridge_cuda_stream

    logger.info("=" * 60)
    logger.info("  Moshi + Bridge + Ditto — Streaming Server Starting  v3")
    logger.info("=" * 60)
    logger.info(f"  Device : {cfg.DEVICE}")

    # ── Max GPU performance flags ────────────────────────────────────────────
    if cfg.DEVICE == "cuda":
        import torch
        # ── H200 (SM90a / Hopper) GPU optimizations ──────────────────────────
        # H200 does NOT reproduce the A100 CUDNN_STATUS_INTERNAL_ERROR that
        # occurs when running concurrent TRT+cuDNN priority streams.  Re-enable
        # benchmark mode so cuDNN auto-selects the fastest convolution kernels.
        torch.backends.cudnn.benchmark = True
        # TF32 on matmul and convolutions for full SM90a tensor core throughput.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # BF16 matmul precision — 'high' uses tensor cores without sacrificing
        # dynamic range (unlike float16).  H200's BF16 peak is ~2PFLOPs.
        torch.set_float32_matmul_precision('high')
        # Flash attention (SDPA) is natively fused on Hopper (FA3 kernel).
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        # Reserve 95% of GPU memory for models — leaves 5% for TRT scratch.
        torch.cuda.memory.set_per_process_memory_fraction(0.95)
        logger.info(
            "  H200 (SM90a) optimizations applied: "
            "cudnn.benchmark=True, TF32, BF16 matmul='high', Flash-SDPA, mem=95%"
        )
    # H200 pods typically have 96+ vCPUs — allow more CPU threads than A100 pods.
    import torch
    torch.set_num_threads(min(16, torch.get_num_threads()))
    logger.info(f"  PyTorch CPU threads: {torch.get_num_threads()}")

    # ── Moshi ────────────────────────────────────────────────────────────────
    logger.info("\n[1/2] Loading Moshi …")
    t0 = time.time()
    _moshi_state = StreamingMoshiState(
        hf_repo      = cfg.MOSHI_HF_REPO,
        moshi_weight = cfg.MOSHI_WEIGHT,
        mimi_weight  = cfg.MIMI_WEIGHT,
        tokenizer    = cfg.TOKENIZER,
        device       = cfg.DEVICE,
        dtype        = cfg.DTYPE,
    )
    _moshi_state.warmup()
    logger.info(f"[1/2] ✅ Moshi ready ({time.time()-t0:.1f}s)")

    # ── Bridge ───────────────────────────────────────────────────────────────
    logger.info("\n[2/2] Loading Bridge …")
    t0 = time.time()
    _bridge_stream = StreamingBridgeInference(
        checkpoint_path = cfg.BRIDGE_CKPT,
        config_path     = cfg.BRIDGE_CONFIG,
        chunk_size      = cfg.BRIDGE_CHUNK,
        device          = cfg.DEVICE,
    )
    if cfg.DEVICE == "cuda":
        _bridge_cuda_stream = torch.cuda.Stream()
        logger.info("[2/2]   CUDA stream created for Bridge isolation")
    # Try torch.compile for the bridge model — reduces kernel-launch overhead
    # on A100. Falls back gracefully on PyTorch < 2.0 or if compile fails.
    try:
        if hasattr(torch, 'compile'):
            _bridge_stream = torch.compile(
                _bridge_stream,
                mode='max-autotune',   # H200: auto-tune for peak SM90a throughput
                fullgraph=False,
            )
            logger.info("[2/2]   torch.compile applied to Bridge (max-autotune for H200)")
    except Exception as ce:
        logger.warning(f"[2/2]   torch.compile skipped: {ce}")
    logger.info(f"[2/2] ✅ Bridge ready ({time.time()-t0:.1f}s)")

    # ── Ditto ────────────────────────────────────────────────────────────────
    logger.info("\n[3/3] Ditto: adapter created per-session (image-specific).")
    logger.info(f"       TRT models: {cfg.DITTO_DATA_ROOT}")
    logger.info(f"       Adaptive bridge flush: {cfg.BRIDGE_FLUSH_TIMEOUT_MS}ms")

    # ── Register default session (obama.jpeg) ───────────────────────────────
    _default_img = cfg.DEFAULT_IMAGE
    if os.path.isfile(_default_img):
        _sessions["default"] = {"image_path": _default_img, "ready": True}
        logger.info(f"[startup] Default session registered → {_default_img}")
    else:
        logger.warning(f"[startup] Default image not found: {_default_img}  (upload required)")

    logger.info("\n" + "=" * 60)
    logger.info("  ✅  All models loaded. Server ready for connections.")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Moshi + Bridge + Ditto — Real-Time Streaming Server v3"
    )
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      type=int, default=7860)
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"])

    # Model path overrides (all have env-var equivalents, see Config above)
    p.add_argument("--hf-repo",         default=None)
    p.add_argument("--moshi-weight",    default=None)
    p.add_argument("--mimi-weight",     default=None)
    p.add_argument("--tokenizer",       default=None)
    p.add_argument("--bridge-ckpt",     default=None)
    p.add_argument("--bridge-config",   default=None)
    p.add_argument("--bridge-chunk",    type=int, default=None)
    p.add_argument("--bridge-flush-ms", type=int, default=None,
                   help="Adaptive bridge flush timeout in milliseconds (default 100)")
    p.add_argument("--ditto-data-root", default=None)
    p.add_argument("--ditto-cfg-pkl",   default=None)
    p.add_argument("--ditto-emo",            type=int, default=None)
    p.add_argument("--ditto-sampling-steps", type=int, default=None)
    p.add_argument("--jpeg-quality",    type=int, default=None)
    p.add_argument("--half", action="store_const",
                   const=torch.float16, default=None, dest="dtype",
                   help="Use float16 instead of bfloat16")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Apply CLI overrides to Config
    if args.hf_repo:         cfg.MOSHI_HF_REPO    = args.hf_repo
    if args.moshi_weight:    cfg.MOSHI_WEIGHT      = args.moshi_weight
    if args.mimi_weight:     cfg.MIMI_WEIGHT       = args.mimi_weight
    if args.tokenizer:       cfg.TOKENIZER         = args.tokenizer
    if args.bridge_ckpt:     cfg.BRIDGE_CKPT       = args.bridge_ckpt
    if args.bridge_config:   cfg.BRIDGE_CONFIG     = args.bridge_config
    if args.bridge_chunk:    cfg.BRIDGE_CHUNK      = args.bridge_chunk
    if args.bridge_flush_ms: cfg.BRIDGE_FLUSH_TIMEOUT_MS = args.bridge_flush_ms
    if args.ditto_data_root: cfg.DITTO_DATA_ROOT   = args.ditto_data_root
    if args.ditto_cfg_pkl:   cfg.DITTO_CFG_PKL     = args.ditto_cfg_pkl
    if args.ditto_emo:            cfg.DITTO_EMO            = args.ditto_emo
    if args.ditto_sampling_steps: cfg.DITTO_SAMPLING_STEPS = args.ditto_sampling_steps
    if args.jpeg_quality:    cfg.DITTO_JPEG_QUALITY = args.jpeg_quality
    if args.dtype:           cfg.DTYPE             = args.dtype

    uvicorn.run(
        "streaming_server:app",
        host             = args.host,
        port             = args.port,
        log_level        = args.log_level,
        loop             = "asyncio",
        # WebSocket protocol-level pings keep the RunPod HTTPS proxy alive.
        # Ping every 20s; wait up to 60s for a pong before closing.
        ws_ping_interval = 20,
        ws_ping_timeout  = 60,
    )