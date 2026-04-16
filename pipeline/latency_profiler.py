"""
pipeline/latency_profiler.py
============================
Centralized latency profiling and diagnostics for the Moshi → Bridge → Ditto
real-time streaming pipeline.

Usage
-----
    from pipeline.latency_profiler import PipelineProfiler

    profiler = PipelineProfiler(target_fps=25)

    # Record a timing
    profiler.record("MOSHI", "lm_step", elapsed_ms=14.2)
    profiler.record("BRIDGE", "infer",  elapsed_ms=9.1)

    # Tick FPS counter every time a frame is sent out
    profiler.tick_frame()

    # Print summary every N frames
    if profiler.frame_count % 50 == 0:
        profiler.print_summary()

Structured log format
---------------------
    [MOSHI]      frame=12  | encode=3.1ms  | lm_step=14.2ms | decode=4.0ms  | total=21.3ms
    [BRIDGE]     chunk=12  | wait=48.2ms   | infer=9.1ms    | push=0.8ms    | total=58.1ms  flush=TIMEOUT
    [DITTO:A2M]  batch=6   | infer=38.4ms  | q_wait=1.2ms
    [PIPELINE]   frame=340 | fps=8.2       | total_est=177ms | dropped=2
"""

from __future__ import annotations

import logging
import time
from collections import deque, defaultdict
from threading import Lock
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Budget constants  (all in ms)
# ─────────────────────────────────────────────────────────────────────────────

# Latency budget per module (1 Moshi token = 80ms of audio = 2 Video frames at 25 FPS)
LATENCY_BUDGET_MS: Dict[str, float] = {
    "MOSHI":      80.0,   # Generates 80ms of audio
    "BRIDGE":      8.0,   # wait + infer + push_features
    "DITTO:A2M":  80.0,   # diffusion audio2motion (heaviest)
    "DITTO:STITCH": 2.0,
    "DITTO:WARP":   3.0,
    "DITTO:DEC":    3.0,
    "DITTO:PUT":    2.0,
    "DITTO:WRITE":  1.5,
    "PIPELINE":    80.0,  # total real-time chunk budget
}

TARGET_FPS = 25.0
TARGET_FRAME_MS = 1000.0 / TARGET_FPS    # 40 ms


# ─────────────────────────────────────────────────────────────────────────────
# MovingAverage helper
# ─────────────────────────────────────────────────────────────────────────────

class MovingAverage:
    """Thread-safe sliding-window moving average."""

    def __init__(self, window: int = 30):
        self._window = window
        self._data: deque = deque(maxlen=window)
        self._lock = Lock()

    def add(self, value: float):
        with self._lock:
            self._data.append(value)

    def mean(self) -> float:
        with self._lock:
            if not self._data:
                return 0.0
            return sum(self._data) / len(self._data)

    def p95(self) -> float:
        with self._lock:
            if not self._data:
                return 0.0
            s = sorted(self._data)
            return s[int(0.95 * len(s))]

    def latest(self) -> float:
        with self._lock:
            return self._data[-1] if self._data else 0.0

    def count(self) -> int:
        with self._lock:
            return len(self._data)


# ─────────────────────────────────────────────────────────────────────────────
# FPS Tracker
# ─────────────────────────────────────────────────────────────────────────────

class FPSTracker:
    """Measures real-time FPS using a sliding timestamp window."""

    def __init__(self, window: int = 30):
        self._timestamps: deque = deque(maxlen=window)
        self._lock = Lock()
        self._total = 0

    def tick(self):
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            self._total += 1

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            span = self._timestamps[-1] - self._timestamps[0]
            if span <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / span

    @property
    def total(self) -> int:
        with self._lock:
            return self._total


# ─────────────────────────────────────────────────────────────────────────────
# PipelineProfiler — main class
# ─────────────────────────────────────────────────────────────────────────────

class PipelineProfiler:
    """
    Collects and reports per-module latency timings for the streaming pipeline.

    Parameters
    ----------
    target_fps    : target real-time FPS (default 25)
    window        : sliding window size for moving averages (default 30)
    log_every     : print per-module summary every N frames (default 50)
    enabled       : set False to disable all profiling with zero overhead
    """

    def __init__(
        self,
        target_fps: float = TARGET_FPS,
        window: int = 30,
        log_every: int = 50,
        enabled: bool = True,
    ):
        self.target_fps    = target_fps
        self.target_frame_ms = 1000.0 / target_fps
        self.log_every     = log_every
        self.enabled       = enabled

        # Per-module, per-step moving averages
        # { module_name: { step_name: MovingAverage } }
        self._averages: Dict[str, Dict[str, MovingAverage]] = defaultdict(
            lambda: defaultdict(lambda: MovingAverage(window))
        )

        # FPS trackers
        self._fps_out = FPSTracker(window)

        # Drop counters
        self._drops: Dict[str, int] = defaultdict(int)

        # Queue size snapshots (latest)
        self._queue_sizes: Dict[str, int] = {}

        # Global lock for summary printing
        self._print_lock = Lock()

        # Chunk / frame counters
        self._chunk_count = 0
        self._frame_count = 0

        if enabled:
            logger.info(
                f"[Profiler] Latency profiler enabled — "
                f"target={target_fps} FPS ({self.target_frame_ms:.0f}ms/frame) | "
                f"window={window} frames"
            )

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record(self, module: str, step: str, elapsed_ms: float):
        """Record a timing measurement (thread-safe)."""
        if not self.enabled:
            return
        self._averages[module][step].add(elapsed_ms)

    def record_drop(self, component: str, count: int = 1):
        """Record a dropped frame/token."""
        if not self.enabled:
            return
        with self._print_lock:
            self._drops[component] += count

    def record_queue_size(self, name: str, size: int):
        """Record current queue size snapshot."""
        if not self.enabled:
            return
        self._queue_sizes[name] = size

    def tick_frame(self):
        """Call once per output frame to update FPS counter."""
        if not self.enabled:
            return
        self._fps_out.tick()
        self._frame_count = self._fps_out.total

    def tick_chunk(self):
        """Call once per bridge chunk processed."""
        if not self.enabled:
            return
        self._chunk_count += 1

    @property
    def frame_count(self) -> int:
        return self._fps_out.total

    @property
    def fps(self) -> float:
        return self._fps_out.fps

    # ------------------------------------------------------------------
    # Structured single-event log (call from hot path)
    # ------------------------------------------------------------------

    def log_moshi(
        self,
        frame_id: int,
        encode_ms: float,
        lm_step_ms: float,
        decode_ms: float,
        token_q_size: int = 0,
        dropped_tokens: int = 0,
    ):
        """Emit a structured [MOSHI] log line and record averages."""
        if not self.enabled:
            return
        total = encode_ms + lm_step_ms + decode_ms
        self.record("MOSHI", "encode",   encode_ms)
        self.record("MOSHI", "lm_step",  lm_step_ms)
        self.record("MOSHI", "decode",   decode_ms)
        self.record("MOSHI", "total",    total)
        self.record_queue_size("token_q", token_q_size)

        budget = LATENCY_BUDGET_MS.get("MOSHI", 15.0)
        flag = " ⚠ OVER BUDGET" if total > budget else ""
        logger.debug(
            f"[MOSHI]      frame_id={frame_id:>5} | "
            f"encode={encode_ms:>6.1f}ms | lm_step={lm_step_ms:>6.1f}ms | "
            f"decode={decode_ms:>6.1f}ms | total={total:>6.1f}ms | "
            f"token_q={token_q_size} | dropped={dropped_tokens}{flag}"
        )

    def log_bridge(
        self,
        chunk_id: int,
        batch_size: int,
        wait_ms: float,
        infer_ms: float,
        push_ms: float,
        flush_reason: str = "SIZE",   # "SIZE" or "TIMEOUT"
        token_q_size: int = 0,
    ):
        """Emit a structured [BRIDGE] log line and record averages."""
        if not self.enabled:
            return
        total = wait_ms + infer_ms + push_ms
        self.record("BRIDGE", "wait",   wait_ms)
        self.record("BRIDGE", "infer",  infer_ms)
        self.record("BRIDGE", "push",   push_ms)
        self.record("BRIDGE", "total",  total)

        budget = LATENCY_BUDGET_MS.get("BRIDGE", 8.0)
        flag = " ⚠ OVER BUDGET" if total > budget else ""
        logger.info(
            f"[BRIDGE]     chunk_id={chunk_id:>5} | batch={batch_size:>2} | "
            f"wait={wait_ms:>6.1f}ms | infer={infer_ms:>6.1f}ms | "
            f"push={push_ms:>5.1f}ms | total={total:>6.1f}ms | "
            f"flush={flush_reason} | token_q={token_q_size}{flag}"
        )
        self.tick_chunk()

    def log_ditto_worker(
        self,
        worker: str,       # "A2M" | "STITCH" | "WARP" | "DEC" | "PUT" | "WRITE"
        frame_id: int,
        infer_ms: float,
        q_wait_ms: float = 0.0,
        extra: str = "",
    ):
        """Emit a structured [DITTO:WORKER] log line."""
        if not self.enabled:
            return
        mod = f"DITTO:{worker}"
        total = infer_ms + q_wait_ms
        self.record(mod, "infer",   infer_ms)
        self.record(mod, "q_wait",  q_wait_ms)
        self.record(mod, "total",   total)

        budget = LATENCY_BUDGET_MS.get(mod, 5.0)
        flag = " ⚠" if infer_ms > budget else ""
        logger.debug(
            f"[DITTO:{worker:<5}] frame={frame_id:>5} | "
            f"infer={infer_ms:>7.2f}ms | q_wait={q_wait_ms:>6.2f}ms{flag} {extra}"
        )

    def log_adapter(
        self,
        frame_id: int,
        q_wait_ms: float,
        encode_ms: float,
        jpeg_bytes: int = 0,
    ):
        """Emit a structured [ADAPTER] log line."""
        if not self.enabled:
            return
        total = q_wait_ms + encode_ms
        self.record("ADAPTER", "q_wait", q_wait_ms)
        self.record("ADAPTER", "encode", encode_ms)
        self.record("ADAPTER", "total",  total)
        self.tick_frame()

        logger.debug(
            f"[ADAPTER]    frame={frame_id:>5} | "
            f"q_wait={q_wait_ms:>6.2f}ms | encode={encode_ms:>5.2f}ms | "
            f"size={jpeg_bytes//1024}KB | out_fps={self._fps_out.fps:>5.1f}"
        )

    def log_pipeline(
        self,
        frame_id: int,
        frame_q_wait_ms: float,
        send_q_size: int,
        dropped: int = 0,
    ):
        """Emit a [PIPELINE] line with end-to-end FPS and queue diagnostics."""
        if not self.enabled:
            return
        self.record("PIPELINE", "frame_q_wait", frame_q_wait_ms)

        fps = self._fps_out.fps
        flag = " ✓" if fps >= self.target_fps * 0.9 else " ✗ LOW FPS"
        logger.info(
            f"[PIPELINE]   frame={frame_id:>5} | fps={fps:>5.1f}{flag} | "
            f"frame_q_wait={frame_q_wait_ms:>6.1f}ms | "
            f"send_q={send_q_size} | dropped={dropped}"
        )

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def print_summary(self, label: str = ""):
        """Print a full per-module latency breakdown with budget analysis."""
        if not self.enabled:
            return

        with self._print_lock:
            fps = self._fps_out.fps
            total_frames = self._fps_out.total
            q_str = "  ".join(f"{k}={v}" for k, v in self._queue_sizes.items())
            drop_str = "  ".join(f"{k}={v}" for k, v in self._drops.items())

            lines = [
                "",
                "═" * 80,
                f"  PIPELINE LATENCY SUMMARY  {label}",
                f"  Output: {fps:.1f} FPS  (target={self.target_fps} FPS)  |  "
                f"frames={total_frames}  chunks={self._chunk_count}",
                f"  Queue sizes: {q_str or 'N/A'}",
                f"  Drops:       {drop_str or 'none'}",
                "─" * 80,
                f"  {'Module':<18} {'Step':<12} {'Avg(ms)':>9} {'P95(ms)':>9} "
                f"{'Budget':>9} {'Status':>8}",
                "─" * 80,
            ]

            # Modules in pipeline order
            module_order = [
                "MOSHI", "BRIDGE",
                "DITTO:A2M", "DITTO:STITCH", "DITTO:WARP",
                "DITTO:DEC", "DITTO:PUT", "DITTO:WRITE",
                "ADAPTER", "PIPELINE",
            ]

            for mod in module_order:
                if mod not in self._averages:
                    continue
                budget = LATENCY_BUDGET_MS.get(mod, None)
                steps = self._averages[mod]
                for step, avg_obj in steps.items():
                    if avg_obj.count() == 0:
                        continue
                    avg = avg_obj.mean()
                    p95 = avg_obj.p95()
                    bstr = f"{budget:.0f}ms" if budget else "  N/A"
                    if step == "total" and budget:
                        status = "✓ OK" if avg <= budget else "⚠ OVER"
                    else:
                        status = ""
                    lines.append(
                        f"  {mod:<18} {step:<12} {avg:>8.1f}ms {p95:>8.1f}ms "
                        f"{bstr:>9} {status:>8}"
                    )

            # Ideal recommendation
            lines += [
                "─" * 80,
                f"  TARGET for {self.target_fps:.0f} FPS: {self.target_frame_ms:.0f}ms/frame total",
                f"  IDEAL BUDGET:  Moshi≤15ms | Bridge≤8ms | DittoA2M≤30ms | "
                f"Stitch/Warp/Dec/Put≤10ms | Wire≤5ms",
                "═" * 80,
                "",
            ]

            logger.info("\n".join(lines))

    def print_fps_insight(self):
        """Print a one-liner FPS vs. budget insight — good for periodic logging."""
        if not self.enabled:
            return
        fps = self._fps_out.fps
        moshi_total  = self._averages["MOSHI"]["total"].mean()
        bridge_total = self._averages["BRIDGE"]["total"].mean()
        a2m_total    = self._averages["DITTO:A2M"]["total"].mean()

        bottleneck = max(
            [("MOSHI", moshi_total), ("BRIDGE", bridge_total), ("DITTO:A2M", a2m_total)],
            key=lambda x: x[1],
        )
        logger.info(
            f"[PIPELINE INSIGHT] fps={fps:.1f} | "
            f"moshi={moshi_total:.0f}ms | bridge={bridge_total:.0f}ms | "
            f"ditto_a2m={a2m_total:.0f}ms | "
            f"bottleneck={bottleneck[0]}({bottleneck[1]:.0f}ms)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Context-manager timer helper (lightweight, no class dependency)
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """
    Lightweight context manager for measuring elapsed time in ms.

        with Timer() as t:
            do_work()
        print(f"elapsed: {t.ms:.1f}ms")
    """

    def __init__(self):
        self.ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.ms = (time.perf_counter() - self._start) * 1000.0
