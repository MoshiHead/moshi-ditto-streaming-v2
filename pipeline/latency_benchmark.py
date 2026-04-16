"""
pipeline/latency_benchmark.py
==============================
Standalone offline vs. online streaming latency comparison benchmark.

Runs the Bridge and simulates the Ditto Audio2Motion stage in both
offline batch mode and online streaming (KV-cache) mode, then prints
a side-by-side comparison table with FPS and budget analysis.

Usage
-----
    python pipeline/latency_benchmark.py \\
        --bridge-ckpt  checkpoints/bridge_best.pt \\
        --bridge-config bridge_module/config.yaml \\
        --num-frames  100 \\
        --chunk-size  2

    # Also benchmark Ditto's Audio2Motion (requires model files):
    python pipeline/latency_benchmark.py \\
        --bridge-ckpt  checkpoints/bridge_best.pt \\
        --bridge-config bridge_module/config.yaml \\
        --ditto-cfg-pkl   ditto-inference/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \\
        --ditto-data-root ditto-inference/checkpoints/ditto_trt_Ampere_Plus \\
        --num-frames 50

Output example
--------------
    ╔══════════════════════════════════════════════════════════╗
    ║         OFFLINE vs STREAMING LATENCY COMPARISON          ║
    ╠══════════════════════════════════════════════════════════╣
    ║ Mode       │ Module    │ Avg(ms) │ P95(ms) │ FPS   │ Budget │ Status ║
    ╠══════════════════════════════════════════════════════════╣
    ║ OFFLINE    │ Bridge    │   6.2ms │   8.1ms │ 161.3 │  8ms   │   ✓   ║
    ║ STREAMING  │ Bridge    │  58.3ms │  82.4ms │  17.2 │  8ms   │   ✗   ║
    ╚══════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import statistics
from typing import List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("latency_benchmark")

# Ensure paths
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRIDGE_DIR = os.path.join(_ROOT, "bridge_module")
for p in [_ROOT, _BRIDGE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _warmup_gpu():
    """Simple GPU warmup to avoid cold-start bias."""
    if torch.cuda.is_available():
        x = torch.zeros(256, 256, device="cuda")
        for _ in range(10):
            x = x @ x
        torch.cuda.synchronize()


def _make_dummy_tokens(num_frames: int, num_codebooks: int) -> torch.Tensor:
    """Generate random Mimi tokens (num_frames, num_codebooks)."""
    return torch.randint(0, 2048, (num_frames, num_codebooks))


# ─────────────────────────────────────────────────────────────────────────────
# Bridge benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def bench_bridge_offline(
    ckpt: str, cfg: str, num_frames: int, warmup: int = 5
) -> List[float]:
    """
    Offline batch bridge inference.
    Processes all `num_frames` tokens in a single forward pass.
    Returns list of latencies in ms (one per call = one per batch).
    """
    from inference import BridgeInference
    model = BridgeInference(ckpt, cfg)
    tokens = _make_dummy_tokens(num_frames, model.num_codebooks)

    logger.info(f"[offline-bridge] Warmup ({warmup} runs)...")
    for _ in range(warmup):
        model(tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"[offline-bridge] Benchmarking {num_frames} frames as single batch...")
    times = []
    for _ in range(20):   # repeat 20 times to get stable stats
        t0 = time.perf_counter()
        model(tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(_ms(t0))

    return times


def bench_bridge_streaming(
    ckpt: str, cfg: str, num_frames: int, chunk_size: int, warmup: int = 5
) -> Tuple[List[float], List[float]]:
    """
    Online streaming (KV-cache) bridge inference.
    Processes tokens `chunk_size` at a time.
    Returns:
      chunk_times : latency per chunk (ms)
      frame_times : per-frame latency = chunk_time / chunk_size (ms)
    """
    from inference import StreamingBridgeInference
    model = StreamingBridgeInference(ckpt, cfg, chunk_size=chunk_size)
    all_tokens = _make_dummy_tokens(num_frames, model.num_codebooks)

    logger.info(f"[streaming-bridge] Warmup ({warmup} chunks)...")
    model.reset()
    for i in range(warmup):
        chunk = all_tokens[i * chunk_size: (i + 1) * chunk_size]
        if chunk.shape[0] == 0:
            break
        model.step(chunk)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"[streaming-bridge] Benchmarking {num_frames} frames in chunks of {chunk_size}...")
    model.reset()
    chunk_times = []
    for start in range(0, num_frames, chunk_size):
        chunk = all_tokens[start: start + chunk_size]
        if chunk.shape[0] == 0:
            break
        t0 = time.perf_counter()
        model.step(chunk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        chunk_times.append(_ms(t0))

    # Normalise to per-frame latency
    frame_times = [t / chunk_size for t in chunk_times]
    return chunk_times, frame_times


# ─────────────────────────────────────────────────────────────────────────────
# Ditto Audio2Motion benchmark (optional)
# ─────────────────────────────────────────────────────────────────────────────

def bench_ditto_a2m(
    cfg_pkl: str,
    data_root: str,
    num_frames: int,
    sampling_steps: int,
    feat_dim: int = 1024,
    src_image: Optional[str] = None,
    warmup: int = 3,
) -> Tuple[List[float], List[float]]:
    """
    Benchmark Ditto's Audio2Motion stage.
    Returns (offline_times, streaming_times) in ms per batch.
    """
    _ditto_dir = os.path.join(_ROOT, "ditto-inference")
    if _ditto_dir not in sys.path:
        sys.path.insert(0, _ditto_dir)

    try:
        from stream_pipeline_online import StreamSDK
    except ImportError as e:
        logger.warning(f"[ditto-bench] Cannot import StreamSDK: {e}")
        return [], []

    if src_image is None:
        logger.warning("[ditto-bench] No --source-image provided; skipping Ditto A2M benchmark.")
        return [], []

    dummy_feats = np.random.randn(num_frames, feat_dim).astype(np.float32)

    results_offline   = []
    results_streaming = []

    for mode_name, online_mode in [("OFFLINE", False), ("STREAMING", True)]:
        logger.info(f"[ditto-bench] Running {mode_name} Audio2Motion benchmark...")
        try:
            sdk = StreamSDK(cfg_pkl, data_root)
            sdk.setup(
                source_path        = src_image,
                output_path        = "/tmp/bench_dummy.mp4",
                online_mode        = online_mode,
                N_d                = num_frames,
                sampling_timesteps = sampling_steps,
            )
            a2m = sdk.audio2motion

            # Warm up
            for _ in range(warmup):
                a2m(dummy_feats[:sdk.audio2motion.seq_frames][None])
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            seq_frames = sdk.audio2motion.seq_frames
            times = []
            for start in range(0, num_frames - seq_frames, seq_frames // 2):
                feat_clip = dummy_feats[start: start + seq_frames][None]
                t0 = time.perf_counter()
                a2m(feat_clip, None)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(_ms(t0))

            if mode_name == "OFFLINE":
                results_offline = times
            else:
                results_streaming = times

            sdk.stop_event.set()
        except Exception as exc:
            logger.warning(f"[ditto-bench] {mode_name} A2M benchmark failed: {exc}")

    return results_offline, results_streaming


# ─────────────────────────────────────────────────────────────────────────────
# Table printer
# ─────────────────────────────────────────────────────────────────────────────

TARGET_FRAME_MS = 40.0   # 25 FPS → 40ms/frame

_BUDGETS = {
    "Bridge (batch/frame)":      8.0,
    "Bridge (chunk)":           16.0,
    "Ditto A2M":                30.0,
}


def _row(
    mode: str,
    module: str,
    times: List[float],
    budget_ms: Optional[float] = None,
    is_per_frame: bool = True,
) -> str:
    if not times:
        return f"  {mode:<12} │ {module:<22} │  N/A"
    avg  = statistics.mean(times)
    p95  = sorted(times)[int(0.95 * len(times))]
    fps  = 1000.0 / avg if avg > 0 else 0.0
    bstr = f"{budget_ms:.0f}ms" if budget_ms else "  N/A"
    if budget_ms:
        status = "  ✓ OK" if avg <= budget_ms else "  ✗ OVER"
    else:
        status = ""
    return (
        f"  {mode:<12} │ {module:<22} │ "
        f"{avg:>8.1f}ms │ {p95:>8.1f}ms │ {fps:>7.1f} │ "
        f"{bstr:>7} │{status}"
    )


def print_comparison_table(results: dict):
    """Print the full comparison table from collected results."""
    header = (
        f"  {'Mode':<12} │ {'Module':<22} │ "
        f"{'Avg(ms)':>9} │ {'P95(ms)':>9} │ {'FPS':>7} │ "
        f"{'Budget':>7} │ Status"
    )
    sep = "─" * len(header)

    print()
    print("╔" + "═" * (len(header) - 2) + "╗")
    print("║" + "   OFFLINE vs STREAMING LATENCY COMPARISON".center(len(header) - 2) + "║")
    print("╠" + "═" * (len(header) - 2) + "╣")
    print(header)
    print("╠" + "═" * (len(header) - 2) + "╣")

    for label, rows in results.items():
        for mode, module, times, budget, per_frame in rows:
            print(_row(mode, module, times, budget, per_frame))
        print("╠" + "─" * (len(header) - 2) + "╣")

    print("╠" + "═" * (len(header) - 2) + "╣")
    print("║" + f"  TARGET for 25 FPS: {TARGET_FRAME_MS:.0f}ms total  │  "
          f"Moshi≤15ms  Bridge≤8ms  DittoA2M≤30ms".center(len(header) - 2) + "║")

    print()
    print("  OPTIMIZATION HINTS:")
    print("  ┌─ If Bridge STREAMING >> OFFLINE:")
    print("  │  → Reduce BRIDGE_FLUSH_TIMEOUT_MS (try 20ms)")
    print("  │  → Reduce BRIDGE_CHUNK to 1 for lower per-chunk latency")
    print("  ├─ If Ditto A2M STREAMING >> OFFLINE:")
    print("  │  → Reduce sampling_timesteps (try 6–8)")
    print("  │  → online_mode=True is expected to be 2–3× slower than batch")
    print("  └─ If total > 40ms in streaming mode:")
    print("     → 25 FPS is not achievable; consider 15 FPS target (67ms budget)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Offline vs. Online streaming latency comparison for Moshi-Bridge-Ditto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bridge-ckpt",    required=True, help="Path to bridge .pt checkpoint")
    p.add_argument("--bridge-config",  required=True, help="Path to bridge config.yaml")
    p.add_argument("--num-frames",     type=int, default=100, help="Number of frames to benchmark")
    p.add_argument("--chunk-size",     type=int, default=2,   help="Streaming chunk size (Mimi frames)")
    p.add_argument("--warmup",         type=int, default=5,   help="Warmup iterations")
    p.add_argument("--ditto-cfg-pkl",  default=None, help="Ditto config .pkl (optional)")
    p.add_argument("--ditto-data-root",default=None, help="Ditto TRT model dir (optional)")
    p.add_argument("--ditto-steps",    type=int, default=10,  help="Ditto sampling steps")
    p.add_argument("--source-image",   default=None, help="Portrait image for Ditto benchmark")
    args = p.parse_args()

    logger.info("=" * 60)
    logger.info("  LATENCY BENCHMARK: Offline vs. Streaming")
    logger.info(f"  frames={args.num_frames} | chunk={args.chunk_size} | warmup={args.warmup}")
    logger.info("=" * 60)

    _warmup_gpu()

    results = {}

    # ── Bridge ───────────────────────────────────────────────────────────────
    logger.info("\n[1/2] Bridge benchmark ...")

    off_times = bench_bridge_offline(
        args.bridge_ckpt, args.bridge_config,
        args.num_frames, args.warmup,
    )
    # Offline: per-frame = total / num_frames
    off_per_frame = [t / args.num_frames for t in off_times]

    chunk_times, str_per_frame = bench_bridge_streaming(
        args.bridge_ckpt, args.bridge_config,
        args.num_frames, args.chunk_size, args.warmup,
    )

    results["Bridge"] = [
        ("OFFLINE",   "Bridge (per frame)",  off_per_frame, _BUDGETS["Bridge (batch/frame)"], True),
        ("OFFLINE",   "Bridge (full batch)", off_times,     None, False),
        ("STREAMING", "Bridge (per frame)",  str_per_frame, _BUDGETS["Bridge (batch/frame)"], True),
        ("STREAMING", "Bridge (per chunk)",  chunk_times,   _BUDGETS["Bridge (chunk)"], False),
    ]

    # ── Ditto A2M (optional) ──────────────────────────────────────────────────
    if args.ditto_cfg_pkl and args.ditto_data_root:
        logger.info("\n[2/2] Ditto Audio2Motion benchmark ...")
        off_a2m, str_a2m = bench_ditto_a2m(
            args.ditto_cfg_pkl, args.ditto_data_root,
            args.num_frames, args.ditto_steps,
            src_image=args.source_image,
            warmup=args.warmup,
        )
        results["Ditto A2M"] = [
            ("OFFLINE",   "Ditto A2M (per batch)", off_a2m, _BUDGETS["Ditto A2M"], False),
            ("STREAMING", "Ditto A2M (per batch)", str_a2m, _BUDGETS["Ditto A2M"], False),
        ]
    else:
        logger.info("\n[2/2] Ditto A2M benchmark skipped (no --ditto-cfg-pkl provided)")

    # ── Print table ──────────────────────────────────────────────────────────
    print_comparison_table(results)


if __name__ == "__main__":
    main()
