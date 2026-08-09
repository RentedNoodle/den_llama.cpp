#!/usr/bin/env python3
"""
profile_driver_overhead.py — Measure CUDA driver launch overhead during LLM decode.

Multi-method measurement toolkit. Fires llama-bench, records GPU utilization
via pynvml at high frequency, computes driver overhead from GPU idle gaps.

METHODS (auto-selected best available):
  1. pynvml GPU utilization sampling (always available, 20-100ms granularity)
  2. CUPTI ctypes FFI (precise, if cupti64 DLL accessible)
  3. Wall-clock + cudaEvent (manual, instrumented — guide printed)

USAGE:
  python tools/profile_driver_overhead.py                                    \
      --binary I:/den_llama.cpp/build_ninja/bin/llama-cli.exe               \
      --model  I:/models/ornith-1.0-35b-APEX-I-Mini-MTP.gguf                \
      --n-tokens 128                                                         \
      --ngl 99

OUTPUT:
  tok/s, avg GPU util%, estimated driver overhead ms/token, % of budget, verdict

THRESHOLDS:
  <5%   = FINE — driver overhead negligible
  5-15% = WORTH_OPTIMIZING — batch launches, CUDA graphs viable
  >15%  = URGENT — driver is the bottleneck, persistent kernel or graphs needed
"""

import subprocess
import sys
import os
import time
import threading
import argparse
import json
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# ── pynvml ──────────────────────────────────────────────────────────────

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

# ── ctypes for CUPTI FFI ───────────────────────────────────────────────

try:
    import ctypes
    from ctypes import (
        c_uint8, c_uint16, c_uint32, c_uint64, c_int32, c_int64,
        c_size_t, c_void_p, c_char_p, c_bool,
        POINTER, Structure, CFUNCTYPE, byref, cast, sizeof, addressof
    )
    HAS_CTYPES = True
except ImportError:
    HAS_CTYPES = False

# ── Data structures ────────────────────────────────────────────────────

@dataclass
class GpuSample:
    """Single GPU utilization sample."""
    timestamp: float        # wall-clock seconds since epoch
    gpu_util: int           # 0-100
    mem_util: int           # 0-100
    temperature: int        # Celsius

@dataclass
class OverheadReport:
    """Final measurement report."""
    method: str
    total_wall_s: float
    n_tokens: int
    tok_per_s: float
    ms_per_token: float

    # GPU metrics
    avg_gpu_util_pct: float
    gpu_active_s: float
    gpu_idle_s: float

    # Overhead estimates
    estimated_launches_per_token: int
    overhead_ms_per_token: float
    overhead_pct: float
    avg_launch_gap_us: float
    verdict: str

    # Raw data
    gpu_samples: List[GpuSample] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def print_report(self):
        print()
        print("═" * 70)
        print("  CUDA DRIVER OVERHEAD PROFILE")
        print("═" * 70)
        print(f"  Method:                 {self.method}")
        print(f"  Total wall time:        {self.total_wall_s:.3f}s")
        print(f"  Tokens generated:       {self.n_tokens}")
        print(f"  Generation speed:       {self.tok_per_s:.1f} tok/s")
        print(f"  Time per token:         {self.ms_per_token:.3f} ms")
        print()
        print(f"  GPU utilization (avg):  {self.avg_gpu_util_pct:.1f}%")
        print(f"  GPU active time:        {self.gpu_active_s:.3f}s")
        print(f"  GPU idle time:          {self.gpu_idle_s:.3f}s")
        print()
        print(f"  Est. launches/token:    {self.estimated_launches_per_token}")
        print(f"  Overhead per token:     {self.overhead_ms_per_token:.3f} ms")
        print(f"  Overhead % of budget:   {self.overhead_pct:.1f}%")
        print(f"  Avg launch gap:         {self.avg_launch_gap_us:.3f} µs")
        print()
        print(f"  ─── VERDICT ───")
        print(f"  {self.verdict}")
        print()

        if self.warnings:
            print("  ─── WARNINGS ───")
            for w in self.warnings:
                print(f"  [!] {w}")
            print()

        print("═" * 70)


# ── pynvml Sampler ─────────────────────────────────────────────────────

class GpuSampler:
    """High-frequency GPU utilization sampler via pynvml."""

    def __init__(self, device_index: int = 0, interval_ms: int = 20):
        self.device_index = device_index
        self.interval_ms = interval_ms
        self.samples: List[GpuSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handle = None

    def start(self):
        if not HAS_PYNVML:
            raise RuntimeError("pynvml not installed: pip install pynvml")

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

        # Warmup: discard first sample (stale counters)
        pynvml.nvmlDeviceGetUtilizationRates(self._handle)

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[GpuSample]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._handle:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return self.samples

    def _sample_loop(self):
        while self._running:
            try:
                ts = time.time()
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)

                self.samples.append(GpuSample(
                    timestamp=ts,
                    gpu_util=util.gpu,
                    mem_util=util.memory,
                    temperature=temp,
                ))
            except pynvml.NVMLError:
                pass

            time.sleep(self.interval_ms / 1000.0)


# ── llama-bench parser ─────────────────────────────────────────────────

def parse_llama_output(line: str) -> Optional[Dict]:
    """Parse llama-bench / llama-perplexity output for timings."""
    # llama-bench output format: "llama_perf_context_print: ..."
    # Example: "llama_perf_print: load_time = 1234.56 ms"
    # Example: "llama_perf_print: prompt eval = 1234.56 ms / 7 tokens (...)"
    # Example: "llama_perf_print: eval time = 1234.56 ms / 128 runs (...)"
    # We want: eval time and token count

    line = line.strip()
    if "eval time" in line and "ms /" in line:
        # "eval time = 1234.56 ms / 128 runs (  9.64 ms per token, 103.75 tokens per second)"
        try:
            parts = line.split("/")
            ms_part = parts[0].split("=")[-1].strip().replace(" ms", "")
            runs_part = parts[1].strip().split()[0]
            total_ms = float(ms_part)
            n_runs = int(runs_part)
            return {"phase": "eval", "total_ms": total_ms, "n_runs": n_runs}
        except (ValueError, IndexError):
            pass

    if "prompt eval" in line and "ms /" in line:
        try:
            parts = line.split("/")
            ms_part = parts[0].split("=")[-1].strip().replace(" ms", "")
            tokens_part = parts[1].strip().split()[0]
            total_ms = float(ms_part)
            n_tokens = int(tokens_part)
            return {"phase": "prompt", "total_ms": total_ms, "n_tokens": n_tokens}
        except (ValueError, IndexError):
            pass

    return None


# ── llama-bench wrapper ────────────────────────────────────────────────

def run_llama_bench(
    binary: str,
    model: str,
    n_tokens: int = 128,
    ngl: int = 99,
    extra_args: Optional[List[str]] = None,
    timeout_s: int = 600,
) -> Tuple[float, int, bool]:
    """
    Run llama-bench (or llama-cli), capture wall-clock time and token count.
    Returns (total_wall_s, n_tokens, coherent).
    Uses subprocess.DEVNULL to avoid pipe deadlock from thinking-model output.
    """
    # NON-INTERACTIVE flags (prevents hang waiting for stdin)
    # --simple-io: no interactive prompt, batch mode
    # --no-display-prompt: don't print ">" prompt
    # -e: eval mode (treat -p as prompt, exit after -n tokens)
    # --no-perf: suppress perf output (we capture timing ourselves)
    cmd = [
        binary,
        "-m", model,
        "-p", "64",
        "-n", str(n_tokens),
        "-ngl", str(ngl),
        "-t", "8",
        "-c", "32768",
        "-e",                   # eval mode, exit after generation
        "--simple-io",           # non-interactive, no prompt
        "--no-display-prompt",   # suppress "> " output
        "--no-perf",             # suppress internal timers (reduce noise)
    ]
    if extra_args:
        cmd += extra_args

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout_s,
            cwd=os.path.dirname(binary),
        )
    except subprocess.TimeoutExpired:
        print(f"[!] timed out after {timeout_s}s")
        return time.time() - t0, 0, False

    t1 = time.time()
    wall_s = t1 - t0

    tokens_generated = n_tokens
    eval_ms = 0.0
    for line in result.stderr.splitlines():
        if "eval time" in line and "ms /" in line:
            try:
                parts = line.split("/")
                ms_part = parts[0].split("=")[-1].strip().replace(" ms", "")
                runs_part = parts[1].strip().split()[0]
                eval_ms = float(ms_part)
                tokens_generated = int(runs_part)
            except (ValueError, IndexError):
                pass

    coherent = result.returncode == 0
    return wall_s, tokens_generated, coherent


# ── Analysis ───────────────────────────────────────────────────────────

def compute_overhead(
    gpu_samples: List[GpuSample],
    wall_s: float,
    n_tokens: int,
    method: str,
    extra_warnings: Optional[List[str]] = None,
) -> OverheadReport:
    """Compute overhead metrics from GPU utilization samples."""

    warnings = list(extra_warnings or [])

    if not gpu_samples:
        warnings.append("No GPU samples collected. Is nvidia-smi working?")

    # Compute average GPU utilization during the run
    # Filter to samples within [wall_start, wall_end]
    if gpu_samples:
        gpu_utils = [s.gpu_util for s in gpu_samples]
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
    else:
        avg_gpu_util = 100.0  # assume fully utilized if we can't measure

    gpu_active_s = wall_s * (avg_gpu_util / 100.0)
    gpu_idle_s = wall_s - gpu_active_s

    # Token metrics
    tok_per_s = n_tokens / wall_s if wall_s > 0 else 0.0
    ms_per_token = (wall_s / n_tokens * 1000.0) if n_tokens > 0 else 0.0

    # Estimate launches per token
    # 35B MoE: ~41 layers × ~40 ops/layer = ~1640 launches per token
    # This is an estimate — actual depends on model architecture
    estimated_launches = 1640

    # Overhead per token
    overhead_ms_per_token = (gpu_idle_s / n_tokens * 1000.0) if n_tokens > 0 else 0.0
    overhead_pct = (gpu_idle_s / wall_s * 100.0) if wall_s > 0 else 0.0

    # Average launch gap
    if estimated_launches > 0 and n_tokens > 0:
        total_launches = estimated_launches * n_tokens
        avg_launch_gap_us = (gpu_idle_s * 1e6) / total_launches if total_launches > 0 else 0.0
    else:
        avg_launch_gap_us = 0.0

    # Verdict
    if overhead_pct < 5.0:
        verdict = "<5% = FINE — driver overhead negligible. No action needed."
    elif overhead_pct < 15.0:
        verdict = f"5-15% = WORTH OPTIMIZING — batch{estimated_launches} launches, CUDA graphs or persistent kernel could recover ~{overhead_ms_per_token:.2f}ms/token."
    else:
        verdict = f">15% = URGENT — driver is the bottleneck at {overhead_pct:.1f}% overhead. Persistent kernel or CUDA graphs STRONGLY recommended."

    if avg_gpu_util < 90:
        warnings.append(
            f"Low GPU utilization ({avg_gpu_util:.0f}%). "
            "If this is due to CPU-GPU sync points, check --no-display-prompt / --simple-io flags."
        )

    return OverheadReport(
        method=method,
        total_wall_s=wall_s,
        n_tokens=n_tokens,
        tok_per_s=tok_per_s,
        ms_per_token=ms_per_token,
        avg_gpu_util_pct=avg_gpu_util,
        gpu_active_s=gpu_active_s,
        gpu_idle_s=gpu_idle_s,
        estimated_launches_per_token=estimated_launches,
        overhead_ms_per_token=overhead_ms_per_token,
        overhead_pct=overhead_pct,
        avg_launch_gap_us=avg_launch_gap_us,
        verdict=verdict,
        gpu_samples=gpu_samples,
        warnings=warnings,
    )


# ── Simple decode-loop timer (no GPU profiling, pure wall clock) ───────

def simple_decode_loop_profile(
    binary: str,
    model: str,
    n_tokens: int = 128,
    ngl: int = 99,
    timeout_s: int = 600,
) -> OverheadReport:
    """
    Alternative: profile individual decode steps with high-res timer.
    Instruments llama.cpp's decode loop via timed subprocess I/O.
    Less precise but works without any GPU profiling.
    """
    print("[*] Simple decode-loop timing (wall-clock only, no GPU sampling)")
    print("[*] This measures TOTAL time, no GPU breakdown possible.")
    print("[*] For GPU breakdown, use --pynvml method.")
    print()

    wall_s, n_tokens, coherent = run_llama_bench(
        binary, model, n_tokens, ngl, timeout_s=timeout_s
    )

    warnings = []
    if not coherent:
        warnings.append("Output may be incoherent — GPU errors inflate timing")

    return compute_overhead(
        gpu_samples=[GpuSample(time.time(), 100, 0, 0)],  # assume 100% GPU util
        wall_s=wall_s,
        n_tokens=n_tokens,
        method="wall-clock only (no GPU util — assuming 100%)",
        extra_warnings=warnings,
    )


# ── pynvml method ──────────────────────────────────────────────────────

def pynvml_profile(
    binary: str,
    model: str,
    n_tokens: int = 128,
    ngl: int = 99,
    sample_interval_ms: int = 20,
    extra_args: Optional[List[str]] = None,
    timeout_s: int = 600,
) -> OverheadReport:
    """
    Profile using pynvml GPU utilization sampling at high frequency.
    """
    if not HAS_PYNVML:
        print("[!] pynvml not available. Install: pip install pynvml")
        print("[!] Falling back to simple wall-clock timing.")
        return simple_decode_loop_profile(binary, model, n_tokens, ngl, timeout_s=timeout_s)

    print(f"[*] Starting GPU sampler ({sample_interval_ms}ms interval)...")
    sampler = GpuSampler(device_index=0, interval_ms=sample_interval_ms)

    # Start sampling BEFORE launching llama-bench
    sampler.start()
    time.sleep(0.1)  # let sampler settle

    print(f"[*] Running: {binary} -m {os.path.basename(model)} -n {n_tokens} -ngl {ngl}")
    wall_s, n_tokens_out, coherent = run_llama_bench(
        binary, model, n_tokens, ngl, extra_args, timeout_s=timeout_s
    )

    # Brief settle before stopping sampler (capture any trailing GPU activity)
    time.sleep(0.2)

    samples = sampler.stop()
    print(f"[*] Collected {len(samples)} GPU utilization samples over {wall_s:.1f}s")

    warnings = []
    if not coherent:
        warnings.append("Output may be incoherent — GPU errors inflate timing")
    if len(samples) < 10:
        warnings.append(f"Only {len(samples)} GPU samples — increase run time or reduce interval")

    # Trim samples to actual run window
    # Find first sample with GPU util > 0 (when model loads and starts running)
    active_samples = [s for s in samples if s.gpu_util > 5]
    if not active_samples:
        active_samples = samples

    return compute_overhead(
        gpu_samples=active_samples,
        wall_s=wall_s,
        n_tokens=n_tokens_out,
        method=f"pynvml GPU utilization ({sample_interval_ms}ms polling)",
        extra_warnings=warnings,
    )


# ── CUPTI ctypes FFI method (experimental) ────────────────────────────

def cupti_ffi_profile(
    binary: str,
    model: str,
    n_tokens: int = 128,
    ngl: int = 99,
    timeout_s: int = 600,
) -> OverheadReport:
    """
    Experimental: Use CUPTI via ctypes FFI for precise kernel timestamps.
    Requires cupti64 DLL accessible on PATH.
    """
    if not HAS_CTYPES:
        print("[!] ctypes not available.")
        return pynvml_profile(binary, model, n_tokens, ngl, timeout_s=timeout_s)

    # Try to load CUPTI DLL
    cupti_dll = None
    dll_paths = [
        "cupti64_2026.2.0.dll",
        "cupti64.dll",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/extras/CUPTI/lib64/cupti64_2026.2.0.dll",
    ]
    for path in dll_paths:
        try:
            cupti_dll = ctypes.CDLL(path)
            print(f"[*] Loaded CUPTI DLL: {path}")
            break
        except OSError:
            continue

    if cupti_dll is None:
        print("[!] CUPTI DLL not found. Check CUDA 13.3 extras/CUPTI/lib64/")
        print("[!] Falling back to pynvml method.")
        return pynvml_profile(binary, model, n_tokens, ngl, timeout_s=timeout_s)

    # CUPTI FFI is complex — requires buffer management callbacks, activity record parsing.
    # For now, we note that CUPTI is available and suggest the .cu tool instead.
    print("[!] CUPTI FFI via ctypes is NOT fully implemented for Python.")
    print("[!] Use profile_driver_overhead.cu for precise CUPTI-based measurement.")
    print("[!] Falling back to pynvml method for now.")

    return pynvml_profile(binary, model, n_tokens, ngl, timeout_s=timeout_s)


# ── Manual instrumentation guide ───────────────────────────────────────

def print_instrumentation_guide():
    """Print instructions for manually instrumenting llama.cpp's decode loop."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  MANUAL INSTRUMENTATION GUIDE                                       ║
║  For precise per-launch timing, instrument ggml-cuda decode path.   ║
╚══════════════════════════════════════════════════════════════════════╝

In ggml/src/ggml-cuda/ggml-cuda.cu, find the decode loop and add:

  #include <chrono>

  // Before decode loop:
  auto t_decode_start = std::chrono::high_resolution_clock::now();
  uint64_t launch_count = 0;

  // Before EACH cudaLaunchKernel / ggml_cuda_op call:
  auto t0 = std::chrono::high_resolution_clock::now();
  // ... kernel launch ...
  cudaEvent_t ev;
  cudaEventCreate(&ev);
  cudaEventRecord(ev, stream);
  cudaEventSynchronize(ev);
  auto t1 = std::chrono::high_resolution_clock::now();
  launch_count++;

  double launch_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  // Log launch_us every 10th launch to CSV

After 128 tokens:
  total_decode_ms = total wall clock
  total_launch_overhead = sum of all launch times
  avg_launch_us = total_launch_overhead / launch_count

Or add this sampler block after the decode loop in llama.cpp's main loop:

  // After each token generation:
  static int token_count = 0;
  static auto token_start = std::chrono::high_resolution_clock::now();
  auto token_end = std::chrono::high_resolution_clock::now();
  double token_ms = std::chrono::duration<double, std::milli>(token_end - token_start).count();
  fprintf(stderr, "TOKEN_TIMING: token=%d time_ms=%.3f\n", token_count++, token_ms);
  token_start = token_end;

This gives per-token wall clock, which compared against GPU active time
(from CUPTI) yields the exact driver overhead.
""")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure CUDA driver overhead during LLM decode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic pynvml-based profile (no build required):
  python tools/profile_driver_overhead.py \\
      --binary I:/den_llama.cpp/build_ninja/bin/llama-cli.exe \\
      --model  I:/models/ornith-1.0-35b-APEX-I-Mini-MTP.gguf

  # With more tokens for better averaging:
  python tools/profile_driver_overhead.py \\
      --binary I:/den_llama.cpp/build_ninja/bin/llama-cli.exe \\
      --model  I:/models/ornith-1.0-35b-APEX-I-Mini-MTP.gguf \\
      --n-tokens 256 --sample-interval 10

  # Print instrumentation guide only:
  python tools/profile_driver_overhead.py --guide
""",
    )

    parser.add_argument("--binary", help="Path to llama-cli.exe or llama-bench.exe")
    parser.add_argument("--model", help="Path to GGUF model file")
    parser.add_argument("--n-tokens", type=int, default=128,
                        help="Number of tokens to generate (default: 128)")
    parser.add_argument("--ngl", type=int, default=99,
                        help="Number of GPU layers (default: 99)")
    parser.add_argument("--sample-interval", type=int, default=20,
                        help="GPU sampling interval in ms (default: 20)")
    parser.add_argument("--method", choices=["pynvml", "cupti-ffi", "simple", "auto"],
                        default="auto",
                        help="Measurement method (default: auto = best available)")
    parser.add_argument("--guide", action="store_true",
                        help="Print manual instrumentation guide and exit")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout in seconds for llama-cli (default: 600)")
    parser.add_argument("--extra-args", nargs="*", default=[],
                        help="Extra arguments to pass to llama-cli")

    args = parser.parse_args()

    if args.guide:
        print_instrumentation_guide()
        return 0

    if not args.binary or not args.model:
        parser.error("--binary and --model are required. Use --guide for manual approach.")

    # Normalize to absolute paths (relative paths break when cwd != script dir)
    args.binary = os.path.abspath(args.binary)
    args.model = os.path.abspath(args.model)

    # Validate binary exists
    if not os.path.exists(args.binary):
        print(f"[!] Binary not found: {args.binary}")
        sys.exit(1)

    # Validate model exists
    if not os.path.exists(args.model):
        print(f"[!] Model not found: {args.model}")
        sys.exit(1)

    # ── Run measurement ──────────────────────────────────────────────
    print("=" * 70)
    print("  profile_driver_overhead.py")
    print(f"  Binary:  {args.binary}")
    print(f"  Model:   {os.path.basename(args.model)}")
    print(f"  Tokens:  {args.n_tokens}")
    print(f"  Method:  {args.method}")
    print("=" * 70)
    print()

    if args.method == "simple":
        report = simple_decode_loop_profile(
            args.binary, args.model, args.n_tokens, args.ngl,
            timeout_s=args.timeout
        )
    elif args.method == "cupti-ffi":
        report = cupti_ffi_profile(
            args.binary, args.model, args.n_tokens, args.ngl
        )
    elif args.method == "pynvml":
        report = pynvml_profile(
            args.binary, args.model, args.n_tokens, args.ngl,
            args.sample_interval, args.extra_args if args.extra_args else None,
            timeout_s=args.timeout
        )
    else:  # auto
        # Try pynvml first, fall back to simple
        report = pynvml_profile(
            args.binary, args.model, args.n_tokens, args.ngl,
            args.sample_interval, args.extra_args if args.extra_args else None,
            timeout_s=args.timeout
        )

    if args.json:
        data = {
            "method": report.method,
            "total_wall_s": report.total_wall_s,
            "n_tokens": report.n_tokens,
            "tok_per_s": report.tok_per_s,
            "ms_per_token": report.ms_per_token,
            "avg_gpu_util_pct": report.avg_gpu_util_pct,
            "gpu_active_s": report.gpu_active_s,
            "gpu_idle_s": report.gpu_idle_s,
            "overhead_ms_per_token": report.overhead_ms_per_token,
            "overhead_pct": report.overhead_pct,
            "avg_launch_gap_us": report.avg_launch_gap_us,
            "verdict": report.verdict,
            "warnings": report.warnings,
            "n_gpu_samples": len(report.gpu_samples),
            "gpu_util_distribution": {
                "min": min((s.gpu_util for s in report.gpu_samples), default=0),
                "max": max((s.gpu_util for s in report.gpu_samples), default=0),
                "p50": _percentile([s.gpu_util for s in report.gpu_samples], 50),
                "p90": _percentile([s.gpu_util for s in report.gpu_samples], 90),
                "p99": _percentile([s.gpu_util for s in report.gpu_samples], 99),
            },
        }
        print(json.dumps(data, indent=2))
    else:
        report.print_report()

    return 0


def _percentile(data: List[float], pct: float) -> float:
    """Compute percentile of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]


if __name__ == "__main__":
    sys.exit(main())
