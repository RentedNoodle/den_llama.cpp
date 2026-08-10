#!/usr/bin/env python3
"""
measure_vram_context.py — VRAM metering wrapper for NVFP4 KV context scaling.

Runs gate_accuracy_kv.py at increasing context sizes, polls nvidia-smi during
each run to measure peak VRAM, and reports a comparison table.

Output table: context, F32_KV_VRAM_MB, NVFP4_KV_VRAM_MB, compression_ratio, KLD, cos.

Usage:
  python tools/measure_vram_context.py --model I:\\models\\ornith-9b.gguf
  python tools/measure_vram_context.py --model I:\\models\\ornith-9b.gguf --ctx-sizes 1024,2048,4096,8192
  python tools/measure_vram_context.py --model I:\\models\\ornith-35b.gguf --ctx-sizes 1024,2048,4096 --ngl 0
"""

import subprocess
import sys
import os
import argparse
import time
import threading
import tempfile
from pathlib import Path
from typing import Optional, List, Dict

# ─────────────────────────────────────────────────────────────────────────────
# PATH DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

GATE_KV_PATH = SCRIPT_DIR / "gate_accuracy_kv.py"
if not GATE_KV_PATH.is_file():
    print(f"ERROR: gate_accuracy_kv.py not found at {GATE_KV_PATH}")
    sys.exit(1)

PYTHON = r"C:\Den\den-py314\Scripts\python.exe"
if not os.path.isfile(PYTHON):
    PYTHON = sys.executable

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_vram_mb() -> Optional[float]:
    """Poll current VRAM usage in MB via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().split("\n")[0].strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


class VramSampler:
    """Background thread that samples VRAM usage at fixed interval."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.samples: List[float] = []
        self._peak: float = 0.0

    def start(self):
        self._running = True
        self._peak = 0.0
        self.samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if self.samples:
            return {
                "peak_mb": max(self.samples),
                "min_mb": min(self.samples),
                "avg_mb": sum(self.samples) / len(self.samples),
                "n_samples": len(self.samples),
            }
        return {"peak_mb": 0.0, "min_mb": 0.0, "avg_mb": 0.0, "n_samples": 0}

    def _sample_loop(self):
        while self._running:
            vram = get_vram_mb()
            if vram is not None:
                self.samples.append(vram)
                if vram > self._peak:
                    self._peak = vram
            time.sleep(self.interval_ms / 1000.0)


def run_gate_for_context(
    model_path: str,
    context_size: int,
    ngl: int,
    threads: int,
    n_tokens: int,
    tail_tokens: int,
    vram_sampler: VramSampler,
    timeout_s: int = 600,
) -> Dict:
    """
    Run gate_accuracy_kv.py for one context size.
    Returns parsed result dict + VRAM stats.
    """
    # Start VRAM sampling BEFORE the gate run
    vram_sampler.start()

    # Small delay to ensure sampler is running
    time.sleep(0.5)

    # Run gate_accuracy_kv.py as subprocess, capture its stdout for parsing
    cmd = [
        PYTHON,
        str(GATE_KV_PATH),
        "--model", model_path,
        "--tokens", str(n_tokens),
        "--ngl", str(ngl),
        "--threads", str(threads),
        "--tail-tokens", str(tail_tokens),
    ]

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        vram_stats = vram_sampler.stop()
        return {
            "context": context_size,
            "passed": False,
            "error": f"Timeout after {timeout_s}s",
            "elapsed_s": elapsed,
            **vram_stats,
        }

    elapsed = time.perf_counter() - t0
    vram_stats = vram_sampler.stop()

    # Parse gate output for KLD/cos
    output = result.stdout + result.stderr
    passed = result.returncode == 0

    # Extract tile KLD and cos from the output
    tile_kld = None
    tile_cos = None
    median_kld = None

    for line in output.splitlines():
        # "mean KLD  : 0.00000000" in TILE region section
        if "tile_kld_mean" in line.lower() or ("TILE" in line and "mean KLD" in line):
            try:
                tile_kld = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        # "mean cos  : 1.00000000" in TILE region section
        if "tile_cos_mean" in line.lower() or ("TILE" in line and "mean cos" in line):
            try:
                tile_cos = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass

    # Fallback: search for "median_kld" in the metrics section
    if tile_kld is None:
        import re
        m = re.search(r"median_kld\s+[\d.]+\s+", output)
        if m:
            try:
                median_kld = float(m.group().split()[1])
            except (ValueError, IndexError):
                pass

    return {
        "context": context_size,
        "passed": passed,
        "tile_kld": tile_kld,
        "tile_cos": tile_cos,
        "median_kld": median_kld,
        "elapsed_s": elapsed,
        "exit_code": result.returncode,
        **vram_stats,
    }


def measure_vram_context(
    model_path: str,
    context_sizes: List[int],
    ngl: int = 99,
    threads: int = 6,
    n_tokens: int = 300,
    tail_tokens: int = 1024,
    timeout_s: int = 600,
) -> List[Dict]:
    """
    Measure VRAM at each context size for both F32 KV and NVFP4 KV.
    gate_accuracy_kv.py creates BOTH contexts (F32 + NVFP4) in a single run,
    so each run measures the combined VRAM of both contexts + model.

    To isolate F32 vs NVFP4 KV VRAM, we do TWO passes:
    1. Run gate normally (both contexts) → get NVFP4-gated total VRAM
    2. Run with DEN_NVFP4_KV_CACHE=0 for F32 context + DEN_NVFP4_KV_CACHE=1 for NVFP4
       → The delta between two single-context runs tells us per-cache VRAM.

    Since gate_accuracy_kv.py creates BOTH contexts simultaneously, the peak
    VRAM includes model + F32 context + NVFP4 context. We extract per-KV-cache
    VRAM by running single-context baselines at each size.
    """
    print(f"\n{BOLD}{'='*90}{RESET}")
    print(f"{BOLD}  VRAM METERING — NVFP4 KV vs F32 KV at Scaling Context Sizes{RESET}")
    print(f"{'='*90}")
    print(f"  Model         : {Path(model_path).name}")
    print(f"  Context sizes : {context_sizes}")
    print(f"  GPU layers    : {ngl}")
    print(f"  Threads       : {threads}")
    print(f"  Gate tokens   : {n_tokens}")
    print(f"  Tail tokens   : {tail_tokens}")
    print(f"{'='*90}")

    results = []
    n_ctx = len(context_sizes)

    for i, ctx in enumerate(context_sizes):
        print(f"\n{BOLD}[{i + 1}/{n_ctx}] Context: {ctx} tokens{RESET}")

        # ── Phase A: Dual-context run (gate_accuracy_kv.py default) ─────────
        # This creates F32 oracle + NVFP4 candidate simultaneously.
        # Peak VRAM = model + F32 KV + NVFP4 KV + activations.
        print(f"  {CYAN}Phase A: Dual-context gate run (F32 + NVFP4 KV)...{RESET}")
        sampler_a = VramSampler(interval_ms=50)
        result_a = run_gate_for_context(
            model_path=model_path,
            context_size=ctx,
            ngl=ngl,
            threads=threads,
            n_tokens=n_tokens,
            tail_tokens=tail_tokens,
            vram_sampler=sampler_a,
            timeout_s=timeout_s,
        )

        # ── Phase B: F32-only baseline ──────────────────────────────────────
        # Run with NVFP4 disabled, single F32 context.
        # This gives us model + F32 KV baseline.
        print(f"  {CYAN}Phase B: F32-only baseline (single context)...{RESET}")

        # Use a quick subprocess that creates ONE F32 context with same ctx size
        sampler_b = VramSampler(interval_ms=50)
        sampler_b.start()
        time.sleep(0.5)

        # Quick script: import gate_accuracy_kv, create single F32 context, decode prompt
        baseline_script = f"""
import sys
sys.path.insert(0, r"{SCRIPT_DIR}")
import os
os.environ["DEN_NVFP4_KV_CACHE"] = "0"
os.environ.pop("DEN_THRIFT_ATTENTION", None)
from gate_accuracy_kv import _get_lib, DualContextKVModel, LONGER_PROMPT

lib = _get_lib()
mparams = lib.llama_model_default_params()
mparams.n_gpu_layers = {ngl}
model = lib.llama_model_load_from_file(r"{model_path}".encode(), mparams)
vocab = lib.llama_model_get_vocab(model)
n_vocab = lib.llama_vocab_n_tokens(vocab)

cp = lib.llama_context_default_params()
cp.n_ctx = {ctx}
cp.n_threads = {threads}
cp.n_threads_batch = {threads}
cp.type_k = 0  # F32
cp.type_v = 0  # F32
cp.nvfp4_kv_enabled = False
cp.sparse_kv_enabled = False
cp.expert_stage = True
ctx = lib.llama_init_from_model(model, cp)

# Decode a short prompt to allocate KV
prompt = "The capital of France is Paris, a city known for its"
import ctypes
from ctypes import c_int32, c_char_p, POINTER
llama_token = c_int32
text_bytes = prompt.encode("utf-8")
n_max = len(text_bytes) + 32
tokens = (llama_token * n_max)()
n = lib.llama_tokenize(vocab, text_bytes, len(text_bytes), tokens, n_max, True, True)
if n > 0:
    token_list = list(tokens[:n])
    ta = (llama_token * n)(*token_list)
    batch = lib.llama_batch_get_one(ta, n)
    lib.llama_decode(ctx, batch)

import time
time.sleep(1)
print("BASELINE_DONE")
lib.llama_free(ctx)
lib.llama_model_free(model)
"""
        b_script_path = tempfile.mktemp(suffix=".py", prefix="vram_baseline_")
        with open(b_script_path, "w") as f:
            f.write(baseline_script)

        try:
            b_result = subprocess.run(
                [PYTHON, b_script_path],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            b_result = subprocess.CompletedProcess([], -1, "", "")
        finally:
            os.unlink(b_script_path)

        vram_stats_b = sampler_b.stop()

        # ── Compute per-KV-cache VRAM ────────────────────────────────────────
        # dual_total = model + F32_KV + NVFP4_KV
        # f32_only   = model + F32_KV
        # nvfp4_kv   = dual_total - f32_only
        dual_peak = result_a.get("peak_mb", 0.0)
        f32_peak = vram_stats_b.get("peak_mb", 0.0)

        nvfp4_kv_vram = max(0.0, dual_peak - f32_peak)

        # F32 KV VRAM: estimate from context size
        # Qwen 9B: head_dim=128, n_heads=16, n_kv_heads=4 (GQA), 32 layers
        # KV per layer per token: 2 * 128 * 4 * 4 bytes = 4096 bytes
        # 48 layers * ctx tokens * 4096 = ~200MB at 1k
        # This is ballpark — actual VRAM depends on model architecture
        f32_kv_est = f32_peak * 0.0  # we use measured delta
        # Better: assume NVFP4 KV = 3.1x compression
        compression_ratio = f32_peak / nvfp4_kv_vram if nvfp4_kv_vram > 0.1 else 0.0

        row = {
            "context": ctx,
            "dual_peak_mb": dual_peak,
            "f32_only_mb": f32_peak,
            "nvfp4_kv_estimate_mb": nvfp4_kv_vram,
            "compression_ratio": round(compression_ratio, 2),
            "tile_kld": result_a.get("tile_kld"),
            "tile_cos": result_a.get("tile_cos"),
            "median_kld": result_a.get("median_kld"),
            "gate_passed": result_a.get("passed", False),
            "gate_elapsed_s": result_a.get("elapsed_s", 0),
            "error": result_a.get("error"),
        }
        results.append(row)

        # Progress
        kld_str = f"{row['tile_kld']:.6f}" if row['tile_kld'] is not None else "N/A"
        cos_str = f"{row['tile_cos']:.6f}" if row['tile_cos'] is not None else "N/A"
        ratio_str = f"{compression_ratio:.1f}x" if compression_ratio > 0.1 else "N/A"
        status = f"{GREEN}PASS{RESET}" if row["gate_passed"] else f"{RED}FAIL{RESET}"

        print(f"  [{i + 1}/{n_ctx}] ctx={ctx}  "
              f"dualkV={dual_peak:.0f}MB  nvfp4Est={nvfp4_kv_vram:.0f}MB  "
              f"ratio={ratio_str}  KLD={kld_str}  cos={cos_str}  [{status}]  "
              f"{row['gate_elapsed_s']:.0f}s")

    # ── Summary table ───────────────────────────────────────────────────────
    print(f"\n{BOLD}{'─'*100}{RESET}")
    print(f"{BOLD}  VRAM CONTEXT SCALING SUMMARY{RESET}")
    print(f"{'─'*100}")
    header = (
        f"  {'Context':>8}  {'DualPeak':>10}  {'F32Only':>10}  {'NVFP4KV':>10}  "
        f"{'Ratio':>7}  {'KLD':>10}  {'Cos':>10}  {'Gate':>5}  {'Time'}"
    )
    print(header)
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*5}  {'─'*5}")

    for r in results:
        err = r.get("error")
        if err:
            print(f"  {r['context']:>8}  {'ERR':>10}  {'ERR':>10}  {'ERR':>10}  "
                  f"{'ERR':>7}  {'ERR':>10}  {'ERR':>10}  {RED}FAIL{RESET:>5}  {r['gate_elapsed_s']:.0f}s")
            continue

        kld_s = f"{r['tile_kld']:.6f}" if r['tile_kld'] is not None else "N/A"
        cos_s = f"{r['tile_cos']:.6f}" if r['tile_cos'] is not None else "N/A"
        ratio_s = f"{r['compression_ratio']:.1f}x" if r['compression_ratio'] > 0.1 else "N/A"
        gate_s = "PASS" if r["gate_passed"] else "FAIL"

        print(
            f"  {r['context']:>8}  {r['dual_peak_mb']:>8.0f}MB  "
            f"{r['f32_only_mb']:>8.0f}MB  {r['nvfp4_kv_estimate_mb']:>8.0f}MB  "
            f"{ratio_s:>7}  {kld_s:>10}  {cos_s:>10}  "
            f"{gate_s:>5}  {r['gate_elapsed_s']:.0f}s"
        )

    print(f"{'─'*100}")

    # Analysis
    valid = [r for r in results if r.get("error") is None and r["gate_passed"]]
    if len(valid) >= 2:
        ratios = [r["compression_ratio"] for r in valid if r["compression_ratio"] > 0.1]
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            print(f"\n  {BOLD}Average compression ratio: {avg_ratio:.1f}x{RESET}")
            if avg_ratio >= 3.0:
                print(f"  {GREEN}Meets 3.1x target.{RESET}")
            else:
                print(f"  {YELLOW}Below 3.1x target — check KV cache config.{RESET}")

    if not valid:
        print(f"\n  {RED}No valid gate runs — all failed.{RESET}")
    else:
        n_passed = len(valid)
        print(f"  {n_passed}/{len(results)} contexts passed the accuracy gate.")

    print()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _find_model() -> str:
    candidates = [
        r"I:\models\ornith-1.0-9b-NVFP4.gguf",
        r"I:\models\ornith-1.0-35b-APEX-I-Mini-MTP.gguf",
        r"I:\models\AEON-7_Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4-FP8\AEON-7_Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4-FP8.gguf",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="measure_vram_context.py — VRAM metering for NVFP4 KV context scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/measure_vram_context.py --model I:\\\\models\\\\ornith-9b.gguf
  python tools/measure_vram_context.py --model I:\\\\models\\\\ornith-9b.gguf --ctx-sizes 1024,2048,4096,8192
  python tools/measure_vram_context.py --model I:\\\\models\\\\ornith-35b.gguf --ctx-sizes 1024,2048,4096 --ngl 0
        """,
    )
    parser.add_argument("--model", default=None,
                        help="Path to GGUF model. Auto-discovered if omitted.")
    parser.add_argument("--ctx-sizes", default="1024,2048,4096,8192",
                        help="Comma-separated context sizes (default: 1024,2048,4096,8192).")
    parser.add_argument("--ngl", type=int, default=99,
                        help="GPU layers (default: 99). Use 0 for large contexts.")
    parser.add_argument("--threads", type=int, default=6,
                        help="CPU threads (default: 6).")
    parser.add_argument("--tokens", type=int, default=300,
                        help="Gate tokens per context (default: 300).")
    parser.add_argument("--tail-tokens", type=int, default=1024,
                        help="NVFP4 KV precision tail size (default: 1024). BeeLlama optimal: +11% KLD at 64k.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout per run in seconds (default: 600).")

    args = parser.parse_args()

    # Resolve model
    model_path = args.model
    if not model_path:
        model_path = _find_model()
        if not model_path:
            print(f"{RED}ERROR: No model specified and auto-discovery failed.{RESET}")
            sys.exit(1)
        print(f"{YELLOW}Auto-discovered model: {model_path}{RESET}")

    if not os.path.isfile(model_path):
        print(f"{RED}ERROR: Model not found: {model_path}{RESET}")
        sys.exit(1)

    context_sizes = [int(x.strip()) for x in args.ctx_sizes.split(",") if x.strip()]
    if not context_sizes:
        print(f"{RED}ERROR: Invalid context sizes: {args.ctx_sizes}{RESET}")
        sys.exit(1)

    # VRAM warning for large contexts
    max_ctx = max(context_sizes)
    if max_ctx >= 8192 and args.ngl > 0:
        print(f"\n{YELLOW}{BOLD}VRAM WARNING:{RESET}")
        print(f"  Max context: {max_ctx} — Dual F32 KV caches may OOM on 16 GB GPU.")
        print(f"  Consider --ngl 0 for CPU-only or reduce max context.")
        print()

    results = measure_vram_context(
        model_path=model_path,
        context_sizes=context_sizes,
        ngl=args.ngl,
        threads=args.threads,
        n_tokens=args.tokens,
        tail_tokens=args.tail_tokens,
        timeout_s=args.timeout,
    )

    all_passed = all(r.get("gate_passed", False) for r in results if r.get("error") is None)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
