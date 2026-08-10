#!/usr/bin/env python3
"""
regression_baseline.py — NVFP4 KV accuracy regression database.

Runs gate_accuracy_kv.py at 1024 tokens (~2 min), captures key metrics,
compares against stored baseline. Flags regressions >1% change in KLD or cos.

Usage:
  python tools/regression_baseline.py --baseline-store   # record new baseline
  python tools/regression_baseline.py --baseline-check    # verify against existing
  python tools/regression_baseline.py --baseline-store --model I:\\models\\ornith-9b.gguf
"""

import sys
import os
import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BASELINE_PATH = SCRIPT_DIR / "regression_baseline.json"
GATE_SCRIPT = SCRIPT_DIR / "gate_accuracy_kv.py"

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROBES
# ─────────────────────────────────────────────────────────────────────────────


def _run(cmd: list[str], timeout: int = 30) -> str:
    """Run command, return stdout stripped, empty string on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def get_commit_hash() -> str:
    """Return HEAD commit hash from repo root."""
    out = _run(["git", "rev-parse", "HEAD"], timeout=10)
    if out:
        return out
    # Try from project root
    out = _run(["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"], timeout=10)
    return out if out else "unknown"


def get_cuda_version() -> str:
    """Return nvcc version string, e.g. '13.3'."""
    # Try nvcc from PATH
    out = _run(["nvcc", "--version"], timeout=10)
    if out:
        for line in out.splitlines():
            if "release" in line:
                # "Cuda compilation tools, release 13.3, V13.3.33"
                parts = line.split("release")[-1].split(",")[0].strip()
                return parts
    # Try WSL path
    out = _run(["wsl", "bash", "-c", "nvcc --version 2>/dev/null"], timeout=15)
    if out:
        for line in out.splitlines():
            if "release" in line:
                return line.split("release")[-1].split(",")[0].strip()
    # Try pip nvcc
    pip_nvcc = Path(os.environ.get("CUDA_PATH", "")) / "bin" / "nvcc.exe"
    if pip_nvcc.is_file():
        out = _run([str(pip_nvcc), "--version"], timeout=10)
        if out:
            for line in out.splitlines():
                if "release" in line:
                    return line.split("release")[-1].split(",")[0].strip()
    return "unknown"


def get_driver_version() -> str:
    """Return nvidia-smi driver version, e.g. '610.47'."""
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], timeout=15)
    if out:
        return out.splitlines()[0].strip()
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# GATE RUNNER
# ─────────────────────────────────────────────────────────────────────────────


def run_gate_1024(model_path: str = None, ngl: int = 99, threads: int = 6) -> dict:
    """Run gate_accuracy_kv.py at 1024 tokens, parse JSON metrics from stdout."""
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--tokens", "1024",
        "--ngl", str(ngl),
        "--threads", str(threads),
    ]
    if model_path:
        cmd.extend(["--model", model_path])

    print(f"  Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "elapsed_s": 600}
    except Exception as e:
        return {"error": str(e), "elapsed_s": time.perf_counter() - t0}

    # Parse stdout for the metrics we need (gate_accuracy_kv.py prints them)
    stdout = r.stdout
    stderr = r.stderr

    # Extract key values from the printed table format
    metrics = _parse_gate_output(stdout)
    metrics["elapsed_s"] = elapsed
    metrics["returncode"] = r.returncode
    metrics["stderr_tail"] = stderr[-500:] if stderr else ""

    return metrics


def _parse_gate_output(text: str) -> dict:
    """Parse gate_accuracy_kv.py output for key metrics."""
    result = {
        "mean_kld": None,
        "mean_cos": None,
        "top1_rate": None,
        "tile_positions": 0,
        "tail_positions": 0,
        "median_kld": None,
        "passed": False,
    }

    for line in text.splitlines():
        line = line.strip()
        # "Mean KLD            : 0.000123"
        # "Mean KLD            : N/A"
        if "Mean KLD" in line:
            try:
                parts = line.split(":")[-1].strip()
                if parts.lower() != "n/a":
                    result["mean_kld"] = float(parts)
            except ValueError:
                pass
        # "Median cosine       : 0.999876"
        elif "Median cosine" in line:
            try:
                parts = line.split(":")[-1].strip()
                if parts.lower() != "n/a":
                    result["mean_cos"] = float(parts)
            except ValueError:
                pass
        # "mean cos  : 0.999987" (from TILE region block)
        elif "tile_cos_mean" in line or ("mean cos" in line.lower() and "tile" in line.lower()):
            pass  # handled above
        # "Top-1" / "top1_rate" in the result table
        elif "TILE region" in line:
            # Next lines contain tile data
            pass
        # "tile_positions" in regions dict or "TILE region (N positions)"
        elif "TILE region" in line and "positions" in line:
            try:
                # "TILE region (744 positions): ..."
                num = line.split("(")[1].split(" positions")[0].strip()
                result["tile_positions"] = int(num)
            except (ValueError, IndexError):
                pass
        # "Positions evaluated : 1024 (tail=256, tile=768)"
        elif "Positions evaluated" in line:
            try:
                tail_part = line.split("tail=")[1].split(",")[0].strip()
                tile_part = line.split("tile=")[1].split(")")[0].strip()
                result["tail_positions"] = int(tail_part)
                result["tile_positions"] = int(tile_part)
            except (ValueError, IndexError):
                pass
        # "mean KLD  : 0.000000" — from region breakdown (TAIL or TILE)
        elif "mean KLD" in line.lower():
            try:
                parts = line.split(":")[-1].strip()
                if parts.lower() != "n/a":
                    result["mean_kld"] = float(parts)
            except ValueError:
                pass
        # "mean cos  : 0.999999"
        elif "mean cos" in line.lower() and "cosine" not in line.lower():
            try:
                parts = line.split(":")[-1].strip()
                if parts.lower() != "n/a":
                    result["mean_cos"] = float(parts)
            except ValueError:
                pass
        # "PASS" or "FAIL" overall
        elif "Overall:" in line and "PASS" in line:
            result["passed"] = True
        # "Median KLD" from gate results table
        elif "median_kld " in line.lower() or "median KLD" in line:
            pass  # we prefer mean_kld from diagnostics

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE STORE / CHECK
# ─────────────────────────────────────────────────────────────────────────────


def gather_run_data(model_path: str = None, ngl: int = 99, threads: int = 6) -> dict:
    """Run gate + gather all metadata into one record."""
    print(f"\n  Gathering system metadata...")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_hash": get_commit_hash(),
        "cuda_version": get_cuda_version(),
        "driver_version": get_driver_version(),
        "python_version": sys.version.split()[0],
        "gate_tokens": 1024,
        "ngl": ngl,
        "threads": threads,
    }

    # Determine model name
    if model_path:
        record["model_name"] = Path(model_path).name
    else:
        record["model_name"] = "auto-detected"

    # Run the gate
    print(f"  Running accuracy gate at 1024 tokens...")
    metrics = run_gate_1024(model_path=model_path, ngl=ngl, threads=threads)
    record["metrics"] = metrics

    return record


def store_baseline(record: dict) -> None:
    """Append record to baseline JSON array."""
    records = []
    if BASELINE_PATH.exists():
        try:
            records = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                records = []
        except (json.JSONDecodeError, ValueError):
            records = []

    # Don't store if error
    if record.get("metrics", {}).get("error"):
        print(f"\n  ERROR: gate failed, not storing baseline.")
        print(f"    {record['metrics']['error']}")
        return

    records.append(record)
    BASELINE_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Baseline stored: {BASELINE_PATH}")
    print(f"  Records in database: {len(records)}")


def check_baseline(record: dict) -> int:
    """Compare current run against latest baseline. Return 0 on pass, 1 on regression."""
    if not BASELINE_PATH.exists():
        print(f"\n  {RED}No baseline found. Run --baseline-store first.{RESET}")
        return 1

    try:
        records = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if not records:
            print(f"\n  {RED}Baseline file exists but is empty.{RESET}")
            return 1
    except (json.JSONDecodeError, ValueError):
        print(f"\n  {RED}Baseline file is corrupt.{RESET}")
        return 1

    baseline = records[-1]  # latest stored baseline
    baseline_metrics = baseline.get("metrics", {})
    current_metrics = record.get("metrics", {})

    if current_metrics.get("error"):
        print(f"\n  {RED}CURRENT RUN FAILED: {current_metrics['error']}{RESET}")
        return 1

    if baseline_metrics.get("error"):
        print(f"\n  {RED}BASELINE ITSELF HAD ERROR: {baseline_metrics['error']}{RESET}")
        return 1

    # ── Compare metrics ──────────────────────────────────────────────────
    regressions = []

    for metric_key, metric_label, tolerance_pct in [
        ("mean_kld", "mean_KLD", 1.0),
        ("mean_cos", "mean_cos", 1.0),
        ("top1_rate", "top1_rate", 1.0),
    ]:
        baseline_val = baseline_metrics.get(metric_key)
        current_val = current_metrics.get(metric_key)

        if baseline_val is None or current_val is None:
            continue

        if baseline_val == 0.0:
            # For KLD, any increase from 0 is a regression
            if current_val > 0.0:
                pct_change = float("inf") if current_val > 0 else 0.0
                regressions.append({
                    "metric": metric_label,
                    "baseline": baseline_val,
                    "current": current_val,
                    "pct_change": float("+inf") if current_val > 0 else 0.0,
                })
        else:
            pct_change = abs((current_val - baseline_val) / baseline_val) * 100.0
            if pct_change > tolerance_pct:
                regressions.append({
                    "metric": metric_label,
                    "baseline": baseline_val,
                    "current": current_val,
                    "pct_change": pct_change,
                })

    # ── Print comparison ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  REGRESSION CHECK")
    print(f"{'='*70}")
    print(f"  Baseline: {baseline.get('timestamp', 'unknown')}")
    print(f"  Commit  : {baseline.get('commit_hash', 'unknown')[:8]}")
    print(f"  Model   : {baseline.get('model_name', 'unknown')}")
    print(f"  Current : {record.get('timestamp', 'unknown')}")
    print(f"  Commit  : {record.get('commit_hash', 'unknown')[:8]}")
    print(f"  Model   : {record.get('model_name', 'unknown')}")
    print(f"{'='*70}")
    print(f"  {'Metric':<16} {'Baseline':<14} {'Current':<14} {'Change':>10} {'Status':>10}")
    print(f"  {'-'*16} {'-'*14} {'-'*14} {'-'*10} {'-'*10}")

    for metric_key, metric_label, _ in [
        ("mean_kld", "mean_KLD", 1.0),
        ("mean_cos", "mean_cos", 1.0),
        ("top1_rate", "top1_rate", 1.0),
    ]:
        bv = baseline_metrics.get(metric_key)
        cv = current_metrics.get(metric_key)
        bv_s = f"{bv:.8f}" if bv is not None else "N/A"
        cv_s = f"{cv:.8f}" if cv is not None else "N/A"

        if bv is not None and cv is not None:
            if bv == 0.0:
                pct = "+inf" if cv > 0 else "0.00%"
            else:
                pct = f"{(cv - bv) / bv * 100:+.2f}%"
        else:
            pct = "N/A"

        # Determine if this is a regression
        is_regression = any(r["metric"] == metric_label for r in regressions)
        status = f"{RED}REGRESSION{RESET}" if is_regression else f"{GREEN}OK{RESET}"

        print(f"  {metric_label:<16} {bv_s:<14} {cv_s:<14} {pct:>10}  {status:>10}")

    print(f"  {'-'*16} {'-'*14} {'-'*14} {'-'*10} {'-'*10}")

    if regressions:
        print(f"\n  {RED}{BOLD}REGRESSION DETECTED:{RESET}")
        for r in regressions:
            print(f"    {r['metric']}: {r['baseline']:.8f} -> {r['current']:.8f} "
                  f"({r['pct_change']:+.2f}%)")
        return 1
    else:
        print(f"\n  {GREEN}{BOLD}PASS: within baseline tolerance{RESET}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(
        description="regression_baseline.py — NVFP4 KV accuracy regression database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/regression_baseline.py --baseline-store
  python tools/regression_baseline.py --baseline-check
  python tools/regression_baseline.py --baseline-store --model I:\\\\models\\\\ornith-9b.gguf
        """,
    )
    parser.add_argument("--baseline-store", action="store_true",
                        help="Run gate and save as new baseline entry.")
    parser.add_argument("--baseline-check", action="store_true",
                        help="Run gate and compare against stored baseline.")
    parser.add_argument("--model", default=None,
                        help="Model path (auto-detected if omitted).")
    parser.add_argument("--ngl", type=int, default=99,
                        help="GPU layers (default: 99).")
    parser.add_argument("--threads", type=int, default=6,
                        help="CPU threads (default: 6).")

    args = parser.parse_args()

    if not args.baseline_store and not args.baseline_check:
        print(f"{YELLOW}No action specified. Use --baseline-store or --baseline-check.{RESET}")
        print(f"  --baseline-store : run gate and save as new baseline")
        print(f"  --baseline-check : run gate and compare against stored baseline")
        sys.exit(0)

    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  REGRESSION BASELINE — NVFP4 KV Accuracy{RESET}")
    print(f"{'='*70}")

    record = gather_run_data(model_path=args.model, ngl=args.ngl, threads=args.threads)

    exit_code = 0

    if args.baseline_store:
        store_baseline(record)
        # Also check against previous if one exists
        if BASELINE_PATH.exists():
            records = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            if len(records) > 1:
                print(f"\n  {CYAN}Comparing against previous baseline...{RESET}")
                check_baseline(record)

    if args.baseline_check:
        exit_code = check_baseline(record)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
