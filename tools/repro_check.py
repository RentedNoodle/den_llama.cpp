#!/usr/bin/env python3
"""
repro_check.py — Quick-start reproducibility check for den_llama.cpp.

Prints system state (CUDA, driver, GPU, commit, Python) then runs a
10-token gate quick-check (<30s). Outputs JSON for automation.

Purpose: any developer (or future you) can run this to verify they can
reproduce results from the same commit on the same hardware.

Usage:
  python tools/repro_check.py
  python tools/repro_check.py --model I:\\models\\ornith-9b.gguf
  python tools/repro_check.py --json-only   # machine-readable output
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
GATE_SCRIPT = SCRIPT_DIR / "gate_accuracy_kv.py"

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM STATE PROBES
# ─────────────────────────────────────────────────────────────────────────────


def _run(cmd: list[str], timeout: int = 30) -> str:
    """Run command, return stdout stripped, empty string on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def probe_system() -> dict:
    """Collect all system metadata into a dict."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": os.environ.get("COMPUTERNAME", "unknown"),
        "python_version": sys.version,
        "python_executable": sys.executable,
    }

    # ── Git commit ────────────────────────────────────────────────────────
    commit = _run(["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"], timeout=10)
    info["commit_hash"] = commit if commit else "unknown"

    # Also get branch
    branch = _run(["git", "-C", str(PROJECT_DIR), "rev-parse", "--abbrev-ref", "HEAD"], timeout=10)
    info["branch"] = branch if branch else "unknown"

    # Dirty check
    status = _run(["git", "-C", str(PROJECT_DIR), "status", "--porcelain"], timeout=10)
    info["dirty"] = bool(status)

    # ── nvcc version ──────────────────────────────────────────────────────
    for nvcc_cmd in [
        ["nvcc", "--version"],
        ["wsl", "bash", "-c", "nvcc --version 2>/dev/null"],
    ]:
        out = _run(nvcc_cmd, timeout=15)
        if out:
            info["nvcc_version_raw"] = out
            for line in out.splitlines():
                if "release" in line.lower():
                    info["cuda_version"] = line.split("release")[-1].split(",")[0].strip()
                    break
            break
    if "cuda_version" not in info:
        info["cuda_version"] = "unknown"

    # ── nvidia-smi ────────────────────────────────────────────────────────
    smi = _run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"], timeout=15)
    if smi:
        parts = smi.split(",")
        if len(parts) >= 3:
            info["driver_version"] = parts[0].strip()
            info["gpu_name"] = parts[1].strip()
            info["gpu_memory"] = parts[2].strip()
    if "driver_version" not in info:
        info["driver_version"] = "unknown"
        info["gpu_name"] = "unknown"
        info["gpu_memory"] = "unknown"

    # ── CUDA_PATH / toolkit detection ─────────────────────────────────────
    cuda_paths = []
    for envkey in ["CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT"]:
        v = os.environ.get(envkey, "")
        if v:
            cuda_paths.append({envkey: v})
    info["cuda_env"] = cuda_paths if cuda_paths else "none"

    # ── llama.dll detection ───────────────────────────────────────────────
    build_dirs = [
        PROJECT_DIR / "build_ninja" / "bin" / "llama.dll",
        PROJECT_DIR / "build_bench" / "bin" / "llama.dll",
        PROJECT_DIR / "build" / "bin" / "llama.dll",
    ]
    for d in build_dirs:
        if d.is_file():
            info["llama_dll"] = str(d)
            # Get file size + modification time
            try:
                stat = d.stat()
                info["llama_dll_size"] = stat.st_size
                info["llama_dll_mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            except Exception:
                pass
            break
    else:
        info["llama_dll"] = "NOT FOUND — build first"

    # ── Model auto-discovery ──────────────────────────────────────────────
    candidates = [
        r"I:\models\ornith-1.0-35b-APEX-I-Mini-MTP.gguf",
        r"I:\models\AEON-7_Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4-FP8\AEON-7_Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4-FP8.gguf",
        r"I:\models\ornith-1.0-9b-NVFP4.gguf",
    ]
    found = [c for c in candidates if os.path.isfile(c)]
    info["available_models"] = found if found else ["none found"]

    # ── Environment flags ─────────────────────────────────────────────────
    den_env = {k: v for k, v in os.environ.items() if k.startswith("DEN_") or k.startswith("GGML_")}
    info["den_env"] = den_env if den_env else "none"

    return info


# ─────────────────────────────────────────────────────────────────────────────
# QUICK GATE CHECK (10 tokens)
# ─────────────────────────────────────────────────────────────────────────────


def run_quick_gate(model_path: str = None) -> dict:
    """Run gate_accuracy_kv.py at 10 tokens for fast sanity check."""
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--tokens", "10",
        "--ngl", "99",
        "--threads", "6",
    ]
    if model_path:
        cmd.extend(["--model", model_path])

    print(f"  Running quick gate: {' '.join(cmd)}")

    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "timeout (120s)", "elapsed_s": 120}
    except Exception as e:
        return {"passed": False, "error": str(e), "elapsed_s": time.perf_counter() - t0}

    # Parse pass/fail from exit code + stdout
    passed = r.returncode == 0

    # Try to find key metrics in output
    result = {
        "passed": passed,
        "returncode": r.returncode,
        "elapsed_s": elapsed,
        "error": None,
    }

    for line in r.stdout.splitlines():
        if "Mean KLD" in line:
            try:
                result["mean_kld"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Median cosine" in line:
            try:
                result["median_cos"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    if not passed:
        result["error"] = r.stderr[-500:] if r.stderr else f"exit code {r.returncode}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_system_state(info: dict) -> None:
    """Pretty-print system state."""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  SYSTEM STATE{RESET}")
    print(f"{'='*70}")
    print(f"  Host             : {info['hostname']}")
    print(f"  Python           : {info['python_version'].split()[0]}")
    print(f"  Commit           : {info['commit_hash'][:12] if len(info.get('commit_hash', '')) > 12 else info.get('commit_hash', 'unknown')}")
    print(f"  Branch           : {info['branch']}")
    print(f"  Dirty            : {info['dirty']}")
    print(f"  CUDA             : {info.get('cuda_version', 'unknown')}")
    print(f"  Driver           : {info.get('driver_version', 'unknown')}")
    print(f"  GPU              : {info.get('gpu_name', 'unknown')}")
    print(f"  VRAM             : {info.get('gpu_memory', 'unknown')}")
    print(f"  llama.dll        : {info.get('llama_dll', 'NOT FOUND')}")
    print(f"  Models available : {len(info.get('available_models', []))}")
    for m in info.get("available_models", [])[:3]:
        print(f"    {m}")
    if info.get("den_env") and info["den_env"] != "none":
        print(f"  DEN_ env vars    :")
        for k, v in info["den_env"].items():
            print(f"    {k}={v}")
    print(f"{'='*70}")


def print_quick_gate_result(result: dict) -> None:
    """Pretty-print quick gate result."""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  QUICK GATE CHECK (10 tokens){RESET}")
    print(f"{'='*70}")
    status = f"{GREEN}PASS{RESET}" if result["passed"] else f"{RED}FAIL{RESET}"
    print(f"  Status     : [{status}]")
    if result.get("mean_kld") is not None:
        print(f"  Mean KLD   : {result['mean_kld']:.8f}")
    if result.get("median_cos") is not None:
        print(f"  Median cos : {result['median_cos']:.8f}")
    print(f"  Elapsed    : {result['elapsed_s']:.1f}s")
    if result.get("error"):
        print(f"  {RED}Error     : {result['error'][:200]}{RESET}")
    print(f"{'='*70}")

    if not result["passed"]:
        print(f"\n  {RED}QUICK GATE FAILED — build may be broken or model missing.{RESET}")
        print(f"  {RED}Run full gate: python tools/gate_accuracy_kv.py --tokens 1024{RESET}")
    else:
        print(f"\n  {GREEN}Quick gate passed. System can reproduce results.{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="repro_check.py — Quick-start reproducibility check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/repro_check.py
  python tools/repro_check.py --model I:\\\\models\\\\ornith-9b.gguf
  python tools/repro_check.py --json-only
  python tools/repro_check.py --no-gate        # just system state, skip gate
        """,
    )
    parser.add_argument("--model", default=None,
                        help="Model path (auto-detected if omitted).")
    parser.add_argument("--json-only", action="store_true",
                        help="Output JSON only (machine-readable).")
    parser.add_argument("--no-gate", action="store_true",
                        help="Skip gate check, print system state only.")
    parser.add_argument("--output", default=None,
                        help="Write JSON report to file.")

    args = parser.parse_args()

    # Probe system
    info = probe_system()

    if args.no_gate:
        gate_result = None
    else:
        print(f"{CYAN}Running 10-token quick gate...{RESET}")
        gate_result = run_quick_gate(args.model)
        info["quick_gate"] = gate_result

    # Build report
    report = {
        "system_state": info,
    }

    if args.json_only:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_system_state(info)
        if gate_result:
            print_quick_gate_result(gate_result)

        # Summary
        if gate_result and gate_result["passed"]:
            print(f"\n  {GREEN}{BOLD}READY: System can reproduce NVFP4 KV results.{RESET}")
        elif gate_result:
            print(f"\n  {RED}{BOLD}NOT READY: Quick gate failed.{RESET}")
        else:
            print(f"\n  {YELLOW}System state captured. Run without --no-gate for full check.{RESET}")

    # Write to file if requested
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  Report written: {args.output}")

    sys.exit(0 if (gate_result is None or gate_result["passed"]) else 1)


if __name__ == "__main__":
    main()
