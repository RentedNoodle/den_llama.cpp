#!/usr/bin/env python3
"""
test_needle_retrieval.py — Needle-in-haystack retrieval test for NVFP4 KV.

Plants a secret fact deep in filler text, then queries the model at the end.
Tests BOTH F32 KV (oracle) and NVFP4 KV (candidate). Both must find the needle
for the "lossless" claim to hold.

Uses llama-cli subprocess — functional test, not logit-level. Measures
retrieval accuracy, not distribution divergence.

Usage:
  python tools/test_needle_retrieval.py --model I:\\models\\ornith-9b.gguf
  python tools/test_needle_retrieval.py --model I:\\models\\ornith-35b.gguf --context-size 32768
  python tools/test_needle_retrieval.py --model I:\\models\\ornith-9b.gguf --context-size 65536 --ngl 0

Exit: 0 = both F32 and NVFP4 pass, 1 = failure.
"""

import subprocess
import sys
import os
import re
import argparse
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# PATH DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BUILD_DIRS = [
    PROJECT_DIR / "build_ninja" / "bin",
    PROJECT_DIR / "build_bench" / "bin",
    PROJECT_DIR / "build" / "bin",
]

def find_llama_cli() -> Path:
    for d in BUILD_DIRS:
        candidate = d / "llama-cli.exe"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "llama-cli.exe not found. Build llama.cpp first."
    )

LLAMA_CLI = find_llama_cli()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NEEDLE_FACT = "The secret passphrase is XKCD-2026"
NEEDLE_MARKER = "XKCD-2026"
NEEDLE_POSITION = 100  # plant at position 100 in context

# Filler text chunks that tokenize to ~5-10 tokens each for Qwen tokenizer.
# Repeating a natural paragraph so generated tokens are diverse enough
# to form a real KV cache but not a special pattern.
FILLER_PARAGRAPH = (
    "The history of artificial intelligence spans decades of research and development. "
    "Early pioneers like Alan Turing posed fundamental questions about machine thinking. "
    "The Dartmouth Conference of 1956 is widely considered the birth of AI as a field. "
    "Since then, the field has seen multiple waves of optimism and funding, known as AI summers, "
    "followed by periods of reduced interest called AI winters. "
    "Modern deep learning, powered by large datasets and GPU computing, has transformed "
    "the landscape of what machines can accomplish. "
    "Neural networks with many layers can learn hierarchical representations of data. "
    "These representations have proven remarkably effective for vision, language, and reasoning tasks. "
)

QUERY_TEXT = "\n\nQuestion: What is the secret passphrase mentioned earlier in this document? Answer with just the passphrase.\n\n"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def estimate_filler_tokens(text: str) -> int:
    """Conservative estimate: ~1.3 chars per token for English text (subword tokenizer)."""
    return max(1, len(text) // 3)


def build_test_prompt(context_size: int) -> str:
    """
    Build a prompt with:
    - Filler text filling context_size - 110 tokens (accounting for needle + query)
    - Needle planted at position ~100
    - Query at the end
    """
    target_filler_tokens = max(0, context_size - 110)

    # Build filler by repeating the paragraph
    para_tokens = estimate_filler_tokens(FILLER_PARAGRAPH)
    repeats = max(1, (target_filler_tokens // para_tokens) + 1)
    filler = FILLER_PARAGRAPH * repeats

    # Build the prompt: filler + needle + more filler + query
    # Needle goes at position ~100. We'll insert it after a short prefix.
    prefix = filler[:300]  # ~100 tokens of prefix
    suffix = filler[300:]  # rest of filler
    prompt = prefix + "\n\n" + NEEDLE_FACT + "\n\n" + suffix + QUERY_TEXT

    return prompt


def run_llama_cli(
    model_path: str,
    prompt: str,
    n_tokens: int,
    ngl: int,
    context_size: int,
    nvfp4_kv: bool,
    threads: int = 6,
    timeout_s: int = 600,
) -> Tuple[str, int, float]:
    """
    Run llama-cli, return (output_text, exit_code, elapsed_seconds).
    nvfp4_kv=True enables DEN_NVFP4_KV_CACHE=1 + DEN_THRIFT_ATTENTION=1.
    """
    env = os.environ.copy()
    if nvfp4_kv:
        env["DEN_NVFP4_KV_CACHE"] = "1"
        env["DEN_THRIFT_ATTENTION"] = "1"
    else:
        env["DEN_NVFP4_KV_CACHE"] = "0"
        env.pop("DEN_THRIFT_ATTENTION", None)

    # Use temp file for output (avoids pipe deadlock with large output)
    fd, output_path = tempfile.mkstemp(suffix=".txt", prefix="needle_out_")
    os.close(fd)

    # Build prompt file (avoids quoting issues with large prompts)
    fd2, prompt_path = tempfile.mkstemp(suffix=".txt", prefix="needle_prompt_")
    with os.fdopen(fd2, "w", encoding="utf-8") as f:
        f.write(prompt)

    cmd = [
        str(LLAMA_CLI),
        "-m", model_path,
        "-f", prompt_path,
        "-n", str(n_tokens),
        "-ngl", str(ngl),
        "-t", str(threads),
        "-c", str(context_size),
        "--temp", "0",          # greedy for determinism
        "--top-k", "1",
        "-s", "42",             # fixed seed
        "--simple-io",
        "--no-display-prompt",
        "--no-perf",
        "-e",                   # eval mode, exit after generation
        "-o", output_path,
    ]

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            cwd=str(LLAMA_CLI.parent),
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        # Clean up temp files
        try:
            os.unlink(output_path)
            os.unlink(prompt_path)
        except OSError:
            pass
        return f"[TIMEOUT after {timeout_s}s]", -1, elapsed

    elapsed = time.perf_counter() - t0

    # Read output
    output_text = ""
    if os.path.isfile(output_path):
        try:
            output_text = Path(output_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            output_text = ""
        os.unlink(output_path)

    # Clean up prompt file
    try:
        os.unlink(prompt_path)
    except OSError:
        pass

    return output_text, result.returncode, elapsed


def check_needle_found(output: str) -> bool:
    """Check if the needle passphrase appears in the output."""
    # Case-insensitive search for the marker
    return NEEDLE_MARKER.lower() in output.lower()


def print_result(label: str, found: bool, elapsed: float, output_preview: str):
    status = f"{GREEN}PASS{RESET}" if found else f"{RED}FAIL{RESET}"
    preview = output_preview[:200].replace("\n", "\\n")
    print(f"  {label:<24} [{status}]  {elapsed:>6.1f}s  |  {preview}")


def test_needle_retrieval(
    model_path: str,
    context_size: int = 4096,
    ngl: int = 99,
    threads: int = 6,
    timeout_s: int = 600,
) -> dict:
    """
    Run needle retrieval test for BOTH F32 KV and NVFP4 KV.
    Returns dict with results.
    """
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  NEEDLE-IN-HAYSTACK RETRIEVAL TEST{RESET}")
    print(f"{'='*80}")
    print(f"  Model        : {Path(model_path).name}")
    print(f"  Context size : {context_size}")
    print(f"  Needle pos   : ~{NEEDLE_POSITION}")
    print(f"  Needle       : \"{NEEDLE_FACT}\"")
    print(f"  GPU layers   : {ngl}")
    print(f"  Threads      : {threads}")
    print(f"{'='*80}")

    # Warn for large contexts
    if context_size >= 32768:
        print(f"\n  {YELLOW}WARNING: {context_size} context may be slow. Timeout={timeout_s}s.{RESET}")
    if context_size >= 65536 and ngl > 0:
        print(f"  {YELLOW}WARNING: {context_size} context with GPU layers may OOM. Consider --ngl 0.{RESET}")

    # Build prompt
    print(f"\n  {CYAN}Building test prompt...{RESET}")
    prompt = build_test_prompt(context_size)
    prompt_chars = len(prompt)
    print(f"  Prompt chars  : {prompt_chars:,}")
    print(f"  Est. tokens   : ~{prompt_chars // 3:,} (target ~{context_size - 50})")

    # ── F32 KV (oracle) ────────────────────────────────────────────────────
    print(f"\n  {CYAN}[1/2] Running F32 KV oracle (NVFP4_KV_CACHE=0)...{RESET}")
    f32_output, f32_rc, f32_elapsed = run_llama_cli(
        model_path=model_path,
        prompt=prompt,
        n_tokens=32,  # enough for "XKCD-2026" response
        ngl=ngl,
        context_size=context_size,
        nvfp4_kv=False,
        threads=threads,
        timeout_s=timeout_s,
    )
    f32_found = check_needle_found(f32_output if f32_rc == 0 else "")

    # ── NVFP4 KV (candidate) ───────────────────────────────────────────────
    print(f"\n  {CYAN}[2/2] Running NVFP4 KV candidate (NVFP4_KV_CACHE=1)...{RESET}")
    nvfp4_output, nvfp4_rc, nvfp4_elapsed = run_llama_cli(
        model_path=model_path,
        prompt=prompt,
        n_tokens=32,
        ngl=ngl,
        context_size=context_size,
        nvfp4_kv=True,
        threads=threads,
        timeout_s=timeout_s,
    )
    nvfp4_found = check_needle_found(nvfp4_output if nvfp4_rc == 0 else "")

    # ── Results ─────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  RESULTS{RESET}")
    print(f"{'='*80}")

    print_result("F32 KV (oracle)", f32_found, f32_elapsed,
                 f32_output if f32_rc == 0 else f"[exit={f32_rc}]")
    print_result("NVFP4 KV (candidate)", nvfp4_found, nvfp4_elapsed,
                 nvfp4_output if nvfp4_rc == 0 else f"[exit={nvfp4_rc}]")

    print(f"  {'─'*70}")

    both_pass = f32_found and nvfp4_found
    overall = f"{GREEN}PASS{RESET}" if both_pass else f"{RED}FAIL{RESET}"
    print(f"  Overall: [{overall}]  F32={'PASS' if f32_found else 'FAIL'}  "
          f"NVFP4={'PASS' if nvfp4_found else 'FAIL'}")

    if not f32_found:
        print(f"  {RED}F32 KV oracle failed needle retrieval — test is invalid.{RESET}")
        print(f"  {RED}  Possible causes: context too large for model, prompt broken,{RESET}")
        print(f"  {RED}  or model cannot follow the instruction. Fix F32 first.{RESET}")
    elif not nvfp4_found:
        print(f"  {RED}NVFP4 KV lost the needle while F32 kept it.{RESET}")
        print(f"  {RED}  NVFP4 KV quantization is LOSSLESS — causing retrieval failure.{RESET}")

    print(f"{'='*80}\n")

    result = {
        "test": "NEEDLE_RETRIEVAL",
        "context_size": context_size,
        "needle": NEEDLE_FACT,
        "needle_position": NEEDLE_POSITION,
        "f32": {
            "found": f32_found,
            "exit_code": f32_rc,
            "elapsed_s": f32_elapsed,
            "output_preview": f32_output[:200] if f32_rc == 0 else f"[exit={f32_rc}]",
        },
        "nvfp4": {
            "found": nvfp4_found,
            "exit_code": nvfp4_rc,
            "elapsed_s": nvfp4_elapsed,
            "output_preview": nvfp4_output[:200] if nvfp4_rc == 0 else f"[exit={nvfp4_rc}]",
        },
        "passed": both_pass,
    }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DISCOVERY
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="test_needle_retrieval.py — Needle-in-haystack for NVFP4 KV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/test_needle_retrieval.py --model I:\\models\\ornith-9b.gguf
  python tools/test_needle_retrieval.py --model I:\\models\\ornith-9b.gguf --context-size 32768
  python tools/test_needle_retrieval.py --model I:\\models\\ornith-35b.gguf --context-size 65536 --ngl 0
  python tools/test_needle_retrieval.py --model I:\\models\\ornith-9b.gguf --context-size 1024,4096,16384
        """,
    )
    parser.add_argument("--model", default=None,
                        help="Path to GGUF model. Auto-discovered if omitted.")
    parser.add_argument("--context-size", default="4096",
                        help="Context size(s) to test. Comma-separated for sweep "
                             "(e.g. 1024,4096,16384,32768,65536). Single value for one test.")
    parser.add_argument("--ngl", type=int, default=99,
                        help="GPU layers (default: 99). Use 0 for large contexts on CPU.")
    parser.add_argument("--threads", type=int, default=6,
                        help="CPU threads (default: 6).")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout per run in seconds (default: 600).")

    args = parser.parse_args()

    # Resolve model
    model_path = args.model
    if not model_path:
        model_path = _find_model()
        if not model_path:
            print(f"{RED}ERROR: No model specified and auto-discovery failed.{RESET}")
            print("  Specify --model or place a model at:")
            for c in [
                r"I:\models\ornith-1.0-9b-NVFP4.gguf",
                r"I:\models\ornith-1.0-35b-APEX-I-Mini-MTP.gguf",
            ]:
                print(f"    {c}")
            sys.exit(1)
        print(f"{YELLOW}Auto-discovered model: {model_path}{RESET}")

    if not os.path.isfile(model_path):
        print(f"{RED}ERROR: Model not found: {model_path}{RESET}")
        sys.exit(1)

    # Parse context sizes
    context_sizes = [int(x.strip()) for x in args.context_size.split(",") if x.strip()]
    if not context_sizes:
        print(f"{RED}ERROR: Invalid context size: {args.context_size}{RESET}")
        sys.exit(1)

    # Run tests
    all_passed = True
    for ctx in context_sizes:
        result = test_needle_retrieval(
            model_path=model_path,
            context_size=ctx,
            ngl=args.ngl,
            threads=args.threads,
            timeout_s=args.timeout,
        )
        if not result["passed"]:
            all_passed = False

    # ── Sweep summary (if multiple context sizes) ─────────────────────────
    if len(context_sizes) > 1:
        print(f"\n{BOLD}SWEEP SUMMARY:{RESET}")
        print(f"  Context sizes tested: {context_sizes}")
        print(f"  Overall: {f'{GREEN}ALL PASS{RESET}' if all_passed else f'{RED}FAILURES DETECTED{RESET}'}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
