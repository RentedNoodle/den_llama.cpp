#!/usr/bin/env python3
"""
test_cross_prompt_robustness.py — NVFP4 KV accuracy across diverse prompt types.

Different prompts produce different KV cache magnitude distributions, which
stress-tests the NVFP4 LUT-based quantization differently:
- Code: structured tokens with repetitive patterns
- Math: numeric tokens with low entropy
- Roleplay: natural language with high variance
- Adversarial: repeated punctuation/symbols (extreme magnitudes)
- Long-form: sustained narrative (accumulated attention)

Runs gate_accuracy_kv.py with each prompt type and reports per-prompt metrics
to identify if NVFP4 KV accuracy is prompt-dependent.

Usage:
  python tools/test_cross_prompt_robustness.py --model I:\\models\\ornith-35b-NVFP4.gguf
  python tools/test_cross_prompt_robustness.py --model I:\\models\\ornith-35b-NVFP4.gguf --tokens 200 --csv robustness.csv
  python tools/test_cross_prompt_robustness.py --model I:\\models\\ornith-9b-NVFP4.gguf --ngl 0

Exit: 0 = all prompts pass, 1 = one or more prompts fail.
"""

import sys
import os
import argparse
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_accuracy_kv import (
    test_accuracy_kv,
    NVFP4_KV_TAIL_TOKENS,
    GREEN, RED, YELLOW, CYAN, BOLD, RESET,
    _find_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT CATALOG — Diverse prompt types stress-test KV quantization LUT
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = {
    "code": (
        "def fibonacci(n):\n"
        "    \"\"\"Return the nth Fibonacci number using dynamic programming.\"\"\"\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    prev, curr = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        prev, curr = curr, prev + curr\n"
        "    return curr\n"
        "\n"
        "# The fibonacci sequence has many interesting properties.\n"
        "# For large n, the ratio of consecutive terms approaches the golden ratio.\n"
        "# This function runs in O(n) time and O(1) space."
    ),
    "math": (
        "Solve: 3x + 5 = 20\n"
        "Step 1: Subtract 5 from both sides: 3x = 15\n"
        "Step 2: Divide both sides by 3: x = 5\n"
        "Verify: 3(5) + 5 = 15 + 5 = 20. Correct.\n"
        "\n"
        "Now consider the quadratic equation: x^2 - 5x + 6 = 0\n"
        "Factoring: (x - 2)(x - 3) = 0\n"
        "Solutions: x = 2 or x = 3\n"
        "The discriminant is: b^2 - 4ac = 25 - 24 = 1"
    ),
    "roleplay": (
        "You are a pirate captain sailing the Caribbean Sea in 1720. "
        "Your ship, the Crimson Kraken, has just spotted a Spanish galleon "
        "on the horizon. The wind is at your back, the crew is eager, and "
        "the treasure holds of that galleon are rumored to contain gold "
        "doubloons from the New World. You turn to your first mate and say, "
        "\"Ready the cannons, but hold fire until we're alongside. I want "
        "that ship intact — and her captain alive to negotiate.\""
    ),
    "adversarial": (
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        "????????????????????????????????????????????????????????????????????"
        "????????????????????????????????????????????????????????????????????"
        "####################################################################"
        "####################################################################"
        "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        "...................................................................."
        "...................................................................."
    ),
    "longform": (
        "The Industrial Revolution, which began in Britain in the late 18th "
        "century, marked a fundamental transformation in human society. Prior "
        "to this period, most people lived in rural areas and worked in "
        "agriculture. The invention of the steam engine by James Watt, "
        "combined with advances in iron production and textile manufacturing, "
        "enabled the creation of factories that could produce goods at "
        "unprecedented scale. This shift from agrarian to industrial economies "
        "had profound social consequences: urbanization accelerated as workers "
        "moved to cities, new social classes emerged including the industrial "
        "bourgeoisie and the urban proletariat, and labor movements began "
        "organizing for better working conditions. The revolution also drove "
        "imperial expansion as European powers sought raw materials and new "
        "markets for their manufactured goods. By the mid-19th century, the "
        "Industrial Revolution had spread to continental Europe and North "
        "America, setting the stage for the modern global economy. Railways "
        "connected distant regions, steamships crossed oceans in record time, "
        "and the telegraph enabled near-instantaneous communication across "
        "continents. Each of these innovations reinforced the others, creating "
        "a feedback loop of technological progress that continues to this day."
    ),
}

# Expected KV characteristics for each prompt type (for diagnostics)
PROMPT_CHARACTERISTICS = {
    "code":        "structured tokens, repetitive patterns, moderate variance",
    "math":        "numeric tokens, low entropy, small token set",
    "roleplay":    "natural language, high variance, wide token distribution",
    "adversarial": "extreme repetition, punctuation saturation, outlier magnitudes",
    "longform":    "sustained narrative, accumulated attention, high token diversity",
}


def extract_prompt_metrics(result: dict, prompt_type: str, elapsed_s: float) -> dict:
    """Extract key metrics from gate_accuracy_kv result for cross-prompt comparison."""
    regions = result.get("regions", {})
    metrics = result.get("metrics", {})
    extra = result.get("extra", {})

    tile_kld = regions.get("tile_kld_mean")
    tile_cos = regions.get("tile_cos_mean")
    tail_kld = regions.get("tail_kld_mean")
    tail_cos = regions.get("tail_cos_mean")

    return {
        "prompt_type": prompt_type,
        "characteristic": PROMPT_CHARACTERISTICS.get(prompt_type, "unknown"),
        "n_prompt": result.get("n_prompt_tokens", 0),
        "n_tokens": result.get("n_tokens", 0),
        "tail_pos": regions.get("tail_positions", 0),
        "tile_pos": regions.get("tile_positions", 0),
        "tile_kld": tile_kld,
        "tile_cos": tile_cos,
        "tail_kld": tail_kld,
        "tail_cos": tail_cos,
        "median_kld": metrics.get("median_kld", {}).get("value"),
        "p99_kld": extra.get("p99_kld"),
        "max_kld": extra.get("max_kld"),
        "mean_cos": metrics.get("mean_cos", {}).get("value"),
        "min_cos": metrics.get("min_cos", {}).get("value"),
        "top1_rate": metrics.get("top1_rate", {}).get("value"),
        "worst_step": extra.get("worst_step"),
        "passed": result.get("passed", False),
        "error": result.get("error"),
        "elapsed_s": elapsed_s,
    }


def print_cross_prompt_report(results: list[dict]):
    """Print cross-prompt robustness comparison table."""
    print(f"\n{BOLD}{'='*110}{RESET}")
    print(f"{BOLD}  CROSS-PROMPT ROBUSTNESS — NVFP4 KV Accuracy vs F32 Oracle{RESET}")
    print(f"{'='*110}")

    # Sort by prompt_type for consistent display
    results_sorted = sorted(results, key=lambda r: r["prompt_type"])

    # Table header
    header = (
        f"  {'Prompt':>14}  {'tile_KLD':>10}  {'tile_cos':>10}  "
        f"{'med_KLD':>10}  {'max_KLD':>10}  {'mean_cos':>10}  "
        f"{'top1%':>7}  {'tile_pos':>9}  {'verdict':>10}"
    )
    print(header)
    print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*9}  {'-'*10}")

    all_passed = True
    worst_kld = -1.0
    worst_prompt = ""
    worst_cos = 1.0
    worst_cos_prompt = ""

    for r in results_sorted:
        err = r.get("error")
        if err:
            print(f"  {r['prompt_type']:>14}  {'ERR':>10}  {'ERR':>10}  {'ERR':>10}  "
                  f"{'ERR':>10}  {'ERR':>10}  {'—':>7}  {'—':>9}  {RED}ERROR{RESET}")
            print(f"    {RED}{err[:120]}{RESET}")
            all_passed = False
            continue

        tile_kld_s = f"{r['tile_kld']:.6f}" if r['tile_kld'] is not None else "N/A"
        tile_cos_s = f"{r['tile_cos']:.6f}" if r['tile_cos'] is not None else "N/A"
        med_kld_s = f"{r['median_kld']:.6f}" if r['median_kld'] is not None else "N/A"
        max_kld_s = f"{r['max_kld']:.6f}" if r['max_kld'] is not None else "N/A"
        mean_cos_s = f"{r['mean_cos']:.6f}" if r['mean_cos'] is not None else "N/A"
        top1_s = f"{r['top1_rate']*100:.1f}" if r['top1_rate'] is not None else "N/A"

        verdict = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"

        print(f"  {r['prompt_type']:>14}  {tile_kld_s:>10}  {tile_cos_s:>10}  "
              f"{med_kld_s:>10}  {max_kld_s:>10}  {mean_cos_s:>10}  "
              f"{top1_s:>6}%  {r['tile_pos']:>9}  {verdict}")

        if not r["passed"]:
            all_passed = False

        # Track worst-case
        if r.get("tile_kld") is not None and r["tile_kld"] > worst_kld:
            worst_kld = r["tile_kld"]
            worst_prompt = r["prompt_type"]
        if r.get("tile_cos") is not None and r["tile_cos"] < worst_cos:
            worst_cos = r["tile_cos"]
            worst_cos_prompt = r["prompt_type"]

    # Per-prompt characteristics
    print(f"\n{BOLD}  PROMPT CHARACTERISTICS & EXPECTED STRESS:{RESET}")
    for r in results_sorted:
        marker = " <<<" if r["prompt_type"] == worst_prompt else ""
        cos_marker = " <<<" if r["prompt_type"] == worst_cos_prompt else ""
        print(f"    {r['prompt_type']:>14}: {r['characteristic']}{marker}{cos_marker}")
    if worst_prompt:
        print(f"\n  {YELLOW}Worst KLD: {worst_prompt} ({worst_kld:.6f}){RESET}")
    if worst_cos_prompt:
        print(f"  {YELLOW}Worst cos: {worst_cos_prompt} ({worst_cos:.6f}){RESET}")

    # Robustness analysis
    print(f"\n{BOLD}  ROBUSTNESS ANALYSIS:{RESET}")
    valid_results = [r for r in results_sorted if r.get("tile_kld") is not None]
    if len(valid_results) >= 2:
        klds = [r["tile_kld"] for r in valid_results]
        coses = [r["tile_cos"] for r in valid_results if r["tile_cos"] is not None]
        kld_spread = max(klds) - min(klds)
        cos_spread = max(coses) - min(coses) if coses else 0

        if kld_spread < 0.001:
            print(f"    {GREEN}KLD spread < 0.001 — NVFP4 KV is prompt-INVARIANT (robust).{RESET}")
        elif kld_spread < 0.005:
            print(f"    {YELLOW}KLD spread {kld_spread:.4f} — minor prompt sensitivity.{RESET}")
        else:
            print(f"    {RED}KLD spread {kld_spread:.4f} — NVFP4 KV is prompt-DEPENDENT (fragile).{RESET}")

        if cos_spread < 0.001:
            print(f"    {GREEN}Cosine spread < 0.001 — stable across all prompt types.{RESET}")
        else:
            print(f"    {YELLOW}Cosine spread {cos_spread:.4f} — varies by prompt.{RESET}")

        # Identify risky prompt types
        risky = [r["prompt_type"] for r in valid_results if r.get("tile_kld", 0) > 0.01]
        if risky:
            print(f"    {RED}Risky prompts (tile_KLD > 0.01): {', '.join(risky)}{RESET}")

        n_passed = sum(1 for r in valid_results if r["passed"])
        n_total = len(valid_results)
        print(f"    Passed: {n_passed}/{n_total} prompts")

    # Overall verdict
    print(f"\n{BOLD}  OVERALL VERDICT:{RESET}")
    if all_passed:
        print(f"    {GREEN}ALL PROMPTS PASS — NVFP4 KV is prompt-robust.{RESET}")
    else:
        failed = [r["prompt_type"] for r in valid_results if not r["passed"]]
        print(f"    {RED}FAILURES: {', '.join(failed)} — NVFP4 KV NOT prompt-robust.{RESET}")
    print(f"{'='*110}\n")


def write_csv(results: list[dict], csv_path: str):
    """Write results to CSV."""
    if not results:
        return
    fields = [
        "prompt_type", "characteristic", "n_prompt", "n_tokens",
        "tail_pos", "tile_pos",
        "tile_kld", "tile_cos", "tail_kld", "tail_cos",
        "median_kld", "p99_kld", "max_kld", "mean_cos", "min_cos",
        "top1_rate", "worst_step", "passed", "elapsed_s",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  CSV written: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="test_cross_prompt_robustness.py — NVFP4 KV accuracy across prompt types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/test_cross_prompt_robustness.py --model I:\\models\\ornith-35b-NVFP4.gguf
  python tools/test_cross_prompt_robustness.py --model I:\\models\\ornith-35b-NVFP4.gguf --tokens 200
  python tools/test_cross_prompt_robustness.py --model I:\\models\\ornith-9b-NVFP4.gguf --csv robustness.csv
  python tools/test_cross_prompt_robustness.py --model I:\\models\\ornith-35b.gguf --prompts code,math,adversarial
        """,
    )
    parser.add_argument("--model", default=None,
                        help="Path to model. Auto-discovered if omitted.")
    parser.add_argument("--tokens", type=int, default=500,
                        help="Tokens per prompt (default: 500).")
    parser.add_argument("--ngl", type=int, default=99,
                        help="GPU layers (default: 99).")
    parser.add_argument("--threads", type=int, default=6,
                        help="CPU threads (default: 6).")
    parser.add_argument("--no-expert-stage", action="store_true",
                        help="Disable expert_stage (for dense models).")
    parser.add_argument("--tail-tokens", type=int, default=NVFP4_KV_TAIL_TOKENS,
                        help=f"KV tail size (default: {NVFP4_KV_TAIL_TOKENS}).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Evaluate KLD/cos every Nth step (default: 1).")
    parser.add_argument("--prompts", default=None,
                        help="Comma-separated prompt types to test (default: all).")
    parser.add_argument("--csv", default=None,
                        help="Write results to CSV.")

    args = parser.parse_args()

    # Resolve model
    model_path = args.model
    if not model_path:
        model_path = _find_model()
        if not model_path:
            print(f"{RED}ERROR: No model and auto-discovery failed.{RESET}")
            sys.exit(1)
        print(f"{YELLOW}Auto-discovered model: {model_path}{RESET}")

    if not os.path.isfile(model_path):
        print(f"{RED}ERROR: Model not found: {model_path}{RESET}")
        sys.exit(1)

    # Select prompts
    if args.prompts:
        selected = [p.strip() for p in args.prompts.split(",")]
        unknown = [p for p in selected if p not in PROMPTS]
        if unknown:
            print(f"{RED}ERROR: Unknown prompt types: {unknown}{RESET}")
            print(f"  Available: {list(PROMPTS.keys())}")
            sys.exit(1)
    else:
        selected = list(PROMPTS.keys())

    expert_stage = not args.no_expert_stage

    # Banner
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  CROSS-PROMPT ROBUSTNESS TEST{RESET}")
    print(f"{'='*70}")
    print(f"  Model        : {Path(model_path).name}")
    print(f"  Tokens/prompt: {args.tokens}")
    print(f"  GPU layers   : {args.ngl}")
    print(f"  Threads      : {args.threads}")
    print(f"  Expert stage : {expert_stage}")
    print(f"  Tail tokens  : {args.tail_tokens}")
    print(f"  Stride       : {args.stride}")
    print(f"  Prompts      : {selected}")
    print(f"  Total runs   : {len(selected)}")
    print(f"{'='*70}")

    results = []
    n_runs = len(selected)
    for i, prompt_type in enumerate(selected):
        prompt_text = PROMPTS[prompt_type]
        print(f"\n{BOLD}[{i + 1}/{n_runs}] Prompt: {prompt_type} "
              f"({len(prompt_text)} chars, {PROMPT_CHARACTERISTICS[prompt_type]}){RESET}")

        t0 = time.perf_counter()
        try:
            result = test_accuracy_kv(
                model_path=model_path,
                n_tokens=args.tokens,
                ngl=args.ngl,
                stride=args.stride,
                n_threads=args.threads,
                expert_stage=expert_stage,
                prompt=prompt_text,
                tail_tokens=args.tail_tokens,
            )
        except Exception as e:
            import traceback
            result = {
                "gate": "ACCURACY_KV",
                "passed": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "n_prompt_tokens": 0,
                "n_tokens": args.tokens,
                "n_positions": 0,
                "regions": {},
                "metrics": {},
                "extra": {},
            }
            print(f"  {RED}Exception: {e}{RESET}")

        elapsed = time.perf_counter() - t0
        row = extract_prompt_metrics(result, prompt_type, elapsed)
        results.append(row)

        if not row["error"]:
            status = f"{GREEN}PASS{RESET}" if row["passed"] else f"{RED}FAIL{RESET}"
            print(f"  [{i + 1}/{n_runs}] {prompt_type:>14}  "
                  f"tile_KLD={row['tile_kld']:.6f}  tile_cos={row['tile_cos']:.6f}  "
                  f"[{status}]  {elapsed:.0f}s")
        else:
            print(f"  [{i + 1}/{n_runs}] {prompt_type:>14}  "
                  f"{RED}ERROR: {row['error'][:80]}{RESET}  {elapsed:.0f}s")

    # Report
    print_cross_prompt_report(results)

    # CSV
    if args.csv:
        write_csv(results, args.csv)

    # Exit
    all_passed = all(r.get("passed", False) for r in results if "error" not in r or not r["error"])
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
