#!/usr/bin/env python3
"""
gate_accuracy_context_scaling.py
Context-scaling wrapper for NVFP4 KV accuracy gate.

Runs gate_accuracy_kv at increasing context sizes to determine if
NVFP4 KV accuracy (KLD=0, cos=1.0) holds or degrades with context length.

The 500-token gate passed at KLD=0 but context was ~600 tokens (105 prompt + 500 gen).
This tests at 1k/2k/4k/8k/16k to find where KLD starts accumulating.

KEY INSIGHT: At 1k, 1024 tail (F32) = all tokens exact (no tile test). At 2k, still 1024 tail
+ ~976 tile (NVFP4). At 16k, 1024 tail + ~15k tile positions. KLD might be zero
at small scale but diverge at scale. This is the reviewers' #1 concern.

VRAM WARNING: Two F32 KV caches at large contexts eat VRAM fast.
  - 9B model at 8k: ~12.8 GB KV + ~2.5 GB model = ~15.3 GB (tight)
  - 35B model at 4k: similar (fewer layers but larger model)
  - 16k: use --ngl 0 (CPU-only) or test with small model

Usage:
  python tools/gate_accuracy_context_scaling.py --model I:\\models\\ornith-9b.gguf
  python tools/gate_accuracy_context_scaling.py --model I:\\models\\ornith-9b.gguf --ctx-sizes 1024,2048,4096
  python tools/gate_accuracy_context_scaling.py --model I:\\models\\ornith-35b.gguf --ctx-sizes 1024,2048 --ngl 0
  python tools/gate_accuracy_context_scaling.py --model I:\\models\\ornith-9b.gguf --csv results.csv
"""

import sys
import os
import argparse
import time
import csv
from pathlib import Path

# Add tools/ to path so we can import gate_accuracy_kv
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_accuracy_kv import (
    test_accuracy_kv,
    NVFP4_KV_TAIL_TOKENS,
    LONGER_PROMPT,
    GREEN, RED, YELLOW, CYAN, BOLD, RESET,
    _find_model,
)

DEFAULT_CONTEXT_SIZES = [1024, 2048, 4096, 8192]

# Conservative estimate for LONGER_PROMPT token count (actual ~105 with Qwen tokenizer).
# Adds headroom so generated tokens fill to target context without overflow.
PROMPT_TOKEN_ESTIMATE = 200


def extract_row(result: dict, ctx_target: int, elapsed_s: float) -> dict:
    """Pull relevant metrics from a test_accuracy_kv result dict."""
    regions = result.get("regions", {})
    metrics = result.get("metrics", {})
    extra = result.get("extra", {})

    # tile_kld_mean, tile_cos_mean, tail_kld_mean, tail_cos_mean come from regions
    tail_kld = regions.get("tail_kld_mean")
    tile_kld = regions.get("tile_kld_mean")
    tail_cos = regions.get("tail_cos_mean")
    tile_cos = regions.get("tile_cos_mean")

    # median_kld and top1_rate are nested in metrics dict
    median_kld = metrics.get("median_kld", {}).get("value")
    top1 = metrics.get("top1_rate", {}).get("value")

    return {
        "context_target": ctx_target,
        "n_prompt": result.get("n_prompt_tokens", 0),
        "n_tokens": result.get("n_tokens", 0),
        "total_pos": result.get("n_positions", 0),
        "tail_pos": regions.get("tail_positions", 0),
        "tile_pos": regions.get("tile_positions", 0),
        "tail_kld": tail_kld,
        "tile_kld": tile_kld,
        "tail_cos": tail_cos,
        "tile_cos": tile_cos,
        "median_kld": median_kld,
        "top1_rate": top1,
        "passed": result.get("passed", False),
        "error": result.get("error"),
        "elapsed_s": elapsed_s,
    }


def print_summary_table(results: list[dict]) -> None:
    """Print compact context-scaling summary table."""
    print(f"\n{BOLD}{'='*95}{RESET}")
    print(f"{BOLD}  CONTEXT SCALING SUMMARY — NVFP4 KV Accuracy vs F32 Oracle{RESET}")
    print(f"{'='*95}")
    header = (
        f"  {'Context':>8}  {'Prompt':>6}  {'Tail':>5}  {'Tile':>6}  "
        f"{'tail_KLD':>10}  {'tile_KLD':>10}  {'tile_cos':>10}  {'top1%':>7}  {'Pass':>5}  {'Time'}"
    )
    print(header)
    print(f"  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*5}  {'-'*6}")

    for r in results:
        err = r.get("error")
        if err:
            print(
                f"  {r['context_target']:>8}  {'—':>6}  {'—':>5}  {'—':>6}  "
                f"{'ERR':>10}  {'ERR':>10}  {'ERR':>10}  {'—':>7}  "
                f"{RED}FAIL{RESET:>5}  {r['elapsed_s']:.0f}s"
            )
            print(f"    {RED}Error: {err[:100]}{RESET}")
            continue

        tail_kld_s = f"{r['tail_kld']:.6f}" if r['tail_kld'] is not None else "N/A"
        tile_kld_s = f"{r['tile_kld']:.6f}" if r['tile_kld'] is not None else "N/A"
        tile_cos_s = f"{r['tile_cos']:.6f}" if r['tile_cos'] is not None else "N/A"
        top1_s = f"{r['top1_rate']*100:.1f}" if r['top1_rate'] is not None else "N/A"

        pass_str = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"
        # ANSI codes mess up column alignment — pad manually
        print(
            f"  {r['context_target']:>8}  {r['n_prompt']:>6}  "
            f"{r['tail_pos']:>5}  {r['tile_pos']:>6}  "
            f"{tail_kld_s:>10}  {tile_kld_s:>10}  {tile_cos_s:>10}  "
            f"{top1_s:>6}%  {'PASS' if r['passed'] else 'FAIL':>5}  {r['elapsed_s']:.0f}s"
        )

    print(f"{'='*95}\n")

    # ── Analysis ──────────────────────────────────────────────────────────
    tile_klds = [r["tile_kld"] for r in results if r.get("tile_kld") is not None]
    tile_cos_vals = [r["tile_cos"] for r in results if r.get("tile_cos") is not None]

    if not tile_klds:
        print(f"  {RED}No valid tile measurements — all runs failed.{RESET}")
        return

    max_kld = max(tile_klds)
    min_cos = min(tile_cos_vals) if tile_cos_vals else 1.0

    n_runs = len(tile_klds)
    n_passed = sum(1 for r in results if r["passed"])

    print(f"  {BOLD}Analysis:{RESET}")
    print(f"    Runs         : {n_runs} ({n_passed} passed, {n_runs - n_passed} failed)")
    print(f"    Max tile KLD : {max_kld:.8f}")
    print(f"    Min tile cos : {min_cos:.8f}")

    if n_runs >= 2:
        # Check if KLD is increasing with context (degradation trend)
        diffs = [tile_klds[i] - tile_klds[i - 1] for i in range(1, len(tile_klds))]
        all_increasing = all(d >= -1e-9 for d in diffs)  # allow tiny noise
        if all_increasing:
            print(f"    {YELLOW}KLD trend  : INCREASING with context — possible degradation{RESET}")
        else:
            print(f"    KLD trend  : stable / non-monotonic")

    if max_kld < 0.001:
        print(f"\n  {GREEN}{BOLD}KLD < 0.001 at ALL context sizes — NVFP4 KV is LOSSESS up to "
              f"{results[-1]['context_target']} tokens.{RESET}")
    elif max_kld < 0.005:
        print(f"\n  {YELLOW}{BOLD}KLD < 0.005 — minor divergence, within BeeLlama q8_0 tier.{RESET}")
    elif max_kld < 0.01:
        print(f"\n  {YELLOW}{BOLD}KLD < 0.01 — noticeable but may be acceptable for most use cases.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}KLD >= 0.01 — NVFP4 KV accuracy degrades significantly with context.{RESET}")
        print(f"  {RED}Review tile layout, scale quantization, or per-layer thresholds.{RESET}")

    print()


def write_csv(results: list[dict], csv_path: str) -> None:
    """Write results to CSV file."""
    if not results:
        return
    fields = [
        "context_target", "n_prompt", "n_tokens", "total_pos",
        "tail_pos", "tile_pos",
        "tail_kld", "tile_kld", "tail_cos", "tile_cos",
        "median_kld", "top1_rate", "passed", "elapsed_s",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  CSV written: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="gate_accuracy_context_scaling.py — Multi-context NVFP4 KV accuracy gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/gate_accuracy_context_scaling.py --model I:\\\\models\\\\ornith-9b.gguf
  python tools/gate_accuracy_context_scaling.py --ctx-sizes 1024,2048,4096,8192,16384 --ngl 0
  python tools/gate_accuracy_context_scaling.py --model I:\\\\models\\\\ornith-9b.gguf --csv ctx_results.csv --prompt "Once upon a time"
        """,
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to model. Auto-discovered if omitted (prefers 9B NVFP4 for VRAM efficiency).",
    )
    parser.add_argument(
        "--ctx-sizes", default="1024,2048,4096,8192",
        help="Comma-separated context sizes to test (default: 1024,2048,4096,8192). "
             "Use --ngl 0 for 16384+ (F32 KV cache OOM on GPU).",
    )
    parser.add_argument(
        "--ngl", type=int, default=99,
        help="GPU layers (default: 99). Use 0 for CPU-only (large contexts).",
    )
    parser.add_argument(
        "--threads", type=int, default=6,
        help="CPU threads (default: 6).",
    )
    parser.add_argument(
        "--no-expert-stage", action="store_true",
        help="Disable expert_stage context flag (for dense models).",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Custom prompt text (default: transformer architecture paragraph).",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Write results to CSV file.",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Only evaluate KLD/cos every Nth token (default: 1 = all). "
             "5-10x speedup for long contexts. Curve is empirically flat.",
    )
    parser.add_argument(
        "--tail-tokens", type=int, default=NVFP4_KV_TAIL_TOKENS,
        help=f"Precision tail size (default: {NVFP4_KV_TAIL_TOKENS}). "
             "Must match DEN_NVFP4_KV_TAIL env var.",
    )

    args = parser.parse_args()

    # Resolve model
    model_path = args.model
    if not model_path:
        model_path = _find_model()
        if not model_path:
            print(f"{RED}ERROR: No model specified and auto-discovery failed.{RESET}")
            print("  Specify --model or ensure a model exists at:")
            for c in [
                r"I:\models\ornith-1.0-35b-APEX-I-Mini-MTP.gguf",
                r"I:\models\ornith-1.0-9b-NVFP4.gguf",
            ]:
                print(f"    {c}")
            sys.exit(1)
        print(f"{YELLOW}Auto-discovered model: {model_path}{RESET}")

    if not os.path.isfile(model_path):
        print(f"{RED}ERROR: Model not found: {model_path}{RESET}")
        sys.exit(1)

    # Parse context sizes
    context_sizes = [int(x.strip()) for x in args.ctx_sizes.split(",") if x.strip()]

    expert_stage = not args.no_expert_stage

    # ── VRAM warning for large contexts ───────────────────────────────────
    max_ctx = max(context_sizes)
    if max_ctx >= 8192 and args.ngl > 0:
        print(f"\n{YELLOW}{BOLD}VRAM WARNING:{RESET}")
        print(f"  Max context: {max_ctx} — two F32 KV caches may OOM on GPU.")
        print(f"  F32 KV cache at {max_ctx}: ~{(max_ctx * 2 * 128 * 16 * 4 * 48) / (1024**3):.1f} GB per context (9B est).")
        print(f"  Two contexts + model weights may exceed 16 GB.")
        print(f"  If OOM, retry with: --ngl 0 (CPU-only) or --ctx-sizes 1024,2048,4096")
        print()

    # ── Run tests ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  CONTEXT SCALING TEST — NVFP4 KV Accuracy Gate{RESET}")
    print(f"{'='*70}")
    print(f"  Model        : {Path(model_path).name}")
    print(f"  Context sizes: {context_sizes}")
    print(f"  GPU layers   : {args.ngl}")
    print(f"  Threads      : {args.threads}")
    print(f"  Expert stage : {expert_stage}")
    print(f"  Stride       : {args.stride}")
    print(f"  Tail tokens  : {args.tail_tokens}")
    print(f"  Total runs   : {len(context_sizes)}")
    print(f"{'='*70}")

    results = []
    n_ctx = len(context_sizes)
    for i, ctx_target in enumerate(context_sizes):
        # Tokens to generate = target context - estimated prompt tokens.
        # Must be at least tail+1 to reach tile region.
        n_tokens = max(args.tail_tokens + 1, ctx_target - PROMPT_TOKEN_ESTIMATE)

        print(f"\n{BOLD}[{i + 1}/{n_ctx}] Target context: {ctx_target} — generating {n_tokens} tokens...{RESET}")

        t0 = time.perf_counter()
        try:
            result = test_accuracy_kv(
                model_path=model_path,
                n_tokens=n_tokens,
                ngl=args.ngl,
                n_threads=args.threads,
                expert_stage=expert_stage,
                prompt=args.prompt,
                tail_tokens=args.tail_tokens,
                stride=args.stride,
            )
        except Exception as e:
            import traceback
            result = {
                "gate": "ACCURACY_KV",
                "passed": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "n_prompt_tokens": 0,
                "n_tokens": n_tokens,
                "n_positions": 0,
                "regions": {},
                "metrics": {},
                "extra": {},
            }
            print(f"  {RED}Exception: {e}{RESET}")

        elapsed = time.perf_counter() - t0
        row = extract_row(result, ctx_target, elapsed)
        results.append(row)

        # Progress
        if not row["error"]:
            status = f"{GREEN}PASS{RESET}" if row["passed"] else f"{RED}FAIL{RESET}"
            print(
                f"  [{i + 1}/{n_ctx}] ctx={ctx_target}  "
                f"tail={row['tail_pos']}  tile={row['tile_pos']}  "
                f"tile_KLD={row['tile_kld']:.6f}  tile_cos={row['tile_cos']:.6f}  "
                f"[{status}]  {elapsed:.0f}s"
            )
        else:
            print(f"  [{i + 1}/{n_ctx}] ctx={ctx_target}  {RED}ERROR: {row['error'][:80]}{RESET}  {elapsed:.0f}s")

    # ── Summary table ─────────────────────────────────────────────────────
    print_summary_table(results)

    # ── CSV output ────────────────────────────────────────────────────────
    if args.csv:
        write_csv(results, args.csv)

    # ── Exit code ─────────────────────────────────────────────────────────
    all_passed = all(r["passed"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
