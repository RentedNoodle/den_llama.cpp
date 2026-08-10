#!/usr/bin/env python3
"""
test_ssm_state_drift.py — SSM/GDN recurrent state drift vs context length.

The NVFP4 KV gate (gate_accuracy_kv.py) tests attention KV cache quality.
GDN/SSM recurrence state may drift INDEPENDENTLY at long context — each
attention perturbation feeds into the SSM recurrence, which can amplify
small errors over many steps.

Since we cannot directly read SSM internal state without instrumentation,
we use LOGIT comparison as a proxy. Logit divergence that GROWS with context
(accelerating KLD) indicates SSM state drift. Constant divergence = KV-only.

For Ornith 35B (qwen35moe), SSM layers are interleaved with attention layers.
Key diagnostic: if median_KLD at 16k >> 4x median_KLD at 4k, SSM is drifting.

Usage:
  python tools/test_ssm_state_drift.py --model I:\\models\\ornith-35b-NVFP4.gguf
  python tools/test_ssm_state_drift.py --model I:\\models\\ornith-35b-NVFP4.gguf --ctx-sizes 1024,4096,16384 --ngl 0
  python tools/test_ssm_state_drift.py --model I:\\models\\ornith-9b-NVFP4.gguf --csv drift_results.csv

Exit: 0 = no SSM drift detected (linear KLD growth), 1 = SSM drift detected (super-linear KLD growth).
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

DEFAULT_CTX_SIZES = [1024, 4096, 16384]
PROMPT_TOKEN_ESTIMATE = 200

# Known SSM layer indices for qwen35moe (35B-A3B).
# Layer 0 = dense attn + GDN, layers 1-47 have interleaved full-attention + GDN.
# For Ornith 35B, SSM recurrence (gated_delta_net / GDN) runs on every layer.
# We estimate SSM-layer boundary positions based on hidden_dim ratios.
# In qwen35moe: 48 layers, SSM state size = 128 * 4 heads = 512 dims.
SSM_LAYERS_QWEN35 = list(range(48))  # every layer has GDN in qwen35moe


def extract_ssm_metrics(result: dict, ctx_target: int, elapsed_s: float) -> dict:
    """Extract SSM-drift-relevant metrics from a gate_accuracy_kv result."""
    regions = result.get("regions", {})
    metrics = result.get("metrics", {})
    extra = result.get("extra", {})

    return {
        "context_target": ctx_target,
        "n_prompt": result.get("n_prompt_tokens", 0),
        "n_tokens": result.get("n_tokens", 0),
        "total_pos": result.get("n_positions", 0),
        "tail_pos": regions.get("tail_positions", 0),
        "tile_pos": regions.get("tile_positions", 0),
        "tile_kld_mean": regions.get("tile_kld_mean"),
        "tile_cos_mean": regions.get("tile_cos_mean"),
        "tail_kld_mean": regions.get("tail_kld_mean"),
        "tail_cos_mean": regions.get("tail_cos_mean"),
        "median_kld": metrics.get("median_kld", {}).get("value"),
        "p99_kld": extra.get("p99_kld"),
        "max_kld": extra.get("max_kld"),
        "mean_cos": metrics.get("mean_cos", {}).get("value"),
        "min_cos": metrics.get("min_cos", {}).get("value"),
        "top1_rate": metrics.get("top1_rate", {}).get("value"),
        "worst_step": extra.get("worst_step"),
        "worst_kld": extra.get("worst_kld"),
        "passed": result.get("passed", False),
        "error": result.get("error"),
        "elapsed_s": elapsed_s,
    }


def compute_drift_diagnostic(results: list[dict]) -> dict:
    """
    Analyze KLD growth pattern across context sizes.

    - LINEAR growth: KLD ~ O(n) — static KV quantization error, no SSM drift.
    - SUPER-LINEAR growth: KLD ~ O(n^k), k > 1 — SSM recurrence amplifies error.
    - EXPONENTIAL growth: KLD doubles every fixed interval — SSM state collapse.

    Returns diagnostic dict with drift_verdict.
    """
    valid = [r for r in results if r.get("tile_kld_mean") is not None and r.get("tile_pos", 0) > 0]
    if len(valid) < 2:
        return {
            "drift_verdict": "INSUFFICIENT_DATA",
            "drift_detected": False,
            "kld_per_token": None,
            "growth_factor": None,
            "details": f"Need >= 2 valid context runs, got {len(valid)}",
        }

    # Normalize KLD per tile position (per-position average divergence)
    kld_per_pos = []
    ctx_sizes = []
    for r in valid:
        tile_pos = max(r["tile_pos"], 1)
        kld_per_pos.append(r["tile_kld_mean"] / tile_pos)
        ctx_sizes.append(r["context_target"])

    # Growth factor: ratio of KLD-per-position at max context vs min context
    min_kld_pp = kld_per_pos[0]
    max_kld_pp = kld_per_pos[-1]
    growth_factor = max_kld_pp / (min_kld_pp + 1e-12)

    # If KLD-per-position INCREASES with context, that's SSM state drift.
    # Constant KLD-per-position = KV-only error (the tile error per position
    # is independent of context length).
    if len(kld_per_pos) >= 3:
        diffs = [kld_per_pos[i] - kld_per_pos[i - 1] for i in range(1, len(kld_per_pos))]
        all_increasing = all(d >= -1e-12 for d in diffs)
    else:
        all_increasing = kld_per_pos[-1] > kld_per_pos[0] * 1.1

    # Verdict
    if growth_factor < 1.5:
        verdict = "NO_DRIFT"
        detected = False
        detail = f"KLD/pos stable ({min_kld_pp:.8f} -> {max_kld_pp:.8f}, {growth_factor:.1f}x)"
    elif growth_factor < 3.0:
        verdict = "MILD_DRIFT"
        detected = True
        detail = f"Mild KLD/pos growth ({min_kld_pp:.8f} -> {max_kld_pp:.8f}, {growth_factor:.1f}x)"
    elif growth_factor < 10.0:
        verdict = "MODERATE_DRIFT"
        detected = True
        detail = f"Moderate KLD/pos growth ({min_kld_pp:.8f} -> {max_kld_pp:.8f}, {growth_factor:.1f}x)"
    else:
        verdict = "SEVERE_DRIFT"
        detected = True
        detail = f"Severe KLD/pos growth ({min_kld_pp:.8f} -> {max_kld_pp:.8f}, {growth_factor:.1f}x)"

    # Check cosine degradation too
    cos_values = [r.get("tile_cos_mean", 1.0) for r in valid if r.get("tile_cos_mean") is not None]
    cos_degrading = False
    if len(cos_values) >= 2:
        cos_degrading = cos_values[-1] < cos_values[0] - 0.001

    return {
        "drift_verdict": verdict,
        "drift_detected": detected,
        "kld_per_pos_min": min_kld_pp,
        "kld_per_pos_max": max_kld_pp,
        "growth_factor": growth_factor,
        "all_increasing": all_increasing,
        "cos_degrading": cos_degrading,
        "details": detail,
        "context_sizes": ctx_sizes,
        "kld_per_pos_series": kld_per_pos,
    }


def print_drift_report(results: list[dict], diagnostic: dict):
    """Print SSM state drift diagnostic report."""
    print(f"\n{BOLD}{'='*95}{RESET}")
    print(f"{BOLD}  SSM STATE DRIFT TEST — NVFP4 KV vs F32 Oracle (Logit Proxy){RESET}")
    print(f"{'='*95}")
    print(f"  Method: Compare logit distributions at increasing context lengths.")
    print(f"  Theory: If SSM recurrence amplifies KV quantization error, KLD/position")
    print(f"          will GROW with context length (super-linear).")
    print(f"          If only KV error, KLD/position stays CONSTANT (linear).")
    print(f"  Note:   GDN/SSM internal state NOT readable via public API.")
    print(f"          Logit divergence is the proxy signal.")
    print(f"{'='*95}")

    # Per-context results table
    header = (
        f"  {'Context':>8}  {'Prompt':>6}  {'Tail':>5}  {'Tile':>6}  "
        f"{'tile_KLD':>10}  {'KLD/pos':>10}  {'tile_cos':>10}  {'max_KLD':>10}  {'Pass':>5}  {'Time'}"
    )
    print(f"\n{BOLD}  PER-CONTEXT RESULTS:{RESET}")
    print(header)
    print(f"  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*5}  {'-'*6}")

    for r in results:
        err = r.get("error")
        if err:
            print(f"  {r['context_target']:>8}  {'—':>6}  {'—':>5}  {'—':>6}  "
                  f"{'ERR':>10}  {'ERR':>10}  {'ERR':>10}  {'ERR':>10}  {RED}FAIL{RESET}  {r['elapsed_s']:.0f}s")
            print(f"    {RED}{err[:100]}{RESET}")
            continue

        tile_kld = r.get("tile_kld_mean")
        tile_pos = max(r.get("tile_pos", 1), 1)
        kld_per_pos = tile_kld / tile_pos if tile_kld is not None else None

        tile_kld_s = f"{tile_kld:.6f}" if tile_kld is not None else "N/A"
        kpp_s = f"{kld_per_pos:.8f}" if kld_per_pos is not None else "N/A"
        tile_cos_s = f"{r['tile_cos_mean']:.6f}" if r.get("tile_cos_mean") is not None else "N/A"
        max_kld_s = f"{r['max_kld']:.6f}" if r.get("max_kld") is not None else "N/A"

        print(f"  {r['context_target']:>8}  {r['n_prompt']:>6}  "
              f"{r['tail_pos']:>5}  {r['tile_pos']:>6}  "
              f"{tile_kld_s:>10}  {kpp_s:>10}  {tile_cos_s:>10}  "
              f"{max_kld_s:>10}  {'PASS' if r['passed'] else 'FAIL':>5}  {r['elapsed_s']:.0f}s")

    print(f"\n{BOLD}{'='*95}{RESET}")
    print(f"{BOLD}  SSM DRIFT DIAGNOSTIC{RESET}")
    print(f"{'='*95}")

    d = diagnostic
    verdict_color = GREEN if not d["drift_detected"] else (YELLOW if d["drift_verdict"] == "MILD_DRIFT" else RED)
    print(f"  Verdict          : {verdict_color}{BOLD}{d['drift_verdict']}{RESET}")
    print(f"  Drift detected   : {d['drift_detected']}")
    print(f"  Growth factor    : {d['growth_factor']:.2f}x (KLD/pos max vs min context)")
    print(f"  KLD/pos monotonic: {d['all_increasing']}")
    print(f"  Cosine degrading : {d['cos_degrading']}")
    print(f"  {d['details']}")

    if d.get("context_sizes") and d.get("kld_per_pos_series"):
        print(f"\n  {CYAN}KLD-per-position series:{RESET}")
        for ctx, kpp in zip(d["context_sizes"], d["kld_per_pos_series"]):
            bar_len = min(int(kpp * 100000), 50)
            bar = "#" * bar_len if bar_len > 0 else "."
            print(f"    ctx={ctx:>5}: {kpp:.8f}  {bar}")

    # Interpretation guide
    print(f"\n{BOLD}  INTERPRETATION:{RESET}")
    print(f"    Growth factor < 1.5x  : {GREEN}NO DRIFT{RESET} — KV error is constant per position.")
    print(f"    Growth factor 1.5-3x  : {YELLOW}MILD DRIFT{RESET} — SSM slightly amplifies error.")
    print(f"    Growth factor 3-10x   : {YELLOW}MODERATE DRIFT{RESET} — SSM recurrence is compounding error.")
    print(f"    Growth factor > 10x   : {RED}SEVERE DRIFT{RESET} — SSM state trajectory diverges.")
    print(f"")
    print(f"    If cosine ALSO degrades with context, SSM state collapse is likely.")
    print(f"    If cosine stays flat but KLD grows, it's attention perturbation amplification.")
    print(f"{'='*95}\n")

    return d["drift_detected"]


def write_csv(results: list[dict], diagnostic: dict, csv_path: str):
    """Write results to CSV."""
    fields = [
        "context_target", "n_prompt", "n_tokens", "total_pos",
        "tail_pos", "tile_pos",
        "tile_kld_mean", "tile_cos_mean", "tail_kld_mean", "tail_cos_mean",
        "median_kld", "p99_kld", "max_kld", "mean_cos", "min_cos",
        "top1_rate", "worst_step", "worst_kld",
        "passed", "elapsed_s",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  CSV written: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="test_ssm_state_drift.py — SSM/GDN state drift vs context length",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/test_ssm_state_drift.py --model I:\\models\\ornith-35b-NVFP4.gguf
  python tools/test_ssm_state_drift.py --model I:\\models\\ornith-35b-NVFP4.gguf --ctx-sizes 1024,4096,16384 --ngl 0
  python tools/test_ssm_state_drift.py --model I:\\models\\ornith-9b-NVFP4.gguf --csv drift.csv
        """,
    )
    parser.add_argument("--model", default=None,
                        help="Path to model. Auto-discovered if omitted.")
    parser.add_argument("--ctx-sizes", default="1024,4096,16384",
                        help="Comma-separated context sizes (default: 1024,4096,16384).")
    parser.add_argument("--ngl", type=int, default=99,
                        help="GPU layers (default: 99). Use 0 for CPU-only for large contexts.")
    parser.add_argument("--threads", type=int, default=6,
                        help="CPU threads (default: 6).")
    parser.add_argument("--no-expert-stage", action="store_true",
                        help="Disable expert_stage (for dense models).")
    parser.add_argument("--tail-tokens", type=int, default=NVFP4_KV_TAIL_TOKENS,
                        help=f"KV tail size (default: {NVFP4_KV_TAIL_TOKENS}).")
    parser.add_argument("--csv", default=None,
                        help="Write per-context results to CSV.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Evaluate KLD/cos every Nth step (default: 1).")
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt text.")

    args = parser.parse_args()

    # Resolve model
    model_path = args.model
    if not model_path:
        model_path = _find_model()
        if not model_path:
            print(f"{RED}ERROR: No model specified and auto-discovery failed.{RESET}")
            print("  Specify --model or place a model at one of:")
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

    context_sizes = [int(x.strip()) for x in args.ctx_sizes.split(",") if x.strip()]
    expert_stage = not args.no_expert_stage

    # VRAM warning
    max_ctx = max(context_sizes)
    if max_ctx >= 8192 and args.ngl > 0:
        print(f"\n{YELLOW}{BOLD}VRAM WARNING:{RESET}")
        print(f"  Max context: {max_ctx} — dual F32 KV caches may OOM on GPU.")
        print(f"  If OOM, retry with: --ngl 0 (CPU-only) or --ctx-sizes 1024,2048,4096")
        print()

    # Banner
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  SSM STATE DRIFT TEST{RESET}")
    print(f"{'='*70}")
    print(f"  Model        : {Path(model_path).name}")
    print(f"  Context sizes: {context_sizes}")
    print(f"  GPU layers   : {args.ngl}")
    print(f"  Threads      : {args.threads}")
    print(f"  Expert stage : {expert_stage}")
    print(f"  Tail tokens  : {args.tail_tokens}")
    print(f"  Stride       : {args.stride}")
    print(f"{'='*70}")

    # Run gate at each context size
    results = []
    n_ctx = len(context_sizes)
    for i, ctx_target in enumerate(context_sizes):
        n_tokens = max(args.tail_tokens + 1, ctx_target - PROMPT_TOKEN_ESTIMATE)

        print(f"\n{BOLD}[{i + 1}/{n_ctx}] Target context: {ctx_target} — generating {n_tokens} tokens...{RESET}")

        t0 = time.perf_counter()
        try:
            result = test_accuracy_kv(
                model_path=model_path,
                n_tokens=n_tokens,
                ngl=args.ngl,
                stride=args.stride,
                n_threads=args.threads,
                expert_stage=expert_stage,
                prompt=args.prompt,
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
                "n_tokens": n_tokens,
                "n_positions": 0,
                "regions": {},
                "metrics": {},
                "extra": {},
            }
            print(f"  {RED}Exception: {e}{RESET}")

        elapsed = time.perf_counter() - t0
        row = extract_ssm_metrics(result, ctx_target, elapsed)
        results.append(row)

        if not row["error"]:
            status = f"{GREEN}PASS{RESET}" if row["passed"] else f"{RED}FAIL{RESET}"
            print(f"  [{i + 1}/{n_ctx}] ctx={ctx_target}  "
                  f"tile={row['tile_pos']}  tile_KLD={row['tile_kld_mean']:.6f}  "
                  f"tile_cos={row['tile_cos_mean']:.6f}  [{status}]  {elapsed:.0f}s")
        else:
            print(f"  [{i + 1}/{n_ctx}] ctx={ctx_target}  {RED}ERROR: {row['error'][:80]}{RESET}  {elapsed:.0f}s")

    # Compute drift diagnostic
    diagnostic = compute_drift_diagnostic(results)

    # Print report
    drift_detected = print_drift_report(results, diagnostic)

    # CSV
    if args.csv:
        write_csv(results, diagnostic, args.csv)

    # Exit: 0 = no drift, 1 = drift detected
    sys.exit(1 if drift_detected else 0)


if __name__ == "__main__":
    main()
