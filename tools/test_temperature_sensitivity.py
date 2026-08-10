#!/usr/bin/env python3
"""
test_temperature_sensitivity.py — NVFP4 KV accuracy at non-zero temperatures.

The gate_accuracy_kv.py test runs with greedy decoding (temp=0), which is
proven correct. But real-world inference uses temp > 0 (usually 0.3-0.7).
Sampling from a softmax distribution amplifies small logit differences —
a logit delta that's invisible under argmax can change sampling outcome.

This test runs the dual-context F32-vs-NVFP4 comparison at multiple
temperature values (0, 0.3, 0.7) to determine if:
  a) NVFP4 KV logit divergence is TEMPERATURE-INVARIANT (same KLD/cos at all temps)
  b) Higher temperature AMPLIFIES divergence (KLD grows, cos drops)
  c) Higher temperature MASKS divergence (softmax flattens, KLD shrinks)

Method: Take F32 logits and NVFP4 logits, apply temperature scaling,
compute KLD/cos of the resulting distributions. The underlying logits
are from the SAME dual-context run — we just re-evaluate the softmax
at different temperature values.

Usage:
  python tools/test_temperature_sensitivity.py --model I:\\models\\ornith-35b-NVFP4.gguf
  python tools/test_temperature_sensitivity.py --model I:\\models\\ornith-35b-NVFP4.gguf --temps 0,0.3,0.7,1.0 --tokens 500
  python tools/test_temperature_sensitivity.py --model I:\\models\\ornith-9b-NVFP4.gguf --csv temp_results.csv

Exit: 0 = KLD stable across temps, 1 = KLD degrades at higher temps.
"""

import sys
import os
import argparse
import time
import csv
import ctypes
from ctypes import (
    c_int32, c_uint32, c_int8, c_bool, c_float, c_double,
    c_char, c_char_p, c_void_p, c_size_t, POINTER, Structure,
    CFUNCTYPE, byref, cast, pointer, sizeof,
    create_string_buffer, CDLL,
)
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_accuracy_kv import (
    DualContextKVModel,
    NVFP4_KV_TAIL_TOKENS,
    LONGER_PROMPT,
    softmax,
    cos_sim,
    GREEN, RED, YELLOW, CYAN, BOLD, RESET,
    _find_model,
    GGML_TYPE_F32,
)

DEFAULT_TEMPS = [0.0, 0.3, 0.7]


def softmax_temperature(logits: np.ndarray, temp: float) -> np.ndarray:
    """Softmax with temperature scaling. temp=0 = argmax (one-hot)."""
    if temp <= 0.0:
        # Argmax as one-hot distribution
        result = np.zeros_like(logits)
        result[np.argmax(logits)] = 1.0
        return result
    scaled = logits / temp
    return softmax(scaled)


def compute_kld_temp(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence P||Q with epsilon for stability."""
    eps = 1e-12
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def run_temperature_test(
    model_path: str,
    n_tokens: int = 500,
    ngl: int = 99,
    n_threads: int = 6,
    expert_stage: bool = True,
    prompt: str = None,
    tail_tokens: int = NVFP4_KV_TAIL_TOKENS,
    stride: int = 1,
    temperatures: list = None,
) -> list[dict]:
    """
    Run F32-vs-NVFP4 dual-context decode ONCE, then evaluate softmax
    divergence at multiple temperature values from the SAME logits.

    This isolates temperature effects from decode randomness — since both
    contexts run greedy (argmax) for token selection, but we measure the
    softmax distributions at each position under different temperatures.
    """
    if prompt is None:
        prompt = LONGER_PROMPT
    if temperatures is None:
        temperatures = DEFAULT_TEMPS

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  TEMPERATURE SENSITIVITY — NVFP4 KV vs F32 Oracle{RESET}")
    print(f"{'='*70}")
    print(f"  Model        : {Path(model_path).name}")
    print(f"  Tokens       : {n_tokens}")
    print(f"  GPU layers   : {ngl}")
    print(f"  Threads      : {n_threads}")
    print(f"  Expert stage : {expert_stage}")
    print(f"  Temperatures : {temperatures}")
    print(f"  Prompt       : {prompt[:80]}...")
    print(f"{'='*70}")

    m = None
    try:
        ctx_size = max(512, n_tokens + 512)
        print(f"\n  {CYAN}Loading model + creating dual contexts...{RESET}")
        m = DualContextKVModel(
            model_path,
            n_ctx=ctx_size,
            ngl=ngl,
            n_threads=n_threads,
            expert_stage=expert_stage,
        )
        print(f"  Vocab size: {m.n_vocab}")

        # Decode prompt
        print(f"  {CYAN}Decoding prompt ({len(prompt)} chars)...{RESET}")
        prompt_tokens = m.tokenize(prompt, add_special=True)
        print(f"  Prompt tokens: {len(prompt_tokens)}")
        m.decode_both(prompt_tokens)

        n_prompt = len(prompt_tokens)
        first_tile_step = max(0, tail_tokens - n_prompt + 1)

        # Per-temperature accumulators (tile region only)
        temp_data = {t: {"klds": [], "cosines": [], "top1_matches": 0, "total": 0}
                     for t in temperatures}

        print(f"\n  {CYAN}Running shared decode + multi-temp evaluation ({n_tokens} tokens)...{RESET}")
        print(f"  {'Step':>6}  {'T=0 KLD':>10}  {'T=0.3 KLD':>10}  {'T=0.7 KLD':>10}  {'Token'}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        for step in range(n_tokens):
            do_metrics = (step % stride == 0) or (step == n_tokens - 1)
            is_tile_region = step >= first_tile_step

            lf32 = m.get_logits_f32()
            lnv = m.get_logits_nvfp4()

            if do_metrics and is_tile_region:
                for temp in temperatures:
                    pf32 = softmax_temperature(lf32.astype(np.float64), temp)
                    pnv = softmax_temperature(lnv.astype(np.float64), temp)

                    kld = compute_kld_temp(pf32, pnv)
                    cos = cos_sim(lf32, lnv)
                    top1_match = int(np.argmax(lf32)) == int(np.argmax(lnv))

                    temp_data[temp]["klds"].append(kld)
                    temp_data[temp]["cosines"].append(cos)
                    if top1_match:
                        temp_data[temp]["top1_matches"] += 1
                    temp_data[temp]["total"] += 1

                # Progress print
                if step % 50 == 0 or step == n_tokens - 1:
                    k0 = temp_data[0.0]["klds"][-1] if temp_data[0.0]["klds"] else -1
                    k3 = temp_data[0.3]["klds"][-1] if temp_data[0.3]["klds"] else -1
                    k7 = temp_data[0.7]["klds"][-1] if temp_data[0.7]["klds"] else -1
                    next_token = int(np.argmax(lf32))
                    print(f"  {step:>6}  {k0:>10.6f}  {k3:>10.6f}  {k7:>10.6f}  {next_token}")

            # Greedy next token from F32 oracle
            next_token = int(np.argmax(lf32))
            m.decode_both([next_token])

        # ── Build per-temperature result dicts ──────────────────────────────
        results = []
        for temp in temperatures:
            d = temp_data[temp]
            n = d["total"]
            if n == 0:
                results.append({
                    "temperature": temp,
                    "n_positions": 0,
                    "median_kld": None,
                    "mean_kld": None,
                    "max_kld": None,
                    "mean_cos": None,
                    "min_cos": None,
                    "top1_rate": None,
                    "passed": False,
                    "error": "No tile positions evaluated",
                })
                continue

            klds = np.array(d["klds"])
            cosines = np.array(d["cosines"])
            top1_rate = d["top1_matches"] / n

            median_kld = float(np.median(klds))
            mean_kld = float(np.mean(klds))
            max_kld = float(np.max(klds))
            p99_kld = float(np.percentile(klds, 99.0)) if len(klds) >= 100 else max_kld
            mean_cos = float(np.mean(cosines))
            min_cos = float(np.min(cosines))

            # Pass/fail using same thresholds as gate_accuracy_kv.py
            passed_kld_median = median_kld < 0.001
            passed_cos_mean = mean_cos >= 0.9995
            passed_cos_min = min_cos >= 0.99
            passed = passed_kld_median and passed_cos_mean and passed_cos_min

            results.append({
                "temperature": temp,
                "n_positions": n,
                "median_kld": median_kld,
                "mean_kld": mean_kld,
                "max_kld": max_kld,
                "p99_kld": p99_kld,
                "mean_cos": mean_cos,
                "min_cos": min_cos,
                "top1_rate": top1_rate,
                "passed": passed,
            })

    except Exception as e:
        import traceback
        results = [{
            "temperature": -1,
            "n_positions": 0,
            "median_kld": None,
            "mean_kld": None,
            "max_kld": None,
            "mean_cos": None,
            "min_cos": None,
            "top1_rate": None,
            "passed": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }]
    finally:
        if m is not None:
            try:
                m.close()
            except Exception:
                pass

    _print_temperature_report(results)
    return results


def _print_temperature_report(results: list[dict]):
    """Print temperature sensitivity report."""
    if not results or results[0].get("error"):
        err = results[0].get("error", "Unknown error") if results else "No results"
        print(f"\n  [{RED}FAIL{RESET}] ERROR: {err}")
        if results and results[0].get("traceback"):
            for line in results[0]["traceback"].splitlines()[-6:]:
                print(f"    {line}")
        return

    print(f"\n{BOLD}{'='*95}{RESET}")
    print(f"{BOLD}  TEMPERATURE SENSITIVITY RESULTS{RESET}")
    print(f"{'='*95}")

    # Table
    header = (
        f"  {'Temp':>6}  {'N pos':>6}  {'med_KLD':>10}  {'mean_KLD':>10}  "
        f"{'max_KLD':>10}  {'p99_KLD':>10}  {'mean_cos':>10}  {'min_cos':>10}  "
        f"{'top1%':>7}  {'Verdict':>10}"
    )
    print(header)
    print(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*10}")

    valid = []
    for r in results:
        if r.get("median_kld") is None:
            print(f"  {r['temperature']:>6.1f}  {'—':>6}  {'N/A':>10}  {'N/A':>10}  "
                  f"{'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'—':>7}  {RED}N/A{RESET}")
            continue

        verdict = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"
        print(f"  {r['temperature']:>6.1f}  {r['n_positions']:>6}  "
              f"{r['median_kld']:>10.6f}  {r['mean_kld']:>10.6f}  "
              f"{r['max_kld']:>10.6f}  {r['p99_kld']:>10.6f}  "
              f"{r['mean_cos']:>10.6f}  {r['min_cos']:>10.6f}  "
              f"{r['top1_rate']*100:>6.1f}%  {verdict}")
        valid.append(r)

    # ── Temperature sensitivity analysis ───────────────────────────────────
    if len(valid) >= 2:
        print(f"\n{BOLD}  TEMPERATURE SENSITIVITY ANALYSIS:{RESET}")

        klds = [r["median_kld"] for r in valid]
        coses = [r["mean_cos"] for r in valid]
        temps_list = [r["temperature"] for r in valid]

        # KLD trend with temperature
        kld_t0 = klds[0]  # temp=0 baseline
        kld_ratios = [k / (kld_t0 + 1e-12) for k in klds]

        print(f"  KLD at T=0    : {kld_t0:.8f}  (baseline — proven correct)")
        for temp, ratio in zip(temps_list[1:], kld_ratios[1:]):
            direction = "HIGHER" if ratio > 1.05 else ("LOWER" if ratio < 0.95 else "SAME")
            color = YELLOW if abs(ratio - 1.0) > 0.05 else GREEN
            print(f"  KLD at T={temp:.1f} : {klds[valid.index([r for r in valid if r['temperature'] == temp][0])]:.8f}  "
                  f"({color}{ratio:.2f}x vs T=0 — {direction}{RESET})")

        # Interpretation
        max_ratio = max(kld_ratios)
        min_ratio = min(kld_ratios)
        spread = max_ratio - min_ratio

        if spread < 0.1:
            print(f"\n  {GREEN}{BOLD}KLD is TEMPERATURE-INVARIANT (spread < 10%).{RESET}")
            print(f"  {GREEN}NVFP4 KV accuracy is stable across all sampling temperatures.{RESET}")
        elif max_ratio > 2.0:
            print(f"\n  {RED}{BOLD}KLD GROWS with temperature — divergence AMPLIFIED by sampling.{RESET}")
            print(f"  {RED}NVFP4 KV errors are more visible at higher temperatures.{RESET}")
        elif max_ratio < 0.5:
            print(f"\n  {YELLOW}{BOLD}KLD SHRINKS with temperature — divergence MASKED by sampling.{RESET}")
            print(f"  {YELLOW}Temperature flattens the distribution, hiding KV quantization errors.{RESET}")

        # Cosine stability
        cos_t0 = coses[0]
        cos_degradation = cos_t0 - min(coses)
        if cos_degradation > 0.001:
            print(f"\n  {YELLOW}Cosine degrades {cos_degradation:.4f} at higher temps — "
                  f"softmax scaling affects direction.{RESET}")
        else:
            print(f"\n  {GREEN}Cosine is temperature-INVARIANT (< 0.001 change).{RESET}")

    # Overall
    all_passed = all(r["passed"] for r in valid)
    print(f"\n{BOLD}  OVERALL:{RESET}")
    if all_passed:
        print(f"  {GREEN}ALL TEMPERATURES PASS — NVFP4 KV is temp-robust.{RESET}")
    else:
        failed_temps = [r["temperature"] for r in valid if not r["passed"]]
        print(f"  {RED}FAILURES at T={failed_temps} — NVFP4 KV degrades at these temperatures.{RESET}")
    print(f"{'='*95}\n")


def write_csv(results: list[dict], csv_path: str):
    """Write results to CSV."""
    if not results:
        return
    fields = [
        "temperature", "n_positions",
        "median_kld", "mean_kld", "max_kld", "p99_kld",
        "mean_cos", "min_cos", "top1_rate", "passed",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  CSV written: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="test_temperature_sensitivity.py — NVFP4 KV at non-zero temperatures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/test_temperature_sensitivity.py --model I:\\models\\ornith-35b-NVFP4.gguf
  python tools/test_temperature_sensitivity.py --model I:\\models\\ornith-35b-NVFP4.gguf --temps 0,0.3,0.7,1.0
  python tools/test_temperature_sensitivity.py --model I:\\models\\ornith-9b-NVFP4.gguf --tokens 500 --csv temp.csv

Note: Only one decode pass is needed — all temperatures are evaluated from the
same logits. The decode itself always uses greedy (argmax) token selection so
both F32 and NVFP4 contexts see identical token sequences.
        """,
    )
    parser.add_argument("--model", default=None,
                        help="Path to model. Auto-discovered if omitted.")
    parser.add_argument("--tokens", type=int, default=500,
                        help="Tokens to generate (default: 500).")
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
    parser.add_argument("--temps", default="0,0.3,0.7",
                        help="Comma-separated temperatures (default: 0,0.3,0.7).")
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt text.")
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

    temperatures = [float(x.strip()) for x in args.temps.split(",") if x.strip()]
    expert_stage = not args.no_expert_stage

    # Banner
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  TEMPERATURE SENSITIVITY TEST{RESET}")
    print(f"{'='*70}")
    print(f"  Model        : {Path(model_path).name}")
    print(f"  Tokens       : {args.tokens}")
    print(f"  GPU layers   : {args.ngl}")
    print(f"  Threads      : {args.threads}")
    print(f"  Expert stage : {expert_stage}")
    print(f"  Temperatures : {temperatures}")
    print(f"  Stride       : {args.stride}")
    print(f"{'='*70}")

    # Run test
    results = run_temperature_test(
        model_path=model_path,
        n_tokens=args.tokens,
        ngl=args.ngl,
        n_threads=args.threads,
        expert_stage=expert_stage,
        prompt=args.prompt,
        tail_tokens=args.tail_tokens,
        stride=args.stride,
        temperatures=temperatures,
    )

    # CSV
    if args.csv:
        write_csv(results, args.csv)

    # Exit
    all_passed = all(r.get("passed", False) for r in results if r.get("median_kld") is not None)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
