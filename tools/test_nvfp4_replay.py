#!/usr/bin/env python3
"""
B2: Deterministic Replay Test — NVFP4 KV Quality Guard

Generates N tokens with F32 KV + NVFP4 KV on same model/prompt.
Compares token-by-token. Any divergence = regression.
Saves golden reference file for CI comparison.

Usage:
  python tools/test_nvfp4_replay.py --model <model.gguf> --tokens 1000
"""
import subprocess, sys, os, json, hashlib, argparse
from pathlib import Path

CLI = Path(__file__).parent.parent / "build_ninja" / "bin" / "llama-cli.exe"

def run_bench(model: str, n_tokens: int, nvfp4: bool, prompt: str = "The capital of France is") -> str:
    """Run llama-cli and return generated text."""
    env = os.environ.copy()
    if nvfp4:
        env["DEN_NVFP4_KV_CACHE"] = "1"
    else:
        env.pop("DEN_NVFP4_KV_CACHE", None)

    args = [
        str(CLI), "-m", model,
        "-p", prompt,
        "-n", str(n_tokens),
        "-ngl", "99", "-t", "4",
        "-no-cnv", "-st",
        "-c", "2048",
    ]
    if nvfp4:
        args += ["-ctk", "nvfp4_kv", "-ctv", "nvfp4_kv"]
    else:
        args += ["-ctk", "f32", "-ctv", "f32"]

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except FileNotFoundError:
        print(f"ERROR: {CLI} not found. Build first: build_now.bat")
        sys.exit(1)

    if result.returncode != 0:
        print(f"ERROR: llama-cli exit {result.returncode}")
        print(result.stderr[-500:])
        return "ERROR"

    return result.stdout.strip()

def extract_tokens(text: str, prompt: str) -> list:
    """Extract generated tokens after the prompt."""
    # llama-cli output format: prompt + generated text
    idx = text.find(prompt)
    if idx >= 0:
        return text[idx + len(prompt):].strip().split()
    return text.strip().split()

def compute_token_match(tokens_a: list, tokens_b: list) -> dict:
    """Compare two token lists, return match stats."""
    total = min(len(tokens_a), len(tokens_b))
    matches = sum(1 for i in range(total) if tokens_a[i] == tokens_b[i])
    first_diff = -1

    for i in range(total):
        if tokens_a[i] != tokens_b[i]:
            first_diff = i
            break

    return {
        "total_tokens": total,
        "matches": matches,
        "match_rate": matches / total if total > 0 else 0.0,
        "first_divergence": first_diff,
        "identical": matches == total,
    }

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def main():
    parser = argparse.ArgumentParser(description="NVFP4 KV replay quality test")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--tokens", type=int, default=256, help="Tokens to generate")
    parser.add_argument("--prompt", default="The capital of France is", help="Input prompt")
    parser.add_argument("--golden", help="Path to golden reference file (for CI comparison)")
    parser.add_argument("--save-golden", action="store_true", help="Save golden reference file")
    args = parser.parse_args()

    print(f"=== NVFP4 KV REPLAY TEST ===")
    print(f"Model: {args.model}")
    print(f"Tokens: {args.tokens}")

    # If golden file provided, compare against it only
    if args.golden and not args.save_golden:
        print(f"Comparing against golden: {args.golden}")
        nvfp4_text = run_bench(args.model, args.tokens, nvfp4=True, prompt=args.prompt)
        nvfp4_hash = compute_hash(nvfp4_text)

        try:
            with open(args.golden) as f:
                golden = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Golden file {args.golden} not found")
            sys.exit(1)

        if nvfp4_hash == golden.get("hash", ""):
            print(f"PASS: hash match {nvfp4_hash}")
            sys.exit(0)
        else:
            print(f"FAIL: hash mismatch. Current={nvfp4_hash}, golden={golden.get('hash')}")
            sys.exit(1)

    # Generate both F32 and NVFP4 output
    print("Running F32 KV baseline...")
    f32_text = run_bench(args.model, args.tokens, nvfp4=False, prompt=args.prompt)
    if f32_text in ("TIMEOUT", "ERROR"):
        print("FATAL: F32 baseline failed")
        sys.exit(1)

    print("Running NVFP4 KV...")
    nvfp4_text = run_bench(args.model, args.tokens, nvfp4=True, prompt=args.prompt)
    if nvfp4_text in ("TIMEOUT", "ERROR"):
        print("FATAL: NVFP4 run failed")
        sys.exit(1)

    f32_tokens = extract_tokens(f32_text, args.prompt)
    nvfp4_tokens = extract_tokens(nvfp4_text, args.prompt)

    result = compute_token_match(f32_tokens, nvfp4_tokens)
    result["f32_hash"] = compute_hash(f32_text)
    result["nvfp4_hash"] = compute_hash(nvfp4_text)

    print(f"\nF32 tokens: {len(f32_tokens)}, NVFP4 tokens: {len(nvfp4_tokens)}")
    print(f"Match rate: {result['match_rate']*100:.1f}% ({result['matches']}/{result['total_tokens']})")

    if result["first_divergence"] >= 0:
        print(f"First divergence at token {result['first_divergence']}")
        ctx = 5
        a = max(0, result["first_divergence"] - ctx)
        b = min(result["total_tokens"], result["first_divergence"] + ctx)
        print(f"  F32 context: {' '.join(f32_tokens[a:b])}")
        print(f"  NVFP4 context: {' '.join(nvfp4_tokens[a:b])}")

    if result["identical"]:
        print("\nPASS: 100% token match — NVFP4 KV = F32 KV (token-identical)")
        golden_path = args.golden or "test_nvfp4_golden.json"
        if args.save_golden:
            with open(golden_path, "w") as f:
                json.dump({
                    "model": args.model,
                    "tokens": args.tokens,
                    "prompt": args.prompt,
                    "hash": result["nvfp4_hash"],
                    "match_rate": result["match_rate"],
                    "timestamp": subprocess.run(["date", "/t"], capture_output=True, text=True, shell=True).stdout.strip(),
                }, f, indent=2)
            print(f"Golden reference saved: {golden_path}")
        sys.exit(0)
    else:
        print(f"\nFAIL: {result['match_rate']*100:.1f}% match — NVFP4 KV diverges from F32 KV")
        sys.exit(1)

if __name__ == "__main__":
    main()
