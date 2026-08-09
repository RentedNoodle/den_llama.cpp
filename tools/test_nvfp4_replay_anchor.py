#!/usr/bin/env python3
"""
B2-ANCHOR: NVFP4 KV Replay — ANCHOR-ONLY Discriminating Probe  (DECISIVE revision)

Diagnostic for the "tile quantization vs structural bug" question.

A short prompt (<= 3 tokens) fills cache positions 0-(prompt_len-1). The NVFP4
KV cache keeps the first 4 cache positions in F32 anchor buffers
(DEN_NVFP4_KV_ANCHOR_TOKENS == 4); positions >= 4 live in quantized NVFP4
tiles. If F32 KV vs NVFP4 KV are then BIT-EXACT (100% token match) well past
the first divergence of a long-prompt run, it proves the anchor read/write
path is correct and the residual divergence is TILE QUANTIZATION error.

CRITICAL CACHE-POSITION MAPPING (FIXED):
  The prompt occupies cache positions 0 .. prompt_len-1.
  Generated output token i occupies cache position  cache_pos = i + prompt_len.
  So for prompt "The cap" (prompt_len = 2):
    output token 0 -> cache pos 2  (ANCHOR)
    output token 1 -> cache pos 3  (ANCHOR)
    output token 2 -> cache pos 4  (FIRST NVFP4 TILE)
  The OLD code compared the output-token index against 4 directly, which was
  WRONG: divergence at output-token 2 was mislabeled "still in anchors" when it
  is actually cache pos 4 = the first tile. This revision reports divergence by
  CACHE POSITION (output index + prompt_len), so the anchor/tile boundary is
  interpreted correctly.

VERIFIED (store-path inspection, ggml-cuda.cu + fattn-nvfp4-kv.cu):
  The SET_ROWS hook (ggml-cuda.cu ~2076-2108) calls den_nvfp4_kv_store() for
  EVERY cache_k_l/cache_v_l write, and kv_store_quantize_kernel()
  (fattn-nvfp4-kv.cu ~454-501) is keyed on seq_pos only: seq_pos < 4 writes the
  F32 anchor buffer, seq_pos >= 4 quantizes a tile. It does NOT distinguish
  prefill from generation. Therefore anchors ARE written for generated tokens
  at cache pos 2,3 during generation — they are NOT left as garbage. The store
  path is phase-agnostic and correct for generated anchors.

Usage:
  python tools/test_nvfp4_replay_anchor.py --model <model.gguf> [--tokens 200]
"""
import subprocess, sys, os, json, hashlib, argparse, re
from pathlib import Path

CLI = Path(__file__).parent.parent / "build_ninja" / "bin" / "llama-cli.exe"

ANCHOR_TOKENS = 4  # DEN_NVFP4_KV_ANCHOR_TOKENS in fattn-nvfp4-kv.cuh

# "llama_perf_context_print: prompt eval time = 123.45 ms /     7 tokens (..."
# The perf line is emitted via LOG_INF (stderr by default). / N tokens = prompt token count.
_PERF_RE = re.compile(r"prompt eval time = [\d.]+ ms / (\d+) tokens")


def run_bench(model: str, n_tokens: int, nvfp4: bool, prompt: str) -> tuple:
    """Run llama-cli and return (generated_text, full_stdout, full_stderr)."""
    env = os.environ.copy()
    if nvfp4:
        env["DEN_NVFP4_KV_CACHE"] = "1"
        env["DEN_THRIFT_ATTENTION"] = "1"  # K8V8: 8-bit K and V (V was the worst error link)
    else:
        env.pop("DEN_NVFP4_KV_CACHE", None)
        env.pop("DEN_THRIFT_ATTENTION", None)

    args = [
        str(CLI), "-m", model,
        "-p", prompt,
        "-n", str(n_tokens),
        "-ngl", "99", "-t", "4",
        "-no-cnv", "-st",
        "-c", "2048",
        "--temp", "0", "-s", "42",
    ]
    if nvfp4:
        args += ["-ctk", "nvfp4_kv", "-ctv", "nvfp4_kv"]
    else:
        args += ["-ctk", "f32", "-ctv", "f32"]

    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300, env=env)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "", ""
    except FileNotFoundError:
        print(f"ERROR: {CLI} not found. Build first: build_now.bat")
        sys.exit(1)

    if result.returncode != 0:
        print(f"ERROR: llama-cli exit {result.returncode}")
        print(result.stderr[-500:])
        return "ERROR", "", ""

    return result.stdout.strip(), result.stdout, result.stderr


def get_prompt_token_count(stderr: str, stdout: str = "") -> int:
    """Parse llama-cli perf output for the number of prompt tokens.

    The perf line 'prompt eval time = X ms / N tokens' reports N = number of
    prompt tokens actually tokenized/evaluated (includes BOS). Falls back to a
    whitespace-split heuristic only if the perf line is missing.
    """
    for blob in (stderr, stdout):
        m = _PERF_RE.search(blob)
        if m:
            return int(m.group(1))
    # Fallback heuristic: approximate prompt token count by whitespace words.
    # Do NOT trust this — it is only a guard; the perf line is the source of truth.
    return -1


def extract_tokens(text: str, prompt: str) -> list:
    """Extract generated tokens after the prompt."""
    idx = text.find(prompt)
    if idx >= 0:
        return text[idx + len(prompt):].strip().split()
    return text.strip().split()


def compute_token_match(tokens_a: list, tokens_b: list, prompt_len: int) -> dict:
    """Compare two token lists.

    Reports divergence by CACHE POSITION = output_token_index + prompt_len, so
    the anchor (0..3) vs tile (>=4) boundary is interpreted correctly. The
    prompt tokens themselves (cache pos 0..prompt_len-1) are identical by
    construction and are excluded from the comparison.
    """
    total = min(len(tokens_a), len(tokens_b))
    matches = 0
    divergences = []  # list of (output_index, cache_pos, region)
    for i in range(total):
        cache_pos = i + prompt_len
        if tokens_a[i] == tokens_b[i]:
            matches += 1
        else:
            divergences.append((i, cache_pos, "ANCHOR" if cache_pos < ANCHOR_TOKENS else "TILE"))

    first_div = divergences[0] if divergences else None
    first_div_cache = first_div[1] if first_div else -1

    return {
        "total_tokens": total,
        "matches": matches,
        "match_rate": matches / total if total > 0 else 0.0,
        "first_divergence": first_div,           # (output_index, cache_pos, region)
        "first_divergence_cache_pos": first_div_cache,
        "divergences": divergences,
        "identical": matches == total,
    }


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def interpret(first_div_cache: int, prompt_len: int) -> str:
    """Decisive interpretation of WHERE error begins, in cache-position space.

    first_div_cache == -1 -> identical, no divergence.
    first_div_cache <  ANCHOR_TOKENS -> the first differing cache position is in
        the FP32 anchor region -> the ANCHOR read/write path is structurally
        wrong (kernel bug), not tile precision.
    first_div_cache >= ANCHOR_TOKENS -> the anchors all matched and the first
        difference is an NVFP4 tile -> TILE QUANTIZATION error (scale
        granularity / E2M1 spacing), the kernel structure is fine.
    """
    if first_div_cache < 0:
        return "IDENTICAL"
    if first_div_cache < ANCHOR_TOKENS:
        return ("ANCHOR_PATH_BUG")
    return "TILE_QUANTIZATION"


def main():
    parser = argparse.ArgumentParser(description="NVFP4 KV ANCHOR-ONLY discriminating probe (cache-pos corrected)")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--tokens", type=int, default=200, help="Tokens to generate")
    parser.add_argument("--prompt", default="The cap", help="SHORT prompt (<=3 tokens) -> fills anchors only")
    args = parser.parse_args()

    print(f"=== NVFP4 KV REPLAY TEST — ANCHOR-ONLY PROBE (CACHE-POS CORRECTED) ===")
    print(f"Model: {args.model}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Tokens: {args.tokens}")
    print(f"Anchor boundary: cache positions 0..{ANCHOR_TOKENS-1} = F32 anchors; >= {ANCHOR_TOKENS} = NVFP4 tiles")
    print()

    print("Running F32 KV baseline...")
    f32_text, f32_out, f32_err = run_bench(args.model, args.tokens, nvfp4=False, prompt=args.prompt)
    if f32_text in ("TIMEOUT", "ERROR"):
        print("FATAL: F32 baseline failed")
        sys.exit(1)

    print("Running NVFP4 KV...")
    nvfp4_text, nvfp4_out, nvfp4_err = run_bench(args.model, args.tokens, nvfp4=True, prompt=args.prompt)
    if nvfp4_text in ("TIMEOUT", "ERROR"):
        print("FATAL: NVFP4 run failed")
        sys.exit(1)

    # ── Actual prompt token count (NOT assumed) ─────────────────────────────
    prompt_len = get_prompt_token_count(f32_err, f32_out)
    if prompt_len < 0:
        prompt_len = get_prompt_token_count(nvfp4_err, nvfp4_out)
    if prompt_len < 0:
        # Guard: prompt "The cap" is 2 whitespace words; use that only as last resort.
        prompt_len = len(args.prompt.split())
        print("WARNING: could not parse 'prompt eval time = ... / N tokens' from llama-cli "
              f"output; using heuristic prompt token count = {prompt_len} (VERIFY ME).")
    print(f"\nACTUAL PROMPT TOKEN COUNT: {prompt_len}")
    print(f"  -> prompt occupies cache positions 0..{prompt_len-1}")
    print(f"  -> output token i occupies cache position i + {prompt_len}")
    print(f"  -> first NVFP4 tile (if any) at cache position {ANCHOR_TOKENS} (output token {ANCHOR_TOKENS - prompt_len})")
    print()

    f32_tokens = extract_tokens(f32_text, args.prompt)
    nvfp4_tokens = extract_tokens(nvfp4_text, args.prompt)

    result = compute_token_match(f32_tokens, nvfp4_tokens, prompt_len)
    result["f32_hash"] = compute_hash(f32_text)
    result["nvfp4_hash"] = compute_hash(nvfp4_text)

    print(f"F32 tokens: {len(f32_tokens)}, NVFP4 tokens: {len(nvfp4_tokens)}")
    print(f"Match rate: {result['match_rate']*100:.1f}% ({result['matches']}/{result['total_tokens']})")

    if result["identical"]:
        print("\nPASS: 100% token match — anchors AND tiles exact, no divergence observed.")
        print("  (Cache positions beyond the prompt all matched bit-for-bit.)")
        sys.exit(0)

    fd = result["first_divergence"]  # (output_index, cache_pos, region)
    print(f"\nFirst divergence:")
    print(f"  output token {fd[0]}  ->  cache position {fd[1]}  ->  {fd[2]} region")

    # ── Granular output: first 10 divergent cache positions ────────────────
    print(f"\nFirst {min(10, len(result['divergences']))} divergent cache positions:")
    for i, (out_idx, cpos, region) in enumerate(result["divergences"][:10]):
        marker = " <== first divergence" if i == 0 else ""
        print(f"  #{i+1}: output token {out_idx:>3} -> cache pos {cpos:>3} [{region}]{marker}")

    ctx = 5
    a = max(0, fd[0] - ctx)
    b = min(result["total_tokens"], fd[0] + ctx)
    print(f"\n  F32 context:   {' '.join(f32_tokens[a:b])}")
    print(f"  NVFP4 context: {' '.join(nvfp4_tokens[a:b])}")

    # ── Decisive interpretation in cache-position space ─────────────────────
    verdict = interpret(result["first_divergence_cache_pos"], prompt_len)
    print(f"\n=== INTERPRETATION: {verdict} ===")
    if verdict == "IDENTICAL":
        print("  No divergence. Anchor + tile paths both bit-exact.")
    elif verdict == "ANCHOR_PATH_BUG":
        print(f"  First divergent CACHE POSITION = {result['first_divergence_cache_pos']} < {ANCHOR_TOKENS}.")
        print("  The error starts INSIDE the FP32 anchor region -> the anchor read/write")
        print("  path is structurally WRONG (kernel bug), NOT tile quantization.")
        print("  This would explain a short-prompt run diverging EARLY/WORSE than a long one:")
        print("  generated anchor positions (cache 2,3) are being read as garbage.")
    elif verdict == "TILE_QUANTIZATION":
        print(f"  First divergent CACHE POSITION = {result['first_divergence_cache_pos']} >= {ANCHOR_TOKENS}.")
        print("  All FP32 anchor positions matched exactly; the first difference is an")
        print("  NVFP4 tile -> residual error is TILE PRECISION (scale granularity / E2M1")
        print("  spacing), NOT the kernel structure. Future fix targets tile precision.")
        # Sanity for the short-vs-long puzzle:
        if prompt_len < ANCHOR_TOKENS and result["first_divergence_cache_pos"] == ANCHOR_TOKENS:
            print("  NOTE: with this short prompt the very FIRST tile (cache pos "
                  f"{ANCHOR_TOKENS}) diverges. If the long-prompt run matches better, the tile "
                  "error here is position-locality-sensitive (e.g. scale-train/outlier "
                  "dependence on the few preceding keys) — still tile precision, not anchor.")
    sys.exit(1)


if __name__ == "__main__":
    main()
