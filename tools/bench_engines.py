#!/usr/bin/env python3
"""bench_engines.py — multi-engine benchmark harness for Project Den.

Runs the SAME GGUF through each engine binary, measures eval tok/s from
llama_print_timings, and captures generated text for a coherence check.

MODEL SEMANTICS (HARD — do not cross):
  * 9B  Ornith-1.0-9B-NVFP4-v2  = DENSE.  Benchmark as dense. No MoE flags.
  * 35B Ornith-1.0-35B-NVFP4-v2 = MoE.    Benchmark as MoE. MoE target.

Usage:
  python bench_engines.py --engine <name> [--model dense|moe] [--dry-run]
  python bench_engines.py --all --dry-run     # print commands only
"""

import argparse
import os
import re
import subprocess
import sys
import time

PY = r"C:\Den\den-py314\Scripts\python.exe"

# CUDA runtime DLLs (cudart64_13.dll, cublas64_13.dll, ...) live in the pip
# cu13 package's bin\x86_64 subdir, NOT in bin or System32. Engines fail with
# STATUS_DLL_NOT_FOUND unless this is on PATH at launch.
PIP_CUDA_BIN64 = r"C:\Users\james\AppData\Local\Programs\Python\Python314\Lib\site-packages\nvidia\cu13\bin\x86_64"
PIP_CUDA_BIN = r"C:\Users\james\AppData\Local\Programs\Python\Python314\Lib\site-packages\nvidia\cu13\bin"


def _env_with_cuda(engine_bin_dir=None):
    env = os.environ.copy()
    parts = []
    if engine_bin_dir:
        parts.append(engine_bin_dir)
    parts += [PIP_CUDA_BIN64, PIP_CUDA_BIN]
    env["PATH"] = os.pathsep.join(parts) + os.pathsep + env.get("PATH", "")
    return env

# 4 benchmark models (2026-08-04). Each run on all 4 engines unless SKIP.
MODELS = {
    # gemma-26B A4B Q4_K_M: gemma4 arch — den's flash-attn mma kernel does NOT
    # dispatch it (garbles, not n/a) → excluded for den. 2026-08-04: gemma12
    # Q5_K_M REMOVED (user: keep gemmaQAT QAT over it).
    "gemma26":  r"I:\models\google-gemma-4-26B-A4B-it-Q4_K_M.gguf",
    # ornith-9B Q4_K_M: standard quant, all 4 engines run it (den normal workhorse).
    "ornith9":  r"I:\models\Ornith-1.0-9B-heretic-MTP-Q4_K_M.gguf",
    # ornith-35B mini APEX MoE: loads+runs on all 4 engines.
    "ornith35": r"I:\models\ornith-1.0-35b-APEX-I-Mini-MTP.gguf",
}

# Per-engine model exclusions → n/a (engine can't run that model this round).
# Den skips gemma4-arch models (flash-attn kernel gap). Marked n/a, not failed.
SKIP = {
    "den":      {"gemma26"},
    "beellama": set(),
    "ik":       set(),
    "mainline": set(),
}

# Engine binary paths. den = baseline.
ENGINES = {
    "den":      r"I:\den_llama.cpp\build_ninja\bin\llama-cli.exe",
    "beellama": r"I:\beellama.cpp\build_bench\bin\llama-cli.exe",
    "ik":       r"I:\ik_llama.cpp\build_bench\bin\llama-cli.exe",
    "mainline": r"I:\den_llama.cpp\llama_upstream\build_bench\bin\llama-cli.exe",
}

# beellama + mainline are DLL-stub executables — their impl DLLs live in their
# own bin dir; must be on PATH at launch or you get exit 53 / DLL-not-found.
ENGINE_BIN_DIRS = {
    "den":      r"I:\den_llama.cpp\build_ninja\bin",
    "beellama": r"I:\beellama.cpp\build_bench\bin",
    "ik":       r"I:\ik_llama.cpp\build_bench\bin",
    "mainline": r"I:\den_llama.cpp\llama_upstream\build_bench\bin",
}

PROMPT = "The capital of France is"
N_TOKENS = 32  # default generation length (average tok/s)
PEAK_N = 12    # short-run length -> peak/highest tok/s (2026-08-04, user order)

# Offload per model: gemma/ornith9 fit 16 GB -> full offload; ornith35 (14.4GB) -> partial.
# Offload per model: gemma12/ornith9 fit 16 GB -> full offload; gemma26 (16.8GB) +
# ornith35 (14.4GB) are partial. 2026-08-04: gemmaQAT swapped for gemma26.
DEFAULT_NGL = {"gemma26": 40, "ornith9": 99, "ornith35": 40}

# Extra per-engine flags (e.g. beellama/ik specific decode paths). Empty by
# default — add ONLY after confirming the flag is valid for that engine.
EXTRA_FLAGS = {
    # den runs --mode normal (the fork workhorse; super/NVFP4 backburnered)
    "den":      ["--mode", "normal"],
    "beellama": [],
    "ik":       [],
    "mainline": [],
}


def parse_timings(text):
    """Extract llama_print_timings eval + prompt tok/s. Returns dict."""
    out = {}
    m = re.search(r"eval time\s*=\s*([\d.]+) ms /\s*(\d+) runs?\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)", text)
    if m:
        out["eval_ms"], out["eval_runs"], out["ms_per_tok"], out["eval_tok_s"] = \
            float(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
    # ik_llama.cpp prints "decode time" for the token-eval stage (not "eval
    # time"). Fall back to it so ik's real tok/s isn't misreported as n/a/0.
    if "eval_tok_s" not in out:
        m = re.search(r"decode time\s*=\s*([\d.]+) ms /\s*(\d+) runs?\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)", text)
        if m:
            out["eval_ms"], out["eval_runs"], out["ms_per_tok"], out["eval_tok_s"] = \
                float(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
    m = re.search(r"prompt eval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens?\s*\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)", text)
    if m:
        out["prompt_ms"], out["prompt_toks"], out["prompt_tok_s"] = \
            float(m.group(1)), int(m.group(2)), float(m.group(3))
    return out


def _kill_stale_llama():
    """Nuke any leftover llama-cli.exe so jobs stay truly sequential and VRAM
    is freed between jobs. The prior subprocess.run-timeout bug orphaned hung
    children (holding VRAM) that crashed subsequent jobs; kill them up front."""
    try:
        subprocess.run(["taskkill", "/F", "/T", "/IM", "llama-cli.exe"],
                       capture_output=True, text=True, timeout=30)
    except Exception:  # noqa
        pass


def run_one(engine, model_key, n_tokens=N_TOKENS, ngl=None, extra=None, timeout=900, t2=None):
    if t2 is not None:
        timeout = t2
    if model_key in SKIP.get(engine, set()):
        return {"engine": engine, "model": model_key, "n/a": True, "elapsed": 0}
    binary = ENGINES[engine]
    model = MODELS[model_key]
    if ngl is None:
        ngl = DEFAULT_NGL[model_key]
    extra = extra or EXTRA_FLAGS.get(engine, [])

    cmd = [binary, "-m", model, "-p", PROMPT, "-n", str(n_tokens),
           "-ngl", str(ngl)] + extra
    print("CMD:", " ".join(cmd))
    t0 = time.time()

    # Ensure a clean GPU before launching (kills any orphaned llama-cli from a
    # prior timed-out job or a stray parallel harness).
    _kill_stale_llama()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=_env_with_cuda(ENGINE_BIN_DIRS[engine]))
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()          # kill the child instead of orphaning it
        proc.communicate()   # reap it
        _kill_stale_llama()  # nuke any grandchildren too
        return {"engine": engine, "model": model_key, "error": "TIMEOUT", "elapsed": round(time.time() - t0, 1)}
    elapsed = time.time() - t0
    timings = parse_timings(out)
    timings.update({
        "engine": engine, "model": model_key, "rc": proc.returncode,
        "elapsed_s": round(elapsed, 1),
    })
    # coherence: strip the echoed prompt, take first generated line
    gen = [ln for ln in (out or "").splitlines()
           if ln.strip() and "llama_" not in ln and "print_info" not in ln and "load" not in ln.lower()]
    timings["gen_sample"] = " | ".join(gen[-3:])[:200]
    return timings


def run_microbench(engine, model_key, ngl=None):
    """Canonical llama-bench microbench for an engine+model (2026-08-04, user order).
    Runs pp512/1024/2048 + tg128/256, parses the markdown table into {test: tok_s}.
    """
    import re
    binary = os.path.join(ENGINE_BIN_DIRS[engine], "llama-bench.exe")
    if not os.path.exists(binary):
        return {"engine": engine, "model": model_key, "error": "no llama-bench binary"}
    model = MODELS[model_key]
    if ngl is None:
        ngl = DEFAULT_NGL[model_key]
    cmd = [binary, "-m", model, "-p", "512", "1024", "2048",
           "-n", "128", "256", "-ngl", str(ngl)]
    print("CMD:", " ".join(cmd))
    _kill_stale_llama()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=_env_with_cuda(ENGINE_BIN_DIRS[engine]))
    try:
        out, _ = proc.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.communicate(); _kill_stale_llama()
        return {"engine": engine, "model": model_key, "error": "TIMEOUT"}
    rows = {}
    for ln in (out or "").splitlines():
        parts = [p.strip() for p in ln.split("|")] if "|" in ln else []
        if len(parts) < 3:
            continue
        test, ts = parts[-2], parts[-1]
        if re.fullmatch(r"(pp|tg)\d+", test) and re.fullmatch(r"\d+(\.\d+)?", ts):
            rows[test] = float(ts)
    return {"engine": engine, "model": model_key, "rc": proc.returncode, "bench": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=list(ENGINES))
    ap.add_argument("--model", choices=list(MODELS), default="dense")
    ap.add_argument("--all", action="store_true", help="run every engine on every model")
    ap.add_argument("--dry-run", action="store_true", help="print commands only, don't run")
    ap.add_argument("--ngl", type=int, default=None, help="override layers offloaded")
    ap.add_argument("--n", type=int, default=N_TOKENS)
    ap.add_argument("--timeout", type=int, default=900, help="per-job timeout seconds")
    ap.add_argument("--engines", default=None, help="comma-separated engines to run (e.g. beellama,ik); default all")
    ap.add_argument("--extra", nargs="*", default=[], help="extra flags passed to binary")
    ap.add_argument("--micro", action="store_true",
                    help="run canonical llama-bench microbench (pp512/1024/2048, tg128/256)")
    args = ap.parse_args()

    # validate binaries exist up front
    for name, path in ENGINES.items():
        import os
        if not os.path.exists(path):
            print(f"[warn] {name}: binary NOT built: {path}")

    jobs = []
    engine_set = list(ENGINES)
    if args.engines:
        engine_set = [e.strip() for e in args.engines.split(",") if e.strip() in ENGINES]
    if args.all:
        for eng in engine_set:
            for mod in MODELS:
                jobs.append((eng, mod))
    else:
        jobs = [(args.engine or "den", args.model)]

    if args.dry_run:
        for eng, mod in jobs:
            cmd = [ENGINES[eng], "-m", MODELS[mod], "-p", PROMPT,
                   "-n", str(args.n), "-ngl", str(args.ngl or DEFAULT_NGL[mod])] \
                   + EXTRA_FLAGS.get(eng, []) + args.extra
            print(" ".join(cmd))
        return 0

    results = []
    for eng, mod in jobs:
        print(f"\n=== {eng} / {mod} ===")
        if args.micro:
            try:
                r = run_microbench(eng, mod, ngl=args.ngl)
                results.append(r)
                if "error" in r:
                    print("ERROR:", r["error"])
                    continue
                for k in ("pp512", "pp1024", "pp2048", "tg128", "tg256"):
                    print(f"  {k:6s} {r['bench'].get(k, 0):8.2f} tok/s")
            except Exception as e:  # noqa
                print("EXC:", repr(e))
            continue
        try:
            avg = run_one(eng, mod, n_tokens=args.n, ngl=args.ngl, extra=args.extra, t2=args.timeout)
            if avg.get("n/a"):
                results.append(avg)
                print("  n/a (feature-set gap)")
                continue
            if "error" in avg:
                results.append(avg)
                print("ERROR:", avg["error"])
                continue
            # Peak/highest tok/s via a short run (PEAK_N=12). 2026-08-04.
            peak = run_one(eng, mod, n_tokens=PEAK_N, ngl=args.ngl, extra=args.extra, t2=args.timeout)
            avg["peak_tok_s"] = peak.get("eval_tok_s", 0) if "error" not in peak else 0
            results.append(avg)
            print(f"  avg eval: {avg.get('eval_tok_s', 'N/A')} tok/s  "
                  f"peak: {avg.get('peak_tok_s', 'N/A')} tok/s  "
                  f"prompt: {avg.get('prompt_tok_s', 'N/A')} tok/s  "
                  f"rc={avg['rc']}  {avg['elapsed_s']}s")
            print(f"  gen: {avg['gen_sample']}")
        except Exception as e:  # noqa
            print("EXC:", repr(e))

    if args.micro:
        print("\n=== MICRO SUMMARY (llama-bench tok/s; n/a = gap) ===")
        for r in results:
            if "error" in r:
                print(f"  {r['engine']:10s} {r['model']:8s}  {r['error']}")
            else:
                b = r.get("bench", {})
                print(f"  {r['engine']:10s} {r['model']:8s}  " +
                      "  ".join(f"{k} {b.get(k, 0):7.2f}" for k in ("pp512", "pp1024", "pp2048", "tg128", "tg256")))
    else:
        print("\n=== SUMMARY (avg eval tok/s | peak tok/s; n/a = feature-set gap) ===")
        for r in results:
            if r.get("n/a"):
                print(f"  {r['engine']:10s} {r['model']:8s}  n/a")
            elif "error" in r:
                print(f"  {r['engine']:10s} {r['model']:8s}  {r['error']}")
            else:
                print(f"  {r['engine']:10s} {r['model']:8s}  avg {r.get('eval_tok_s', 0):7.2f}  peak {r.get('peak_tok_s', 0):7.2f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
