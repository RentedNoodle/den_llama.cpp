#!/usr/bin/env python3
"""Parity gate — CPU (-ngl 0) vs GPU (-ngl 99) bit-exact coherence check.

Phase B2 of the Borg plan: the regression gate for the NVFP4 decode fix.
Runs the SAME prompt on the CPU path and the GPU path with DEN_EVAL_DUMP=1,
parses the "EVAL <name> n=<elems> rms=... h=<fnv1a>" lines emitted at
src/llama.cpp:4706, and asserts:

  (a) the greedy token streams (stdout) are identical, AND
  (b) every EVAL FNV-1a hash matches per (tensor name, occurrence index).

A divergence fails the gate (non-zero exit) — coherence is non-regressable.
Bit-exact h= equality is strictly stronger than cosine: equal hashes mean equal
float bit-patterns mean identical argmax. This is the permanent safety net for
the Blocker-1 fix once it lands.

Usage:
  python ci/parity_gate.py --exe build_win/bin/Release/llama-cli.exe \
      --model I:/models/Ornith-1.0-9B-Q8_0.gguf \
      --prompt "The capital of France is" -n 16
"""

import argparse
import os
import re
import subprocess
import sys

EVAL_RE = re.compile(
    r"^EVAL\s+(?P<name>\S+)\s+n=(?P<n>\d+)\s+rms=(?P<rms>[\d.eE+-]+)\s+"
    r"l1=(?P<l1>[\d.eE+-]+)\s+min=(?P<min>[\d.eE+-]+)\s+max=(?P<max>[\d.eE+-]+)\s+"
    r"h=(?P<h>[0-9a-fA-F]{8})\s+fp=\["
)


def run_once(exe, model, prompt, tokens, ngl, ngl_d):
    """Run llama-cli once with DEN_EVAL_DUMP=1; return (stdout, eval_lines)."""
    env = dict(os.environ)
    env["DEN_EVAL_DUMP"] = "1"
    cmd = [
        exe,
        "-m", model,
        "-p", prompt,
        "-n", str(tokens),
        "--temp", "0",
        "--top-k", "1",
        "--seed", "1",  # numeric — common.cpp:944 stoul() throws on the old "FIXED" sentinel
        "--spec-type", "none",
        "-ngld", str(ngl_d),
        "-ngl", str(ngl),
    ]
    print(f"[parity] running: ngl={ngl} tokens={tokens} prompt={prompt!r}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, env=env
        )
    except subprocess.TimeoutExpired:
        print("[parity] FATAL: run timed out (1800s)")
        sys.exit(2)
    out = proc.stdout or ""
    err = proc.stderr or ""
    eval_lines = EVAL_RE.findall(err)
    print(f"[parity] ngl={ngl}: stdout {len(out)} chars, EVAL nodes {len(eval_lines)}")
    if proc.returncode != 0:
        print(f"[parity] WARN: ngl={ngl} exit={proc.returncode}; continuing to parse")
    return out, eval_lines


def by_name(eval_lines):
    """{name: [h, ...]} preserving occurrence order."""
    d = {}
    for name, n, rms, l1, mn, mx, h in eval_lines:
        d.setdefault(name, []).append(h.lower())
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", default="build_win/bin/Release/llama-cli.exe")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("-n", "--tokens", type=int, default=16)
    ap.add_argument("--cpu-ngl", type=int, default=0)
    ap.add_argument("--gpu-ngl", type=int, default=99)
    ap.add_argument("--ngld", type=int, default=0)
    args = ap.parse_args()

    # --- CPU run ---
    cpu_out, cpu_evals = run_once(
        args.exe, args.model, args.prompt, args.tokens, args.cpu_ngl, args.ngld
    )
    # --- GPU run ---
    gpu_out, gpu_evals = run_once(
        args.exe, args.model, args.prompt, args.tokens, args.gpu_ngl, args.ngld
    )

    failures = []

    # (a) Token-stream / argmax identity: the full stdout must match.
    cpu_txt = cpu_out.strip()
    gpu_txt = gpu_out.strip()
    if cpu_txt != gpu_txt:
        # Show the first divergence for diagnosis.
        common = os.path.commonprefix([cpu_txt, gpu_txt])
        print("[parity] FAIL: token streams diverge after:")
        print(f"  ...{common[-80:]!r}")
        print(f"  CPU: ...{cpu_txt[len(common):len(common)+60]!r}")
        print(f"  GPU: ...{gpu_txt[len(common):len(common)+60]!r}")
        failures.append("token-stream")
    else:
        print("[parity] OK: token streams identical (argmax identity)")

    # (b) Per-node EVAL FNV-1a hash equality, aligned by (name, occurrence).
    cpu_n = by_name(cpu_evals)
    gpu_n = by_name(gpu_evals)
    all_names = sorted(set(cpu_n) | set(gpu_n))
    total_cmp = 0
    mism = 0
    for name in all_names:
        ch = cpu_n.get(name, [])
        gh = gpu_n.get(name, [])
        if len(ch) != len(gh):
            print(f"[parity] FAIL: node '{name}' occurrence count differs "
                  f"(CPU {len(ch)} vs GPU {len(gh)})")
            failures.append(f"count:{name}")
            continue
        for i, (a, b) in enumerate(zip(ch, gh)):
            total_cmp += 1
            if a != b:
                mism += 1
                if mism <= 12:
                    print(f"[parity] FAIL: {name}#{i} h=0x{a} (CPU) vs 0x{b} (GPU)")
    print(f"[parity] compared {total_cmp} node-evals, {mism} hash mismatches")
    if mism:
        failures.append(f"{mism} hash mismatches")

    if failures:
        print(f"[parity] GATE FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("[parity] GATE PASSED — CPU/GPU bit-exact parity holds")
    sys.exit(0)


if __name__ == "__main__":
    main()
