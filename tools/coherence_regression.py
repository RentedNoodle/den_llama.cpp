#!/usr/bin/env python3
"""Coherence regression suite — Paris + 4 unique tests (math/reasoning/logic/code).
Spawns llama-cli per prompt (temp 0, deterministic), checks for expected tokens.
Usage: python tools/coherence_regression.py [engine] [model_path]
"""
import os, subprocess, sys

BINS = {
    "den":      r"I:\den_llama.cpp\build_ninja\bin",
    "ik":       r"I:\ik_llama.cpp\build_bench\bin",
    "beellama": r"I:\beellama.cpp\build_bench\bin",
    "mainline": r"I:\den_llama.cpp\llama_upstream\build_bench\bin",
}
CUDA = r"C:\Users\james\AppData\Local\Programs\Python\Python314\Lib\site-packages\nvidia\cu13\bin\x86_64"
DEFAULT_MODEL = r"I:\models\Ornith-1.0-9B-heretic-MTP-Q4_K_M.gguf"

# (name, prompt, [expected substrings — any match = pass])
TESTS = [
    ("fact",      "The capital of France is",                       ["Paris"]),
    ("math",      "What is 17 times 23?",                           ["391"]),
    ("reasoning", "A train travels 60 km/h for 3 hours. How far does it travel in km?",
                                                                    ["180"]),
    ("logic",     "If all birds have feathers and penguins are birds, do penguins have feathers? Answer yes or no.",
                                                                    ["yes"]),
    ("code",      "Complete the Python function so it returns a + b:\ndef add(a, b):",
                                                                    ["a + b", "a+b", "return a + b", "return a+b"]),
]


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "den"
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    bindir = BINS[engine]
    exe = os.path.join(bindir, "llama-cli.exe")
    if not os.path.exists(exe):
        print(f"ERROR: no llama-cli at {exe}"); return 1
    env = os.environ.copy()
    env["PATH"] = bindir + ";" + CUDA + ";" + env.get("PATH", "")

    passed = total = 0
    for name, prompt, expects in TESTS:
        total += 1
        try:
            p = subprocess.run([exe, "-m", model, "-p", prompt, "-n", "24",
                                "--temp", "0", "-ngl", "99"],
                               capture_output=True, text=True, env=env, timeout=120)
            out = p.stdout + p.stderr
            ok = any(e.lower() in out.lower() for e in expects)
        except Exception as ex:  # noqa
            ok, out = False, f"EXC {ex!r}"
        passed += ok
        tail = out.strip()[-140:].replace("\n", " ")
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<10} expect {expects}  ->  ...{tail!r}")

    print(f"\nCOHERENCE: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
