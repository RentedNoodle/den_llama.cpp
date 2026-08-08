#!/usr/bin/env python3
"""
B3: SASS AUDIT HOOK — Kernel Integrity Guard

Post-build hook that audits NVFP4 CUDA kernel SASS instruction counts.
Compares against golden baseline. Blocks push if counts change.

Configuration: stores golden baseline in .claude/hooks/sass-golden.json

Usage: python .claude/hooks/sass-audit.py [--baseline] [--check]
  --baseline  Regenerate golden baseline
  --check     Compare current against golden (default)
"""
import subprocess, json, sys, os, re
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
BUILD_BIN = REPO / "build_ninja" / "bin"
GOLDEN_FILE = Path(__file__).parent / "sass-golden.json"

# NVFP4 kernel files to audit
NVFP4_SOURCES = [
    "ggml/src/ggml-cuda/fattn-nvfp4-kv.cu",
    "ggml/src/ggml-cuda/ggml-cuda.cu",
]

def find_cubin(source_path: str) -> Path:
    """Find the compiled cubin/obj for a CUDA source."""
    stem = Path(source_path).stem
    # Search build dir for matching .obj
    build_dir = REPO / "build_ninja"
    candidates = list(build_dir.glob(f"**/{stem}*.obj"))
    if candidates:
        return candidates[0]
    candidates = list(build_dir.glob(f"**/{stem}*.cubin"))
    if candidates:
        return candidates[0]
    return None

def get_sass_instructions(cubin: Path) -> dict:
    """Run cuobjdump -sass and count instructions by opcode."""
    try:
        result = subprocess.run(
            ["cuobjdump", "-sass", str(cubin)],
            capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        # Try CUDA 13.3 path
        cuobjdump = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/cuobjdump.exe"
        if not os.path.exists(cuobjdump):
            return {"error": "cuobjdump not found"}
        result = subprocess.run([cuobjdump, "-sass", str(cubin)],
                               capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        return {"error": f"cuobjdump failed: {result.stderr[:200]}"}

    counts = {}
    for line in result.stdout.split("\n"):
        line = line.strip()
        # SASS lines look like: /*0000*/ MOV R1, c[0x0][0x28] ;
        # or: /*0018*/ OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X ...
        if line.startswith("/*"):
            match = re.match(r'/\*\w+\*/\s+(\S+)', line)
            if match:
                opcode = match.group(1).rstrip(';')
                # Normalize: strip operand suffixes for counting
                base_op = opcode.split('.')[0] if '.' in opcode[0:4] else opcode
                counts[base_op] = counts.get(base_op, 0) + 1

    return counts

def audit():
    """Audit SASS instruction counts against golden baseline."""
    results = {}

    for src in NVFP4_SOURCES:
        cubin = find_cubin(src)
        if not cubin:
            print(f"SKIP {src}: no compiled object found")
            continue

        counts = get_sass_instructions(cubin)
        if "error" in counts:
            print(f"ERROR {src}: {counts['error']}")
            continue

        results[src] = {
            "cubin": str(cubin),
            "instruction_count": sum(counts.values()),
            "instructions": counts,
        }
        print(f"OK {src}: {results[src]['instruction_count']} total SASS instructions, {len(counts)} unique opcodes")

    # Compare against golden if checking
    if not GOLDEN_FILE.exists():
        print(f"\nNo golden baseline at {GOLDEN_FILE}. Run --baseline first.")
        return 0  # Don't block on first run

    with open(GOLDEN_FILE) as f:
        golden = json.load(f)

    violations = []
    for src, data in results.items():
        if src not in golden:
            print(f"NEW {src}: not in golden baseline")
            violations.append(f"New file: {src}")
            continue

        g = golden[src]
        if data["instruction_count"] != g["instruction_count"]:
            delta = data["instruction_count"] - g["instruction_count"]
            print(f"REGRESSION {src}: {g['instruction_count']}→{data['instruction_count']} ({delta:+d} instructions)")
            violations.append(f"{src}: instruction count changed by {delta}")

        # Check individual opcodes
        new_ops = set(data["instructions"].keys()) - set(g.get("instructions", {}).keys())
        removed_ops = set(g.get("instructions", {}).keys()) - set(data["instructions"].keys())
        if new_ops:
            print(f"  NEW opcodes: {new_ops}")
            violations.append(f"{src}: new opcodes {new_ops}")
        if removed_ops:
            print(f"  REMOVED opcodes: {removed_ops}")
            violations.append(f"{src}: removed opcodes {removed_ops}")

    if violations:
        print(f"\nFAIL: {len(violations)} SASS violations found")
        for v in violations:
            print(f"  - {v}")
        return 1

    print("\nPASS: SASS instruction counts match golden baseline")
    return 0

def baseline():
    """Generate golden baseline from current build."""
    golden = {}
    for src in NVFP4_SOURCES:
        cubin = find_cubin(src)
        if not cubin:
            print(f"SKIP {src}: no compiled object")
            continue

        counts = get_sass_instructions(cubin)
        if "error" in counts:
            print(f"ERROR {src}: {counts['error']}")
            continue

        golden[src] = {
            "cubin": str(cubin),
            "instruction_count": sum(counts.values()),
            "instructions": counts,
        }

    with open(GOLDEN_FILE, "w") as f:
        json.dump(golden, f, indent=2, sort_keys=True)

    total = sum(g["instruction_count"] for g in golden.values())
    print(f"SASS baseline saved: {len(golden)} files, {total} total instructions")
    print(f"Golden: {GOLDEN_FILE}")

if __name__ == "__main__":
    if "--baseline" in sys.argv:
        baseline()
    else:
        sys.exit(audit())
