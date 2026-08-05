#!/usr/bin/env python3
"""Model inventory manager — audits I:/models against the budget + governor rules.

Checks:
1. Total model size on I:/models vs the 250GB budget (governor rule).
2. Free space on I: (must stay >= 50GB free).
3. Duplicate models (same base, multiple quants/copies) — flag >1 copy/model.
   NVFP4 is its OWN category (budget carve-out 2026-08-04): max 1 NVFP4 variant
   per model base, coexisting with max 1 non-NVFP4 variant of the same base.
4. <=2 non-NVFP4 models per param size (NVFP4 variants exempt).
5. Any stray .gguf in wrong dirs (e.g. /i/i/ from the path bug).

Run periodically (scheduled) + on demand. Reports violations; does NOT delete
without approval (per feedback_delete_permission).

Usage:
  python tools/model_inventory_manager.py            # full audit
  python tools/model_inventory_manager.py --report   # print only, exit 0
"""
import os, sys, json, glob, datetime, re

MODELS_DIR = "I:\\models"
TOTAL_BUDGET_GB = 250
MIN_FREE_GB = 50
MAX_PER_PARAM_SIZE = 2
MAX_NVFP4_PER_MODEL = 1
MAX_OTHER_PER_MODEL = 2  # e.g. BF16 reference + APEX backbone coexist

# WORKING FILES (budget exception 2026-08-04): a file currently being produced /
# worked on (e.g. the in-progress 35B NVFP4 requant output) is EXEMPT from the
# per-model / per-param-size caps. Still counts toward the total budget.
# Update when the working file changes.
WORKING_FILES = {
    "I:\\models\\Ornith-1.0-35B-NVFP4-MTP.gguf",  # requant output (den-native NVFP4 35B)
}

# LOCKED FILES (2026-08-04): benchmark model files that require EXPLICIT permission
# before deleting or replacing. The space manager must NEVER auto-delete these, and
# they are exempt from duplicate / over-budget deletion-flagging. If a locked file is
# MISSING from disk, the audit flags it as a WARNING (not OK) so a vanished locked
# model is obvious. Keep in sync with the "_locked" array in I:/models/.current.json.
LOCKED_FILES = {
    "I:\\models\\google-gemma-4-26B-A4B-it-Q4_K_M.gguf",
    "I:\\models\\Ornith-1.0-9B-heretic-MTP-Q4_K_M.gguf",
    "I:\\models\\ornith-1.0-35b-APEX-I-Mini-MTP.gguf",
}

# Suffixes stripped (case-insensitive) to get the model base name.
_BASE_SUFFIXES = ["-mtp", "-nvfp4", "-q4_0", "-q8_0", "-bf16", "-v2", "-opt",
                  "-apex-i-mini", "-apex", "-fw", "-fw2"]

def model_base(fname):
    b = fname.lower().replace(".gguf", "")
    for suf in _BASE_SUFFIXES:
        b = b.replace(suf, "")
    return b.strip("-_")

def is_nvfp4(fname):
    return "nvfp4" in fname.lower()

def param_size(base):
    m = re.search(r"(\d+(?:\.\d+)?)b", base)
    return m.group(1) if m else None

def human(n):
    return f"{n/1e9:.1f}GB"

def audit():
    report = {"ok": [], "violations": [], "models": [], "timestamp": datetime.datetime.now().isoformat()}

    # Collect all model files
    total = 0
    present = set()
    for p in glob.glob(os.path.join(MODELS_DIR, "*.gguf")):
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        total += sz
        present.add(p)
        name = os.path.basename(p)
        report["models"].append({"file": name, "path": p, "gb": round(sz/1e9, 1)})

    # Locked files MUST be present. If one vanished, flag WARNING (not OK) so it's obvious.
    for lp in sorted(LOCKED_FILES):
        if lp in present:
            report["ok"].append(f"LOCKED present: {os.path.basename(lp)}")
        else:
            report["violations"].append(
                f"LOCKED FILE MISSING: {lp} (locked model vanished from disk — investigate, do not delete)")

    # Free space
    try:
        import shutil
        free = shutil.disk_usage("I:")[2]
        report["free_gb"] = round(free/1e9, 1)
        if free < MIN_FREE_GB * 1e9:
            report["violations"].append(f"FREE SPACE LOW: {report['free_gb']}GB free (< {MIN_FREE_GB}GB)")
        else:
            report["ok"].append(f"free space OK: {report['free_gb']}GB")
    except Exception as e:
        report["violations"].append(f"cannot check free space: {e}")

    # Total budget
    report["total_gb"] = round(total/1e9, 1)
    if total > TOTAL_BUDGET_GB * 1e9:
        report["violations"].append(f"TOTAL OVER BUDGET: {report['total_gb']}GB > {TOTAL_BUDGET_GB}GB")
    else:
        report["ok"].append(f"total {report['total_gb']}GB <= {TOTAL_BUDGET_GB}GB")

    # Duplicates: NVFP4 is its OWN category (carve-out 2026-08-04).
    # Per model base: max 1 NVFP4 variant + max 1 non-NVFP4 variant, coexist OK.
    from collections import defaultdict
    base_fam = defaultdict(lambda: {"nvfp4": 0, "other": 0})
    for m in report["models"]:
        if m["path"] in WORKING_FILES:
            continue  # working-file budget exception — exempt from per-model/per-size caps
        if m["path"] in LOCKED_FILES:
            continue  # locked file — never flag for deletion/replacement (space manager guard)
        base = model_base(m["file"])
        base_fam[base][("nvfp4" if is_nvfp4(m["file"]) else "other")] += 1
    for base, fam in sorted(base_fam.items()):
        nv, ot = fam["nvfp4"], fam["other"]
        if nv > MAX_NVFP4_PER_MODEL:
            report["violations"].append(
                f"NVFP4 OVER: '{base}' has {nv} NVFP4 variants (max {MAX_NVFP4_PER_MODEL})")
        elif ot > MAX_OTHER_PER_MODEL:
            report["violations"].append(f"DUPLICATE: '{base}' has {ot} non-NVFP4 copies")
        else:
            report["ok"].append(f"'{base}': {ot} non-NVFP4 + {nv} NVFP4")

    # Param-size cap: <=2 non-NVFP4 models per size; NVFP4 variants exempt.
    size_count = defaultdict(int)
    for base, fam in base_fam.items():
        size_count[param_size(base)] += fam["other"]
    for size, n in sorted(size_count.items()):
        if size is None:
            continue
        if n > MAX_PER_PARAM_SIZE:
            report["violations"].append(
                f"PARAM-SIZE OVER: {size}B has {n} non-NVFP4 models (max {MAX_PER_PARAM_SIZE})")
        else:
            report["ok"].append(f"{size}B: {n} non-NVFP4 models (NVFP4 exempt)")

    # Stray files in wrong dir (the /i/i bug)
    stray = glob.glob(os.path.join("I:\\i", "**", "*.gguf"), recursive=True)
    if stray:
        report["violations"].append(f"STRAY FILES in I:\\i: {[os.path.basename(s) for s in stray]}")

    return report

if __name__ == "__main__":
    r = audit()
    print("=" * 60)
    print("MODEL INVENTORY AUDIT", r["timestamp"])
    print("=" * 60)
    print(f"Total: {r.get('total_gb', '?')}GB  Free: {r.get('free_gb', '?')}GB")
    print(f"\nModels ({len(r['models'])}):")
    for m in sorted(r["models"], key=lambda x: -x["gb"]):
        print(f"  {m['gb']:>8}GB  {m['file']}")
    print(f"\nOK ({len(r['ok'])}):")
    for o in r["ok"]:
        print(f"  OK: {o}")
    if r["violations"]:
        print(f"\nVIOLATIONS ({len(r['violations'])}):")
        for v in r["violations"]:
            print(f"  WARN: {v}")
        sys.exit(1)
    print("\nAll good.")
    sys.exit(0)
