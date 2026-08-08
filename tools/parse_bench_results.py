#!/usr/bin/env python3
"""
parse_bench_results.py — Parse llama-bench output and build comparison tables.

Reads llama-bench output (JSON via -o json, or Markdown via -o md) from stdin
or a file. Groups results by model, extracts KV config details, detects NVFP4
status, and prints a formatted markdown comparison table with winners highlighted.

Usage:
    llama-bench -m model.gguf -o json 2>&1 | python parse_bench_results.py
    llama-bench -m model.gguf -o md   2>&1 | python parse_bench_results.py
    python parse_bench_results.py bench_output.json
    python parse_bench_results.py bench_output_1.json bench_output_2.md ...
"""
import sys
import re
import json
import os
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    model_name: str = ""
    model_filename: str = ""
    model_type: str = ""
    model_size: int = 0
    type_k: str = ""
    type_v: str = ""
    n_prompt: int = 0
    n_gen: int = 0
    avg_ts: float = 0.0
    stddev_ts: float = 0.0
    nvfp4_kv_enabled: bool = False
    n_cpu_moe: int = 0
    n_gpu_layers: int = 0
    source_file: str = ""


@dataclass
class ModelGroup:
    model_key: str  # short identifier for grouping
    display_name: str
    results: List[BenchResult] = field(default_factory=list)
    nvfp4_enabled: bool = False
    nvfp4_compression_ratio: Optional[float] = None
    nvfp4_note: str = ""


# ─── Input parsing ───────────────────────────────────────────────────────────

def decode_raw(path: str) -> str:
    """Auto-detect encoding and read file."""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", errors="replace")


def extract_nvfp4_info(text: str) -> Tuple[bool, Optional[float], str]:
    """
    Scan stderr/prose lines for NVFP4 KV cache status and compression info.
    Returns (enabled, compression_ratio, note).
    """
    enabled = False
    ratio = None
    note_parts = []

    # Patterns for NVFP4 status lines
    nvfp4_enable_patterns = [
        r"KV\s+NVFP4\s*:\s*ENABLED",
        r"NVFP4\s+KV\s+cache\s*:\s*ENABLED",
        r"nvfp4_kv\s*=\s*1",
        r"KVarN\S*\s*:\s*ENABLED",
        r"TurboQuant\s+KV\s*:\s*ENABLED",
    ]
    for pat in nvfp4_enable_patterns:
        if re.search(pat, text, re.IGNORECASE):
            enabled = True
            note_parts.append("NVFP4 KV enabled")
            break

    # Patterns for compression ratio
    ratio_patterns = [
        r"(?:KV\s+)?compression(?:\s+ratio)?\s*[:=]\s*(\d+\.?\d*)\s*[x×]",
        r"(\d+\.?\d*)\s*[x×]\s*(?:KV\s+)?compression",
        r"KVarN\S*\s+(\d+\.?\d*)\s*[x×]",
        r"TurboQuant.*?(\d+\.?\d*)\s*[x×]",
        r"NVFP4\s+KV.*?(\d+\.?\d*)\s*[x×]",
    ]
    for pat in ratio_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ratio = float(m.group(1))
            note_parts.append(f"{ratio:.1f}x compression")
            break

    note = "; ".join(note_parts) if note_parts else ""
    return enabled, ratio, note


def short_model_name(filename: str, model_type: str) -> str:
    """Derive a short, readable model name from filename and type."""
    # Prefer model_type if it's meaningful
    if model_type:
        # Strip common suffixes
        name = model_type.strip()
        # Remove size info in parentheses if present
        name = re.sub(r'\s*\(.*?\)', '', name)
        return name

    # Fallback: derive from filename
    base = os.path.basename(filename)
    # Remove extension(s)
    base = re.sub(r'\.(gguf|den|bin)$', '', base)
    # Shorten common patterns
    base = re.sub(r'-heretic', '', base)
    base = re.sub(r'-uncensored', '', base)
    base = re.sub(r'-abliterated', '-abl', base)
    # Trim very long names
    if len(base) > 50:
        # Keep key parts: model family + size + quant
        parts = base.split('-')
        kept = []
        for p in parts:
            if any(kw in p.lower() for kw in ['qwen', 'ornith', 'gemma', 'infatoshi',
                                                 'aeon', 'llama', 'mistral', 'mixtral']):
                kept.append(p)
            elif re.match(r'^\d+[Bb]$', p):
                kept.append(p)
            elif re.match(r'^[QK]\d', p.upper()) or 'nvfp4' in p.lower() or 'fp8' in p.lower():
                kept.append(p)
        if kept:
            base = '-'.join(kept)
    return base


def infer_model_key(filename: str, model_type: str) -> Tuple[str, str]:
    """
    Return (grouping_key, display_name).
    Key groups different KV configs of the same model together.
    """
    display = short_model_name(filename, model_type)

    # Normalize to a grouping key: strip quant suffixes, MTP variants
    key = display
    # Remove quantization variants from key (but not from display)
    key = re.sub(r'-(?:Q[2348]_[KLM]|IQ\d_\w+|NVFP4\S*|FP\d+|BF16|F32|F16)\b', '', key, flags=re.IGNORECASE)
    # Remove MTP / spec-decode variants
    key = re.sub(r'-(?:MTP|spec|draft)', '', key, flags=re.IGNORECASE)
    # Remove -v2, -v3 etc
    key = re.sub(r'-v\d+', '', key)
    # Collapse multiple dashes
    key = re.sub(r'-+', '-', key).strip('-')

    if not key:
        # Fallback: use first 40 chars of filename
        key = os.path.basename(filename)[:40]
        key = re.sub(r'\.(gguf|den|bin)$', '', key)

    return key, display


# ─── JSON parsing ────────────────────────────────────────────────────────────

def parse_json_input(text: str, source_file: str) -> List[BenchResult]:
    """Parse llama-bench JSON output."""
    results = []

    # Find JSON array start — skip any stderr preamble
    start = text.find("[")
    if start < 0:
        return results

    try:
        data = json.loads(text[start:])
    except json.JSONDecodeError:
        # Try extracting just the JSON array with balanced brackets
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return results
        else:
            return results

    if not isinstance(data, list):
        return results

    for entry in data:
        if not isinstance(entry, dict):
            continue

        r = BenchResult()
        r.model_filename = str(entry.get("model_filename", ""))
        r.model_type = str(entry.get("model_type", ""))
        r.model_size = int(entry.get("model_size", 0))
        r.type_k = str(entry.get("type_k", ""))
        r.type_v = str(entry.get("type_v", ""))
        r.n_prompt = int(entry.get("n_prompt", 0))
        r.n_gen = int(entry.get("n_gen", 0))
        r.avg_ts = float(entry.get("avg_ts", 0.0))
        r.stddev_ts = float(entry.get("stddev_ts", 0.0))
        r.nvfp4_kv_enabled = bool(int(entry.get("nvfp4_kv_enabled", 0)))
        r.n_cpu_moe = int(entry.get("n_cpu_moe", 0))
        r.n_gpu_layers = int(entry.get("n_gpu_layers", 0))
        r.source_file = source_file

        key, display = infer_model_key(r.model_filename, r.model_type)
        r.model_name = display

        results.append(r)

    return results


# ─── Markdown table parsing ──────────────────────────────────────────────────

def parse_markdown_input(text: str, source_file: str) -> List[BenchResult]:
    """
    Parse llama-bench markdown table output.
    Expected columns: model | size | params | backend | ngl | threads |
                      n_batch | n_ubatch | type_k | type_v | test | t/s
    """
    results = []

    lines = text.splitlines()
    table_lines: List[str] = []

    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            # Skip separator lines like |---|---|
            if re.match(r'^\|[\s\-:|]+\|$', stripped):
                if table_lines:
                    # Previous line was header
                    header = table_lines[-1]
                    table_lines = [header]  # keep only header
                continue
            table_lines.append(stripped)
        else:
            # End of table
            if table_lines:
                results.extend(_parse_md_table_lines(table_lines, source_file))
                table_lines = []

    # Handle final table
    if table_lines:
        results.extend(_parse_md_table_lines(table_lines, source_file))

    return results


def _parse_md_table_lines(table_lines: List[str], source_file: str) -> List[BenchResult]:
    """Parse a single markdown table block."""
    if len(table_lines) < 2:
        return []

    # Parse header
    header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
    col_map = {}
    for i, h in enumerate(header_cells):
        col_map[h.lower().replace(" ", "_")] = i

    results = []
    for line in table_lines[1:]:
        cells = [c.strip() for c in line.strip("|").split("|")]

        def cell_val(col_name: str) -> str:
            idx = col_map.get(col_name.lower().replace(" ", "_"), -1)
            if 0 <= idx < len(cells):
                return cells[idx]
            return ""

        r = BenchResult()
        r.model_name = cell_val("model")
        r.model_filename = r.model_name
        r.model_type = cell_val("model_type") if "model_type" in col_map else r.model_name
        r.type_k = cell_val("type_k")
        r.type_v = cell_val("type_v")
        r.source_file = source_file

        # Parse test column: e.g. "pp64", "tg128", "pp512/tg128"
        test_str = cell_val("test")
        pp_match = re.search(r'pp(\d+)', test_str)
        tg_match = re.search(r'tg(\d+)', test_str)
        r.n_prompt = int(pp_match.group(1)) if pp_match else 0
        r.n_gen = int(tg_match.group(1)) if tg_match else 0

        # Parse t/s
        ts_str = cell_val("t/s")
        try:
            r.avg_ts = float(ts_str)
        except (ValueError, TypeError):
            # Try to parse ± format: "50.1 ± 2.3"
            ts_match = re.match(r'([\d.]+)', ts_str)
            if ts_match:
                r.avg_ts = float(ts_match.group(1))
            else:
                r.avg_ts = 0.0

        # Try to get stddev from test-specific columns if available
        # Some versions have separate pp/tg columns with stdev
        if not r.avg_ts and "s_tg_t/s" in col_map:
            if r.n_gen > 0:
                try:
                    r.avg_ts = float(cell_val("s_tg_t/s"))
                except (ValueError, TypeError):
                    pass
            elif r.n_prompt > 0:
                try:
                    r.avg_ts = float(cell_val("s_pp_t/s"))
                except (ValueError, TypeError):
                    pass

        if r.avg_ts > 0 or r.model_name:
            results.append(r)

    return results


# ─── Aggregation and table building ──────────────────────────────────────────

def build_model_groups(all_results: List[BenchResult],
                       nvfp4_info: Dict[str, Tuple[bool, Optional[float], str]]) -> Dict[str, ModelGroup]:
    """
    Group results by model, separating by KV config.
    Returns dict keyed by "model_key|type_k|type_v".
    """
    groups: Dict[str, ModelGroup] = {}

    for r in all_results:
        # Create a grouping key: model + KV config
        kv_tag = f"{r.type_k}/{r.type_v}"
        if r.nvfp4_kv_enabled:
            kv_tag = "NVFP4"
        elif r.type_k in ("f32", "float32") and r.type_v in ("f32", "float32"):
            kv_tag = "F32"
        elif "q8_0" in r.type_k or "q8_0" in r.type_v:
            kv_tag = "q8_0"
        elif "nvfp4" in r.type_k.lower() or "nvfp4" in r.type_v.lower():
            kv_tag = "NVFP4"
        elif "kvarn" in r.type_k.lower() or "kvarn" in r.type_v.lower():
            kv_tag = "KVarN4"

        model_key, display = infer_model_key(r.model_filename, r.model_type)
        group_key = f"{model_key}|{kv_tag}"

        if group_key not in groups:
            g = ModelGroup(model_key=model_key, display_name=display)
            groups[group_key] = g

        groups[group_key].results.append(r)

    # Apply NVFP4 info
    for key, g in groups.items():
        # Check if any result has NVFP4 enabled
        for r in g.results:
            if r.nvfp4_kv_enabled:
                g.nvfp4_enabled = True
                break

        # Check NVFP4 info from stderr for this model
        for info_key, (enabled, ratio, note) in nvfp4_info.items():
            if g.model_key.lower() in info_key.lower() or info_key.lower() in g.model_key.lower():
                if enabled:
                    g.nvfp4_enabled = True
                if ratio is not None:
                    g.nvfp4_compression_ratio = ratio
                if note:
                    g.nvfp4_note = note

    return groups


def build_comparison_table(groups: Dict[str, ModelGroup]) -> str:
    """Build a markdown comparison table from grouped results."""
    # Group by model_key (across KV configs)
    model_entries: Dict[str, List[Tuple[str, ModelGroup]]] = defaultdict(list)
    for key, g in groups.items():
        model_entries[g.model_key].append((key, g))

    lines = []
    lines.append("## Benchmark Comparison")
    lines.append("")

    for model_key in sorted(model_entries.keys()):
        entries = model_entries[model_key]
        display_name = entries[0][1].display_name

        lines.append(f"### {display_name}")
        lines.append("")

        # Build table rows
        header = "| KV Config | tg64 t/s | pp64 t/s | StdDev | Notes |"
        sep = "|-----------|:--------:|:--------:|:------:|-------|"
        lines.append(header)
        lines.append(sep)

        best_tg = 0.0
        best_pp = 0.0
        rows_data: List[Tuple[str, float, float, float, str, str]] = []

        for _, g in entries:
            # Extract KV config label
            kv_label = ""
            tg64_val = 0.0
            pp64_val = 0.0
            stddev = 0.0

            for r in g.results:
                if r.type_k and r.type_v:
                    kv_label = f"{r.type_k} / {r.type_v}"
                if r.nvfp4_kv_enabled:
                    kv_label += " + NVFP4"

                if r.n_gen == 64 and r.n_prompt == 0:
                    tg64_val = r.avg_ts
                    stddev = r.stddev_ts
                elif r.n_prompt == 64 and r.n_gen == 0:
                    pp64_val = r.avg_ts
                elif r.n_gen > 0:
                    tg64_val = r.avg_ts
                    stddev = r.stddev_ts
                elif r.n_prompt > 0:
                    pp64_val = r.avg_ts

            # Build notes
            note_parts = []
            if g.nvfp4_enabled:
                note_parts.append("NVFP4 KV")
            if g.nvfp4_compression_ratio:
                note_parts.append(f"{g.nvfp4_compression_ratio:.1f}x compression")
            if g.nvfp4_note and g.nvfp4_note not in " ".join(note_parts):
                note_parts.append(g.nvfp4_note)

            # Detect expert offload
            for r in g.results:
                if r.n_cpu_moe > 0:
                    note_parts.append(f"ncmoe={r.n_cpu_moe}")
                    break

            note = "; ".join(note_parts) if note_parts else "-"

            best_tg = max(best_tg, tg64_val)
            best_pp = max(best_pp, pp64_val)

            rows_data.append((kv_label, tg64_val, pp64_val, stddev, note, ""))

        # Detect winners
        for i, (kv_label, tg64_val, pp64_val, stddev, note, _) in enumerate(rows_data):
            row = list(rows_data[i])
            is_best_tg = (tg64_val == best_tg and best_tg > 0)
            is_best_pp = (pp64_val == best_pp and best_pp > 0)

            if is_best_tg:
                row[5] = "**WINNER** (tg64)"
            elif is_best_pp:
                row[5] = f"**BEST PP** ({pp64_val:.0f})"

            rows_data[i] = tuple(row)

        # Print rows
        for kv_label, tg64_val, pp64_val, stddev, note, flag in rows_data:
            tg_str = f"**{tg64_val:.1f}**" if "WINNER" in flag else f"{tg64_val:.1f}"
            pp_str = f"**{pp64_val:.1f}**" if "BEST PP" in flag else f"{pp64_val:.1f}"

            display_note = f"{note} {flag}".strip()
            if not display_note or display_note == "-":
                display_note = "-"

            lines.append(f"| {kv_label} | {tg_str} | {pp_str} | {stddev:.2f} | {display_note} |")

        lines.append("")

    return "\n".join(lines)


def build_summary_table(groups: Dict[str, ModelGroup]) -> str:
    """Build a compact summary table — one row per model+KV combo."""
    lines = []
    lines.append("## Summary (all models)")
    lines.append("")
    header = "| Model | KV Config | tg64 | pp64 | StdDev | NVFP4 | Notes |"
    sep = "|-------|-----------|:----:|:----:|:------:|:-----:|-------|"
    lines.append(header)
    lines.append(sep)

    for key in sorted(groups.keys()):
        g = groups[key]

        tg64_val = 0.0
        pp64_val = 0.0
        stddev = 0.0
        kv_label = ""

        for r in g.results:
            if r.type_k and r.type_v and not kv_label:
                kv_label = f"{r.type_k}/{r.type_v}"
            if r.nvfp4_kv_enabled:
                kv_label = "NVFP4"
            if r.n_gen >= 64:
                tg64_val = r.avg_ts
                stddev = r.stddev_ts
            if r.n_prompt >= 64:
                pp64_val = r.avg_ts

        tg_str = f"{tg64_val:.1f}" if tg64_val else "-"
        pp_str = f"{pp64_val:.1f}" if pp64_val else "-"
        nvfp4_str = "YES" if g.nvfp4_enabled else "-"

        note_parts = []
        if g.nvfp4_compression_ratio:
            note_parts.append(f"{g.nvfp4_compression_ratio:.1f}x")
        for r in g.results:
            if r.n_cpu_moe > 0:
                note_parts.append(f"ncmoe={r.n_cpu_moe}")
                break
        note = "; ".join(note_parts) if note_parts else "-"

        lines.append(f"| {g.display_name} | {kv_label} | {tg_str} | {pp_str} | {stddev:.2f} | {nvfp4_str} | {note} |")

    lines.append("")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    files = sys.argv[1:] if len(sys.argv) > 1 else []

    all_results: List[BenchResult] = []
    global_nvfp4_info: Dict[str, Tuple[bool, Optional[float], str]] = {}

    if files:
        for fpath in files:
            if not os.path.isfile(fpath):
                print(f"WARNING: file not found: {fpath}", file=sys.stderr)
                continue

            text = decode_raw(fpath)

            # Always scan for NVFP4 info from stderr/side-channel text
            nvfp4_enabled, ratio, note = extract_nvfp4_info(text)
            if nvfp4_enabled or ratio is not None:
                global_nvfp4_info[fpath] = (nvfp4_enabled, ratio, note)

            # Try JSON first (it's richer), then markdown
            json_results = parse_json_input(text, fpath)
            if json_results:
                all_results.extend(json_results)
            else:
                md_results = parse_markdown_input(text, fpath)
                if md_results:
                    all_results.extend(md_results)
                else:
                    # Last resort: try parsing as bare text for any table-like content
                    print(f"WARNING: no recognized benchmark data in {fpath}", file=sys.stderr)
    else:
        # Read from stdin
        text = sys.stdin.read()

        nvfp4_enabled, ratio, note = extract_nvfp4_info(text)
        if nvfp4_enabled or ratio is not None:
            global_nvfp4_info["stdin"] = (nvfp4_enabled, ratio, note)

        json_results = parse_json_input(text, "stdin")
        if json_results:
            all_results.extend(json_results)
        else:
            md_results = parse_markdown_input(text, "stdin")
            if md_results:
                all_results.extend(md_results)
            else:
                print("ERROR: no recognizable benchmark data found in input", file=sys.stderr)
                print("Expected llama-bench JSON (-o json) or markdown (-o md) output", file=sys.stderr)
                sys.exit(1)

    if not all_results:
        print("ERROR: no benchmark results parsed", file=sys.stderr)
        sys.exit(1)

    # Build groups
    groups = build_model_groups(all_results, global_nvfp4_info)

    # Print summary table
    print(build_summary_table(groups))

    # Print detailed per-model comparison tables
    print(build_comparison_table(groups))

    # Print quick stats
    print("## Quick Stats")
    print("")
    print(f"- Total benchmark entries parsed: **{len(all_results)}**")
    print(f"- Unique model+KV groups: **{len(groups)}**")
    models = set(g.model_key for g in groups.values())
    print(f"- Unique models: **{len(models)}**")
    nvfp4_models = [g for g in groups.values() if g.nvfp4_enabled]
    if nvfp4_models:
        print(f"- NVFP4 KV enabled: **{len(nvfp4_models)}** group(s)")
        for g in nvfp4_models:
            print(f"  - {g.display_name}: {g.nvfp4_note or 'enabled'}")
    print("")


if __name__ == "__main__":
    main()
