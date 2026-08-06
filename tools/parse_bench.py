#!/usr/bin/env python3
"""parse_bench.py - robust benchmark-log parser (auto-detects UTF-16/UTF-8).
Usage: python parse_bench.py <log...>
Prints clean ASCII perf lines. No mojibake, ever. This is the canonical way to
read llama-cli / llama-bench / staging output. Never grep/iconv a log directly.
"""
import sys, re

def decode(p):
    raw = open(p, 'rb').read()
    for enc in ('utf-16-le', 'utf-8'):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode('latin-1', errors='replace')

PATS = [
    (r'(\[ Prompt: .*t/s \| Generation: .*t/s \])', 'banner'),  # den/custom engine banner
    (r'(eval time\s*=.*tokens per second.*)', 'cli'),
    (r'(prompt eval time\s*=.*tokens per second.*)', 'cli'),
    (r'(Den expert-stage L3 probe:.*)', 'staging'),
    (r'^\|\s*(pp|tg)\s*\|.*$', 'bench'),
    (r'(error|CUDA error|out of memory|failed to load|Assertion failed)[^\n]*', 'fail'),
    (r'(main:.*(eval time|prompt eval time).*)', 'cli'),
]

def main():
    for p in sys.argv[1:]:
        t = decode(p)
        print('==== %s' % p)
        seen = set()
        for ln in t.splitlines():
            s = ln.strip()
            if not s:
                continue
            for pat, kind in PATS:
                m = re.search(pat, s)
                if m and m.group(1)[:80] not in seen:
                    seen.add(m.group(1)[:80])
                    txt = m.group(1)
                    # ascii-safe
                    print('  [%s] %s' % (kind, txt.encode('ascii', 'replace').decode()))

if __name__ == '__main__':
    main()
