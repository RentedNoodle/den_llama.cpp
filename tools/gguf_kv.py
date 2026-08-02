#!/usr/bin/env python3
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
for k, f in r.fields.items():
    if k.startswith("GGUF."): continue
    parts = []
    for p in f.parts:
        try:
            parts.append(p.tolist())
        except Exception:
            parts.append(str(p)[:60])
    print(k, "=", parts)
