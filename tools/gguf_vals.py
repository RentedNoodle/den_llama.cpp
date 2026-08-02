#!/usr/bin/env python3
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
for k, f in r.fields.items():
    v = f.contents()
    if isinstance(v, (int, float)) or (hasattr(v, "__len__") and not isinstance(v, str) and len(v) <= 8):
        print("%s = %s" % (k, v))
