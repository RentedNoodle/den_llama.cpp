#!/usr/bin/env python3
import sys, re
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
names = set()
for t in r.tensors:
    m = re.match(r"blk\.(\d+)\.(.*)$", t.name)
    if m:
        names.add(m.group(2))
    else:
        names.add(t.name)
for n in sorted(names):
    print(n)
