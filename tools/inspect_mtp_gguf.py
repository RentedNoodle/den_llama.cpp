#!/usr/bin/env python3
"""Inspect a GGUF for MTP (nextn.*) tensors + arch metadata using gguf-py GGUFReader.
Rule #0: script on disk.
"""
import sys
sys.path.insert(0, '/root/den_final/gguf-py')
from gguf import GGUFReader

def inspect(path):
    r = GGUFReader(path)
    print(f"TENSORS: {len(r.tensors)}")
    for k in r.fields:
        if any(s in k for s in ['arch', 'general', 'expert', 'nextn', 'mtp',
                                'layer_count', 'block_count', 'head_count',
                                'feed_forward', 'tokenizer.ggml.model',
                                'context_length']):
            v = r.fields[k]
            try:
                print(f"  {k} = {v.parts[v.data[0]] if v.data else v.contents()[:120]}")
            except Exception as e:
                print(f"  {k} = (unreadable: {e})")
    print("\n--- nextn / exps tensors ---")
    n_nextn = 0
    for t in r.tensors:
        name = t.name
        if 'nextn' in name:
            n_nextn += 1
            if n_nextn <= 25:
                print(f"  NEXTN {name} shape={t.shape} type={t.tensor_type}")
    print(f"\nTotal nextn tensors: {n_nextn}")
    # exps count
    n_exps = sum(1 for t in r.tensors if 'exps' in t.name)
    print(f"Expert tensors: {n_exps}")

if __name__ == '__main__':
    inspect(sys.argv[1])
