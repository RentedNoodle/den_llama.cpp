import sys, struct
from collections import Counter

sys.path.insert(0, '/root/den_final/llama_upstream/gguf-py')
import gguf

path = '/root/den_final/lostgentoo_nvfp4/out.gguf'
reader = gguf.GGUFReader(path)

# GGUF header basics
import os
fsize = os.path.getsize(path)
print(f'file_size: {fsize} bytes ({fsize/1e9:.2f} GB)')

type_counts = Counter()
nvfp4_block_bytes = []
example_names = {}
mxfp4 = gguf.GGMLQuantizationType.MXFP4
nvfp4 = gguf.GGMLQuantizationType.NVFP4
total_nvfp4_bytes = 0

for t in reader.tensors:
    tn = t.tensor_type
    type_counts[tn.name] += 1
    numel = 1
    for d in t.shape:
        numel *= d
    if tn == nvfp4:
        # bytes per 256 elements
        bpb = t.n_bytes / (numel / 256.0)
        nvfp4_block_bytes.append(bpb)
        total_nvfp4_bytes += t.n_bytes
        if tn.name not in example_names:
            example_names.setdefault('nvfp4', []).append((str(t.name), str(list(t.shape)), t.n_bytes))
    elif tn == mxfp4:
        bpb = t.n_bytes / (numel / 256.0)
        example_names.setdefault('mxfp4', []).append((str(t.name), str(list(t.shape)), t.n_bytes))

print('tensor type counts:')
for k, v in type_counts.most_common():
    print(f'  {k:12s}: {v}')

if nvfp4_block_bytes:
    print(f'NVFP4 tensors: {len(nvfp4_block_bytes)}')
    print(f'  bytes-per-block (per 256 elems): min={min(nvfp4_block_bytes):.3f} max={max(nvfp4_block_bytes):.3f} unique={set(round(x,3) for x in nvfp4_block_bytes)}')
    print(f'  total NVFP4 bytes: {total_nvfp4_bytes} ({total_nvfp4_bytes/1e9:.2f} GB)')
    print('  NVFP4 examples:')
    for ex in example_names.get('nvfp4', [])[:5]:
        print('   ', ex)
if example_names.get('mxfp4'):
    print('  MXFP4 examples:')
    for ex in example_names['mxfp4'][:5]:
        print('   ', ex)

# Show the type_size table from gguf constants for reference
print('gguf constant NVFP4 type_size:', gguf.constants.GGML_QUANT_SIZES.get(nvfp4))
