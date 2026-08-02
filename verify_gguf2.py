import sys, struct
sys.path.insert(0, '/root/den_final/llama_upstream/gguf-py')
import gguf

path = '/root/den_final/lostgentoo_nvfp4/out.gguf'
reader = gguf.GGUFReader(path)

# Raw magic/version
f = open(path, 'rb')
magic = f.read(4)
ver = struct.unpack('<I', f.read(4))[0]
print('GGUF magic:', magic, 'version:', ver)
f.close()

# Characterize BF16 and F32 tensors
bf16_ex = []
f32_ex = []
for t in reader.tensors:
    if t.tensor_type == gguf.GGMLQuantizationType.BF16 and len(bf16_ex) < 10:
        bf16_ex.append(str(t.name))
    if t.tensor_type == gguf.GGMLQuantizationType.F32 and len(f32_ex) < 12:
        f32_ex.append(str(t.name))

print('BF16 example tensors:', bf16_ex)
print('F32 example tensors:', f32_ex)

# Sanity: exact byte math for one NVFP4 tensor
nvfp4 = gguf.GGMLQuantizationType.NVFP4
for t in reader.tensors:
    if t.tensor_type == nvfp4:
        numel = 1
        for d in t.shape: numel *= d
        nblocks = numel // 64   # QK_NVFP4=64
        print(f'  {t.name}: shape={list(t.shape)} numel={numel} blocks64={nblocks} n_bytes={t.n_bytes} expected={nblocks*36}  -> bytes/256elem={t.n_bytes/(numel/256):.1f}')
        break
