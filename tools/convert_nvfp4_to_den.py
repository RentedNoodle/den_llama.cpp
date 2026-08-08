#!/usr/bin/env python3
"""One-shot: RedHatAI NVFP4 safetensors -> .den NULLGLASS tiles.
Takes llm-compressor NVFP4 model, repacks as 160B tiles, writes .den.
Hardest path: direct binary manipulation, no modelopt, no PyTorch required.

Usage:
  python tools/convert_nvfp4_to_den.py --model RedHatAI/Qwen3.6-35B-A3B-NVFP4 --output model.den

PORTED from C:\Den\den-nvfp4-optimizations\tools\_dead_python\convert_nvfp4_to_den.py
-> I:\den_llama.cpp\tools\convert_nvfp4_to_den.py
"""
import argparse, json, os, struct, sys, numpy as np
from pathlib import Path
from safetensors import safe_open
import torch  # needed for weight shapes only

# -- Constants from den_format.h ------------------------------------------
DEN_MAGIC = 0x4E454400
DEN_VERSION = 0x00050000
DEN_HEADER_SIZE = 4096
DEN_TENSOR_ENTRY_SIZE = 128
DEN_TARGET_NVFP4 = 1
DEN_TARGET_BF16 = 2
TILE_K = 256
GROUP_SIZE = 16
TILE_BYTES = 160
TILE_GROUPS = 16
NIBBLES_PER_TILE = 128

def e2m1_encode(values):
    """Branchless E2M1 quantization -- matches kernel."""
    vals = np.asarray(values, dtype=np.float32).flatten()
    bits = vals.view(np.uint32)
    signs = ((bits >> 28) & 8).astype(np.uint8)
    abs_bits = bits & 0x7FFFFFFF
    indices = ((abs_bits > 0x3E800000).astype(np.uint8) +
               (abs_bits >= 0x3F400000).astype(np.uint8) +
               (abs_bits > 0x3FA00000).astype(np.uint8) +
               (abs_bits >= 0x3FE00000).astype(np.uint8) +
               (abs_bits > 0x40200000).astype(np.uint8) +
               (abs_bits >= 0x40600000).astype(np.uint8) +
               (abs_bits > 0x40A00000).astype(np.uint8))
    return signs | indices

def pack_nullglass_tile(w_f32):
    """Pack 256 float32 weights into a 160B NULLGLASS tile."""
    tile = np.zeros(TILE_BYTES, dtype=np.uint8)
    mx = np.max(np.abs(w_f32))
    if mx < 1e-8: mx = 1.0
    for g in range(TILE_GROUPS):
        block = w_f32[g * GROUP_SIZE:(g + 1) * GROUP_SIZE]
        sc = np.max(np.abs(block)) / 6.0
        if sc < 1e-8: sc = 1.0 / 6.0
        tile[g] = int(np.clip(np.log2(sc) + 127, 0, 255))  # UE4M3 scale
        quant = block / (sc * 6.0)
        nibbles = e2m1_encode(quant)
        off = TILE_GROUPS + g * 8
        for i in range(0, GROUP_SIZE, 2):
            tile[off + i // 2] = nibbles[i] | (nibbles[i + 1] << 4)
    tile[144:148] = np.float32(mx).tobytes()
    return tile

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-name", default="Qwen3.6-35B-A3B-NVFP4")
    args = p.parse_args()

    # Load safetensors
    model_dir = args.model
    if not os.path.isdir(model_dir):
        # Try HF cache
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.model)

    config = json.load(open(os.path.join(model_dir, "config.json")))
    cfg = config.get("text_config", config)

    st_files = sorted(Path(model_dir).glob("*.safetensors"))
    st_files = [f for f in st_files if os.path.getsize(f) > 100]  # skip 0-byte stubs
    print(f"Found {len(st_files)} safetensors files in {model_dir}")

    # Collect tensors
    tensors = {}
    for sf in st_files:
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                tensors[key] = t

    print(f"Loaded {len(tensors)} tensors")

    # Identify tensor metadata
    n_layers = cfg.get("num_hidden_layers", cfg.get("n_layers", 40))
    hidden_size = cfg.get("hidden_size", 2048)
    vocab_size = cfg.get("vocab_size", 151936)
    n_heads = cfg.get("num_attention_heads", cfg.get("n_heads", 16))
    n_kv_heads = cfg.get("num_key_value_heads", cfg.get("n_kv_heads", 4))
    head_dim = hidden_size // n_heads
    ffn_size = cfg.get("intermediate_size", cfg.get("ffn_size", 0))
    n_experts = cfg.get("num_experts", 0)
    n_experts_used = cfg.get("num_experts_per_tok", 8)
    n_rot = head_dim

    # Compute FFN size from first MLP gate
    for key in tensors:
        if "mlp.gate" in key and "expert" not in key:
            ffn_size = tensors[key].shape[0]
            break

    # Build .den header
    header = bytearray(DEN_HEADER_SIZE)
    struct.pack_into("<I", header, 0, DEN_MAGIC)
    struct.pack_into("<I", header, 4, DEN_VERSION)
    struct.pack_into("<I", header, 8, 1)  # arch = QWEN35
    struct.pack_into("<I", header, 12, 0)  # flags
    struct.pack_into("<I", header, 16, n_layers)
    struct.pack_into("<I", header, 20, n_heads)
    struct.pack_into("<I", header, 24, n_kv_heads)
    struct.pack_into("<I", header, 28, hidden_size)
    struct.pack_into("<I", header, 32, ffn_size)
    struct.pack_into("<I", header, 36, vocab_size)
    struct.pack_into("<I", header, 40, 262144)  # max_seq_len
    struct.pack_into("<I", header, 44, n_rot)
    struct.pack_into("<I", header, 48, n_experts)
    struct.pack_into("<I", header, 52, n_experts_used)
    struct.pack_into("<f", header, 56, 1000000.0)  # rope_theta
    struct.pack_into("<f", header, 60, 1e-6)       # rms_norm_eps

    # Build tensor entries + data
    entries = bytearray()
    data = bytearray()
    index_offset = DEN_HEADER_SIZE
    data_offset = DEN_HEADER_SIZE  # will be recalculated
    cur_data_off = 0
    tensor_count = 0

    for key, t in sorted(tensors.items()):
        shape = list(t.shape)
        ndim = len(shape)
        if ndim > 4: shape = shape[:4]  # truncate
        while len(shape) < 4: shape.append(1)

        w = t.float().numpy()
        hw_target = DEN_TARGET_NVFP4
        is_firewalled = any(p in key for p in [
            "embed_tokens", "lm_head", "norm", "linear_attn", "shared_expert_gate",
            "mtp", "self_attn.q_norm", "self_attn.k_norm"
        ])

        if is_firewalled:
            # BF16 passthrough
            hw_target = DEN_TARGET_BF16
            bf16 = np.zeros(w.size, dtype=np.uint16)
            bf16[:] = (w.flatten().view(np.uint32) >> 16).astype(np.uint16)
            data_bytes = bf16.tobytes()
            data_size = len(data_bytes)
            scale_size = 0
        elif ndim == 2 and shape[0] > 1 and shape[1] > 1:
            # NVFP4 quantize to NULLGLASS tiles
            N, K = shape[0], shape[1]
            pad_k = (TILE_K - (K % TILE_K)) % TILE_K
            if pad_k > 0:
                w = np.pad(w, ((0, 0), (0, pad_k)))
            Kp = w.shape[1]
            tpr = Kp // TILE_K
            all_tiles = bytearray()
            for r in range(N):
                for t in range(tpr):
                    tile_w = w[r, t * TILE_K:(t + 1) * TILE_K]
                    tile = pack_nullglass_tile(tile_w)
                    all_tiles.extend(tile.tobytes())
            data_bytes = bytes(all_tiles)
            data_size = len(data_bytes)
            scale_size = 0  # embedded in tiles
        else:
            # Small tensors: keep as F32
            hw_target = 3  # F32
            data_bytes = w.astype(np.float32).tobytes()
            data_size = len(data_bytes)
            scale_size = 0

        # Write entry (128 bytes)
        entry = bytearray(DEN_TENSOR_ENTRY_SIZE)
        struct.pack_into("<I", entry, 0, tensor_count)  # slot
        struct.pack_into("<I", entry, 4, hw_target)
        struct.pack_into("<I", entry, 8, ndim)
        struct.pack_into("<I", entry, 12, 0)  # flags
        for j in range(4):
            struct.pack_into("<q", entry, 16 + j * 8, shape[j] if j < ndim else 1)
        struct.pack_into("<Q", entry, 48, w.size)
        struct.pack_into("<Q", entry, 56, cur_data_off)
        struct.pack_into("<Q", entry, 64, data_size)
        struct.pack_into("<Q", entry, 72, 0)
        struct.pack_into("<Q", entry, 80, scale_size)
        entries.extend(entry)
        data.extend(data_bytes)
        cur_data_off += data_size
        tensor_count += 1

        if tensor_count % 50 == 0:
            status = "FW" if is_firewalled else ("NVFP4" if ndim == 2 else "F32")
            print(f"  [{tensor_count}] {status}: {key} {shape}")

    # Finalize header
    new_data_offset = DEN_HEADER_SIZE + tensor_count * DEN_TENSOR_ENTRY_SIZE
    struct.pack_into("<I", header, 104, tensor_count)  # tensor_count
    struct.pack_into("<I", header, 108, index_offset)   # index_offset
    struct.pack_into("<Q", header, 116, new_data_offset) # data_offset
    struct.pack_into("<Q", header, 124, cur_data_off)    # total_data_size

    # Write .den file
    out_path = args.output
    print(f"\nWriting: {out_path} ({tensor_count} tensors, {cur_data_off/1e9:.2f} GB data)")
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(entries)
        f.write(data)

    total = DEN_HEADER_SIZE + len(entries) + cur_data_off
    print(f"Done! {total/1e9:.2f} GB total")

if __name__ == "__main__":
    main()
