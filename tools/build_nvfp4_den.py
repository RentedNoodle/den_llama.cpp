#!/usr/bin/env python3
"""
Build NVFP4 .den file from modelopt-calibrated safetensors.

Reads Qwen3.5-4B-NVFP4-Calibrated-v2/ safetensors (modelopt format),
converts NVFP4-quantized weights to NULLGLASS tiles, and writes a .den
DENPACK V4 file using pure Python.

Usage:
  python3 build_nvfp4_den.py \
    --safetensors I:/models/Qwen3.5-4B-NVFP4-Calibrated-v2/model.safetensors \
    --config I:/models/Qwen3.5-4B-NVFP4-Calibrated-v2/config.json \
    --output I:/models/Qwen3.5-4B-NVFP4.den

PORTED from C:\Den\den-nvfp4-optimizations\tools\_converter_archive\build_nvfp4_den.py
-> I:\den_llama.cpp\tools\build_nvfp4_den.py
"""

import argparse, json, math, os, struct, sys, re
import numpy as np
import torch

# --- Format constants (from den_format.h) --------------------------------
DEN_MAGIC = 0x4E454400
DEN_VERSION = 0x00050000
DEN_HEADER_SIZE = 4096
DEN_TENSOR_ENTRY_SIZE = 128
DEN_ARCH_QWEN35 = 1
DEN_TARGET_NVFP4 = 1
DEN_TARGET_BF16 = 2
DEN_TARGET_F32 = 3

# Slot constants
DEN_SLOT_TOKEN_EMBD = 0
DEN_SLOT_OUTPUT_NORM = 1
DEN_SLOT_OUTPUT = 2
DEN_LAYER_STRIDE = 32

def DEN_SLOT_LAYER_BASE(layer):
    return 3 + layer * DEN_LAYER_STRIDE

# --- NVFP4 constants -----------------------------------------------------
TILE_K = 256
GROUP_SIZE = 16
BLOCKS_PER_TILE = TILE_K // GROUP_SIZE  # 16
NIBBLES_PER_TILE = TILE_K // 2  # 128
NVFP4_TILE_SIZE = 160

# UE4M3 LUT (same as in den_gpu_ops.cu)
UE4M3_VALUES = np.array([
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
    1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875
], dtype=np.float32)

# Firewall patterns (keep BF16/F32)
FIREWALL_PATTERNS = [
    r"model\.embed_tokens\.weight",
    r"lm_head\.weight",
    r".*linear_attn\.in_proj_qkv\.weight",
    r".*linear_attn\.in_proj_z\.weight",
    r".*linear_attn\.in_proj_a\.weight",
    r".*linear_attn\.in_proj_b\.weight",
    r".*linear_attn\.out_proj\.weight",
    r".*linear_attn\.conv1d\.weight",
    r".*linear_attn\.norm\.weight",
    r".*self_attn\.q_proj\.weight",
    r".*self_attn\.o_proj\.weight",
    r".*self_attn\.k_norm\.weight",
    r".*self_attn\.q_norm\.weight",
]

F32_PINNED = ["norm", "ssm_", "attn_gate", "ssm_dt", "bias"]

def is_firewalled(name):
    return any(re.match(p, name) for p in FIREWALL_PATTERNS)

def is_f32_pinned(name):
    return any(p in name for p in F32_PINNED)


# --- FP8 decode ----------------------------------------------------------
def float8_e4m3_to_f32(data):
    """Decode float8_e4m3fn bytes to float32 array."""
    as_uint8 = np.frombuffer(data, dtype=np.uint8)
    sign = (as_uint8 >> 7).astype(np.float32)
    exp = ((as_uint8 >> 3) & 0xF).astype(np.int32)
    man = (as_uint8 & 0x7).astype(np.int32)
    normal = (exp > 0)
    value = np.zeros_like(sign)
    value[normal] = np.power(2.0, exp[normal].astype(np.float32) - 7) * (1.0 + man[normal] / 8.0)
    value[~normal] = np.power(2.0, -6.0) * man[~normal] / 8.0
    value = value * (1.0 - 2.0 * sign)
    return value

def ue4m3_encode(value):
    """float32 -> UE4M3 byte (nearest-neighbor in 16-value LUT)."""
    idx = np.argmin(np.abs(UE4M3_VALUES - value))
    return np.uint8(idx)

# --- Tile packing --------------------------------------------------------
def pack_nullglass_tile(nibbles_128b, scales_f32_16):
    """
    160-byte NULLGLASS tile: 16B UE4M3 scales + 128B E2M1 nibbles + 16B reserved.
    nibbles_128b: [128] uint8 -- 256 E2M1 nibbles, 2 per byte
    scales_f32_16: [16] float32 -- one per 16-element block
    """
    tile = np.zeros(NVFP4_TILE_SIZE, dtype=np.uint8)

    # Normalize scales for OMMA range
    tile_norm = max(np.max(scales_f32_16) / 1.4375, 1e-6)
    scales_norm = scales_f32_16 / tile_norm

    # Encode 16 UE4M3 scale bytes in OMMA format
    for g in range(16):
        tile[g] = ue4m3_encode(scales_norm[g])

    # Copy E2M1 nibbles
    tile[16:144] = nibbles_128b

    # Store tile_norm at bytes 144-147 for kernel denormalization
    tile[144:148] = np.frombuffer(struct.pack('<f', float(tile_norm)), dtype=np.uint8)

    return tile

# --- HF name -> slot mapping (from den_convert_heretic.py) ----------------
def name_to_slot(hf_name):
    if hf_name == 'model.embed_tokens.weight':
        return DEN_SLOT_TOKEN_EMBD
    if hf_name == 'model.norm.weight':
        return DEN_SLOT_OUTPUT_NORM
    parts = hf_name.split('.')
    # layers.N.subgroup.subgroup2
    layer = int(parts[2])
    base = DEN_SLOT_LAYER_BASE(layer)
    suffix = '.'.join(parts[3:])

    if suffix == 'linear_attn.A_log':
        return base + 15
    if suffix == 'linear_attn.dt_bias':
        return base + 16
    if suffix == 'linear_attn.conv1d.weight':
        return base + 17
    if suffix == 'linear_attn.in_proj_qkv.weight':
        return base + 24
    if suffix == 'linear_attn.in_proj_a.weight':
        return base + 12
    if suffix == 'linear_attn.in_proj_b.weight':
        return base + 25
    if suffix == 'linear_attn.in_proj_z.weight':
        return base + 13
    if suffix == 'linear_attn.out_proj.weight':
        return base + 14
    if suffix == 'linear_attn.norm.weight':
        return base + 19
    if suffix == 'self_attn.q_proj.weight':
        return base + 1
    if suffix == 'self_attn.k_proj.weight':
        return base + 2
    if suffix == 'self_attn.v_proj.weight':
        return base + 3
    if suffix == 'self_attn.o_proj.weight':
        return base + 4
    if suffix == 'self_attn.q_norm.weight':
        return base + 5
    if suffix == 'self_attn.k_norm.weight':
        return base + 6
    if suffix == 'mlp.gate_proj.weight':
        return base + 8
    if suffix == 'mlp.up_proj.weight':
        return base + 9
    if suffix == 'mlp.down_proj.weight':
        return base + 10
    if suffix == 'input_layernorm.weight':
        return base + 20
    if suffix == 'post_attention_layernorm.weight':
        return base + 21
    raise ValueError(f"Unmapped: {hf_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bf16-ref", help="path to original HF model dir for firewalled BF16 weights")
    args = ap.parse_args()

    # --- Load calibrated safetensors ------------------------------------
    print(f"Loading: {args.safetensors}")
    import safetensors.torch
    import glob
    tensors = {}
    with safetensors.safe_open(args.safetensors, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    print(f"  Loaded {len(tensors)} tensors")
    print(f"  Key types: {set(str(t.dtype) for t in tensors.values())}")

    # --- Load reference BF16 weights for firewalled tensors --------------
    bf16_ref = {}
    if args.bf16_ref:
        ref_dir = args.bf16_ref
        print(f"Loading reference BF16 from: {ref_dir}")
        ref_files = sorted(glob.glob(os.path.join(ref_dir, "model*.safetensors-*")))
        if not ref_files:
            sf_path = os.path.join(ref_dir, "model.safetensors")
            if os.path.exists(sf_path):
                ref_files = [sf_path]
        for rf in ref_files:
            with safetensors.safe_open(rf, framework="pt") as f:
                for key in f.keys():
                    bf16_ref[key] = f.get_tensor(key)
        print(f"  Loaded {len(bf16_ref)} reference tensors")

    # --- Load config ----------------------------------------------------
    with open(args.config) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)

    n_layers = tc["num_hidden_layers"]
    hidden = tc["hidden_size"]
    ffn = tc["intermediate_size"]
    vocab = tc["vocab_size"]
    n_heads = tc["num_attention_heads"]
    n_kv = tc["num_key_value_heads"]
    rope_theta = tc.get("rope_theta", 10000000.0)
    rms_norm_eps = tc.get("rms_norm_eps", 1e-6)

    rope_params = tc.get("rope_parameters", {})
    partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.25) if isinstance(rope_params, dict) else 0.25
    n_rot = int(n_heads * (hidden // n_heads) * partial_rotary_factor)

    ssm_state_size = tc.get("linear_key_head_dim", 128)
    ssm_value_size = tc.get("linear_value_head_dim", ssm_state_size)
    ssm_conv_kernel = tc.get("linear_conv_kernel_dim", 4)
    ssm_inner_size = tc.get("linear_num_key_heads", 16) * ssm_state_size

    print(f"  4B model: {n_layers} layers, hidden={hidden}, ffn={ffn}")

    # --- Build tensor entries -------------------------------------------
    entries = []  # (slot, hw_target, raw_bytes, [N,K], name)
    nvfp4_count = 0
    bf16_count = 0
    f32_count = 0
    skipped = 0

    for name, data in tensors.items():
        dtype = data.dtype

        # Check if this is a quantized weight (has weight_scale companion)
        base_name = name.replace(".weight", "")
        scale_key = base_name + ".weight_scale"
        is_quantized = scale_key in tensors

        # Determine target
        # Skip calibration metadata tensors (not needed for inference)
        if any(s in name for s in ["weight_scale", "input_scale"]):
            skipped += 1
            continue

        if is_quantized:
            # Some quantized weights are firewalled -- keep as BF16
            if is_firewalled(name):
                target = "F32" if is_f32_pinned(name) else "BF16"
            elif is_f32_pinned(name):
                target = "F32"
            else:
                target = "NVFP4"
        elif str(dtype) == "torch.bfloat16" or str(dtype) == "bfloat16":
            target = "BF16"
        else:
            # float32 or other
            target = "F32" if (data.numel() < 1000 or "norm" in name.lower()) else "BF16"

        shape = list(data.shape)
        ndim = len(shape)
        N = shape[0] if len(shape) > 0 else 1
        K = shape[1] if len(shape) >= 2 else 0

        if target == "NVFP4":
            # --- Convert modelopt NVFP4 -> .den NULLGLASS tiles ---
            weight_np = data.numpy()  # uint8
            scales = tensors[scale_key]
            scale2_key = base_name + ".weight_scale_2"
            scales2 = tensors.get(scale2_key, None)

            # modelopt stores weight as [N, K/2] packed nibbles
            in_half = K
            logical_K = K * 2
            logical_N = N

            in_padded = ((logical_K + TILE_K - 1) // TILE_K) * TILE_K
            n_tiles = logical_N * (in_padded // TILE_K)

            all_tiles = np.zeros((n_tiles, NVFP4_TILE_SIZE), dtype=np.uint8)
            tile_idx = 0
            for o in range(logical_N):
                for t_start in range(0, in_padded, TILE_K):
                    # 128 bytes of E2M1 nibbles for this tile
                    nibble_start = t_start // 2
                    nibble_end = min(nibble_start + NIBBLES_PER_TILE, in_half)
                    nibbles = np.zeros(NIBBLES_PER_TILE, dtype=np.uint8)
                    if nibble_start < in_half:
                        actual = nibble_end - nibble_start
                        nibbles[:actual] = weight_np[o, nibble_start:nibble_start + actual]

                    # 16 scale values for this tile
                    scale_start = t_start // GROUP_SIZE
                    scale_end = min(scale_start + BLOCKS_PER_TILE, scales.shape[1])
                    scale_vals = np.ones(BLOCKS_PER_TILE, dtype=np.float32)
                    if scale_start < scales.shape[1]:
                        actual_scales = scale_end - scale_start
                        if actual_scales > 0:
                            # Convert float8_e4m3fn from raw bytes via uint8 view
                            scale_bytes = scales[o, scale_start:scale_end].view(torch.uint8).cpu().numpy().tobytes()
                            scale_data = float8_e4m3_to_f32(scale_bytes)
                            scale_vals[0:actual_scales] = scale_data
                            if scales2 is not None:
                                if scales2.dim() > 0:
                                    s2 = float(scales2[o].item())
                                else:
                                    s2 = float(scales2.item())
                                scale_vals *= s2

                    all_tiles[tile_idx] = pack_nullglass_tile(nibbles, scale_vals)
                    tile_idx += 1

            raw_bytes = all_tiles.tobytes()
            hw_target = DEN_TARGET_NVFP4
            nvfp4_count += 1

        elif target == "F32":
            if str(data.dtype) == "torch.bfloat16":
                raw_bytes = data.float().numpy().tobytes()
            else:
                raw_bytes = data.cpu().numpy().astype(np.float32).tobytes()
            hw_target = DEN_TARGET_F32
            f32_count += 1
        else:  # BF16
            if str(data.dtype) == "torch.bfloat16":
                raw_bytes = data.view(torch.uint16).numpy().tobytes()
            elif data.dtype == torch.float32:
                bf16_data = (data.view(torch.int32) >> 16).to(torch.uint16)
                raw_bytes = bf16_data.numpy().tobytes()
            else:
                raw_bytes = data.cpu().numpy().tobytes()
            hw_target = DEN_TARGET_BF16
            bf16_count += 1

        # For firewalled BF16 tensors that modelopt quantized, use reference weights
        ref_shape = None
        if hw_target == DEN_TARGET_BF16 and is_quantized and bf16_ref:
            ref_key = name
            if ref_key not in bf16_ref:
                ref_key = "model.language_model." + name[len("model."):]
            if ref_key in bf16_ref:
                ref_data = bf16_ref[ref_key]
                raw_bytes = ref_data.view(torch.uint16).numpy().tobytes()
                ref_shape = list(ref_data.shape)

        slot = name_to_slot(name)
        entry_shape = list(data.shape)
        if ref_shape is not None and len(ref_shape) >= 2:
            entry_shape = ref_shape  # use reference shape for firewalled BF16
        elif hw_target == DEN_TARGET_NVFP4 and len(entry_shape) >= 2:
            entry_shape[1] *= 2  # packed nibbles -> logical elements
        entries.append((slot, hw_target, raw_bytes, entry_shape, name.replace("model.", "", 1)))

    print(f"  Packed: {nvfp4_count} NVFP4 + {bf16_count} BF16 + {f32_count} F32 ({skipped} scales skipped)")

    # --- Derive vd from tensor shapes (config may have wrong value) ----
    n_vh = ssm_inner_size // ssm_state_size
    vd_derived = ssm_value_size
    for _, _, _, shape, name in entries:
        if 'in_proj_z' in name:
            vd_derived = shape[0] // n_vh
            if vd_derived != ssm_value_size:
                print(f"  NOTE: Overriding ssm_value_size {ssm_value_size} -> {vd_derived} (derived from {name} shape={shape})")
            break
    if vd_derived != ssm_value_size:
        ssm_value_size = vd_derived

    # --- Write .den file ------------------------------------------------
    print(f"\nWriting: {args.output}")
    num_tensors = len(entries)

    # Sort by slot for deterministic layout
    entries.sort(key=lambda e: e[0])

    # Data starts after header + tensor index, page-aligned (4096)
    index_end = DEN_HEADER_SIZE + num_tensors * DEN_TENSOR_ENTRY_SIZE
    data_offset = ((index_end + 4095) // 4096) * 4096

    with open(args.output, "wb") as f:
        # -- Header --
        hdr = struct.pack("<IIII", DEN_MAGIC, DEN_VERSION, DEN_ARCH_QWEN35, 0)
        hdr += struct.pack("<IIIIIIII", n_layers, n_heads, n_kv, hidden, ffn, vocab,
                           262144, n_rot)
        hdr += struct.pack("<II", 0, 0)  # n_experts, n_experts_used
        hdr += struct.pack("<ff", rope_theta, rms_norm_eps)
        hdr += struct.pack("<IIIIII", ssm_state_size, ssm_conv_kernel, ssm_inner_size,
                           16, 0, 4)
        hdr += struct.pack("<II", 0, ssm_value_size)
        hdr += struct.pack("<II", 0, 0)  # _padding[2]
        hdr += struct.pack("<IIQ", num_tensors, DEN_HEADER_SIZE, data_offset)
        hdr += struct.pack("<Q", 0)  # total_data_size (filled later)
        hdr += struct.pack("<II", 0, 0)  # hot_tier_count, warm_tier_count
        hdr += struct.pack("<I", 0)  # cold_tier_count
        hdr += struct.pack("<QQQ", 0, 0, 0)  # tier sizes
        hdr = hdr.ljust(DEN_HEADER_SIZE, b'\x00')
        f.write(hdr)

        # -- Tensor entries --
        data_pos = data_offset
        for slot, hw_target, raw_bytes, shape, name in entries:
            ndim = min(len(shape), 4)
            dims = list(shape[:4]) + [0] * (4 - len(shape))
            numel = int(np.prod(shape))

            if hw_target == DEN_TARGET_NVFP4:
                data_size = len(raw_bytes)
                scale_size = 0
                scale_offset = 0
                tile_k = TILE_K
                tile_n = 1
                n_tiles = data_size // NVFP4_TILE_SIZE
            else:
                data_size = len(raw_bytes)
                scale_size = 0
                scale_offset = 0
                tile_k = 0
                tile_n = 0
                n_tiles = 0

            scale_count = n_tiles * BLOCKS_PER_TILE if hw_target == DEN_TARGET_NVFP4 else 0

            entry = struct.pack("<IIII", slot, hw_target, ndim, 0)
            entry += struct.pack("<qqqq", dims[0], dims[1], dims[2], dims[3])
            entry += struct.pack("<Q", numel)
            entry += struct.pack("<QQ", data_pos - data_offset, data_size)
            entry += struct.pack("<QQ", scale_offset, scale_size)
            entry += struct.pack("<IIII", tile_k, tile_n, n_tiles, scale_count)
            entry += struct.pack("<QI", 0, 0)  # norm_offset, norm_size
            entry += struct.pack("<II", 256, 0)  # block_size=256, grid_size=0
            entry += struct.pack("<I", 0)  # padding to 128 bytes
            assert len(entry) == DEN_TENSOR_ENTRY_SIZE, f"entry={len(entry)} != {DEN_TENSOR_ENTRY_SIZE}"

            f.write(entry)
            data_pos += data_size

        # Pad to data offset
        pad = data_offset - f.tell()
        assert pad >= 0
        if pad > 0:
            f.write(b'\x00' * pad)

        # -- Tensor data --
        total_data_size = 0
        for slot, hw_target, raw_bytes, shape, name in entries:
            f.write(raw_bytes)
            total_data_size += len(raw_bytes)

    # -- Update total_data_size in header --
    with open(args.output, "r+b") as f:
        hdr_bytes = bytearray(f.read(DEN_HEADER_SIZE))
        struct.pack_into("<Q", hdr_bytes, 120, total_data_size)
        f.seek(0)
        f.write(hdr_bytes)

    file_size = os.path.getsize(args.output)
    print(f"  Written: {file_size:,} bytes ({num_tensors} tensors)")
    print(f"  Data: {total_data_size:,} bytes at offset {data_offset}")

if __name__ == "__main__":
    main()
