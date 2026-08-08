#!/usr/bin/env python3
"""
Convert NVFP4 safetensors (llm-compressor/modelopt) → GGUF with GGML_TYPE_NVFP4.

NVFP4 format (block_size=16, per llm-compressor standard):
  weight:        uint8 [out_features, in_features//2]    — E2M1 nibbles, 2/byte
  weight_scale:  uint8 [out_features, in_features//16]   — UE4M3 per-block scales
  weight_scale_2: uint8 [out_features, in_features//128] — optional second-level scale (UE8M0)

Usage:
  python convert_nvfp4_safetensors_to_gguf.py \
    --input I:\models\Huihui-gemma4-26B-NVFP4\model.safetensors \
    --config I:\models\Huihui-gemma4-26B-NVFP4\config.json \
    --output I:\models\gemma4-26b-nvfp4.gguf
"""
import struct, json, sys, os, argparse, math
from pathlib import Path
import numpy as np

try:
    import safetensors
except ImportError:
    print("pip install safetensors")
    sys.exit(1)

# GGUF constants
GGML_TYPE_NVFP4 = 40
GGML_TYPE_F32   = 0
GGML_TYPE_BF16  = 30
GGML_TYPE_F16   = 1

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# NVFP4 block layout constants (must match ggml)
NVFP4_BLOCK_SIZE = 16  # elements per scale

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)

def load_safetensors(path):
    """Load safetensors, return dict of tensor_name -> numpy array."""
    tensors = {}
    with safetensors.safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def write_gguf_header(f, metadata):
    """Write GGUF header with metadata key-value pairs."""
    f.write(struct.pack('<I', GGUF_MAGIC))
    f.write(struct.pack('<I', GGUF_VERSION))
    f.write(struct.pack('<Q', len(metadata)))  # n_tensors
    f.write(struct.pack('<Q', len(metadata)))  # n_kv (using same for simplicity)

    # Write metadata keys (simplified — just model info)
    # Real implementation would write proper GGUF metadata

def detect_nvfp4_format(tensors):
    """Detect NVFP4 tensor layout from safetensors.
    Returns dict mapping base_name -> (weight_tensor, scale_tensor, scale2_tensor or None)
    """
    # Scan for NVFP4 weight patterns
    weight_tensors = {}
    scale_tensors = {}
    scale2_tensors = {}

    for name in tensors.keys():
        if name.endswith('.weight_scale_2'):
            base = name[:-len('.weight_scale_2')]
            scale2_tensors[base] = name
        elif name.endswith('.weight_scale'):
            base = name[:-len('.weight_scale')]
            scale_tensors[base] = name
        elif name.endswith('.weight'):
            base = name[:-len('.weight')]
            weight_tensors[base] = name

    # Match weights with their scales
    nvfp4_layers = {}
    for base in weight_tensors:
        if base in scale_tensors:
            nvfp4_layers[base] = {
                'weight': weight_tensors[base],
                'scale': scale_tensors[base],
                'scale_2': scale2_tensors.get(base, None),
            }

    return nvfp4_layers

def convert_to_gguf(input_path, config_path, output_path):
    cfg = load_config(config_path)
    print(f"Loading safetensors from {input_path}...")
    tensors = load_safetensors(input_path)
    print(f"Loaded {len(tensors)} tensors")

    # Detect NVFP4 format
    nvfp4_layers = detect_nvfp4_format(tensors)
    print(f"\nDetected {len(nvfp4_layers)} NVFP4-quantized tensors")

    # Print tensor inventory
    print("\n=== TENSOR INVENTORY ===")
    for name, arr in sorted(tensors.items()):
        dtype = str(arr.dtype)
        shape = list(arr.shape)
        size_mb = arr.nbytes / (1024*1024)
        is_nvfp4 = any(name.endswith(s) for s in ['.weight', '.weight_scale', '.weight_scale_2'])
        tag = "NVFP4" if is_nvfp4 else "FP16/BF16"
        print(f"  {name:60s} {str(shape):20s} {dtype:8s} {size_mb:8.1f} MB  [{tag}]")

    # Analyze NVFP4 block format
    if nvfp4_layers:
        sample_base = list(nvfp4_layers.keys())[0]
        w = tensors[nvfp4_layers[sample_base]['weight']]
        s = tensors[nvfp4_layers[sample_base]['scale']]

        print(f"\n=== NVFP4 FORMAT ANALYSIS ===")
        print(f"Sample: {sample_base}")
        print(f"  Weight shape:      {list(w.shape)} ({w.dtype})")
        print(f"  Weight nbytes:     {w.nbytes}")
        print(f"  Scale shape:       {list(s.shape)} ({s.dtype})")
        print(f"  Scale nbytes:      {s.nbytes}")

        # Infer block_size from shapes
        # weight: [out_features, in_features // 2] (2 elements per byte)
        # scale:  [out_features, in_features // block_size]
        out_features = w.shape[0]
        in_features_packed = w.shape[1]
        in_features = in_features_packed * 2  # 2 elements per byte

        if len(s.shape) >= 2 and s.shape[0] == out_features:
            block_size = in_features // s.shape[1]
            print(f"  Inferred block_size = {block_size} (in_features={in_features}, scale_cols={s.shape[1]})")
        else:
            block_size = 16  # default
            print(f"  Using default block_size = {block_size}")

        # Check scale dtype
        scale_dtype = str(s.dtype)
        if 'float8' in scale_dtype or 'e4m3' in scale_dtype.lower():
            print(f"  Scale format: UE4M3 (float8_e4m3fn)")
        elif s.dtype == np.uint8:
            print(f"  Scale format: uint8 (custom UE4M3 LUT)")
        else:
            print(f"  Scale format: {scale_dtype} (raw)")

        # Check if weight is uint8 (packed E2M1)
        if w.dtype == np.uint8:
            print(f"  Weight format: packed E2M1 nibbles (uint8, 2/byte)")
        else:
            print(f"  Weight format: {w.dtype} (unexpected)")

        # Check for scale_2 (two-level quantization)
        has_scale2 = nvfp4_layers[sample_base]['scale_2'] is not None
        if has_scale2:
            s2 = tensors[nvfp4_layers[sample_base]['scale_2']]
            print(f"  Scale_2 shape:     {list(s2.shape)} ({s2.dtype})")
            print(f"  Two-level quantization: YES (scale + scale_2)")
        else:
            print(f"  Two-level quantization: NO (single scale)")

    # Count non-NVFP4 tensors (router, embeddings, lm_head, etc.)
    non_nvfp4 = {k: v for k, v in tensors.items()
                 if not any(k.endswith(s) for s in ['.weight_scale', '.weight_scale_2'])}
    nvfp4_weights = {k: v for k, v in tensors.items() if k.endswith('.weight')
                     and k[:-len('.weight')] in nvfp4_layers}
    nvfp4_scales = {k: v for k, v in tensors.items() if k.endswith('.weight_scale')
                    and k[:-len('.weight_scale')] in nvfp4_layers}

    # Compute model sizes
    total_bytes = sum(v.nbytes for v in tensors.values())
    nvfp4_weight_bytes = sum(v.nbytes for v in nvfp4_weights.values())
    nvfp4_scale_bytes = sum(v.nbytes for v in nvfp4_scales.values())
    non_nvfp4_bytes = sum(v.nbytes for v in non_nvfp4.values())

    print(f"\n=== SIZE BREAKDOWN ===")
    print(f"  Total safetensors:     {total_bytes/1e9:.2f} GB")
    print(f"  NVFP4 weight data:     {nvfp4_weight_bytes/1e9:.2f} GB ({nvfp4_weight_bytes*100/total_bytes:.1f}%)")
    print(f"  NVFP4 scale data:      {nvfp4_scale_bytes/1e9:.2f} GB ({nvfp4_scale_bytes*100/total_bytes:.1f}%)")
    print(f"  Non-NVFP4 (FP16/BF16): {non_nvfp4_bytes/1e9:.2f} GB ({non_nvfp4_bytes*100/total_bytes:.1f}%)")

    # Estimate GGUF size
    # NVFP4 weights store as-is (uint8 packed). Scales also as-is.
    # Non-NVFP4 tensors convert to BF16 (2 bytes/element)
    gguf_estimate = nvfp4_weight_bytes + nvfp4_scale_bytes
    for name, arr in non_nvfp4.items():
        if name.endswith('.weight') and name[:-len('.weight')] not in nvfp4_layers:
            # Non-quantized weight — convert to BF16
            gguf_estimate += arr.size * 2
        else:
            gguf_estimate += arr.nbytes

    print(f"\n  Estimated GGUF size:   {gguf_estimate/1e9:.2f} GB")
    print(f"  vs QAT Q4 GGUF:        ~14-16 GB")

    # Model architecture info
    hparams = cfg.get('hparams', cfg)
    arch = cfg.get('architectures', ['unknown'])[0] if 'architectures' in cfg else cfg.get('model_type', 'unknown')
    print(f"\n=== MODEL INFO ===")
    print(f"  Architecture:    {arch}")
    print(f"  Hidden size:     {hparams.get('hidden_size', '?')}")
    print(f"  Num layers:      {hparams.get('num_hidden_layers', '?')}")
    print(f"  Num experts:     {hparams.get('num_local_experts', '?')}")
    print(f"  Num KV heads:    {hparams.get('num_key_value_heads', '?')}")
    print(f"  Head dim:        {hparams.get('head_dim', '?')}")
    print(f"  Vocab size:      {hparams.get('vocab_size', '?')}")

    # Output: generate GGUF metadata for llama.cpp
    print(f"\n=== NEXT STEPS ===")
    print(f"  NVFP4 format: {'compatible' if nvfp4_layers else 'NO NVFP4 TENSORS FOUND'}")
    print(f"  To complete conversion:")
    print(f"  1. Verify block_size matches GGUF NVFP4 expectations (block_size=16)")
    print(f"  2. Write GGUF with GGML_TYPE_NVFP4 for quantized weights")
    print(f"  3. Map tensor names to llama.cpp tensor names")
    print(f"  4. Handle MTP head tensors (mtp.* prefix)")
    print(f"  Output: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='model.safetensors path')
    parser.add_argument('--config', required=True, help='config.json path')
    parser.add_argument('--output', default='model-nvfp4.gguf', help='output GGUF path')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found")
        print("Download the model first from HF:")
        print("  https://huggingface.co/sakamakismile/Huihui-gemma-4-26B-A4B-it-qat-abliterated-MTP-NVFP4")
        sys.exit(1)

    convert_to_gguf(args.input, args.config, args.output)
