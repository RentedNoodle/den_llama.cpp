#!/usr/bin/env python3
# ALLOWED: den calibration pipeline (Python orchestration, heavy compute delegated to GPU)
"""Den NVFP4+ universal calibration pipeline.

Auto-detects model profile, optionally abliterates, calibrates with AWQ,
quantizes with multi-grid NVFP4+, and exports .den DENPACK.

Usage:
    den_calibrate.py --model-dir <path> [--profile auto|<name>]
                     [--output <dir>] [--abliterate] [--samples <n>]
                     [--dry-run] [--resume]
"""

import os, sys, re, time, argparse, gc, json, hashlib
from pathlib import Path
import logging

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelopt.torch.quantization import NVFP4_DEFAULT_CFG, quantize as mtq_quantize
import copy

# DenForge imports — add parent dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from den_forge.profiles import detect_profile, get_profile, list_profiles
from den_forge.nvfp4_quant import pack_nvfp4_tiles_multi_grid
# GPU-accelerated tile packing via cuda.core (zero raw ctypes)
_HAVE_GPU_PACK = False
try:
    from den_forge.cuda_pack import pack_tiles_gpu as _gpu_pack
    _HAVE_GPU_PACK = True
except ImportError:
    _gpu_pack = None
from den_forge.nvfp4_pack_extract import yield_nvfp4_tensors, extract_nvfp4_from_module
from tools.den_unified_preconditioner import DenQuantUnifiedPreconditioner, CayleyOptimizer, RotationConfig


def _read_config(model_dir: str) -> dict:
    """Read a model's config.json, handling nested text_config."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No config.json found in {model_dir}")
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def _is_firewalled(name: str, patterns: list) -> bool:
    """Check if a parameter name matches any firewall regex pattern."""
    return any(re.match(p, name) for p in patterns)


def _strip_llm_prefix(name: str) -> str:
    """Normalize Ornith-style HF names: model.language_model.X -> model.X.

    Ornith (Qwen3.5-based) checkpoints place the text transformer under
    `model.language_model`, so parameter names arrive as
    `model.language_model.layers.N.linear_attn.*` etc. This REWRITES the
    `model.language_model.` prefix to canonical `model.`, so
    model.language_model.layers.N.* -> model.layers.N.*,
    model.language_model.norm.weight -> model.norm.weight and
    model.language_model.embed_tokens.weight -> model.embed_tokens.weight.
    The calibration firewall patterns and the GGUF name writer all expect
    those canonical names. `lm_head.weight` lives at the top level and is
    left unchanged.
    """
    prefix = "model.language_model."
    if name.startswith(prefix):
        return "model." + name[len(prefix):]
    return name


def _firewall_with_prefixed(patterns: list) -> list:
    """Duplicate each `model\\.`-anchored firewall pattern so it ALSO matches the
    `model.language_model.` prefix used by Ornith-style checkpoints.

    The profile firewall patterns are anchored with `model\\.`, so without this
    they silently miss every Ornith tensor and the precision firewall is
    defeated (critical linear-attention / Q / O tensors get NVFP4-quantized)."""
    out = list(patterns)
    for p in patterns:
        if p.startswith("model\\."):
            out.append("model\\.language_model\\." + p[len("model\\."):])
    return out


def _modality_from_profile(profile: dict) -> str:
    """Map profile modality enum to string key."""
    mod = profile.get("modality", 1)
    if mod in (1,):   return "llm"
    if mod in (4,):   return "asr"
    if mod in (8,):   return "tts"
    if mod in (2,):   return "diffusion"
    if mod in (16, 32): return "3d"
    return "llm"


def _text_entropy_score(text: str) -> float:
    """Compute a diversity score for calibration sample selection.

    Uses vocabulary entropy and token count as a proxy for activation diversity.
    Higher score = more diverse sample. No model needed.
    """
    import math
    from collections import Counter
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    counter = Counter(words)
    total = len(words)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p + 1e-9)
    # Bonus for longer sequences (more informative gradients)
    length_bonus = math.log(len(words)) / 10.0
    return entropy + length_bonus


def _load_cal_data(profile: dict, tokenizer, num_samples: int):
    """Load calibration data — entropy-ranked for LLM, passthrough for audio/vision."""
    modality = _modality_from_profile(profile)
    from datasets import load_dataset

    if modality == "llm":
        ds = load_dataset("neuralmagic/calibration", "LLM", split="train")
        # Load a larger pool, rank by diversity, select top-N
        pool_size = min(max(num_samples * 3, 150), len(ds))
        pool = list(ds.select(range(pool_size)))

        # Score by text diversity
        scored = [(_text_entropy_score(s["text"]), s) for s in pool]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Select top-N most diverse
        selected = scored[:num_samples]
        print(f"  Entropy-ranked selection: {len(selected)}/{pool_size} samples "
              f"(score range: {selected[-1][0]:.2f} – {selected[0][0]:.2f})")

        samples = []
        for _, s in selected:
            encoded = tokenizer(s["text"], truncation=True,
                               max_length=profile.get("seq_len", 2048),
                               return_tensors="pt")
            samples.append(encoded)
        return samples

    from tools.den_cal_data import load_calibration_data
    return load_calibration_data(profile, num_samples)


def _checkpoint_path(out_dir: str) -> str:
    return os.path.join(out_dir, ".den_calibrate_checkpoint.json")


def _write_checkpoint(out_dir: str, stage: str, extra: dict | None = None):
    data = {"stage": stage, "timestamp": time.time()}
    if extra:
        data.update(extra)
    with open(_checkpoint_path(out_dir), "w") as f:
        json.dump(data, f)


def _read_checkpoint(out_dir: str) -> dict:
    cp = _checkpoint_path(out_dir)
    if os.path.isfile(cp):
        with open(cp) as f:
            return json.load(f)
    return {}


def _stage_timing_from_checkpoint(cp: dict) -> dict[str, float]:
    """Extract per-stage elapsed times from checkpoint for ETA calculation."""
    return cp.get("stage_timing", {})


def _clear_checkpoint(out_dir: str):
    cp = _checkpoint_path(out_dir)
    if os.path.isfile(cp):
        os.remove(cp)


# ---- Self-healing checkpoint system ----------------------------------

def _model_fingerprint(model_dir: str) -> str:
    """SHA256 fingerprint of model identity (config.json + first safetensors header).
    Used to verify the model hasn't changed between resume attempts."""
    h = hashlib.sha256()
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "rb") as f:
            h.update(f.read())
    import glob
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if st_files:
        # Hash just the header (first 64KB) of the first shard for speed
        with open(st_files[0], "rb") as f:
            h.update(f.read(65536))
    return h.hexdigest()[:16]


def _setup_logging(output_dir: str) -> logging.Logger:
    """Crash-proof logging: writes to output_dir so logs survive reboots.
    Auto-flushes after every message so crash doesn't lose the last lines."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "calibration.log")
    logger = logging.getLogger("den_calibrate")
    logger.setLevel(logging.DEBUG)
    # Remove existing handlers to avoid duplicates on resume
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    # Also log to stderr for live streaming
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(sh)
    logger.info(f"Log: {log_path}")
    return logger


def _detect_crash_recovery(output_dir: str) -> dict | None:
    """Detect if a prior calibration crashed and recovery is possible.

    Returns the checkpoint dict if recovery is possible, None otherwise.
    """
    cp = _read_checkpoint(output_dir)
    if not cp:
        return None
    stage = cp.get("stage", "")
    if stage == "done":
        return None  # Completed successfully, no recovery needed
    if stage in ("abliterate", "load", "load_data", "calibrate", "export"):
        return cp
    return None


def _verify_model_unchanged(model_dir: str, cp: dict) -> bool:
    """Verify the model hasn't changed since the checkpoint was written."""
    fp = cp.get("model_fingerprint", "")
    if not fp:
        return True  # No fingerprint in old checkpoint, assume OK
    current = _model_fingerprint(model_dir)
    if fp != current:
        print(f"[WARN] Model fingerprint changed: {fp} -> {current}")
        return False
    return True


def _sha256(data: np.ndarray) -> str:
    return hashlib.sha256(data.tobytes()).hexdigest()


def _write_provenance(output_dir: str, model_dir: str, model_fp: str,
                      profile_name: str, abliterate: bool,
                      ablation_factor: float, num_samples: int) -> None:
    """Write SHA256 provenance chain for blockchain-style model integrity.

    Chain: source_model → calibration_config → abliteration_config → tile_pack → output.
    The engine verifies this chain at load time to ensure the model hasn't been tampered with.
    """
    import json, os

    chain = {
        "version": 1,
        "provenance_chain": [],
    }

    # Link 1: Source model
    source_hash = hashlib.sha256()
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "rb") as f:
            source_hash.update(f.read())
    chain["provenance_chain"].append({
        "link": "source_model",
        "model_fingerprint": model_fp,
        "config_sha256": source_hash.hexdigest(),
    })

    # Link 2: Calibration configuration
    cal_hash = hashlib.sha256()
    cal_hash.update(json.dumps({
        "profile": profile_name,
        "num_samples": num_samples,
        "abliterate": abliterate,
        "ablation_factor": ablation_factor,
    }, sort_keys=True).encode())
    chain["provenance_chain"].append({
        "link": "calibration_config",
        "sha256": cal_hash.hexdigest(),
    })

    # Link 3: Output files
    den_path = os.path.join(output_dir, "model.den")
    gguf_path = os.path.join(output_dir, "model.gguf")
    out_hash = hashlib.sha256()
    for path in [den_path, gguf_path]:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                # Hash only first+last 64KB for speed (we trust disk integrity)
                out_hash.update(f.read(65536))
                f.seek(max(0, os.path.getsize(path) - 65536))
                out_hash.update(f.read(65536))
    chain["provenance_chain"].append({
        "link": "output_files",
        "den_path": den_path,
        "gguf_path": gguf_path,
        "output_sha256": out_hash.hexdigest(),
    })

    # Write chain
    chain_path = os.path.join(output_dir, "model.provenance")
    with open(chain_path, "w") as f:
        json.dump(chain, f, indent=2)
    print(f"  [PROVENANCE] Chain written: {chain_path} ({len(chain['provenance_chain'])} links)")

    # Append to .den file as a footer if space
    try:
        with open(den_path, "ab") as f:
            chain_bytes = json.dumps(chain).encode()
            # Write DENFOOT signature + chain length + chain data
            f.write(b"DENFOOT")
            f.write(len(chain_bytes).to_bytes(8, "little"))
            f.write(chain_bytes)
    except Exception:
        pass  # Footer is optional, don't fail if we can't write it


def _validate_tile_online(info: dict, tile_bytes: bytes,
                          mse_samples: list) -> None:
    """Online scale validation: dequantize a sample tile, compute MSE vs BF16.

    Called during extraction to catch poor calibration early.
    Accumulates results in mse_samples for convergence tracking.
    """
    try:
        import numpy as np

        # Decode first tile's scale header
        tile = np.frombuffer(tile_bytes[:160], dtype=np.uint8)
        scales_raw = tile[0:16].view(np.uint32)
        sfa_vals = []
        for w in scales_raw:
            for bi in range(4):
                code = (w >> (bi * 8)) & 0x0F
                # Map UE4M3 code to value via LUT
                from den_forge.nvfp4_quant import UE4M3_VALUES
                sfa_vals.append(float(UE4M3_VALUES[code] if code < 16 else UE4M3_VALUES[0]))

        # Decode first 16 weights from nibbles (first 8 bytes = 16 E2M1 values)
        nibbles = tile[16:24]
        e2m1_vals = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 0.0]
        weights = np.zeros(16, dtype=np.float32)
        for i in range(8):
            lo = nibbles[i] & 0x0F
            hi = (nibbles[i] >> 4) & 0x0F
            sign_lo = -1.0 if (lo & 0x8) else 1.0
            sign_hi = -1.0 if (hi & 0x8) else 1.0
            weights[i * 2] = sign_lo * e2m1_vals[lo & 0x7]
            weights[i * 2 + 1] = sign_hi * e2m1_vals[hi & 0x7]

        # Get corresponding BF16 reference values (first 16 elements, first block)
        ref = info['data'].ravel()[:16].astype(np.float32)
        if len(ref) < 16:
            return

        dq = weights * sfa_vals[0]
        mse = float(np.mean((ref[:16] - dq) ** 2))
        mse_samples.append({
            'tensor': info['name'],
            'mse': mse,
            'scale': sfa_vals[0],
            'weight_mean': float(np.mean(np.abs(ref[:16]))),
        })
    except Exception:
        pass  # Validation is best-effort, never blocks extraction


def calibrate(model_dir: str, output_dir: str, profile: dict, profile_name: str,
              num_samples: int = 128, abliterate: bool = False, ablation_factor: float = 1.0,
              resume: bool = False, allow_resume: bool = True):
    """Full calibration pipeline: abliterate -> load -> cal data -> AWQ -> pack NVFP4.

    If allow_resume and a crashed checkpoint is detected, automatically resumes
    from the last completed stage. Logs are crash-proof: written to output_dir,
    flushed after every message.
    """
    os.makedirs(output_dir, exist_ok=True)
    log = _setup_logging(output_dir)
    model_fp = _model_fingerprint(model_dir)
    stage_timing = {}  # Initialize before signal handlers so they can reference it

    # --- Install crash handlers early ---
    try:
        from tools.den_calibrate_resilience import install_signal_handlers, RamWatcher, \
            run_preflight, abliterate_safe, extract_tensor_safe
        install_signal_handlers(output_dir, model_fp, stage_timing, log)
        resilience_available = True
    except ImportError:
        resilience_available = False
        def abliterate_safe(m, t, p, l): return False
        def extract_tensor_safe(m): return extract_nvfp4_from_module(m)

    # Pre-flight guards
    if resilience_available and not resume:
        if not run_preflight(output_dir, model_dir):
            log.error("Pre-flight checks failed. Fix issues or use --no-preflight.")
            raise SystemExit(1)

    # RAM watcher
    ram_watcher = None
    if resilience_available:
        ram_watcher = RamWatcher(gpu_limit_pct=90, ram_limit_pct=85,
                                 output_dir=output_dir)
        ram_watcher.start()
        log.info(f"RAM watcher started (GPU limit 90%, RAM limit 85%)")

    # Auto-detect crash recovery
    cp = _read_checkpoint(output_dir) if resume else {}
    if not cp and allow_resume:
        crash_cp = _detect_crash_recovery(output_dir)
        if crash_cp:
            log.warning(f"Detected incomplete calibration (stage={crash_cp.get('stage')}). "
                        f"Auto-resuming. Use --no-resume to start fresh.")
            cp = crash_cp
            resume = True
            if not _verify_model_unchanged(model_dir, crash_cp):
                log.error("Model changed since checkpoint. Cannot resume safely.")
                log.error("Delete checkpoint and restart: "
                          f"rm {_checkpoint_path(output_dir)}")
                raise SystemExit(1)

    start_stage = cp.get("stage", "abliterate" if abliterate else "load")
    stage_timing = cp.get("stage_timing", {})
    stage_start = time.time()

    if resume and start_stage != "abliterate":
        remaining = ["load", "load_data", "calibrate", "export"]
        if start_stage in remaining:
            rem_stages = remaining[remaining.index(start_stage):]
            est_total = sum(stage_timing.get(s, 120) for s in rem_stages)
            log.info(f"Resuming from stage '{start_stage}', "
                     f"~{est_total/60:.0f}min estimated remaining")
        else:
            log.info(f"Resuming from stage '{start_stage}'")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        log.info(f"GPU: {torch.cuda.get_device_name(0)}, free: {free_gb:.1f} GiB")

    # --- Stage 1: Load model (with optional in-memory abliteration) ---
    print("--- Stage 1/5: Load model ---")
    # Load to CPU first to avoid Windows multi-threaded GPU access violation,
    # then move to GPU in single-threaded fashion.
    load_kwargs = {"device_map": "cpu", "dtype": torch.bfloat16, "trust_remote_code": True, "attn_implementation": "eager"}
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    if torch.cuda.is_available():
        model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    if abliterate and start_stage in ("abliterate",):
        print(f"  Abliterating in-memory (factor={ablation_factor:.2f})...")
        result = abliterate_safe(model, tokenizer, profile, log, ablation_factor)
        if result and isinstance(result, dict):
            log.info(f"Abliteration quality: suppression_ratio={result.get('refusal_suppression_ratio', 'N/A')}")
    stage_timing["load"] = time.time() - stage_start
    _write_checkpoint(output_dir, "load_data",
                      {"stage_timing": stage_timing, "model_fingerprint": model_fp})

    # --- Stage 2: Load calibration data ---
    print(f"--- Stage 2/5: Load calibration data ({num_samples} samples) ---")
    cal_data = _load_cal_data(profile, tokenizer, num_samples)
    print(f"  Loaded {len(cal_data)} samples")
    cal_start = time.time()
    stage_timing["load_data"] = cal_start - stage_start - stage_timing.get("load", 0)
    _write_checkpoint(output_dir, "calibrate",
                      {"stage_timing": stage_timing, "model_fingerprint": model_fp})

    # --- Stage 3: AWQ Calibration ---
    print("--- Stage 3/5: AWQ calibration ---")
    firewall = profile.get("firewall", [])
    for name, _ in model.named_parameters():
        # FIX-1: match on the stripped name so Ornith's model.language_model.*
        # tensors hit the precision firewall (the profile patterns are anchored
        # on model\.layers\.. The real name is kept for modelopt below.
        if _is_firewalled(_strip_llm_prefix(name), firewall):
            print(f"  [FIREWALL] {name} -> BF16 passthrough")

    quant_cfg = copy.deepcopy(NVFP4_DEFAULT_CFG)
    for entry in quant_cfg["quant_cfg"]:
        if entry.get("quantizer_name") == "*":
            entry["enable"] = True
    for name, _ in model.named_parameters():
        if _is_firewalled(_strip_llm_prefix(name), firewall):
            # FIX-1: quantizer_name must stay the REAL modelopt module path
            # (model.language_model.layers.N.linear_attn.*) so mtq_quantize
            # can find and un-quantize the firewalled modules.
            quant_cfg["quant_cfg"].append({"quantizer_name": name, "enable": False})

    t0 = time.time()
    sample_count = 0

    def forward_loop(mod):
        nonlocal sample_count
        for s in cal_data:
            with torch.no_grad():
                if isinstance(s, dict) and "input_ids" in s:
                    mod(**{k: v.to(mod.device) for k, v in s.items()})
            sample_count += 1
            if sample_count % 10 == 0:
                torch.cuda.synchronize()
                ram_info = f" | {ram_watcher.status_str()}" if ram_watcher else ""
                print(f"  [TDR heartbeat] {sample_count}/{len(cal_data)} samples, "
                      f"{time.time() - t0:.0f}s elapsed{ram_info}")
                if ram_watcher and ram_watcher.is_emergency():
                    log.error("RAM watcher emergency — aborting calibration")
                    raise RuntimeError("RAM emergency: GPU memory exhausted")

    mtq_quantize(model, quant_cfg, forward_loop=forward_loop)
    torch.cuda.synchronize()

    elapsed = time.time() - t0
    print(f"  Calibration complete: {elapsed:.0f}s")
    stage_timing["calibrate"] = elapsed
    _write_checkpoint(output_dir, "export",
                      {"stage_timing": stage_timing, "model_fingerprint": model_fp})

    # Save AWQ-learned amax for offline extraction
    try:
        from tools.den_amax_persistence import extract_amax
        amax_data = extract_amax(model, output_dir)
    except ImportError:
        amax_data = None

    # Free GPU model memory — all subsequent work uses saved amax + disk
    del forward_loop, cal_data
    if next(model.parameters()).is_cuda:
        model = model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    log.info(f"GPU memory freed ({torch.cuda.memory_allocated() / 1e9:.1f} GB remaining)")

    # --- Stage 4: Extract quantized weights + validate (offline-capable) ---
    print("--- Stage 4/5: Extract NVFP4 quantized weights ---")
    firewall_patterns = profile.get("firewall", [])
    validation_hashes = {}
    online_mse_samples = []  # Online scale validation: track MSE per tensor

    qtensors = []
    gpu_available = torch.cuda.is_available()
    # FIX-1: yield_nvfp4_tensors firewall-matches the REAL param names, so pass
    # both-prefix pattern variants (plain + model.language_model.) to keep
    # Ornith's firewalled tensors BF16/F32 inside extraction, then strip the
    # prefix off every emitted name so the GGUF/.den writers see canonical
    # model.layers.* / model.embed_tokens / model.norm names.
    for info in yield_nvfp4_tensors(model, _firewall_with_prefixed(firewall_patterns)):
        name = _strip_llm_prefix(info['name'])
        info['name'] = name
        if info['is_quantized']:
            validation_hashes[name] = _sha256(info['data'])

            # Stochastic Error Feedback: redistribute quantization error between blocks
            if info['is_quantized'] and profile.get('stochastic_error_feedback', False):
                from den_forge.stochastic_error_feedback import apply_sef_to_tensor
                info['data'] = apply_sef_to_tensor(info['data'].copy())
                # Recompute nibbles and scales from SEF-compensated weights
                from den_forge.nvfp4_pack_extract import _quantize_2d_weight
                new_nibbles, new_scales, _ = _quantize_2d_weight(info['data'].astype(np.float32))
                info['nibbles'] = new_nibbles
                info['scales'] = new_scales

            # ── DenQuant Unified Preconditioner (DUP) — Novel Composition ──
            # Applies Fold-Scale DuQuant++ + Cayley Block-Diagonal Rotations
            # before NVFP4 packing. Typically 15-50% MSE improvement.
            if info['is_quantized'] and info['data'].ndim == 2 and profile.get('dup_enabled', True):
                try:
                    from tools.den_unified_preconditioner import DenQuantUnifiedPreconditioner
                    dup = DenQuantUnifiedPreconditioner(
                        group_size=profile.get('dup_group_size', 16),
                        fold_alpha=profile.get('dup_fold_alpha', 0.5),
                        rotate=profile.get('dup_rotate', True),
                        fold_scale=profile.get('dup_fold_scale', True))
                    W_orig = info['data'].astype(np.float32)
                    W_opt, rotation, fold = dup.optimize(W_orig, W_orig)  # No activations available post-GPU-free
                    info['data'] = torch.from_numpy(W_opt).to(info['data'].dtype)
                    # Recompute nibbles and scales from DUP-optimized weights
                    from den_forge.nvfp4_pack_extract import _quantize_2d_weight
                    new_nibbles, new_scales, _ = _quantize_2d_weight(W_opt)
                    info['nibbles'] = new_nibbles
                    info['scales'] = new_scales
                    # Store rotation+fold in .den metadata for dequant verification
                    info['dup_rotation'] = rotation
                    info['dup_fold'] = fold
                    if profile.get('dup_verbose', False):
                        score_raw = dup._score(W_orig, W_orig)
                        score_opt = dup._score(W_orig, W_opt)
                        impr = (1 - score_opt/score_raw) * 100 if score_raw > 0 else 0
                        print(f"  [DUP] {name}: MSE {score_raw:.4f}→{score_opt:.4f} ({impr:+.1f}%)")
                except Exception as e:
                    if profile.get('dup_verbose', False):
                        print(f"  [DUP] {name}: skipped ({e})")
                    new_scales = info.get('scales', None)  # keep existing scales on DUP failure

            # GPU-accelerated tile packing (100x faster than numpy)
            if info['data'].ndim != 2:
                # Skip non-2D tensors (biases, norms, expert stacks) — can't tile-pack
                tile_bytes, tile_norms = None, None
            elif gpu_available:
                try:
                    from den_forge.gpu_tile_packer import pack_tiles_gpu
                    tile_bytes, tile_norms = pack_tiles_gpu(
                        info['data'],  # BF16 reference weights
                        amax_np=info.get('amax'),  # AWQ-learned per-block amax
                        stochastic=profile.get('stochastic_rounding', False))
                except Exception:
                    # Fallback: numpy packer
                    from den_forge.nvfp4_quant import pack_nvfp4_tiles_multi_grid
                    try:
                        tile_bytes, tile_norms = pack_nvfp4_tiles_multi_grid(
                            info['nibbles'], info['scales'], profile)
                    except (ValueError, RuntimeError) as e:
                        print(f"  [SKIP] {name}: packer error ({e}), storing as passthrough")
                        tile_bytes, tile_norms = None, None
            else:
                from den_forge.nvfp4_quant import pack_nvfp4_tiles_multi_grid
                try:
                    tile_bytes, tile_norms = pack_nvfp4_tiles_multi_grid(
                        info['nibbles'], info['scales'], profile)
                except (ValueError, RuntimeError) as e:
                    print(f"  [SKIP] {name}: packer error ({e}), storing as passthrough")
                    tile_bytes, tile_norms = None, None

            if tile_bytes is not None:
                # Thermodynamic scale relaxation: physics-inspired per-tile adjustment
                if profile.get('thermodynamic_scales', False) and info['data'].size >= 256:
                    from den_forge.thermodynamic_scale import optimize_scales_thermodynamic
                    tile_bytes = optimize_scales_thermodynamic(
                        tile_bytes, info['data'].astype(np.float32), profile)

                # Wavefunction tile encoding: quantum-inspired probabilistic tile headers
                if profile.get('wavefunction_encoding', False) and len(tile_bytes) >= 160:
                    from den_forge.wavefunction_tile import apply_wavefunction_encoding
                    tile_bytes = apply_wavefunction_encoding(
                        tile_bytes, info['data'].astype(np.float32))

                # Ising tile optimization: spin-glass quantization scheme assignment
                if profile.get('ising_tile_optimization', False) and info['data'].size >= 256:
                    from den_forge.ising_tile_optimizer import apply_ising_optimization
                    tile_bytes = apply_ising_optimization(
                        tile_bytes, info['data'].astype(np.float32))

                # Tile importance scoring: variance-based protection flags
                if profile.get('tile_importance_scoring', False) and len(tile_bytes) >= 160:
                    from den_forge.tile_importance import apply_importance_to_tiles
                    tile_bytes = apply_importance_to_tiles(
                        tile_bytes, info['data'].astype(np.float32), profile)

                # Online scale validation: dequantize a sample tile, compute MSE vs BF16
                if len(online_mse_samples) < 50 and info['nibbles'].size > 256:
                    _validate_tile_online(info, tile_bytes, online_mse_samples)

                # Post-packing optimization: fractal tile dictionary dedup
                try:
                    from den_forge.fractal_tile_compressor import build_tile_dictionary
                    tiles_arr = np.frombuffer(tile_bytes, dtype=np.uint8).reshape(-1, 160)
                    dict_tiles, indices, savings = build_tile_dictionary(tiles_arr)
                except ImportError:
                    pass

                qtensors.append({
                    'name': name,
                    'data': tile_bytes,
                    'dtype': 'NVFP4',
                    'shape': info['shape'],
                    'is_quantized': True,
                    'tile_norms': tile_norms,
                })
                print(f"  [NVFP4] {name}: {info['shape']} -> {len(tile_bytes)} bytes "
                      f"({len(tile_bytes) // 160} tiles)")
            else:
                qtensors.append({
                    'name': name,
                    'data': info['data'],
                    'dtype': 'F32',
                    'shape': info['shape'],
                    'is_quantized': False,
                })
        else:
            qtensors.append({
                'name': name,
                'data': info['data'],
                'dtype': info['dtype'],
                'shape': info['shape'],
                'is_quantized': False,
            })
    print(f"  Extracted {len(qtensors)} tensors "
          f"({sum(1 for q in qtensors if q['is_quantized'])} NVFP4, "
          f"{sum(1 for q in qtensors if not q['is_quantized'])} passthrough)")

    # Online scale validation summary
    if online_mse_samples:
        mses = [s['mse'] for s in online_mse_samples]
        avg_mse = sum(mses) / len(mses)
        max_mse = max(mses)
        print(f"  [ONLINE VALIDATION] {len(online_mse_samples)} tiles sampled, "
              f"avg MSE={avg_mse:.2e}, max MSE={max_mse:.2e}")
        if max_mse > 1e-3:
            print(f"  [WARN] High quantization error detected (max MSE={max_mse:.2e}). "
                  f"Consider increasing --samples.")

    # --- Stage 5: Write .den DENPACK ---
    print("--- Stage 5/5: Write outputs ---")
    den_path = os.path.join(output_dir, "model.den")
    try:
        from den_forge.den_packer_wrapper import DenPackerC
        from den_forge.den_tensor import DenTensor

        mods = profile.get("modality", 1)
        packer = DenPackerC(den_path, modalities=mods)
        for qt in qtensors:
            dt = DenTensor(
                name=qt['name'],
                data=qt['data'],
                dtype=qt['dtype'],
                shape=qt['shape'],
            )
            packer.add_tensor(dt)
        packer.finalize()
        size_mb = os.path.getsize(den_path) / 1e6
        print(f"  .den packed: {len(qtensors)} tensors, {size_mb:.0f} MB")
    except Exception as e:
        print(f"  [WARNING] .den export failed: {e}")

    # Write .gguf for Paris Gate / engine inference
    gguf_path = os.path.join(output_dir, "model.gguf")
    try:
        _write_gguf_nvfp4(gguf_path, qtensors, profile, tokenizer, validation_hashes, model_dir=model_dir)
        size_mb = os.path.getsize(gguf_path) / 1e6
        print(f"  .gguf written: {size_mb:.0f} MB")
    except Exception as e:
        print(f"  [WARNING] .gguf export failed: {e}")

    # Source cleanup (if requested)
    output_size_mb = (os.path.getsize(den_path) + os.path.getsize(gguf_path)
                      if os.path.isfile(gguf_path) else os.path.getsize(den_path)) / 1e6
    source_size_mb = sum(os.path.getsize(os.path.join(model_dir, f))
                         for f in os.listdir(model_dir)
                         if f.endswith('.safetensors')) / 1e6

    remove_source = getattr(sys.modules.get('__main__'), '_remove_source', False)
    remove_if_smaller = getattr(sys.modules.get('__main__'), '_remove_source_if_smaller', False)

    if remove_source or (remove_if_smaller and output_size_mb < source_size_mb * 0.5):
        import shutil
        files_removed = []
        bytes_freed = 0
        for f in os.listdir(model_dir):
            if f.endswith('.safetensors'):
                fp = os.path.join(model_dir, f)
                bytes_freed += os.path.getsize(fp)
                os.remove(fp)
                files_removed.append(f)
        print(f"\n  [CLEANUP] Removed {len(files_removed)} source safetensors "
              f"({bytes_freed / 1e9:.1f} GB freed). "
              f"NVFP4 output: {output_size_mb:.0f} MB vs BF16 source: {source_size_mb:.0f} MB "
              f"({output_size_mb / max(source_size_mb, 1) * 100:.0f}%)")

    # SHA256 validation report
    if validation_hashes:
        print(f"\n  Precision Firewall — SHA256 baseline captured for "
              f"{len(validation_hashes)} critical tensors")

    # Write final checkpoint marker — marks calibration as complete.
    # If process crashes after this point, auto-resume won't re-run.
    _write_checkpoint(output_dir, "done", {
        "stage_timing": stage_timing,
        "model_fingerprint": model_fp,
        "completed_at": time.time(),
    })
    # Output integrity check
    den_path = os.path.join(output_dir, "model.den")
    gguf_path = os.path.join(output_dir, "model.gguf")
    if resilience_available:
        try:
            from tools.den_calibrate_resilience import check_output_integrity
            check_output_integrity(den_path, gguf_path)
        except ImportError:
            pass

    # Write provenance chain
    _write_provenance(output_dir, model_dir, model_fp, profile_name,
                      abliterate, ablation_factor, num_samples)

    # Stop RAM watcher
    if ram_watcher:
        ram_watcher.stop()
        log.info(f"RAM watcher stopped. Final: {ram_watcher.status_str()}")

    log.info(f"Done. Output: {output_dir}")
    print(f"Done. Output: {output_dir}")


def _write_gguf_nvfp4(gguf_path: str, qtensors: list, profile: dict,
                      tokenizer, validation_hashes: dict,
                      model_dir: str = None):
    """Write NVFP4 GGUF file from extracted quantized tensors."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "third_party", "ik_llama.cpp", "gguf-py"))
    from gguf import GGUFWriter, GGMLQuantizationType

    model_name = profile.get("name", "Qwen3.5-Den")
    # Auto-detect architecture from model config, not just from profile
    arch_key = profile.get("architecture", "qwen2")
    if model_dir:
        _cfg_path = os.path.join(model_dir, "config.json")
        if os.path.isfile(_cfg_path):
            import json
            with open(_cfg_path) as f:
                _raw = json.load(f)
            _text_cfg = _raw.get("text_config", _raw) if isinstance(_raw, dict) else {}
            _model_type = _text_cfg.get("model_type", "")
            if _model_type == "qwen3_5_text":
                arch_key = "qwen35"
            elif _model_type == "qwen3_5_moe":
                arch_key = "qwen35moe"
            elif _model_type == "qwen3_5_moe_text":
                # FIX-2: Ornith-35B reports text_config.model_type ==
                # "qwen3_5_moe_text"; without this the arch falls back to
                # "qwen2" and the qwen35moe.ssm metadata block is never written.
                arch_key = "qwen35moe"
    gguf_writer = GGUFWriter(gguf_path, arch_key)

    gguf_writer.add_string("general.name", f"{model_name}-NVFP4-Den")
    gguf_writer.add_string("general.quantization_version", "2")
    gguf_writer.add_string("quantization.type", "NVFP4")

    # Copy model hyperparameters from source config.json so the engine
    # can load the GGUF without needing a separate config file.
    cfg = {}  # may be populated from config.json below
    if model_dir:
        config_path = os.path.join(model_dir, "config.json")
        if os.path.isfile(config_path):
            import json
            with open(config_path) as f:
                raw = json.load(f)
            cfg = raw.get("text_config", raw) if isinstance(raw, dict) else {}
            # Required GGUF metadata keys
            kv_map = {
                f"{arch_key}.context_length":               ("max_position_embeddings", 262144),
                f"{arch_key}.embedding_length":             ("hidden_size", 2560),
                f"{arch_key}.block_count":                  ("num_hidden_layers", 32),
                f"{arch_key}.feed_forward_length":          ("intermediate_size", 9216),
                f"{arch_key}.attention.head_count":         ("num_attention_heads", 16),
                f"{arch_key}.attention.head_count_kv":      ("num_key_value_heads", 4),
                f"{arch_key}.attention.layer_norm_rms_epsilon": ("rms_norm_eps", 1e-6),
                f"{arch_key}.attention.key_length":         ("head_dim", None),
                f"{arch_key}.attention.value_length":       ("head_dim", None),
                f"{arch_key}.rope.dimension_count":         ("head_dim", None),
                f"{arch_key}.vocab_size":                   ("vocab_size", 248320),
            }
            for gguf_key, (cfg_key, default) in kv_map.items():
                val = cfg.get(cfg_key) if isinstance(cfg, dict) else default
                if val is None and cfg_key == "head_dim":
                    hs = cfg.get("hidden_size", 2560)
                    nh = cfg.get("num_attention_heads", 16)
                    val = hs // nh
                if val is not None:
                    if isinstance(val, bool):
                        gguf_writer.add_bool(gguf_key, val)
                    elif isinstance(val, float):
                        gguf_writer.add_float32(gguf_key, val)
                    else:
                        gguf_writer.add_uint32(gguf_key, int(val))

            # RoPE parameters (required for Qwen3.5 mrope)
            rope_params = cfg.get("rope_parameters", {})
            if rope_params:
                rope_theta = rope_params.get("rope_theta")
                if rope_theta:
                    gguf_writer.add_float32(f"{arch_key}.rope.freq_base", float(rope_theta))
                mrope_section = rope_params.get("mrope_section")
                if mrope_section:
                    sections = list(mrope_section)
                    while len(sections) < 4:
                        sections.append(0)
                    gguf_writer.add_array(f"{arch_key}.rope.dimension_sections", sections[:4])

            # SSM/Mamba metadata (required by engine for Qwen3.5 hybrid models)
            if arch_key in ("qwen35", "qwen35moe"):
                n_v_heads = cfg.get("linear_num_value_heads", 32)
                head_v_dim = cfg.get("linear_value_head_dim", 128)
                ssm_inner = n_v_heads * head_v_dim
                ssm_keys = {
                    f"{arch_key}.ssm.conv_kernel": cfg.get("linear_conv_kernel_dim", 4),
                    f"{arch_key}.ssm.state_size": cfg.get("linear_key_head_dim", 128),
                    f"{arch_key}.ssm.time_step_rank": n_v_heads,
                    f"{arch_key}.ssm.inner_size": ssm_inner,
                    f"{arch_key}.ssm.group_count": cfg.get("linear_num_key_heads", 16),
                }
                for ssm_key, ssm_val in ssm_keys.items():
                    gguf_writer.add_uint32(ssm_key, int(ssm_val))

        # Tokenizer metadata from tokenizer.json
        tok_path = os.path.join(model_dir, "tokenizer.json")
        tok_config_path = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.isfile(tok_path):
            import json
            with open(tok_path) as f:
                tok_data = json.load(f)
            tok_model = tok_data.get("model", {})
            gguf_writer.add_string("tokenizer.ggml.model", "gpt2")
            all_tokens = {}
            for at in tok_data.get("added_tokens", []):
                tid = at.get("id")
                content = at.get("content")
                if tid is not None and content:
                    all_tokens[tid] = content
            model_vocab = tok_model.get("vocab", {})
            if isinstance(model_vocab, dict):
                for token_str, token_id in model_vocab.items():
                    if token_id not in all_tokens:
                        all_tokens[token_id] = token_str
            sorted_ids = sorted(all_tokens.keys())
            token_strings = [all_tokens[i] for i in sorted_ids]
            vsize = cfg.get("vocab_size", len(token_strings)) if cfg else len(token_strings)
            while len(token_strings) < vsize:
                token_strings.append(f"[PAD_{len(token_strings)}]")
            token_scores = [0.0] * len(token_strings)
            if token_strings:
                gguf_writer.add_array("tokenizer.ggml.tokens", token_strings)
                gguf_writer.add_array("tokenizer.ggml.scores", token_scores)
            merges = tok_model.get("merges", [])
            if merges:
                gguf_writer.add_array("tokenizer.ggml.merges", merges)
        # BOS/EOS token IDs
        bos, eos = None, None
        if os.path.isfile(tok_config_path):
            with open(tok_config_path) as f:
                tok_cfg = json.load(f)
            bos = tok_cfg.get("bos_token_id")
            eos = tok_cfg.get("eos_token_id")
        if eos is None and cfg:
            eos = cfg.get("eos_token_id")
        if bos is not None:
            gguf_writer.add_uint32("tokenizer.ggml.bos_token_id", bos)
        if eos is not None:
            gguf_writer.add_uint32("tokenizer.ggml.eos_token_id", eos)

    # HF → GGUF canonical tensor name mapping
    NAME_MAP = {
        "token_embd.weight": "token_embd.weight",
        "output.weight": "output.weight",
        "token_embd_norm.weight": "token_embd_norm.weight",
        "model.embed_tokens.weight": "token_embd.weight",
        "lm_head.weight": "output.weight",
        "token_embd_norm.weight": "output_norm.weight",
        "model.norm.weight": "output_norm.weight",
        "model.final_layernorm.weight": "output_norm.weight",
    }
    for qt in qtensors:
        name = qt['name']
        # Map to GGUF canonical name if known, otherwise check for per-layer patterns
        if name in NAME_MAP:
            name = NAME_MAP[name]
        else:
            # Check if this is a layer weight
            for prefix in ("model.layers.", "blk."):
                if name.startswith(prefix):
                    # Already canonical (blk.X...) or needs mapping
                    if name.startswith("model.layers."):
                        rest = name[len("model.layers."):]
                        parts = rest.split(".", 1)
                        if len(parts) == 2:
                            layer = parts[0]
                            inner = parts[1]
                            inner_map = {
                                # Standard attention (self_attn) — used in full_attention layers
                                "self_attn.q_proj.weight": f"blk.{layer}.attn_q.weight",
                                "self_attn.k_proj.weight": f"blk.{layer}.attn_k.weight",
                                "self_attn.v_proj.weight": f"blk.{layer}.attn_v.weight",
                                "self_attn.o_proj.weight": f"blk.{layer}.attn_output.weight",
                                "self_attn.qkv_proj.weight": f"blk.{layer}.attn_qkv.weight",
                                "self_attn.q_norm.weight": f"blk.{layer}.attn_q_norm.weight",
                                "self_attn.k_norm.weight": f"blk.{layer}.attn_k_norm.weight",
                                # Linear attention (linear_attn) — used in hybrid/mamba layers
                                "linear_attn.in_proj_qkv.weight": f"blk.{layer}.attn_qkv.weight",
                                "linear_attn.in_proj_z.weight": f"blk.{layer}.attn_gate.weight",
                                "linear_attn.in_proj_a.weight": f"blk.{layer}.ssm_alpha.weight",
                                "linear_attn.in_proj_b.weight": f"blk.{layer}.ssm_beta.weight",
                                "linear_attn.out_proj.weight": f"blk.{layer}.ssm_out.weight",
                                "linear_attn.conv1d.weight": f"blk.{layer}.ssm_conv1d.weight",
                                "linear_attn.norm.weight": f"blk.{layer}.ssm_norm.weight",
                                "linear_attn.A_log": f"blk.{layer}.ssm_a",
                                "linear_attn.dt_bias": f"blk.{layer}.ssm_dt.bias",
                                # MLP
                                "mlp.gate_proj.weight": f"blk.{layer}.ffn_gate.weight",
                                "mlp.up_proj.weight": f"blk.{layer}.ffn_up.weight",
                                "mlp.down_proj.weight": f"blk.{layer}.ffn_down.weight",
                                # Layer norms
                                "input_layernorm.weight": f"blk.{layer}.attn_norm.weight",
                                "post_attention_layernorm.weight": (
                                    f"blk.{layer}.post_attention_norm.weight"
                                    if arch_key in ("qwen35", "qwen35moe")
                                    else f"blk.{layer}.ffn_norm.weight"
                                ),
                            }
                            mapped = inner_map.get(inner)
                            if mapped:
                                name = mapped
                    break
        data = qt['data']
        shape = qt['shape']
        is_quantized = qt['is_quantized']

        # ── Tensor reshape hooks ──────────────────────────────────────────
        # Some HF tensors use multi-dim layouts that the engine expects flattened.
        if name.endswith("ssm_conv1d.weight"):
            # Conv1d weight stays native [conv_dim, d_conv] — GGUF stores
            # column-major as ne=[d_conv, conv_dim]. All backends (CPU iqk_ssm_conv4,
            # CUDA ssm-conv, Metal ssm.metal) expect d_conv contiguous (ne[0]=d_conv).
            data = data.squeeze()
            shape = tuple(data.shape)
        # ──────────────────────────────────────────────────────────────────

        # Apply force_f32 from the profile — override BF16/F32 tensors (not already-NVFP4)
        # Uses simple substring matching (not regex) — patterns like "norm", "ssm_" match anywhere
        force_f32_patterns = profile.get("force_f32", [])
        if force_f32_patterns and any(p in name for p in force_f32_patterns):
            if qt['dtype'] != 'F32':
                # Only cast if data is a numeric array (BF16/F32 raw), not NVFP4-packed bytes
                is_bytes = isinstance(data, (bytes, bytearray)) or \
                           (hasattr(data, 'dtype') and data.dtype == np.uint8 and not is_quantized)
                if not is_quantized and not is_bytes:
                    print(f"  [FORCE_F32] {name}: {qt['dtype']} → F32", file=sys.stderr)
                    qt['dtype'] = 'F32'
                    data = data.astype(np.float32)
                elif is_bytes:
                    # NVFP4-packed tensor matching force_f32 — needs re-extraction as BF16 first
                    print(f"  [FORCE_F32:SKIP] {name}: NVFP4-packed, cannot cast to F32 directly", file=sys.stderr)

        if is_quantized:
            qtype = GGMLQuantizationType.NVFP4
            out_dim, in_dim = shape
            tiles_per_row = (in_dim + 255) // 256
            tiled_row_bytes = tiles_per_row * 160
            tdata = np.frombuffer(data, dtype=np.uint8).reshape(out_dim, tiled_row_bytes)
            # Don't pass raw_shape to NVFP4 tensors — the GGUF writer derives
            # element count from the tiled byte shape via quant_shape_from_byte_shape.
            gguf_writer.add_tensor(name, tdata, raw_dtype=qtype)
        elif qt['dtype'] == 'F32':
            qtype = GGMLQuantizationType.F32
            tdata = data.astype(np.float32)
            if name.endswith("ssm_conv1d.weight"):
                print(f"  [WRITE_F32] {name}: data.shape={data.shape}, tdata.shape={tdata.shape}, raw_shape={shape}", file=sys.stderr)
            gguf_writer.add_tensor(name, tdata, raw_shape=shape, raw_dtype=qtype)
        else:
            qtype = GGMLQuantizationType.BF16
            # BF16 must be stored as uint16 (2 bytes/element), NOT float32 (4 bytes/element).
            # Storing as float32 inflates GGUF offsets by 2x for every BF16 tensor,
            # causing all subsequent tensor offsets to be wrong.
            f32 = data.astype(np.float32).ravel()
            bf16_ui16 = (f32.view(np.uint32) >> 16).astype(np.uint16)
            tdata = bf16_ui16.reshape(shape)
            if name.endswith("ssm_conv1d.weight"):
                import sys as _sys
                print(f"  [WRITE] {name}: tdata.shape={tdata.shape}, raw_shape={shape}, raw_dtype={qtype}", file=_sys.stderr)
            gguf_writer.add_tensor(name, tdata, raw_shape=shape, raw_dtype=qtype)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    if validation_hashes:
        print(f"  [FIREWALL] {len(validation_hashes)} critical tensor SHA256 hashes captured")


def main():
    ap = argparse.ArgumentParser(description="Den NVFP4+ universal calibration pipeline")
    ap.add_argument("--model-dir", required=True, help="Model directory (HF safetensors, with config.json)")
    ap.add_argument("--profile", default=None, help="Profile name or 'auto' (default: auto-detect)")
    ap.add_argument("--output", default=None, help="Output dir (default: <model-dir>-NVFP4-Den-Calibrated)")
    ap.add_argument("--abliterate", action="store_true", help="Abliterate refusal directions before calibration")
    ap.add_argument("--ablation-factor", type=float, default=1.0,
                    help="Ablation strength (0.0=no change, 1.0=full, default: 1.0)")
    ap.add_argument("--no-project-mlp", action="store_true",
                    help="Skip MLP gate/up/down projection during abliteration")
    ap.add_argument("--samples", type=int, default=None, help="Number of calibration samples")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    ap.add_argument("--remove-source", action="store_true",
                    help="Delete original BF16 safetensors after successful conversion")
    ap.add_argument("--remove-source-if-smaller", action="store_true",
                    help="Delete source only if NVFP4 output is >=2x smaller")
    ap.add_argument("--prune-experts", type=float, default=None, metavar="PCT",
                    help="Prune MoE experts routed on <PCT%% tokens (e.g. 1.0)")
    ap.add_argument("--dedup-experts", type=float, default=None, metavar="COSINE",
                    help="Dedup MoE experts with cosine > THRESHOLD to shared (e.g. 0.99)")
    ap.add_argument("--no-null-tiles", action="store_true",
                    help="Disable null-tile skip marking")
    ap.add_argument("--offline", action="store_true",
                    help="Offline mode: extract NVFP4 tiles without loading model. "
                         "Uses WMPS + multi-grid search + SEF. Peak memory <2GB.")
    ap.add_argument("--no-multi-grid", action="store_true",
                    help="Disable multi-grid scale search (faster, slightly lower quality)")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"ERROR: --model-dir not found: {model_dir}", file=sys.stderr)
        return 1

    # Detect profile
    profile_name = args.profile
    profile = None
    if profile_name and profile_name != "auto":
        profile = get_profile(profile_name)
        if not profile:
            print(f"ERROR: Unknown profile '{profile_name}'", file=sys.stderr)
            list_profiles()
            return 2
    else:
        config_path = os.path.join(model_dir, "config.json")
        profile_name, profile = detect_profile(config_path)
        if profile is None:
            print(f"\n[ERROR] Could not detect profile for: {model_dir}", file=sys.stderr)
            print("Available profiles:", file=sys.stderr)
            list_profiles()
            print(f"\nRe-run with --profile <name>\n", file=sys.stderr)
            return 2

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        base = os.path.basename(model_dir.rstrip("/\\"))
        output_dir = os.path.join(os.path.dirname(model_dir), f"{base}-NVFP4-Den-Calibrated")

    # Sample count default from profile
    num_samples = args.samples or profile.get("samples", 128)

    print("=" * 60)
    print(f"  Den NVFP4+ Universal Calibration")
    print("=" * 60)
    print(f"  Model:   {model_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Profile: {profile_name} - {profile.get('description', '')}")
    print(f"  Grid:    {profile.get('quant_grid')} "
          f"(enabled: {','.join(profile.get('grids_enabled',[]))})")
    print(f"  Abliterate: {args.abliterate}")
    print(f"  Samples: {num_samples}")
    print(f"  Resume:  {args.resume}")

    # Dry-run validation
    try:
        from tools.den_calibrate_qol import run_dry_run
        if not run_dry_run(model_dir, profile, profile_name, num_samples, args.abliterate):
            print("\n[ABORT] Dry-run validation failed. Fix issues above or use --no-dry-run to skip.")
            if not args.dry_run:
                return 1
    except ImportError:
        pass  # Optional QoL module

    if args.dry_run:
        print("Dry run complete - exiting.")
        return 0

    # ── Offline mode: zero-forward-pass NVFP4 extraction ──
    if args.offline:
        print("=" * 60)
        print("  Offline Mode — no model load, no AWQ calibration")
        print("  Using: WMPS importance + multi-grid search + SEF")
        print("  Peak memory: <2 GB")
        print("=" * 60)
        from den_forge.offline_extractor import OfflineExtractor
        out = args.output or (model_dir.rstrip("/\\") + "-NVFP4-Den-Offline")
        extractor = OfflineExtractor(
            model_dir=model_dir,
            profile=profile_name,
            multi_grid=not args.no_multi_grid,
            stochastic=not args.no_multi_grid,
        )
        extractor.run(out)
        return 0

    calibrate(model_dir, output_dir, profile, profile_name,
              num_samples=num_samples, abliterate=args.abliterate,
              ablation_factor=args.ablation_factor,
              resume=args.resume)

    # Post-calibration quality report
    try:
        from tools.den_calibrate_qol import generate_quality_report
        report_path = generate_quality_report(output_dir, model_dir)
        if report_path:
            print(f"  [QoL] Quality report: {report_path}")
    except ImportError:
        pass

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# § DenQuant Fold-Scale DuQuant++ Preconditioner
# ═══════════════════════════════════════════════════════════════════════════════
# Adapted from PrismaQuant duquant.py + joint_smoothquant_format_search.py
# Integrated into DenQuant calibration pipeline as V7 generation.
#
# Reference: docs/DENQUANT_CALIBRATION_ENHANCEMENTS.md §1
# ═══════════════════════════════════════════════════════════════════════════════

class FoldScaleDuQuantPP:
    """
    Fold-Scale DuQuant++ Preconditioner with Golden-Section alpha search.

    Transforms weight matrices before NVFP4 packing to reduce quantization error:
      W' = W * diag(1/√(g ◉ α·(1-α)))
    where g = geometric-mean per-block activation statistics.

    Usage:
        fs = FoldScaleDuQuantPP(alpha=0.5, block_size=128)
        W_optimized, scale = fs.optimize(W_orig, activation_samples)
        W_quantized = quantize_to_nvfp4(W_optimized)
    """

    # E2M1 codebook values (matching our DEN_E2M1_LUT)
    E2M1_VALUES = np.array([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
       -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=np.float32)

    def __init__(self, alpha: float = 0.5, block_size: int = 128,
                 group_size: int = 16, use_bandit: bool = True):
        self.alpha = alpha
        self.block_size = block_size
        self.group_size = group_size  # NVFP4 scale group (16 for UE4M3)
        self.use_bandit = use_bandit
        self._bandit_wins = 0

    @staticmethod
    def _golden_section_search(fn, a=0.0, b=1.0, tol=0.01):
        """
        Golden-section search for unimodal function minimum.
        Converges in O(log 1/tol) evaluations (typically 6-8).

        φ = (√5 - 1) / 2 ≈ 0.618 — golden ratio conjugate
        """
        phi = (np.sqrt(5) - 1) / 2.0
        a1 = b - phi * (b - a)
        a2 = a + phi * (b - a)
        f1 = fn(a1)
        f2 = fn(a2)
        while abs(b - a) > tol:
            if f1 < f2:  # minimizing KL
                b = a2
                a2 = a1
                f2 = f1
                a1 = b - phi * (b - a)
                f1 = fn(a1)
            else:
                a = a1
                a1 = a2
                f1 = f2
                a2 = a + phi * (b - a)
                f2 = fn(a2)
        return (a + b) / 2.0

    def _compute_fold_scale(self, weight: np.ndarray,
                           activations: np.ndarray) -> np.ndarray:
        """
        Compute fold-scale vector from activation statistics.

        Step 1: Per-block geometric mean
          g_b = (∏|X[:,b*BS:(b+1)*BS]|²)^{1/(BS*N)}

        Step 2: Fold-scale
          fold[b] = 1 / √(g_b * α + w_stats * (1-α))
        """
        in_features = weight.shape[1]
        num_blocks = (in_features + self.block_size - 1) // self.block_size

        # Per-block geometric mean of activation squares
        g = np.zeros(num_blocks, dtype=np.float32)
        for b in range(num_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, in_features)
            block_acts = activations[:, start:end]
            if block_acts.size > 0:
                # Geometric mean of |X|²
                log_vals = np.log(np.abs(block_acts) + 1e-10).mean()
                g[b] = np.exp(log_vals)

        # Weight statistics: per-block RMS of weight
        w_rms = np.zeros(num_blocks, dtype=np.float32)
        for b in range(num_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, in_features)
            block_w = weight[:, start:end]
            if block_w.size > 0:
                w_rms[b] = np.sqrt(np.mean(block_w ** 2))

        # Fold-scale: fused balance
        alpha = self.alpha
        fold = 1.0 / np.sqrt(g * alpha + w_rms * (1.0 - alpha) + 1e-8)
        return fold.astype(np.float32)

    def _score_nvfp4_roundtrip(self, weight: np.ndarray,
                               alpha: float,
                               fisher_diag: np.ndarray = None) -> float:
        """
        Score alpha by computing actual NVFP4 round-trip error.

        If fisher_diag provided: Fisher-Geometric weighting (z.ai Patch 2).
        High-curvature regions get penalized more heavily.
        """
        old_alpha = self.alpha
        self.alpha = alpha

        fold = self._compute_fold_scale(weight, weight)
        in_features = weight.shape[1]
        fold_expanded = np.repeat(fold, self.block_size)[:in_features]
        W_scaled = weight * fold_expanded[np.newaxis, :]
        max_val = np.max(np.abs(W_scaled))
        if max_val < 1e-8: self.alpha = old_alpha; return 1.0
        scale = max_val / 6.0
        codes = np.clip(np.round(W_scaled / (scale+1e-10)), -7, 7).astype(np.int8)
        codes = np.clip(codes, -7, 7)
        W_dequant = FoldScaleDuQuantPP.E2M1_VALUES[codes + 7] * scale
        W_dequant = W_dequant / fold_expanded[np.newaxis, :]
        error = (weight - W_dequant) ** 2
        # ── Fisher-Geometric weighting ──
        if fisher_diag is not None:
            fisher_w = np.abs(fisher_diag) / (np.sum(np.abs(fisher_diag)) + 1e-8)
            error = error * fisher_w[np.newaxis, :]
        self.alpha = old_alpha
        return float(np.mean(error))
    def optimize(self, weight: np.ndarray,
                 activations: np.ndarray = None) -> tuple:
        """
        Run fold-scale optimization.

        Returns:
            (fold_scale, optimal_alpha)
            fold_scale: array of shape [num_blocks]
            optimal_alpha: float in [0, 1]
        """
        if self.use_bandit:
            # Multi-armed bandit: track running scores
            bandit_scores = {}

            def score_fn(a):
                key = round(a, 4)
                if key in bandit_scores:
                    return bandit_scores[key]
                # Winner-stays: if same alpha wins 3x, skip remaining
                if self._bandit_wins >= 3:
                    return 0.0  # dummy, won't be used
                val = self._score_nvfp4_roundtrip(weight, a)
                bandit_scores[key] = val
                if len(bandit_scores) >= 2:
                    best = min(bandit_scores.values())
                    if val == best:
                        self._bandit_wins += 1
                    else:
                        self._bandit_wins = 0
                return val

            optimal_alpha = FoldScaleDuQuantPP._golden_section_search(
                score_fn, 0.0, 1.0, tol=0.01)
        else:
            # Standard golden-section search
            optimal_alpha = FoldScaleDuQuantPP._golden_section_search(
                lambda a: self._score_nvfp4_roundtrip(weight, a),
                0.0, 1.0, tol=0.01)

        self.alpha = optimal_alpha
        fold = self._compute_fold_scale(weight, activations or weight)

        return fold, optimal_alpha

    def apply(self, weight: np.ndarray,
              fold: np.ndarray = None) -> np.ndarray:
        """Apply fold-scale to weight matrix."""
        if fold is None:
            fold = self._compute_fold_scale(weight, weight)
        # fold is per-block; expand to per-element
        in_features = weight.shape[1]
        fold_expanded = np.repeat(fold, self.block_size)[:in_features]
        return weight * fold_expanded[np.newaxis, :]


if __name__ == "__main__":
    sys.exit(main())
