# den_llama.cpp

A llama.cpp-derived CUDA inference engine, tuned end-to-end for **one RTX 5070 Ti 16 GB (sm_120, Blackwell)**
serving a **dense Qwen3.8-27B hybrid** (16 full-attention + 48 gated-delta-net layers + a block-64 MTP
head) at **IQ3_XXS**, plus a **gemma-4-26B-A4B MoE** lane and a **27B + mmproj vision/OCR** lane.

Every number below was measured on the target machine, with the measurement method stated. Everything is
gated: no change ships without passing the no-regression corpus gate.

---

## Measured performance

**Ship model:** `Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0-qatfa.gguf` (10.44 GB, ~3.06 bpw)
**Ship flags:** `-fa on -ctk q8_0 -ctv q4_0 -ngl all --spec-type draft-mtp --spec-draft-n-max 2`

| metric | value | notes |
|---|---|---|
| decode | **93 t/s** @32K | bandwidth-bound; 93 × 10.3 GB ≈ 958 GB/s vs 896 GB/s peak |
| prefill | **1,378 t/s** @32K | +2.4–4.3% from the multi-column GDN kernel below |
| MTP acceptance | **0.897** | draft-mtp, n_max 2 |
| NIAH | **6/6** | needle at 0.1/0.5/0.9 depth, 16,053-token haystack |
| tool-calling | **12/12** | JSON tool calls + negatives + constraints |
| coherence | **4/4** | long-form, low repetition, no mojibake |

**MoE lane (gemma-4-26B-A4B, Q4_K):** `--fit on` → ~100 t/s decode, tool-calling verified
(`read_file`/`list_dir` with exact paths). `-sm layer` (default) beats `-sm tensor` here.

**Prefill profile (nsys, pp2048):** MMQ **71.7%**, gated_delta_net **13.0%**, attention **~1.2%**,
`quantize_mmq_q8_1` 1.7%. This profile drove the optimisation order — it is why attention-side work was
deprioritised and why the GDN kernel was worth a rewrite.

---

## What this fork adds over stock llama.cpp

| change | measured effect | status |
|---|---|---|
| **Multi-column GDN kernel** — 4 columns per warp; `k`/`q`/gate are indexed by the state *row*, so they are shared across columns and amortised C×, giving C independent FMA chains | **pp512 +2.3%, pp2048 +4.2%, pp4096 +4.3%** | **default ON**; **bit-exact** (llama-perplexity PPL 1.0036 identical both ways, `mtp_accept` unchanged at 0.897) |
| **FA fix** for `q8_0`-K/`q4_0`-V vec instances | long-context prefill ≈150× vs fallback | on |
| **NVFP4** and **`Q4_G64`** KV codecs (INT4 + FP16 scale/64, 4.25 bpv) | Q4_G64 quantizer byte-identical to the CPU reference; dequant bit-identical; KLD 0.0163 | Q4_G64 is the long-context VRAM saver (§Limits) |
| **Layer-wise split-KV** (`--kv-high-precision-layers N`) | N=0 byte-identical; 6/6 NIAH @64K | on |
| **KV headroom guard** (`--kv-headroom-warn/-block/-force`) | reports GPU-resident model+KV+compute and the max safe ctx at reserve time | warn ON, block opt-in |
| **`fit.cpp` MoE-aware margin cap** | `--fit on` auto-placement lands in the fast zone without `-ngl` | on |
| **SER** expert reduction (`-ser`, `--ser-top-p`, `--ser-ramp`) | correct, perf-neutral; density-invariant | on |
| **Adaptive MTP draft depth** (`--spec-type draft-mtp-adaptive`) | **code/verbatim +3.0%** vs best fixed depth; prose neutral | opt-in (workload-dependent by design) |
| **Fast n-gram cache** (`DEN_NGRAM_FAST`) | up to 42× faster drafting; acceptance-neutral | opt-in, default OFF |
| **17 upstream correctness/perf ports** | fusions (RMS_NORM+SCALE, branchless k-quant mmvq, IQ-MoE), jinja/grammar/sampler hardening | gate-verified |
| **Meta-device `offload_op` fix** | `-sm tensor` no longer silently CPU-ifies the MoE | on |
| **`ggml_permute` dim/stride truncation fix** | upstream #29227 | on |

### Verified negatives (recorded so nobody re-tries them)
- MMQ **double-buffered tile pipeline is impossible on sm_120** — a second `tile_x` exceeds `smpbo` at every
  candidate J, and the config selector rejects them all. The kernel already runs the largest tile the SRAM allows.
- `occupancy 2` **−2.2%**; `stream_k` off neutral; tiled `mul_mat` #27851 **−0.5%**; row-split convert #29308 **−2.7%**.
- Upstream GDN `max→rsqrt` #28068 is *reference-correct* but **desyncs our MTP head** (accept 0.897→0.831,
  decode −7%) — the drafter is calibrated to the current GDN numerics, so **any trunk-numeric change must be
  gated**.
- NInfer-style long-context attention reorg: attention is only **1.2%** here, so it cannot pay.

---

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON \
      -DGGML_AVX2=ON -DGGML_AVX512_VNNI=ON
cmake --build build --target llama-server llama-bench -j 8
```

Options of note:
- `DEN_GDN_MULTICOL` (**default ON**) / `DEN_GDN_COLS` (default 4) — the GDN kernel above. Set OFF to revert.
- `DEN_NGRAM_FAST` (default OFF) — the n-gram cache port.
- `GGML_MOE_OFFLOAD_SCALE` (env) — ik-style MoE offload-fanout threshold; **measured −1.5% here**, so off.

Target: sm_120 (Blackwell), CUDA 13.x, Windows. `-j 8` (max `-j 12`); never build with a live server
running — a locked DLL silently keeps the old binary.

---

## Limits (stated plainly)

- **A prompt longer than ~65,536 tokens collapses generation to a single token.** Prefill is normal at every
  length and *decode is fine past 65,536* (verified: prefill 62,920 + 6,000 generated = 68,920 KV, healthy),
  so the fault is in the **prefill attention path for n_kv > ~64K**. It is **inherited, not a regression** —
  the frozen reference engine collapses *earlier* (at 59,534, where this fork still works). Fix path: port
  upstream's post-fork FA work (`#28536` smem swizzle refactor, sparse-FA `#28770`/`#29298`).
  **This currently caps the vision lane at ~64K instead of 196K.**
- **KV codecs at depth:** `q8_0-q4_0` (ship) and `q4_g64` both retrieve at long context; **`q4_g64` uses
  ~1.2 GiB less VRAM**. NVFP4 KV is fine at short context but is *not* the long-context choice.
- Decode is bandwidth-bound (see above) — throughput wins must come from bytes moved or passes, not clocks.

---

## Measurement discipline (learned the hard way)

- **The voice lane spins the GPU at 100% util / ~51 W** and silently wrecks timings. `nvidia-smi` showing
  full utilisation at *idle wattage* is the tell. Pause it before measuring.
- **Single-sample decode gating is unreliable** — the same binary read 89.7 then 92.6 t/s (3.2% spread)
  against a 3% band. Speed is now median-of-3 with the spread reported.
- **Never force-kill a running server mid-work** — it strands the CUDA context (100% util, 0 MiB, ~51 W) and
  only a reboot clears it. Use `llama_lifecycle.ps1`, which starts the server in its own console process
  group and stops it with `CTRL_C_EVENT` so it tears down cleanly (verified: GPU idle immediately after).
- **After any CMake option change, rebuild every target you intend to run** — otherwise a stale exe links
  against fresh DLLs and looks like a crash.

---

## Provenance

Built on ggml-org/llama.cpp (upstream commits are cherry-picked by SHA and noted in the message), with
adaptations from public community forks where noted in the commit history. All code, naming and structure in
this tree are ours; upstream commit authorship is preserved in the commit metadata for traceability.
