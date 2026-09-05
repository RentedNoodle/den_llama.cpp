# DEN feature queue — integrate when convenient, never regressing

Rule: everything lands on `work/*`, passes the four gates
(needle / toolcall-v2 / coherence / MTP-ladder-no-regress vs `main`),
then merges. Mainline frozen during publish + FT windows.

## Queued (ranked)

1. **TurboQuant `turbo4_0` KV** (TheTom/llama-cpp-turboquant, MIT drop-in).
   Why: 3.6x KV compression, +0.4–7.7% PPL. Start asymmetric (4-bit K).
   Gate: needle 6/6 + PPL delta <8% + tok/s vs q4_0. Branch: `work/turbo-kv`.
2. **NVFP4 KV lane.** Upstream paths merged (#21074/#22196). Memory-only win
   for 262K residency. Trial post-rebase. Branch: `work/nvfp4-kv`.
3. **DSpark draft trial** (`draft-dspark` in-tree?). A/B vs DFlash lane.
4. **EAGLE-3 draft path.** Needs SpecForge-trained head. Long-term; revisit
   after native FT head ships (v1.1).
5. **`--spec-draft-conf-min` sweep.** Zero-code knob. With p-min sweep.
6. **ngram-mod shared pool + MTP stack.** Measure combined (community: 4.68x).
7. **kvarn6 KV** (BeeLlama pattern). Compare vs turbo4_0 winner.
8. **Prefix/radix cache for tool loops** (Atlas pattern). Pairs with GDN port.
9. **CUDA-graphs audit** (`GRAPH_OPT`, fit interplay). Metric first.
10. **FP8 KV lane** (Blackwell FA4). Headroom play for 262K.

## Done

- `--fit off`, backend-sampling, b1024 (P0, +38%).
- DFlash draft-ngl fix (40→80 t/s).
- SPEC_N coupling guard (n-max 7 lane; measured n-max 4 wins here).
- iq4_nl KV rejected (no fast kernel, 26.8 t/s).
- MTP3/4 rejected (accept≠speed).

## Never (measured out)

- Q2 weights for agentic lanes (repair-token bill).
- Trees above batch 1. Deep static drafts on prose.
