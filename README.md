# den_llama.cpp

A llama.cpp fork tuned for one job: **Qwen3.8-27B at up to 196K context with native MTP speculative decoding on a single 16 GB GPU** (RTX 5070 Ti, Blackwell GB203).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Qwen3.8--27B%20GSQ--RCO%20IQ3__XXS-yellow)](https://huggingface.co/RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored)
![CUDA Blackwell](https://img.shields.io/badge/CUDA-Blackwell%20sm__120a-76B900)
![Context](https://img.shields.io/badge/context-196K-blue)
![Speculation](https://img.shields.io/badge/speculation-native%20MTP-purple)

Part of [Project Den](https://github.com/RentedNoodle/Project_Den) — a framework for persistent AI companions; this repo is its neural execution engine.

## What this fork is

- **Qwen3.8 GDN-hybrid focus.** Built around the Qwen3.8-27B hybrid architecture (gated-delta-net + attention). This tree carries a delta-net conv-state snapshot bound fix (`src/models/delta-net-base.cpp`) and the Qwen3.5/3.8 model path (`src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`).
- **Native MTP / NextN head work.** The pinned golden master adds an MTP carryover-reset fix (`common/speculative.cpp`: zero stale `pending_h`/`verify_h` on a new prompt) plus server-side MTP hidden-state plumbing (`--embeddings-nextn`, `llama_set_embeddings_nextn()`, per-token `h_nextn` emitted in the non-OAI embeddings JSON).
- **FlashAttention quant-pair enablement.** A 2-file fix (`CMakeLists.txt` + `fattn.cu`, `fa_fix.diff`, +4/−1) compiles the `q8_0-q4_0` FA instance. With `GGML_CUDA_FA_ALL_QUANTS=OFF` — the golden build cache value — the live `-ctk q8_0 -ctv q4_0` pair is not compiled and FA silently falls back to `f16-f16`. The fix lives in a **separate build tree** (same pinned commit + 2-file FA fix), not in this repo's pinned commit; see Results.
- **IQ3_XXS RCO quant pipeline.** Ships a 9.75 GiB `IQ3_XXS` trunk (~3.06 bpw) with a `Q6_K` MTP draft head, packed from ISTA-DASLab's published per-tensor RCO allocation with a custom OrcaRouter-native importance matrix.
- **Gated benchmark suite.** Nothing lands on `main` without passing the repo's gates: needle retrieval, toolcall-v2, coherence, and MTP-ladder speed, with medians-of-N and no-regression checks.
- **Dreya serving target.** The production consumer is a local long-context assistant served on one 16 GB GPU; the adaptive KV streaming path below is what makes 196K fit.

## Results

Only golden-build (`build2`) numbers with recorded conditions are listed. FA-quant-pair numbers are explicitly labeled **experiment (build-fa, not golden)**. Numbers whose conditions are not fully pinned (GPQA-Diamond, toolcall) are intentionally omitted — see the model repo for those.

| Metric | Value | Conditions |
|---|---|---|
| Needle retrieval | **6/6** | golden build2; `Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0.gguf`; 15K-word haystack, depths 0.1–0.9, temp 0.0.¹ |
| WikiText-2 perplexity | **6.1745 ± 0.14889** | golden build2 `llama-perplexity.exe`; ship file; 40×512 chunks; `-fa on`; KV default (no `-ctk`/`-ctv` passed).² |
| Serve t/s @ 32K ctx | **82.73 t/s** (median; 79.4–89.6) | golden build2; ctx 32768; 5 prompts × 300 tok × 2 reps; temp 0.6; seed 424242; np=1; MTP n-max 2; accept 0.6703; prefill 706 t/s.³ |
| Serve t/s @ 65K ctx | **64.16 t/s** (median ALL; 51.11–72.48) | golden build2 (FA A/B control); ship file; ctx 65536; `-b 2048 -ub 2048 -fa on -ctk q8_0 -ctv q4_0 -ngl 99 -t 8 -tb 8`; MTP n-max 2; median accept ALL 0.7675 / CUM 0.7010.⁴ |
| FA build — t/s | **79.48 t/s** (median) | *experiment (build-fa, not golden)*: HEAD `abd589d4c` + `fa_fix.diff`; same flags as the 65K control; +23.9% vs control.⁵ |
| FA build — acceptance | **0.7018 ALL / 0.6768 CUM** | *experiment (build-fa, not golden)*; control 0.7675 / 0.7010.⁵ |
| FA build — ppl / needle | **6.1745 ± 0.14889 / 6/6** | *experiment (build-fa, not golden)*; ppl Δ 0.00% vs control; control needle 5/6 (needle-05 miss).⁵ |

¹ Model card (Hugging Face / ModelScope).
² Project perplexity log (build2 `llama-perplexity.exe`, 40×512-chunk run on the ship file).
³ Project A/B record: model file at run time was `…v2.0-s1-orcaim.gguf`, later consolidated into `v2.0.gguf`; short-ctx KV type was not recorded in the AB source.
⁴ FA A/B battery log (control leg, median-of-3).
⁵ Project FA A/B verdict record §2. Disposition: the pre-registered gate recorded **TESTED-NEGATIVE** (acceptance CUM −0.0242, exceeding the 0.02 bound) and build2 stayed golden; on 2026-09-10 an executive override promoted build-fa to **>128K-regime golden**, retaining build2 as the immutable ≤128K default (override record).

## Models

All repos verified live via the Hugging Face and ModelScope public APIs.

| Artifact | File | Size | SHA256 (prefix) |
|---|---|---|---|
| Main quant — [Hugging Face](https://huggingface.co/RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored) · [ModelScope](https://modelscope.cn/models/RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored) | `Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0.gguf` | 10,466,439,424 B (9.75 GiB) | `41ad7dfb…` |
| Vision projector (BF16) | `mmproj/mmproj-Qwen3.8-27B-BF16.gguf` | 931,146,528 B (~0.87 GiB) | `13cb7beb…` |
| Vision projector (Q8_0) | `mmproj/mmproj-Qwen3.8-27B-Q8_0.gguf` | 629,247,648 B (~0.60 GiB) | `b280c2cb…` |
| Chat template | `froggeric-qwen3.8-tool-use.jinja` | 28,234 B | `e57684ba…` |
| RCO allocation map | `REF-IQ3_XXS-mtp.rco-allocation.txt` | 24,583 B | `2e690030…` |
| imatrix | `imatrix.dat` | 13,642,656 B | `e0fcab28…` |

Base and reference repos: [`orcarouter/Qwen3.8-27B-Uncensored`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored) (uncensored base), [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (architecture + pretrained weights), [`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) (RCO allocation source), [`froggeric/Qwen-Fixed-Chat-Templates`](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (tool-use template). All hashes are from the model repo's `SHA256SUMS.txt`; `REF-…` and `imatrix.dat` prefixes are from the upload record.

## Quick start

### Golden build (Windows, CUDA 13.3 + VS2022 + Ninja)

Pin the golden master commit, then configure with `GGML_CUDA_FA_ALL_QUANTS=OFF` (this is the value the golden `build2` was actually built with):

```powershell
git clone https://github.com/RentedNoodle/den_llama.cpp.git
cd den_llama.cpp
git checkout abd589d4cf6d4d2d221b9532d3861fadbdb7b046

cmake -S . -B build2 -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CUDA=ON `
  -DGGML_CUDA_FA_ALL_QUANTS=OFF `
  -DGGML_CUDA_GRAPHS=ON `
  -DGGML_CUDA_NCCL=ON `
  -DGGML_CUDA_COMPRESSION_MODE=size `
  -DGGML_NATIVE=ON `
  -DGGML_LLAMAFILE=ON `
  -DGGML_OPENMP=ON `
  -DGGML_CPU_REPACK=ON `
  -DGGML_BUILD_EXAMPLES=OFF `
  -DGGML_BUILD_TESTS=OFF

cmake --build build2 --target llama-server --config Release -j 8
```

### Ship command (RTX 5070 Ti 16 GB)

```powershell
build2\bin\llama-server.exe `
  -m Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0.gguf `
  --spec-type draft-mtp --spec-draft-n-max 2 `
  --ctx-size 262144 -fa on -ctk q8_0 -ctv q4_0 -ngl all `
  -b 512 -ub 512 -np 1 --kv-stream-stage-mib 2048 `
  -t 8 -tb 8 --cache-ram 8192 --reasoning-budget 256
```

For depth-0.5 retrieval fidelity, use `-b 2048 -ub 2048` (a documented chunking artifact of the model, not this fork).

## What's in the fork

| Location | What |
|---|---|
| `common/speculative.cpp` | MTP carryover-reset on a new prompt (zeroes stale `pending_h`/`verify_h`), plus an env-gated (`LLAMA_DUMP_MTP`) debug dump. |
| `common/arg.cpp`, `common/common.h`, `common/common.cpp` | `--embeddings-nextn` flag and `common_params::embeddings_nextn`. |
| `include/llama.h`, `src/llama-context.cpp` | `llama_set_embeddings_nextn()` / `llama_get_embeddings_nextn_ith()`; NULL-safe wrapper and env-gated MTP tensor dump. |
| `tools/server/server-task.{h,cpp}`, `tools/server/server-context.cpp` | Server emits per-token MTP head input hidden as `"nextn"` when `embeddings_nextn` is enabled. |
| `src/models/delta-net-base.cpp` | Gated-delta-net conv-state snapshot bound fix (`K = min(n_rs_seq, n_tok)+1`). |
| `benchmarks/` | Adaptive KV streaming benchmark driver (`benchmark_kv_stream.py`) — sweeps context capacity 8K→max, auto-selects the largest practical KV pool, and emits CSV/PNG/SVG. See `benchmarks/README.md`. |
| `BUILD2_GOLDEN.md` | Rebuild/repro record for the golden engine (toolchain, configure command, patch stack). Note: its "Source of truth" commit field names `0875eab41`, one commit behind the current golden HEAD `abd589d4c`. |
| FA quant-pair work | Not in this repo's pinned commit: same HEAD `abd589d4c` + 2-file FA fix (`fa_fix.diff`, +4/−1), maintained in a separate build tree. See Results. |

Branches:

| Branch | Role |
|---|---|
| `main` | Golden master HEAD `abd589d4c` ("MTP carryover fix + server MTP/reasoning plumbing + delta-net fixes + build repro doc"). |
| `den-legacy` | Retired engine history (`a99ff8f9…`). |

## Adaptive KV streaming (contributed)

This branch adds an experimental, block-granular KV cache streaming path to the CUDA `llama-server`. It is intended for running long contexts when model weights leave too little VRAM for the complete KV cache.

With `--kv-stream-stage-mib N`, the authoritative KV tensors are stored in pinned host memory while a bounded CUDA pool is shared by resident KV pages and a transfer ring. The runtime adapts that split as the context grows: it keeps as many pages resident as the budget allows, reclaims resident space for staging when more streaming is required, and prefetches later layers while the current layer computes. This avoids relying on uncontrolled Unified Memory page thrashing and preserves exact attention over the full context.

Detailed project story, design, implementation, and benchmark results are in
[Running Qwen 27B on 16G VRAM with Full Context Length: Building Adaptive KV Cache Streaming for llama.cpp](https://medium.com/@raymond860909/running-qwen-27b-on-16g-vram-with-full-context-length-building-adaptive-kv-cache-streaming-for-bf1e819116e9).

> [!WARNING]
> This is research code tailored to our current NVIDIA CUDA configuration: an RTX 5070 Ti with 16 GB VRAM, `unsloth/Qwen3.8-27B-GGUF` `UD-Q3_K_XL`, a 262144-token context, Flash Attention, a Q8_0 K cache, a Q4_0 V cache, and one server slot. Other models, KV cache quantization combinations, parallel slots, and non-CUDA backends are not yet supported or validated. Expanding model and KV quantization support is follow-up work.

### Build the modified server

Install a C++ compiler, CMake, and the CUDA toolkit, then run this command from the repository root:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release --target llama-server -j
```

The executable is created at `build/bin/llama-server`.

Example using the tested cache configuration:

```bash
./build/bin/llama-server \
  --model /path/to/model.gguf \
  --ctx-size 262144 \
  -fa on \
  -ctk q8_0 \
  -ctv q4_0 \
  -ngl all \
  -np 1 \
  --kv-stream-stage-mib 2304
```

The best value for `--kv-stream-stage-mib` depends on the model, context capacity, GPU, and other VRAM consumers. Start conservatively and increase it while checking startup and peak VRAM use.

#### Optional Unified Memory for model weights

Adaptive KV streaming works with or without Unified Memory. Leave `GGML_CUDA_ENABLE_UNIFIED_MEMORY` unset for ordinary CUDA device allocations. To make GPU-offloaded model buffers CUDA managed allocations, launch the same server with the environment variable enabled:

```bash
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
./build/bin/llama-server \
  --model /path/to/model.gguf \
  --ctx-size 262144 \
  -fa on \
  -ctk q8_0 \
  -ctv q4_0 \
  -ngl all \
  -np 1 \
  --kv-stream-stage-mib 2304
```

With this flag, CUDA-backed model buffers, including GPU-offloaded weights, are allocated with `cudaMallocManaged` and their pages can migrate between VRAM and host memory. The adaptive resident-page and transfer-ring pool is intentionally different: it is still allocated with `cudaMalloc`, so that fixed-size pool remains physically allocated in VRAM instead of becoming managed memory. UVM is therefore optional for this branch and does not change the KV streaming pool into pageable storage.

### Recreate the benchmark graph

The benchmark driver automatically selects the largest practical adaptive KV pool for each configured context capacity, sweeps from 8K through the requested maximum, and generates the CSV, PNG, and SVG results:

```bash
python3 -m pip install matplotlib

python3 benchmarks/benchmark_kv_stream.py \
  --model /path/to/model.gguf \
  --max-context 192K
```

The only required arguments are the model GGUF and maximum context. See [benchmarks/README.md](benchmarks/README.md) for the pool-probing algorithm, generated files, optional settings, and resumable output directories.

---

This fork builds on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT). The upstream README is included unmodified below for attribution.

<details>
<summary>Upstream llama.cpp README (unmodified)</summary>

## Upstream llama.cpp README

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

</details>
