Kernel forge for the-den. Raw PTX tensor core path for Blackwell SM120.
OMMA.SF.16864 cubins, SASS verification, fragment mapping.

Where kernels are proven before promotion. Not just kernels — *kennels*.
The Hydra lives here. One body, many heads. Proven in the forge,
deployed to the dungeon.

GGUF inference engine with NVFP4 native path. Built from ik_llama.cpp
foundation, evolved far beyond — custom NVFP4 stack, MoE dispatch, SSM
kernels, governor FSM, RT core integration.

## Build

Requires CUDA 13.3 (WSL2/Linux). 12.8 is Windows fallback only — no sm_120a.

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120a"
cmake --build build -j$(nproc)
```

## Quick test

```bash
cuobjdump --dump-sass build/ggml/src/libggml.so | grep -c "OMMA.SF.16864"
```

## License

MIT.