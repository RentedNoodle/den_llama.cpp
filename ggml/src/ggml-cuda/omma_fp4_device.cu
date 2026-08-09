// omma_fp4_device.cu — Standalone OMMA 4X device function (Rule 7.9 compliant)
// NO regular CUDA C++ in this translation unit — PTX inline asm ONLY.
// Compiled as standalone --device-c object, device-linked with persistent kernel.
// sm_120a: OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X
// 3-operand scale format {scale_reg}, {block_id, thread_id} — SASS-proven.

#include <cuda_runtime.h>
#include <stdint.h>

// den_omma_fp4_k64 — K=64 OMMA mxf4nvf4 4X tensor core operation
// Accumulates: D += A[NVFP4] × B[NVFP4]  where A,B are E2M1+UE4M3
// Each thread in the warp provides its own A/B fragment.
// A: 4 int regs = 16×8 e2m1 weight matrix (M=16 rows, K=64 innermost)
// B: 2 int regs = 8×8 e2m1 activation matrix (K=64 innermost, N=8 cols)
// D: 4 float regs = partial accumulator fragment (in/out)
// sfa: uint32 packing 4 UE4M3 scales for A's 4 row-groups
// sfb: uint32 packing 4 UE4M3 scales for B's 4 column-groups
// Uses 3-operand scale format verified via SASS audit (all 32 lanes = 64.0 in identity test).
__device__ void den_omma_fp4_k64(
    float D[4],
    const int A[4],
    const int B[2],
    uint32_t sfa,
    uint32_t sfb)
{
    uint16_t bid = 0;      // block id — always 0 for single-block launch
    uint16_t tid = 0;      // thread id for scale group addressing

    float d0, d1, d2, d3;
    float c0 = D[0], c1 = D[1], c2 = D[2], c3 = D[3];

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, "
        "{%17}, {%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
           "r"(B[0]), "r"(B[1]),
           "f"(c0), "f"(c1), "f"(c2), "f"(c3),
           "r"(sfa), "h"(bid), "h"(tid),
           "r"(sfb), "h"(bid), "h"(tid)
    );

    D[0] = d0; D[1] = d1; D[2] = d2; D[3] = d3;
}
