// den_omma_fp4_device.h — OMMA 4X FP4 device function declaration
// Rule 7.9 compliant: declaration ONLY, no implementation, no inline asm.
// Compiled as standalone --device-c TU, device-linked with persistent kernel.
// sm_120a: OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X

#pragma once
#include <stdint.h>

#ifdef __CUDACC__
// K=64 OMMA: A[4] = 4×uint32 (16×8 e2m1 weight nibbles)
//            B[2] = 2×uint32 (8×8 e2m1 activation nibbles)
//            D[4] = 4×float32 accumulators (in/out, carries partial sum)
//            sfa/sfb = 4×UE4M3 packed scales per operand (0x38383838 = all 1.0)
// 3-operand scale format: {scale_reg}, {block_id, thread_id} per side
__device__ void den_omma_fp4_k64(
    float D[4],
    const int A[4],
    const int B[2],
    uint32_t sfa,
    uint32_t sfb);
#endif
