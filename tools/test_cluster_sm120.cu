// test_cluster_sm120.cu — Does sm_120a support Thread Block Clusters + DSMEM?
// Compile: nvcc -arch=sm_120a -o test_cluster_sm120.exe test_cluster_sm120.cu
// Run: test_cluster_sm120.exe
// If "CLUSTER OK" = clusters work. If deadlock/error = do not use.

#include <cuda_runtime.h>
#include <cstdio>

__global__ void cluster_probe(int *d_flag) {
    if (blockIdx.x == 0) { d_flag[0] = 0xDEAD; }
    __syncthreads();
    asm volatile("barrier.cluster.arrive; barrier.cluster.wait;");
    if (blockIdx.x == 1) { d_flag[1] = d_flag[0]; }
}

int main() {
    int *d_flag; cudaMalloc(&d_flag, 8); cudaMemset(d_flag, 0, 8);

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t cfg = {0};
    cfg.gridDim = dim3(2,1,1);
    cfg.blockDim = dim3(1,1,1);
    cfg.stream = 0;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    cudaError_t err = cudaLaunchKernelEx(&cfg, cluster_probe, d_flag);
    if (err != cudaSuccess) {
        printf("CLUSTER LAUNCH FAILED: %s\nNOT on sm_120a\n", cudaGetErrorString(err));
        cudaFree(d_flag); return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CLUSTER DEADLOCK: %s\nNOT on sm_120a\n", cudaGetErrorString(err));
        cudaFree(d_flag); return 1;
    }
    int h[2]; cudaMemcpy(h, d_flag, 8, cudaMemcpyDeviceToHost);
    printf("%s — d_flag[1]=0x%X (expected 0xDEAD)\n",
           h[1]==0xDEAD ? "CLUSTER OK" : "CLUSTER FAIL", h[1]);
    cudaFree(d_flag);
    return h[1]==0xDEAD ? 0 : 1;
}
