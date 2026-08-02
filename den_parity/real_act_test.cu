#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"ERR %s\n",cudaGetErrorString(e));exit(1);} } while(0)
// proven M=1 kernel
extern "C" void run_omma_gemv(const float*, const float*, float*, int, int, cudaStream_t, const float*, int);
extern "C" void den_omma_parity_scale_init(void);
// mid_batch M>1
void den_k1_dense_dispatch(const void*, const float*, float*, int M, int N, int K, cudaStream_t, const float*, int n_norms, bool, float);
static const float KVAL[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};
static float ue4m3_to_f32(uint8_t code) {
    if (code >= 0x7F) return 0.0f;
    int exp=(code>>3)&0xF; int mant=code&0x7;
    if(exp==0) return ldexpf((float)mant/8.0f,-7);
    return ldexpf(1.0f+(float)mant/8.0f,exp-7);
}
static double sw_row(const uint8_t* w, const float* x, int row, int K) {
    const int ktp = K/256; double sum=0;
    const uint8_t* base = w + (size_t)row*ktp*160;
    for(int kt=0;kt<ktp;++kt){ const uint8_t* blk=base+(size_t)kt*160; float tn; memcpy(&tn,blk+152,4);
        for(int k=0;k<256;++k){ float sc=ue4m3_to_f32(blk[k/16])*0.5f; uint8_t nib=(uint8_t)((blk[16+k/2]>>((k%2)*4))&0xF); sum+=(double)(KVAL[nib]*sc*tn)*x[kt*256+k]; } }
    return sum;
}
static double cos_sim(const float* a, const float* b, int n){ double d=0,na=0,nb=0; for(int i=0;i<n;i++){d+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];} return d/(sqrt(na)*sqrt(nb)); }
int main(int argc, char** argv){
    const char* path = argc>1?argv[1]:"attn_qkv_rt.bin";
    const float as = argc>2 ? atof(argv[2]) : 1.0f;
    FILE* f=fopen(path,"rb"); int N,K,nb,bb; float gs;
    fread(&N,4,1,f);fread(&K,4,1,f);fread(&nb,4,1,f);fread(&bb,4,1,f);fread(&gs,4,1,f);
    std::vector<uint8_t> w(nb*160); fread(w.data(),1,w.size(),f); fclose(f);
    den_omma_parity_scale_init();
    const int M=1; std::vector<float> x(K), y_gpu(N), y_sw(N);
    srand(42); for(int i=0;i<K;i++){ float s=0; for(int j=0;j<12;j++) s+=(float)rand()/RAND_MAX; x[i]=(s-6.0f)*as; }
    for(int r=0;r<N;r++) y_sw[r]=(float)sw_row(w.data(),x.data(),r,K);
    uint8_t* wd; float* xd; float* yd; float* tnd;
    CK(cudaMalloc(&wd,w.size()));CK(cudaMalloc(&xd,K*4));CK(cudaMalloc(&yd,N*4));CK(cudaMalloc(&tnd,4));
    CK(cudaMemcpy(wd,w.data(),w.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xd,x.data(),K*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tnd,&gs,4,cudaMemcpyHostToDevice));
    // PROVEN M=1 kernel (den_mxf4nvf4_gemv_launch)
    run_omma_gemv((const float*)wd, xd, yd, N, K, 0, tnd, 1);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(y_gpu.data(),yd,N*4,cudaMemcpyDeviceToHost));
    printf("PROVEN_M1  scale=%.2f cos=%.6f\n", as, cos_sim(y_gpu.data(), y_sw.data(), N));
    // mid_batch M=5 (M=5 -> mid_batch)
    const int M5=5;
    std::vector<float> x5(M5*K), y5(M5*N), y5_sw(M5*N);
    for(int m=0;m<M5;m++) for(int i=0;i<K;i++){ float s=0; for(int j=0;j<12;j++) s+=(float)rand()/RAND_MAX; x5[m*K+i]=(s-6.0f)*as; }
    for(int m=0;m<M5;m++) for(int r=0;r<N;r++) y5_sw[m*N+r]=(float)sw_row(w.data(),&x5[m*K],r,K);
    float* x5d; float* y5d;
    CK(cudaMalloc(&x5d,M5*K*4));CK(cudaMalloc(&y5d,M5*N*4));
    CK(cudaMemcpy(x5d,x5.data(),M5*K*4,cudaMemcpyHostToDevice));
    den_k1_dense_dispatch(wd,x5d,y5d,M5,N,K,0,tnd,1,false,1e-6f);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(y5.data(),y5d,M5*N*4,cudaMemcpyDeviceToHost));
    for(int m=0;m<M5;m++) printf("MID_BATCH_M5 m=%d scale=%.2f cos=%.6f\n", m, as, cos_sim(&y5[m*N], &y5_sw[m*N], N));
    return 0;
}
