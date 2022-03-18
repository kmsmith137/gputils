#include <sstream>
#include <iostream>
#include "../include/gputils/Array.hpp"
#include "../include/gputils/CudaStreamPool.hpp"
#include "../include/gputils/cuda_utils.hpp"
#include "../include/gputils/device_mma.hpp"

using namespace std;
using namespace gputils;


// -------------------------------------------------------------------------------------------------
//
// float16 MMA


template<void (*F)(__half2[], const __half2[], const __half2[], const __half2[]), int Areg, int Breg, int Creg>
__global__ void mma_f16_kernel(float *cdst, const float *asrc, const float *bsrc, int niter)
{
    __half2 a[Areg];
    __half2 b[Breg];
    __half2 c[Creg];

    int t0 = blockIdx.x * blockDim.x;

    #pragma unroll
    for (int i = 0; i < Areg; i++) {
	float x = asrc[t0*2*Areg + (2*i)*blockDim.x + threadIdx.x];
	float y = asrc[t0*2*Areg + (2*i+1)*blockDim.x + threadIdx.x];
	a[i] = __floats2half2_rn(x, y);
    }
    
    #pragma unroll
    for (int i = 0; i < Breg; i++) {
	float x = bsrc[t0*2*Breg + (2*i)*blockDim.x + threadIdx.x];
	float y = bsrc[t0*2*Breg + (2*i+1)*blockDim.x + threadIdx.x];
	b[i] = __floats2half2_rn(x, y);
    }
    
    #pragma unroll
    for (int i = 0; i < Creg; i++)
	c[i] = __floats2half2_rn(0., 0.);  // FIXME

    
    for (int i = 0; i < niter; i++)
	F(c, a, b, c);
    
    #pragma unroll
    for (int i = 0; i < Creg; i++) {
	float2 x = __half22float2(c[i]);
	cdst[t0*2*Creg + (2*i)*blockDim.x + threadIdx.x] = x.x;
	cdst[t0*2*Creg + (2*i+1)*blockDim.x + threadIdx.x] = x.y;
    }
}


template<void (*F)(__half2[], const __half2[], const __half2[], const __half2[]), int M, int N, int K>
static void time_f16_mma(int niter)
{
    constexpr int Areg = (M*K) / 64;
    constexpr int Breg = (N*K) / 64;
    constexpr int Creg = (M*N) / 64;
    
    const int nblocks = 82 * 84 * 4;
    const int nthreads_per_block = 128;
    const int ncallbacks = 15;
    const int nstreams = 2;

    int nth = nblocks * nthreads_per_block;
    Array<float> asrc({nstreams,2*nth*Areg}, af_zero | af_gpu);
    Array<float> bsrc({nstreams,2*nth*Breg}, af_zero | af_gpu);
    Array<float> csrc({nstreams,2*nth*Creg}, af_zero | af_gpu);

    double flops_per_mma = 2*M*N*K;
    double tflops_per_kernel = niter * (nth/32.) * flops_per_mma / pow(2,40.);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    float *a = asrc.data + istream*nth*Areg;
	    float *b = bsrc.data + istream*nth*Breg;
	    float *c = csrc.data + istream*nth*Creg;

	    mma_f16_kernel<F,Areg,Breg,Creg>
		<<< nblocks, nthreads_per_block, 0, stream >>>
		(c, a, b, niter);

	    CUDA_PEEK("mma_f16_kernel");
	};

    stringstream ss;
    ss << "f16 (m=" << M << ", n=" << N << ", k=" << K << ")";
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, ss.str());
    pool.monitor_throughput("Tflops", tflops_per_kernel);
    pool.run();
}



// -------------------------------------------------------------------------------------------------
//
// int4/int8 MMA


template<void (*F)(int[], const int[], const int[], const int[]), int Areg, int Breg, int Creg>
__global__ void mma_int_kernel(int *cdst, const int *asrc, const int *bsrc, int niter)
{
    int a[Areg];
    int b[Breg];
    int c[Creg];

    int t0 = blockIdx.x * blockDim.x;

    #pragma unroll
    for (int i = 0; i < Areg; i++)
	a[i] = asrc[t0*Areg + i*blockDim.x + threadIdx.x];
    
    #pragma unroll
    for (int i = 0; i < Breg; i++)
	b[i] = bsrc[t0*Breg + i*blockDim.x + threadIdx.x];

    #pragma unroll
    for (int i = 0; i < Creg; i++)
	c[i] = 0;

    for (int i = 0; i < niter; i++)
	F(c, a, b, c);
    
    #pragma unroll
    for (int i = 0; i < Breg; i++)
	cdst[t0*Creg + i*blockDim.x + threadIdx.x] = c[i];
}


template<void (*F)(int[], const int[], const int[], const int[]), int BitDepth, int M, int N, int K>
static void time_int_mma(int niter)
{
    constexpr int Areg = (M*K*BitDepth) / 1024;
    constexpr int Breg = (N*K*BitDepth) / 1024;
    constexpr int Creg = (M*N) / 32;
    
    const int nblocks = 82 * 84 * 4;
    const int nthreads_per_block = 128;
    const int ncallbacks = 15;
    const int nstreams = 2;

    int nth = nblocks * nthreads_per_block;
    Array<int> asrc({nstreams,nth*Areg}, af_zero | af_gpu);
    Array<int> bsrc({nstreams,nth*Breg}, af_zero | af_gpu);
    Array<int> csrc({nstreams,nth*Creg}, af_zero | af_gpu);

    double flops_per_mma = 2*M*N*K;
    double tflops_per_kernel = niter * (nth/32.) * flops_per_mma / pow(2,40.);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    int *a = asrc.data + istream*nth*Areg;
	    int *b = bsrc.data + istream*nth*Breg;
	    int *c = csrc.data + istream*nth*Creg;

	    mma_int_kernel<F,Areg,Breg,Creg>
		<<< nblocks, nthreads_per_block, 0, stream >>>
		(c, a, b, niter);

	    CUDA_PEEK("mma_int_kernel");
	};

    stringstream ss;
    ss << "int" << BitDepth << " (m=" << M << ", n=" << N << ", k=" << K << ")";
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, ss.str());
    pool.monitor_throughput("Tflops", tflops_per_kernel);
    pool.run();
}


int main(int argc, char **argv)
{
    time_f16_mma <mma_f16_m16_n8_k8, 16, 8, 8> (256*1024);
    time_f16_mma <mma_f16_m16_n8_k16, 16, 8, 16> (128*1024);
		 
    time_int_mma <mma_s4_m8_n8_k32, 4, 8, 8, 32> (512*1024);
    time_int_mma <mma_s4_m16_n8_k32, 4, 16, 8, 32> (256*1024);
    time_int_mma <mma_s4_m16_n8_k64, 4, 16, 8, 64> (128*1024);
    
    time_int_mma <mma_s8_m8_n8_k16, 8, 8, 8, 16> (512*1024);
    time_int_mma <mma_s8_m16_n8_k16, 8, 16, 8, 16> (256*1024);
    time_int_mma <mma_s8_m16_n8_k32, 8, 16, 8, 32> (128*1024);
    
    return 0;
}
