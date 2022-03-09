#include <sstream>
#include <iostream>
#include "../include/gputils/Array.hpp"
#include "../include/gputils/CudaStreamPool.hpp"
#include "../include/gputils/cuda_utils.hpp"
#include "../include/gputils/device_mma.hpp"

using namespace std;
using namespace gputils;


template<void (*F)(int[], int[], int[], int[]), int Areg, int Breg, int Creg>
__global__ void mma_kernel(int *cdst, const int *asrc, const int *bsrc, int niter)
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


template<void (*F)(int[], int[], int[], int[]), int BitDepth, int M, int N, int K>
static void time_mma(int niter)
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

	    mma_kernel<F,Areg,Breg,Creg>
		<<< nblocks, nthreads_per_block, 0, stream >>>
		(c, a, b, niter);

	    CUDA_PEEK("mma_kernel");
	};

    stringstream ss;
    ss << "int" << BitDepth << " (m=" << M << ", n=" << N << ", k=" << K << ")";
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, ss.str());
    pool.monitor_throughput("Tflops", tflops_per_kernel);
    pool.run();
}


int main(int argc, char **argv)
{
    time_mma<mma_s4_m8_n8_k32, 4, 8, 8, 32> (512*1024);
    time_mma<mma_s4_m16_n8_k32, 4, 16, 8, 32> (256*1024);
    time_mma<mma_s4_m16_n8_k64, 4, 16, 8, 64> (128*1024);
    
    time_mma<mma_s8_m8_n8_k16, 8, 8, 8, 16> (512*1024);
    time_mma<mma_s8_m16_n8_k16, 8, 16, 8, 16> (256*1024);
    time_mma<mma_s8_m16_n8_k32, 8, 16, 8, 32> (128*1024);
    
    return 0;
}
