#include <iostream>
#include <cuda_fp16.h>

#include "../include/gputils/Array.hpp"
#include "../include/gputils/CudaStreamPool.hpp"
#include "../include/gputils/cuda_utils.hpp"

using namespace std;
using namespace gputils;


__device__ __forceinline__ void local_transpose_f16(__half2 &x, __half2 &y)
{
    __half2 xnew = __lows2half2(x, y);
    __half2 ynew = __highs2half2(x, y);
    x = xnew;
    y = ynew;
}


__device__ __forceinline__ void local_transpose_byte_perm(unsigned int &x, unsigned int &y)
{
    unsigned int xnew = __byte_perm(x, y, 0x5410);
    unsigned int ynew = __byte_perm(x, y, 0x7632);
    x = xnew;
    y = ynew;
}


template<typename T, void (*F)(T&,T&)>
__global__ void local_transpose_kernel(T *dst, const T *src, int niter)
{
    int it = threadIdx.x;
    int nt = blockDim.x;

    src += blockIdx.x * 8*nt;
    dst += blockIdx.x * 8*nt;
    
    T x0 = src[it];
    T x1 = src[it + nt];
    T x2 = src[it + 2*nt];
    T x3 = src[it + 3*nt];
    T x4 = src[it + 4*nt];
    T x5 = src[it + 5*nt];
    T x6 = src[it + 6*nt];
    T x7 = src[it + 7*nt];
	
    for (int i = 0; i < niter; i++) {
	F(x0, x1);
	F(x2, x3);
	F(x4, x5);
	F(x6, x7);
	
	F(x0, x2);
	F(x1, x3);
	F(x4, x6);
	F(x5, x7);
	
	F(x0, x4);
	F(x1, x5);
	F(x2, x6);
	F(x3, x7);
    }

    dst[it] = x0;
    dst[it + nt] = x1;
    dst[it + 2*nt] = x2;
    dst[it + 3*nt] = x3;
    dst[it + 4*nt] = x4;
    dst[it + 5*nt] = x5;
    dst[it + 6*nt] = x6;
    dst[it + 7*nt] = x7;
}


template<typename T, void (*F)(T&,T&)>
void time_local_transpose_kernel(const char *name)
{
    const int nblocks = 82 * 84;
    const int nthreads_per_block = 1024;
    const int ncallbacks = 10;
    const int nstreams = 2;
    const int niter = 256 * 1024;
    const double tera_transposes_per_kernel = double(niter) * nblocks * nthreads_per_block * 12 / pow(2.,40.);

    static_assert(sizeof(T) == 4);
    const int ninner = nblocks * nthreads_per_block * 8;
    Array<int> dst({nstreams,ninner}, af_zero | af_gpu);
    Array<int> src({nstreams,ninner}, af_zero | af_gpu);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    T *d = (T *) (dst.data + istream*ninner);
	    T *s = (T *) (dst.data + istream*ninner);

	    local_transpose_kernel<T,F> <<< nblocks, nthreads_per_block >>> (d, s, niter);
	    CUDA_PEEK(name);
	};

    CudaStreamPool pool(callback, ncallbacks, nstreams, name);
    pool.monitor_throughput("teratransposes / sec", tera_transposes_per_kernel);
    pool.run();
}


int main(int argc, char **argv)
{
    cout << "** A puzzle: why is local transpose with __byte_perm() so slow?! **" << endl;
    time_local_transpose_kernel<__half2, local_transpose_f16> ("local_transpose_f16");
    time_local_transpose_kernel<unsigned int, local_transpose_byte_perm> ("local_transpose_byte_perm");
    return 0;
}
