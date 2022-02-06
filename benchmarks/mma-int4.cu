#include <iostream>
#include "../include/gputils.hpp"

using namespace std;
using namespace gputils;


constexpr int nblocks = 3444;
constexpr int nrows_per_warp = 32;
constexpr int ncols_per_warp = 32;
constexpr int nwarps_x = 4;
constexpr int nwarps_y = 4;
constexpr int num_inner_iterations = 128 * 1024;


constexpr int nthreads_per_block = 32 * nwarps_x * nwarps_y;
constexpr int M = nrows_per_warp / 16;
constexpr int N = ncols_per_warp / 8;


__device__ void mma(int c[4], int a[4], int b[2], int d[4])
{
      asm("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
	  "{%0, %1, %2, %3}, "
	  "{%4, %5, %6, %7}, "
	  "{%8, %9}, "
	  "{%10, %11, %12, %13};" :
	   "=r" (c[0]), "=r" (c[1]), "=r" (c[2]), "=r" (c[3]) :
	   "r" (a[0]), "r" (a[1]), "r" (a[2]), "r" (a[3]),
 	   "r" (b[0]), "r" (b[1]),
	   "r" (d[0]), "r" (d[1]), "r" (d[2]), "r" (d[3])
      );
}



__global__ void __launch_bounds__(nthreads_per_block, 1)
mma_int4_kernel(int *dst, const int *src)
{
    int a[M][4];
    int b[N][2];
    int c[M][N][4];

    #pragma unroll
    for (int i = 0; i < M; i++)
        #pragma unroll
	for (int j = 0; j < 4; j++)
	    a[i][j] = src[4*i + j];
    
    #pragma unroll
    for (int i = 0; i < N; i++)
        #pragma unroll
	for (int j = 0; j < 2; j++)
	    b[i][j] = src[4*M + 2*i + j];
    
    #pragma unroll
    for (int i = 0; i < M; i++)
        #pragma unroll
	for (int j = 0; j < N; j++)
            #pragma unroll
	    for (int k = 0; k < 4; k++)
		c[i][j][k] = 0;

    for (int i = 0; i < num_inner_iterations; i++) {
        #pragma unroll
	for (int j = 0; j < M; j++)
            #pragma unroll
	    for (int k = 0; k < N; k++)
		mma(c[j][k], a[j], b[k], c[j][k]);
    }

    int s = 0;
    
    #pragma unroll
    for (int i = 0; i < M; i++)
        #pragma unroll
	for (int j = 0; j < N; j++)
            #pragma unroll
	    for (int k = 0; k < 4; k++)
		s += c[i][j][k];

    int tId = (blockIdx.x * blockDim.x) + threadIdx.x;
    dst[tId] = s;
}


int main(int argc, char **argv)
{
    constexpr int ndst = nblocks * nthreads_per_block;
    constexpr int nsrc = 4*M + 2*N;
    
    double tflops = nblocks * (nthreads_per_block/32.) * nrows_per_warp * ncols_per_warp * 128. * (num_inner_iterations / pow(2.,40.));

    cout << "int4 MMA test\n"
	 << "    nblocks = " << nblocks << "\n"
	 << "    nrows_per_warp = " << nrows_per_warp << "\n"
	 << "    ncols_per_warp = " << ncols_per_warp << "\n"
	 << "    nwarps_x = " << nwarps_x << "\n"
	 << "    nwarps_y = " << nwarps_y << "\n"
	 << "    num_inner_iterations = " << num_inner_iterations << "\n"
	 << "    nthreads_per_block = " << nthreads_per_block << "\n"
	 << "    Total int4 tensor Tops = " << tflops << endl;
    
    Array<int> dst({ndst}, af_gpu | af_zero);
    Array<int> src({nsrc}, af_gpu | af_zero);
    
    CudaTimer t;
    mma_int4_kernel<<<nblocks, nthreads_per_block>>> (dst.data, src.data);
    CUDA_PEEK("mma_int4_kernel");
    float elapsed_time = t.stop();

    cout << "    Elapsed time (sec) = "<< elapsed_time << "\n"
	 << "    Tflops = " << (tflops/elapsed_time) << endl;
    
    return 0;
}
