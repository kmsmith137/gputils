#include <mma.h>
#include <iostream>

#include "../include/gputils/cuda_utils.hpp"
#include "../include/gputils/mem_utils.hpp"

using namespace std;
using namespace nvcuda;
using namespace gputils;


template<int N>
struct IndexMapping
{
    int rmap[N];
    int cmap[N];

    __device__ IndexMapping()
    {
	for (int i = 0; i < N; i++)
	    rmap[i] = cmap[i] = -1;
    }

    template<typename Fragment>
    __device__ void update(const Fragment &frag, int ir, int ic, int nbits)
    {
	assert(sizeof(frag.x[0]) == 4);  // currently assumed but could be relaxed
	assert(frag.num_elements == N);
	assert(frag.num_storage_elements * 32 == N * nbits);

	for (int i = 0; i < N; i++) {
	    int bit = i * nbits;
	    int si = bit / 32;
	    int sv = 1 << (bit % 32);

	    if (frag.x[si] == sv) {
		assert(rmap[i] == -1);
		assert(cmap[i] == -1);
		rmap[i] = ir;
		cmap[i] = ic;
	    }
	}
    }
    
    __device__ void show()
    {
	for (int i = 0; i < blockDim.x; i++) {
	    if (i == threadIdx.x) {
		printf("thread %d:", i);
		for (int j = 0; j < N; j++)
		    printf(" (%d,%d)", rmap[j], cmap[j]);
		printf("\n");
	    }
	    __syncthreads();
	}
    }
};


// Only works if (rstride, cstride) are chosen so that matrix is contiguous.
__device__ void fill_kronecker_matrix(int *out, int ir, int ic, int nr, int nc, int nbits, int rstride, int cstride)
{
    int nbtot = nr * nc * nbits;
    int bit = (ir*rstride + ic*cstride) * nbits;  // set this bit...
    int si = bit / 32;          // ...by setting this index
    int sv = 1 << (bit % 32);   // to this value
    
    for (int i = threadIdx.x; i < (nbtot/32); i += blockDim.x)
	out[i] = (i == si) ? sv : 0;
}


template<typename Fragment, int N>
__global__ void rev_eng_kernel(int *scratch, int nr, int nc, int nbits, int rstride, int cstride, int ldm)
{
    Fragment frag;
    IndexMapping<8> im;
 
    for (int ir = 0; ir < nr; ir++) {
	for (int ic = 0; ic < nc; ic++) {
	    fill_kronecker_matrix(scratch, ir, ic, nr, nc, nbits, rstride, cstride);
	    load_matrix_sync(frag, scratch, ldm);
	    im.update(frag, ir, ic, 4);
	}
    }

    im.show();
}


template<typename Fragment, int N>
void reverse_engineer_fragment(const char *name, int nr, int nc, int nbits, int rstride, int cstride, int ldm)
{
    shared_ptr<int> scratch = af_alloc<int> (1024, af_gpu | af_zero);
    
    cout << name << " (" << nr << "rows, " << nc << "cols)" << endl;
    rev_eng_kernel<Fragment,N> <<<1,32>>> (scratch.get(), nr, nc, nbits, rstride, cstride, ldm);
    CUDA_PEEK(name);
    CUDA_CALL(cudaDeviceSynchronize());
}


int main(int argc, char **argv)
{
    // For the int4 A-fragment, wmma::row_major is required (wmma::col_major gives a CUDA 11.2 compiler error)
    using frag_int4_a = wmma::fragment<wmma::matrix_a, 8, 8, 32, wmma::experimental::precision::s4, wmma::row_major>;
    reverse_engineer_fragment<frag_int4_a,8> ("int4 A-fragment", 8, 32, 4, 32, 1, 32);
    
    // For the int4 B-fragment, wmma::row_major is required (wmma::col_major gives a CUDA 11.2 compiler error)
    using frag_int4_b = wmma::fragment<wmma::matrix_b, 8, 8, 32, wmma::experimental::precision::s4, wmma::col_major>;
    reverse_engineer_fragment<frag_int4_b,8> ("int4 B-fragment", 32, 8, 4, 1, 32, 32);
    
    return 0;
}
