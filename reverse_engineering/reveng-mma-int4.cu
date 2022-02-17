#include <iostream>
#include "../include/gputils.hpp"

using namespace std;
using namespace gputils;


__host__ int slow_ilog2(int n)
{
    assert(n > 0);
    int i = log2(1.5 * n);
    assert(n == (1 << i));
    return i;
}


__host__ Array<int> slow_matmul(const Array<int> &a_, const Array<int> &b_)
{
    Array<int> a = a_.to_host();
    Array<int> b = b_.to_host();
    
    assert(a.ndim == 2);
    assert(b.ndim == 2);
    assert(a.shape[1] == b.shape[0]);

    int m = a.shape[0];
    int p = a.shape[1];
    int n = b.shape[1];

    Array<int> c({m,n});

    for (int i = 0; i < m; i++) {
	for (int j = 0; j < n; j++) {
	    int t = 0;
	    for (int k = 0; k < p; k++)
		t += a.at({i,k}) * b.at({k,j});
	    c.at({i,j}) = t;
	}
    }

    return c;
}


// -------------------------------------------------------------------------------------------------


template<int Nrows, int Ncols, int BitDepth>
struct MatParams
{
    static constexpr int nrows = Nrows;
    static constexpr int ncols = Ncols;
    static constexpr int bit_depth = BitDepth;
    static constexpr int bits_per_fragment = nrows * ncols * bit_depth;

    static_assert(constexpr_is_pow2(nrows));
    static_assert(constexpr_is_pow2(ncols));
    static_assert(constexpr_is_pow2(bit_depth));
    static_assert(bits_per_fragment >= 1024);
    static_assert(bit_depth <= 32);
    
    static constexpr int int32s_per_fragment = bits_per_fragment / 32;
    static constexpr int registers_per_thread = bits_per_fragment / 1024;
    static constexpr int num_bstate_bits = constexpr_ilog2(32 / bit_depth);
    static constexpr int num_regstate_bits = constexpr_ilog2(registers_per_thread);
    static constexpr int num_rowstate_bits = constexpr_ilog2(nrows);
    static constexpr int num_colstate_bits = constexpr_ilog2(ncols);
    static constexpr int num_state_bits = num_rowstate_bits + num_colstate_bits;

    static_assert(num_state_bits == num_bstate_bits + num_regstate_bits + 5);

    // State bit mapping (physical -> logical)
    vector<bool> pbit_isrow;
    vector<int> pbit_lindex;

    // State bit mapping (logical -> physical)
    // FIXME can get rid of this? And finalize()?
    vector<int> lrbit_pindex;
    vector<int> lcbit_pindex;
    bool finalized = false;

    
    MatParams() :
	pbit_isrow(num_state_bits, false),
	pbit_lindex(num_state_bits, -1),
	lrbit_pindex(num_rowstate_bits, -1),
	lcbit_pindex(num_colstate_bits, -1)
    { }

    
    static __host__ Array<int> make_src_matrix()
    {
	Array<int> ret({num_state_bits+1, int32s_per_fragment}, af_zero);  // on cpu
	ret.at({0,0}) = 1;

	for (int b = 0; b < num_state_bits; b++) {
	    int bit_index = bit_depth * (1 << b);
	    int int32_index = bit_index / 32;
	    ret.at({b+1,int32_index}) = (1 << (bit_index % 32));
	}
    
	return ret.to_gpu();
    }

    
    void assign_state_bit(int b, bool isrow, int index)
    {
	assert((b >= 0) && (b < num_state_bits));
	assert(this->pbit_lindex[b] < 0);

	assert(index >= 0);
	assert(index < (isrow ? num_rowstate_bits : num_colstate_bits));
	
	this->pbit_isrow[b] = isrow;
	this->pbit_lindex[b] = index;
    }


    void finalize_layout()
    {
	for (int pb = 0; pb < num_state_bits; pb++) {
	    vector<int> &v = pbit_isrow[pb] ? lrbit_pindex : lcbit_pindex;
	    int lb = pbit_lindex[pb];
	    
	    assert(lb >= 0);
	    assert(lb < v.size());
	    assert(v[lb] < 0);
	    v[lb] = pb;
	}

	// Paranoid
	for (unsigned int lb = 0; lb < lrbit_pindex.size(); lb++)
	    assert(lrbit_pindex[lb] >= 0);
	for (unsigned int lb = 0; lb < lcbit_pindex.size(); lb++)
	    assert(lcbit_pindex[lb] >= 0);

	this->finalized = true;
    }
    
    
    // Helper for show_layout()
    void _show_state_bit(int b, const char *phys_prefix, int phys_index, const char *row_prefix, const char *col_prefix) const
    {
	assert((b >= 0) && (b < num_state_bits));
	cout << "    " << phys_prefix << phys_index << " <-> ";

	if (this->pbit_lindex[b] < 0)
	    cout << "[unassigned]\n";
	else if (this->pbit_isrow[b])
	    cout << row_prefix << this->pbit_lindex[b] << "\n";
	else
	    cout << col_prefix << this->pbit_lindex[b] << "\n";
    }
    
    void show_layout(const char *row_prefix, const char *col_prefix) const
    {
	for (int b = 0; b < num_bstate_bits; b++)
	    _show_state_bit(b, "b", b, row_prefix, col_prefix);
	
	for (int t = 0; t < 5; t++)
	    _show_state_bit(num_bstate_bits + t, "t", t, row_prefix, col_prefix);

	for (int r = 0; r < num_regstate_bits; r++)
	    _show_state_bit(num_bstate_bits + 5 + r, "r", r, row_prefix, col_prefix);
    }


    __host__ void pindex_to_rc(int pindex, int &rindex, int &cindex) const
    {
	assert(finalized);
	assert((pindex >= 0) && (pindex < nrows * ncols));

	rindex = 0;
	cindex = 0;
	
	for (int b = 0; b < num_state_bits; b++)
	    if (pindex & (1 << b))
		(pbit_isrow[b] ? rindex : cindex) += (1 << pbit_lindex[b]);
    }
    

    __host__ int rc_to_pindex(int rindex, int cindex) const
    {
	assert(finalized);
	assert((rindex >= 0) && (rindex < nrows));
	assert((cindex >= 0) && (cindex < ncols));

	int pindex = 0;

	for (int b = 0; b < num_state_bits; b++)
	    if ((pbit_isrow[b] ? rindex : cindex) & (1 << pbit_lindex[b]))
		pindex += (1 << b);

	return pindex;
    }


    __host__ Array<int> make_fragment(int aflags=0) const
    {
	return Array<int> ({int32s_per_fragment}, aflags);
    }

    
    __host__ Array<int> unpack_fragment(const Array<int> &src_) const
    {
	Array<int> src = src_.to_host();
	assert(src.shape_equals({int32s_per_fragment}));
	
	Array<int> dst({nrows, ncols});

	for (int ir = 0; ir < nrows; ir++) {
	    for (int ic = 0; ic < ncols; ic++) {
		int p = rc_to_pindex(ir, ic);
		int j = p / (32/bit_depth);
		int b = (p*bit_depth) - (32*j);

		dst.at({ir,ic}) = (src.data[j] << (32-bit_depth-b)) >> (32-bit_depth);
	    }
	}

	return dst;
    }


    __host__ Array<int> pack_fragment(const Array<int> &src_) const
    {
	int smax = (1U << (bit_depth-1)) - 1U;
	int smin = -smax - 1;
    
	Array<int> src = src_.to_host();
	assert(src.shape_equals({nrows, ncols}));

	Array<int> dst({int32s_per_fragment}, af_zero);

	for (int ir = 0; ir < nrows; ir++) {
	    for (int ic = 0; ic < ncols; ic++) {
		int p = rc_to_pindex(ir, ic);
		int j = p / (32/bit_depth);
		int b = (p*bit_depth) - (32*j);
		
		int s = src.at({ir,ic});
		assert((s >= smin) && (s <= smax));

		unsigned int us = (s << (32-bit_depth));
		s = (us >> (32-bit_depth-b));
		dst.at({j}) |= s;
	    }
	}

	return dst;
    }

    
    __host__ void test_pack_unpack() const
    {
	for (int iouter = 0; iouter < 10; iouter++) {
	    Array<int> a = make_fragment(af_random);
	    Array<int> b = pack_fragment(unpack_fragment(a));

	    assert(a.shape_equals({int32s_per_fragment}));
	    assert(b.shape_equals({int32s_per_fragment}));
	    
	    for (int i = 0; i < int32s_per_fragment; i++)
		assert(a.data[i] == b.data[i]);
	}
    }


    static __device__ void set_zero(int x[registers_per_thread])
    {
	#pragma unroll
	for (int r = 0; r < registers_per_thread; r++)
	    x[r] = 0;
    }

    static __device__ void load(int x[registers_per_thread], const int *p_warp)
    {
	int laneId = threadIdx.x & 0x1f;

	#pragma unroll
	for (int r = 0; r < registers_per_thread; r++)
	    x[r] = p_warp[32*r + laneId];
    }

    static __device__ void store(int x[registers_per_thread], int *p_warp)
    {
	int laneId = threadIdx.x & 0x1f;

	#pragma unroll
	for (int r = 0; r < registers_per_thread; r++)
	    p_warp[32*r + laneId] = x[r];
    }
};


// -------------------------------------------------------------------------------------------------


struct MmaParams
{
    using AParams = MatParams<16, 64, 4>;
    using BParams = MatParams<64, 8, 4>;
    using CParams = MatParams<16, 8, 32>;

    AParams aparams;
    BParams bparams;
    CParams cparams;

    MmaParams() { }

    void reverse_engineer(const Array<int> &dst)
    {
	constexpr int nb1 = AParams::num_state_bits;
	constexpr int nb2 = BParams::num_state_bits;
	    
	assert(dst.shape_equals({nb1+1,nb2+1}));
	assert(dst.at({0,0}) == 0);
	
	int curr_i = 0;
	int curr_j = 0;
	int curr_k = 0;
    
	for (int b = 0; b < nb1; b++) {
	    int d = dst.at({b+1,0});
	    
	    if (d >= 0) {  // index rows in A, C
		aparams.assign_state_bit(b, true, curr_i);
		cparams.assign_state_bit(slow_ilog2(d), true, curr_i);
		curr_i++;
		continue;
	    }

	    bool flag = false;
	    
	    for (int bb = 0; bb < nb2; bb++) {
		d = dst.at({b+1,bb+1});
		if (d < 0)
		    continue;

		// index col in A, row in B
		assert(d == 0);
		assert(!flag);
		aparams.assign_state_bit(b, false, curr_j);
		bparams.assign_state_bit(bb, true, curr_j);
		flag = true;
		curr_j++;
	    }

	    assert(flag);
	}

	for (int b = 0; b < nb2; b++) {
	    int d = dst.at({0,b+1});
	    if (d >= 0) {  // index cols in B, C
		bparams.assign_state_bit(b, false, curr_k);
		cparams.assign_state_bit(slow_ilog2(d), false, curr_k);
		curr_k++;
	    }
	}

	aparams.finalize_layout();
	bparams.finalize_layout();
	cparams.finalize_layout();
	
	this->show_layout();

	for (int ir = 0; ir < nb1+1; ir++) {
	    int i, ja;
	    int pindex_a = (ir > 0) ? (1 << (ir-1)) : 0;
	    aparams.pindex_to_rc(pindex_a, i, ja);
	    
	    for (int ic = 0; ic < nb2+1; ic++) {
		int jb, k;
		int pindex_b = (ic > 0) ? (1 << (ic-1)) : 0;
		bparams.pindex_to_rc(pindex_b, jb, k);

		int d = (ja == jb) ? cparams.rc_to_pindex(i,k) : -1;
		assert(dst.at({ir,ic}) == d);
	    }
	}
    }

    
    void show_layout() const
    {
	cout << "A-matrix\n";
	aparams.show_layout("i", "j");
	cout << "B-matrix\n";
	bparams.show_layout("j", "k");
	cout << "C-matrix\n";
	cparams.show_layout("i", "k");
    }


    void test_pack_unpack() const
    {
	aparams.test_pack_unpack();
	bparams.test_pack_unpack();
	cparams.test_pack_unpack();
	cout << "Yay!!" << endl;
    }
};


// -------------------------------------------------------------------------------------------------


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


__global__ void reveng_mma_kernel(int *dst, const int *asrc, const int *bsrc)
{
    constexpr int a_nrows = MmaParams::AParams::num_state_bits + 1;
    constexpr int b_nrows = MmaParams::BParams::num_state_bits + 1;
    constexpr int a_nreg = MmaParams::AParams::registers_per_thread;
    constexpr int b_nreg = MmaParams::BParams::registers_per_thread;
    constexpr int c_nreg = MmaParams::CParams::registers_per_thread;
    
    int afrag[a_nreg];
    int bfrag[b_nreg];
    int cfrag[c_nreg];

    assert(blockIdx.x == 0);
    int nwarps = blockDim.x >> 5;
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 0x1f;
	
    for (int i = threadIdx.x; i < a_nrows*b_nrows; i += blockDim.x)
	dst[i] = -1;
    
    for (int idst = warpId; idst < a_nrows*b_nrows; idst += nwarps) {
	int arow = idst / b_nrows;
	int brow = idst % b_nrows;

	MmaParams::AParams::load(afrag, asrc + (32*a_nreg)*arow);
	MmaParams::BParams::load(bfrag, bsrc + (32*b_nreg)*brow);
	MmaParams::CParams::set_zero(cfrag);
	
	mma(cfrag, afrag, bfrag, cfrag);

	int c_index = -1;
	int c_count = 0;
	
	#pragma unroll
	for (int r = 0; r < c_nreg; r++) {
	    // Note warp divergence here
	    if (cfrag[r] == 0)
		continue;
	    
	    assert(cfrag[r] == 1);
	    c_index = 32*r + laneId;
	    c_count++;
	}

	c_index = __reduce_max_sync(0xffffffff, c_index);
	c_count = __reduce_add_sync(0xffffffff, c_count);
	assert(c_count <= 1);
	
	if (laneId == 0)
	    dst[arow*b_nrows + brow] = c_index;
    }
}


__global__ void single_mma_kernel(int *cdst, const int *asrc, const int *bsrc)
{
    constexpr int a_nreg = MmaParams::AParams::registers_per_thread;
    constexpr int b_nreg = MmaParams::BParams::registers_per_thread;
    constexpr int c_nreg = MmaParams::CParams::registers_per_thread;

    int warpId = threadIdx.x >> 5;
    if (warpId != 0)
	return;

    int afrag[a_nreg];
    int bfrag[b_nreg];
    int cfrag[c_nreg];

    MmaParams::AParams::load(afrag, asrc);
    MmaParams::BParams::load(bfrag, bsrc);
    MmaParams::CParams::set_zero(cfrag);

    mma(cfrag, afrag, bfrag, cfrag);

    MmaParams::CParams::store(cfrag, cdst);
}


// -------------------------------------------------------------------------------------------------

    
int main(int argc, char **argv)
{
    Array<int> asrc = MmaParams::AParams::make_src_matrix();
    Array<int> bsrc = MmaParams::BParams::make_src_matrix();
    Array<int> dst({asrc.shape[0], bsrc.shape[0]}, af_gpu);

    reveng_mma_kernel<<<1,1024>>> (dst.data, asrc.data, bsrc.data);
    CUDA_PEEK("reveng_mma_kernel");
    CUDA_CALL(cudaDeviceSynchronize());
    dst = dst.to_host();
		   
#if 1
    for (int i = 0; i < dst.shape[0]; i++) {
	for (int j = 0; j < dst.shape[1]; j++)
	    cout << "  " << dst.at({i,j});
	cout << endl;
    }
#endif

    MmaParams params;
    params.reverse_engineer(dst);
    params.test_pack_unpack();

    Array<int> a = params.aparams.make_fragment(af_random);
    Array<int> b = params.bparams.make_fragment(af_random);

    Array<int> au = params.aparams.unpack_fragment(a);
    Array<int> bu = params.bparams.unpack_fragment(b);
    Array<int> ccpu = slow_matmul(au, bu);
    ccpu = params.cparams.pack_fragment(ccpu);

    Array<int> ag = a.to_gpu();
    Array<int> bg = b.to_gpu();
    Array<int> cgpu = params.cparams.make_fragment(af_gpu);

    single_mma_kernel<<<1,32>>> (cgpu.data, ag.data, bg.data);
    CUDA_PEEK("single_mma_kernel");
    CUDA_CALL(cudaDeviceSynchronize());
    cgpu = cgpu.to_host();

    int nc = MmaParams::CParams::int32s_per_fragment;
    assert(ccpu.shape_equals({nc}));
    assert(cgpu.shape_equals({nc}));
    for (int i = 0; i < nc; i++) {
	// cout << "  " << i << " " << ccpu.data[i] << " " << cgpu.data[i] << endl;
	assert(ccpu.data[i] == cgpu.data[i]);
    }
    
    return 0;
}