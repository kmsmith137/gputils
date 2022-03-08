#include <iostream>
#include "../include/gputils.hpp"

using namespace std;
using namespace gputils;



// -------------------------------------------------------------------------------------------------


__device__ __forceinline__
void mma_s4_m16_n8_k64(int c[4], int a[4], int b[2], int d[4])
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


// The 'asrc' array shape is (na, 32*Areg).
// The 'bsrc' array shape is (nb, 32*Breg).
// The 'cdst' array shape is (na, nb, 32*Creg)
//
// This kernel should be launched with (nblocks_x, nblocks_y) = (na, nb),
// and 32 threads/block.


template<void (*F)(int[], int[], int[], int[]), int Areg, int Breg, int Creg>
__global__ void mma_kernel(int *cdst, const int *asrc, const int *bsrc)
{
    assert(blockDim.x == 32);

    int ia = blockIdx.x;
    int ib = blockIdx.y;
    int nb = gridDim.y;
    int laneId = threadIdx.x;

    // Add block offsets
    asrc += ia * 32 * Areg;
    bsrc += ib * 32 * Breg;
    cdst += (ia*nb + ib) * 32 * Creg;

    int afrag[Areg];
    int bfrag[Breg];
    int cfrag[Creg];

    for (int r = 0; r < Areg; r++)
	afrag[r] = asrc[32*r + laneId];
    for (int r = 0; r < Breg; r++)
	bfrag[r] = bsrc[32*r + laneId];    
    for (int r = 0; r < Creg; r++)
	cfrag[r] = 0;

    F(cfrag, afrag, bfrag, cfrag);
    
    for (int r = 0; r < Creg; r++)
	cdst[32*r + laneId] = cfrag[r];
}


// -------------------------------------------------------------------------------------------------


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

    // A "physical" location is identified by (b,r,t), where b indexes a "bit" location within
    // a single register, r indexes a register in a thread, and t indexes a thread.
    static constexpr int num_bstate_bits = constexpr_ilog2(32 / bit_depth);
    static constexpr int num_regstate_bits = constexpr_ilog2(registers_per_thread);

    // A "logical" location is identified by a (row,col).
    static constexpr int num_rowstate_bits = constexpr_ilog2(nrows);
    static constexpr int num_colstate_bits = constexpr_ilog2(ncols);

    // Physical and logical locations should be parameterized by the same number of state bits.
    static constexpr int num_state_bits = num_rowstate_bits + num_colstate_bits;
    static_assert(num_state_bits == num_bstate_bits + num_regstate_bits + 5);

    // State bit mapping (physical -> logical)
    vector<bool> pbit_isrow;
    vector<int> pbit_lindex;
    bool finalized = false;

    
    MatParams() :
	pbit_isrow(num_state_bits, false),
	pbit_lindex(num_state_bits, -1)
    { }

    
    static __host__ Array<int> make_basis_fragments()
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
	assert(!finalized);
	assert((b >= 0) && (b < num_state_bits));
	assert(this->pbit_lindex[b] < 0);

	assert(index >= 0);
	assert(index < (isrow ? num_rowstate_bits : num_colstate_bits));
	
	this->pbit_isrow[b] = isrow;
	this->pbit_lindex[b] = index;
    }


    void finalize()
    {
	for (int pb = 0; pb < num_state_bits; pb++)
	    assert(pbit_lindex[pb] >= 0);
	
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

    
    void reverse_engineer()
    {
	constexpr int na = AParams::num_state_bits;
	constexpr int nb = BParams::num_state_bits;
	constexpr int Areg = AParams::registers_per_thread;
	constexpr int Breg = BParams::registers_per_thread;
	constexpr int Creg = CParams::registers_per_thread;
    
	Array<int> asrc = AParams::make_basis_fragments();
	Array<int> bsrc = BParams::make_basis_fragments();
	Array<int> cdst({na+1, nb+1, CParams::int32s_per_fragment}, af_gpu);
	
	assert(asrc.shape_equals({na+1, AParams::int32s_per_fragment}));
	assert(bsrc.shape_equals({nb+1, BParams::int32s_per_fragment}));

	dim3 nblocks;
	nblocks.x = na+1;
	nblocks.y = nb+1;
	nblocks.z = 1;
	
	mma_kernel<mma_s4_m16_n8_k64,Areg,Breg,Creg> <<<nblocks, 32>>> (cdst.data, asrc.data, bsrc.data);
	
	CUDA_PEEK("mma_kernel [1]");
	cdst = cdst.to_host();

	Array<int> coupling({na+1,nb+1});
	for (int i = 0; i < na+1; i++) {
	    for (int j = 0; j < nb+1; j++) {
		coupling.at({i,j}) = -1;
		for (int k = 0; k < cdst.shape[2]; k++) {
		    if (cdst.at({i,j,k}) != 0) {
			assert(cdst.at({i,j,k}) == 1);
			assert(coupling.at({i,j}) < 0);
			coupling.at({i,j}) = k;
		    }
		}
	    }
	}
	
#if 1
	cout << "Coupling matrix\n";
	for (int i = 0; i < na+1; i++) {
	    for (int j = 0; j < nb+1; j++)
		cout << "  " << coupling.at({i,j});
	    cout << endl;
	}
#endif
	    
	assert(coupling.at({0,0}) == 0);
	
	int curr_i = 0;
	int curr_j = 0;
	int curr_k = 0;
    
	for (int ia = 0; ia < na; ia++) {
	    int d = coupling.at({ia+1,0});
	    
	    if (d >= 0) {  // index rows in A, C
		aparams.assign_state_bit(ia, true, curr_i);
		cparams.assign_state_bit(slow_ilog2(d), true, curr_i);
		curr_i++;
		continue;
	    }

	    bool flag = false;
	    
	    for (int ib = 0; ib < nb; ib++) {
		d = coupling.at({ia+1,ib+1});
		if (d < 0)
		    continue;

		// index col in A, row in B
		assert(d == 0);
		assert(!flag);
		aparams.assign_state_bit(ia, false, curr_j);
		bparams.assign_state_bit(ib, true, curr_j);
		flag = true;
		curr_j++;
	    }

	    assert(flag);
	}

	for (int ib = 0; ib < nb; ib++) {
	    int d = coupling.at({0,ib+1});
	    if (d >= 0) {  // index cols in B, C
		bparams.assign_state_bit(ib, false, curr_k);
		cparams.assign_state_bit(slow_ilog2(d), false, curr_k);
		curr_k++;
	    }
	}

	aparams.finalize();
	bparams.finalize();
	cparams.finalize();
	
	this->show_layout();

	for (int ir = 0; ir < na+1; ir++) {
	    int i, ja;
	    int pindex_a = (ir > 0) ? (1 << (ir-1)) : 0;
	    aparams.pindex_to_rc(pindex_a, i, ja);
	    
	    for (int ic = 0; ic < nb+1; ic++) {
		int jb, k;
		int pindex_b = (ic > 0) ? (1 << (ic-1)) : 0;
		bparams.pindex_to_rc(pindex_b, jb, k);

		int d = (ja == jb) ? cparams.rc_to_pindex(i,k) : -1;
		assert(coupling.at({ir,ic}) == d);
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


    void end_to_end_test() const
    {
	constexpr int Areg = AParams::registers_per_thread;
	constexpr int Breg = BParams::registers_per_thread;
	constexpr int Creg = CParams::registers_per_thread;
	constexpr int nc = CParams::int32s_per_fragment;
	
	aparams.test_pack_unpack();
	bparams.test_pack_unpack();
	cparams.test_pack_unpack();

	Array<int> a = aparams.make_fragment(af_random);
	Array<int> b = bparams.make_fragment(af_random);

	Array<int> au = aparams.unpack_fragment(a);
	Array<int> bu = bparams.unpack_fragment(b);
	Array<int> ccpu = slow_matmul(au, bu);
	ccpu = cparams.pack_fragment(ccpu);

	Array<int> ag = a.to_gpu();
	Array<int> bg = b.to_gpu();
	Array<int> cgpu = cparams.make_fragment(af_gpu);

	mma_kernel<mma_s4_m16_n8_k64,Areg,Breg,Creg> <<<1,32>>> (cgpu.data, ag.data, bg.data);
	CUDA_PEEK("mma_kernel [2]");
	CUDA_CALL(cudaDeviceSynchronize());
	cgpu = cgpu.to_host();

	assert(ccpu.shape_equals({nc}));
	assert(cgpu.shape_equals({nc}));
	for (int i = 0; i < nc; i++) {
	    // cout << "  " << i << " " << ccpu.data[i] << " " << cgpu.data[i] << endl;
	    assert(ccpu.data[i] == cgpu.data[i]);
	}
    }
};


// -------------------------------------------------------------------------------------------------

    
int main(int argc, char **argv)
{
    MmaParams params;
    params.reverse_engineer();
    params.end_to_end_test();
    
    return 0;
}
