#include <iostream>
#include <algorithm>

#include "../include/gputils.hpp"

using namespace std;
using namespace gputils;


// -------------------------------------------------------------------------------------------------


template<typename T>
inline T prod(ssize_t n, const T *arr)
{
    T ret = 1;
    for (ssize_t i = 0; i < n; i++)
	ret *= arr[i];
    return ret;
}


template<typename T>
inline T prod(const vector<T> &v)
{
    return prod(v.size(), &v[0]);
}


static vector<ssize_t> make_random_shape()
{
    int size = 1;
    int ndim = rand_int(1, ArrayMaxDim+1);
    vector<ssize_t> shape(ndim, 1);
    
    while (size < 20000) {
	if (rand_uniform() < 0.1)
	    break;
	
	int i = rand_int(0, ndim);
	int t = rand_int(2, 6);
	
	shape[i] *= t;
	size *= t;
    }
    
    return shape;
}


// FIXME I wrote a function make_random_strides() here, forgot it existed,
// then wrote a similar function in test_utils.cu which I also named
// make_random_strides().
//
// For now I've just renamed this function to xmake_random_strides(), but I'd
// like to come back later and replace it with the "new" make_random_strides().

static vector<ssize_t> xmake_random_strides(const vector<ssize_t> &shape)
{
    int ndim = shape.size();
    vector<ssize_t> perm = rand_permutation(ndim);

    vector<ssize_t> pshape(ndim);
    for (int d = 0; d < ndim; d++)
	pshape[d] = shape[perm[d]];
    
    vector<ssize_t> pstrides(ndim);
    ssize_t min_stride = 1;

    for (int d = ndim-1; d >= 0; d--) {
	ssize_t max_stride = 1000000 / (min_stride * prod(d+1,&pshape[0]));
	max_stride = min(max_stride, min_stride+5);
	max_stride = max(max_stride, min_stride);

	if (rand_uniform() < 0.5)
	    pstrides[d] = min_stride;
	else
	    pstrides[d] = rand_int(min_stride, max_stride + 1);

	// Update min_stride for next iteration of loop
	min_stride = pstrides[d] * pshape[d];
    }

    vector<ssize_t> strides(ndim);
    for (int d = 0; d < ndim; d++)
	strides[perm[d]] = pstrides[d];
    
    return strides;
}


// -------------------------------------------------------------------------------------------------


struct TestArrayAxis {
    ssize_t index = 0;
    ssize_t length = 0;
    ssize_t stride = 0;

    bool operator<(const TestArrayAxis &s) const
    {
	return stride < s.stride;
    }
};


template<typename T>
struct RandomlyStridedArray {
    Array<T> arr;
    int ndim = 0;

    shared_ptr<T> cbase;  // contiguous base array
    shared_ptr<T> cbase_copy;
    ssize_t cbase_len = 0;

    std::vector<TestArrayAxis> axes;
    bool noisy = false;

    
    RandomlyStridedArray(const vector<ssize_t> &shape, const vector<ssize_t> &strides, bool noisy_=false) :
	noisy(noisy_)
    {
	if (noisy) {
	    cout << "RandomlyStridedArray" // << "<" << type_name<T>() << ">"
		 << ": shape=" << tuple_str(shape)
		 << ", strides=" << tuple_str(strides) << endl;
	}
		
	ndim = shape.size();
	assert(ndim > 0);
	assert(ndim <= ArrayMaxDim);
	assert(shape.size() == strides.size());

	cbase_len = 1;
	for (int d = 0; d < ndim; d++) 
	    cbase_len += (shape[d]-1) * strides[d];

	cbase = af_alloc<T> (cbase_len, af_guard | af_random);
	cbase_copy = af_alloc<T> (cbase_len);
	memcpy(cbase_copy.get(), cbase.get(), cbase_len * sizeof(T));

	arr.data = cbase.get();
	arr.ndim = ndim;
	arr.base = cbase;
	arr.size = compute_size(ndim, &shape[0]);
	arr.aflags = 0;
	
	for (int d = 0; d < ArrayMaxDim; d++) {
	    arr.shape[d] = (d < ndim) ? shape[d] : 0;
	    arr.strides[d] = (d < ndim) ? strides[d] : 0;
	}

	arr.check_invariants();
	// check_for_buffer_overflows();

	axes.resize(ndim);
	for (int d = 0; d < ndim; d++) {
	    axes[d].index = d;
	    axes[d].length = shape[d];
	    axes[d].stride = strides[d];
	}

	std::sort(axes.begin(), axes.end());
    }

    RandomlyStridedArray(const vector<ssize_t> &shape_, bool noisy_=false)
	: RandomlyStridedArray(shape_, xmake_random_strides(shape_), noisy_)
    { }
    
    RandomlyStridedArray(bool noisy_=false)
	: RandomlyStridedArray(make_random_shape(), noisy_)
    { }


    // Given a cbase index 0 <= pos < cbase_len, return true (and
    // initialize the length-ndim array ix_out) if in 'arr'.
    
    bool find(ssize_t pos, ssize_t *ix_out) const
    {
	assert(pos >= 0 && pos < cbase_len);

	// Process axes from largest stride to smallest
	for (int d = ndim-1; d >= 0; d--) {
	    const TestArrayAxis &axis = axes[d];
	    int i = axis.index;

	    if (axis.length == 1) {
		ix_out[i] = 0;
		continue;
	    }

	    ssize_t j = pos / axis.stride;
	    if (j >= axis.length)
		return false;
	    
	    pos -= j * axis.stride;		
	    ix_out[i] = j;
	}

	return (pos == 0);
    }


    void check_for_buffer_overflows() const
    {
	T *p1 = cbase.get();
	T *p2 = cbase_copy.get();
	ssize_t ix_out[ndim];
	
	for (ssize_t i = 0; i < cbase_len; i++)
	    if (!find(i, ix_out))
		assert(p1[i] == p2[i]);
    }

    // test_basics(): tests consistency of find(), Array::at(), and
    // Array index enumeration (ix_start, ix_valid, ix_next).
    
    void test_basics() const
    {
	vector<bool> flag(cbase_len, false);
	ssize_t ix_out[ndim];
	ssize_t s = 0;
	
	if (noisy)
	    cout << "    test_basics()" << endl;

	for (auto ix = arr.ix_start(); arr.ix_valid(ix); arr.ix_next(ix)) {
	    ssize_t j = &arr.at(ix) - arr.data;
	    assert(j >= 0 && j < cbase_len);
	    assert(find(j, ix_out));
	    assert(!flag[j]);
	    flag[j] = true;
	    s++;

	    for (int d = 0; d < ndim; d++)
		assert(ix_out[d] == ix[d]);
	}

	assert(s == arr.size);

	for (ssize_t j = 0; j < cbase_len; j++)
	    if (!flag[j])
		assert(!find(j, ix_out));
    }

    
    void test_thin_slice(int axis, int pos) const
    {
	ssize_t ix_out[ndim];

	// Thin-slicing 1-D arrays isn't implemented (see comment in Array.hpp)
	assert(ndim > 1);
	
	if (noisy)
	    cout << "    test_thin_slice(axis=" << axis << ",pos=" << pos << ")" << endl;

	Array<T> s = arr.slice(axis, pos);
	s.check_invariants();

	assert(s.ndim == ndim-1);
	for (int d = 0; d < axis; d++)
	    assert(s.shape[d] == arr.shape[d]);
	for (int d = axis+1; d < ndim; d++)
	    assert(s.shape[d-1] == arr.shape[d]);

	for (auto ix = s.ix_start(); s.ix_valid(ix); s.ix_next(ix)) {
	    ssize_t j = &s.at(ix) - arr.data;
	    assert(j >= 0 && j < cbase_len);
	    assert(find(j, ix_out));

	    assert(ix_out[axis] == pos);
	    for (int d = 0; d < axis; d++)
		assert(ix_out[d] == ix[d]);
	    for (int d = axis+1; d < ndim; d++)
		assert(ix_out[d] == ix[d-1]);
	}
    }

    void test_thin_slice() const
    {
	int axis = rand_int(0, ndim);
	int pos = rand_int(0, arr.shape[axis]);
	test_thin_slice(axis, pos);
    }


    void test_thick_slice(int axis, int start, int stop) const
    {
	ssize_t ix_out[ndim];
	
	if (noisy) {
	    cout << "    test_thick_slice(axis=" << axis
		 << ", start=" << start
		 << ", stop=" << stop << ")" << endl;
	}

	Array<T> s = arr.slice(axis, start, stop);
	s.check_invariants();

	assert(s.ndim == ndim);
	for (int d = 0; d < ndim; d++) {
	    ssize_t t = (d == axis) ? (stop-start) : arr.shape[d];
	    assert(s.shape[d] == t);
	}

	for (auto ix = s.ix_start(); s.ix_valid(ix); s.ix_next(ix)) {
	    ssize_t j = &s.at(ix) - arr.data;
	    assert(j >= 0 && j < cbase_len);
	    assert(find(j, ix_out));

	    for (int d = 0; d < ndim; d++) {
		ssize_t t = (d == axis) ? start : 0;
		assert(ix_out[d] == ix[d] + t);
	    }
	}
    }

    void test_thick_slice() const
    {
	int axis = rand_int(0, ndim);
	int slen = rand_int(0, arr.shape[axis] + 1);
	int start = rand_int(0, arr.shape[axis] - slen + 1);
	int stop = start + slen;
	test_thick_slice(axis, start, stop);
    }

    
    void run_simple_tests() const
    {
	test_basics();
	test_thick_slice();

	if (ndim > 1)
	    test_thin_slice();
    }
};



// -------------------------------------------------------------------------------------------------


template<typename T>
struct FillTestInstance {
    vector<ssize_t> shape;
    vector<ssize_t> strides1;
    vector<ssize_t> strides2;
    
    RandomlyStridedArray<T> rs1;
    RandomlyStridedArray<T> rs2;

    bool noisy = false;
    
    
    FillTestInstance(const vector<ssize_t> &shape_,
		     const vector<ssize_t> &strides1_,
		     const vector<ssize_t> &strides2_) :
	shape(shape_),
	strides1(strides1_),
	strides2(strides2_),
	rs1(shape_, strides1_),
	rs2(shape_, strides2_)
    { }
    
    FillTestInstance(const vector<ssize_t> &shape_) :
	FillTestInstance(shape_, xmake_random_strides(shape_), xmake_random_strides(shape_))
    { }

    FillTestInstance() :
	FillTestInstance(make_random_shape())
    { }

    
    void run()
    {
	if (noisy) {
	    cout << "test_fill: shape=" << tuple_str(shape)
		 << ", strides1=" << tuple_str(strides1)
		 << ", strides2=" << tuple_str(strides2)
		 << ")" << endl;
	}

	Array<T> &arr1 = rs1.arr;
	Array<T> &arr2 = rs2.arr;

	arr1.fill(arr2);
	
	for (auto ix = arr1.ix_start(); arr1.ix_valid(ix); arr1.ix_next(ix))
	    assert(arr1.at(ix) == arr2.at(ix));
	
	rs1.check_for_buffer_overflows();
	rs2.check_for_buffer_overflows();
    }
};

    

// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    bool noisy = false;
    int niter = 1000;

    for (int i = 0; i < niter; i++) {
	if (i % 100 == 0)
	    cout << "test-array: iteration " << i << "/" << niter << endl;
	
	RandomlyStridedArray<float> tf(noisy);
	tf.run_simple_tests();
	
	RandomlyStridedArray<char> tc(noisy);
	tc.run_simple_tests();

	FillTestInstance<float> ff;
	ff.noisy = noisy;
	ff.run();
	
	FillTestInstance<char> fc;
	fc.noisy = noisy;
	fc.run();
    }

    cout << "test-array passed!" << endl;
    return 0;
}
