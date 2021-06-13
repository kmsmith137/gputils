#ifndef _GPUTILS_ARRAY_HPP
#define _GPUTILS_ARRAY_HPP

#include <string>
#include <vector>
#include "mem_utils.hpp"  // af_alloc() and flags


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


static constexpr int ArrayMaxDim = 5;


template<typename T>
struct Array {
    T *data = nullptr;
    
    int ndim = 0;
    ssize_t shape[ArrayMaxDim];
    ssize_t size = 0;

    ssize_t strides[ArrayMaxDim];  // in units sizeof(T), not bytes

    std::shared_ptr<T> base;  // FIXME generalize to shared_ptr<void>?
    int aflags = af_uninitialized;

    // "Empty" arrays are size-zero objects containing null pointers.
    // All Arrays obey the rules:
    //
    //  (data == nullptr)
    //      iff (size == 0)
    //      iff ((ndim==0) || (shape[i]==0 for some 0 <= i < ndim))
    //
    // Note that zero-dimensional arrays are empty (unlike numpy,
    // where zero-dimensional arrays have size 1).

    Array();
    
    // Allocator flags ('aflags') are defined in mem_utils.hpp
    // In particular, the 'af_gpu' flag allocates the array on the GPU
    Array(int ndim, const ssize_t *shape, int aflags=0);
    Array(const std::vector<ssize_t> &shape, int aflags=0);
    
    // Syntactic sugar for constructing array with "inline" dimensions, e.g.
    //    Array<float> arr({m,n}, af_gpu);
    Array(std::initializer_list<ssize_t> shape, int aflags=0);

	  
    // Is array addressable on GPU? On host?
    inline bool on_gpu() const { return !data || af_on_gpu(aflags); }
    inline bool on_host() const { return !data || af_on_host(aflags); }


    inline void fill(const Array<T> &arr);
    inline Array<T> clone(int aflags) const;
    inline Array<T> clone() const;  // retains location_flags of source array
    
    // Returns an Array addressable on GPU (or host), making a copy if necessary.
    inline Array<T> to_gpu() const;
    inline Array<T> to_host(bool page_locked=true) const;

    // Returns number of contiguous dimensions, assuming indices are ordered
    // from slowest to fastest varying. Returns 'ndim' for an empty array.
    int get_ncontig() const;
    bool is_fully_contiguous() const { return get_ncontig() == ndim; }
    
    // at(): range-checked accessor
    // (I'm reserving operator[] for an unchecked accessor.)
    
    inline T& at(int ndim, const ssize_t *ix);
    inline T& at(const std::vector<ssize_t> &ix);
    inline T& at(std::initializer_list<ssize_t> ix);
    
    inline const T& at(int ndim, const ssize_t *ix) const;
    inline const T& at(const std::vector<ssize_t> &ix) const;
    inline const T& at(std::initializer_list<ssize_t> ix) const;

    
    // The new Arrays returned by slice() contain references
    // (not copies) to the data in the original Array.
    inline Array<T> slice(int axis, int start, int stop) const;
    inline Array<T> slice(int axis, int ix) const;   // returns array of dimension (ndim-1)


    // Reshape-by-reference. Throws an exception if either (1) requested shape
    // is incompatible with the current shape; (2) current strides don't permit
    // axes to be combined without copying.
    //
    // FIXME reshape_ref() hasn't been systematically tested, and should be!

    inline Array<T> reshape_ref(int ndim, const ssize_t *shape) const;
    inline Array<T> reshape_ref(const std::vector<ssize_t> &shape) const;
    inline Array<T> reshape_ref(std::initializer_list<ssize_t> shape) const;
    

    // For looping over array indices (not high-performance):
    //
    //    Array<T> arr;
    //    for (auto ix = arr.ix_start(); arr.ix_valid(ix); arr.ix_next(ix)) {
    //        float x = arr.at(ix);
    //        // ...
    //    }
    //    
    // Warning: ix_valid() is not a general-purpose index validator!
    // It only works on the output of ix_start() -> ix_next() -> ...
    
    inline std::vector<ssize_t> ix_start() const;
    inline bool ix_valid(const std::vector<ssize_t> &ix) const;
    inline void ix_next(std::vector<ssize_t> &ix) const;
    
    
    inline bool shape_equals(int ndim, const ssize_t *shape) const;
    inline bool shape_equals(const std::vector<ssize_t> &shape) const;
    inline bool shape_equals(std::initializer_list<ssize_t> shape) const;
    template<typename T2> inline bool shape_equals(const Array<T2> &a) const;    

    inline std::string shape_str() const;
    inline std::string stride_str() const;

    // Throws exception on failure.
    void check_invariants() const;
    
    // "Cheat" accessor, which gives a non-const reference to a const Array
    inline T& _at(int ndim, const ssize_t *ix) const;
};


// -------------------------------------------------------------------------------------------------
//
// Non-inline functions defined elsewhere


extern void check_array_invariants(const void *data, int ndim, const ssize_t *shape,
				   ssize_t size, const ssize_t *strides, int aflags);

extern int compute_ncontig(int ndim, const ssize_t *shape, const ssize_t *strides);

extern ssize_t compute_size(int ndim, const ssize_t *shape);

extern bool shape_eq(int ndim1, const ssize_t *shape1, int ndim2, const ssize_t *shape2);

extern std::string shape_str(int ndim, const ssize_t *shape);

extern void reshape_ref_helper(int src_ndim, const ssize_t *src_shape, const ssize_t *src_strides,
			       int dst_ndim, const ssize_t *dst_shape, ssize_t *dst_strides);

extern void fill_helper(void *dst, const void *src, int nd,
			const ssize_t *shape, const ssize_t *dstride,
			const ssize_t *sstride, ssize_t itemsize, bool noisy=false);


// -------------------------------------------------------------------------------------------------
//
// Inline implementations


template<typename T>
Array<T>::Array()
    : Array(0, nullptr, 0) { }

template<typename T>
Array<T>::Array(const std::vector<ssize_t> &shape_, int aflags)
    : Array(shape_.size(), &shape_[0], aflags) { }

template<typename T>
Array<T>::Array(std::initializer_list<ssize_t> shape_, int aflags)
    : Array(shape_.size(), shape_.begin(), aflags) { }


template<typename T>
Array<T>::Array(int ndim_, const ssize_t *shape_, int aflags_)
    : ndim(ndim_), aflags(aflags_)
{
    assert(ndim >= 0);
    assert(ndim <= ArrayMaxDim);

    for (int i = ndim; i < ArrayMaxDim; i++)
	shape[i] = strides[i] = 0;

    if (ndim == 0) {
	data = nullptr;
	size = 0;
	return;
    }

    size = 1;    
    for (int i = ndim-1; i >= 0; i--) {
	strides[i] = size;
	shape[i] = shape_[i];
	size *= shape_[i];
	assert(shape[i] >= 0);
    }

    if (size != 0) {
	base = af_alloc<T> (size, aflags);
	data = base.get();
    }
    else
	data = nullptr;
}


template<typename T>
void Array<T>::fill(const Array<T> &arr)
{
    assert(this->shape_equals(arr));

    if (size > 0)
	fill_helper(data, arr.data, ndim, shape, strides, arr.strides, sizeof(T), false);
}


template<typename T>
Array<T> Array<T>::clone(int aflags_) const
{
    Array<T> ret(this->ndim, this->shape, aflags_ & ~af_initialization_flags);
    ret.fill(*this);
    return ret;
}

template<typename T>
Array<T> Array<T>::clone() const
{
    return this->clone(this->aflags & af_location_flags);
}

template<typename T>
Array<T> Array<T>::to_gpu() const
{
    return this->on_gpu() ? (*this) : this->clone(af_gpu);
}

template<typename T>
Array<T> Array<T>::to_host(bool page_locked) const
{
    return this->on_host() ? (*this) : this->clone(page_locked ? af_page_locked : 0);
}

template<typename T>
int Array<T>::get_ncontig() const
{
    return compute_ncontig(ndim, shape, strides);
}

template<typename T>
T& Array<T>::_at(int nd, const ssize_t *ix) const
{
    assert(on_host());
    assert(this->ndim == nd);
    
    ssize_t pos = 0;
    for (int d = 0; d < nd; d++) {
	assert(ix[d] >= 0 && ix[d] < shape[d]);
	pos += ix[d] * strides[d];
    }
    
    return data[pos];
}

template<typename T> T& Array<T>::at(int nd, const ssize_t *ix)          { return _at(nd, ix); }
template<typename T> T& Array<T>::at(const std::vector<ssize_t> &ix)     { return _at(ix.size(), &ix[0]); }
template<typename T> T& Array<T>::at(std::initializer_list<ssize_t> ix)  { return _at(ix.size(), ix.begin()); }

template<typename T> const T& Array<T>::at(int nd, const ssize_t *ix) const          { return _at(nd, ix); }
template<typename T> const T& Array<T>::at(const std::vector<ssize_t> &ix) const     { return _at(ix.size(), &ix[0]); }
template<typename T> const T& Array<T>::at(std::initializer_list<ssize_t> ix) const  { return _at(ix.size(), ix.begin()); }


template<typename T>
Array<T> Array<T>::slice(int axis, int ix) const
{
    assert((axis >= 0) && (axis < ndim));
    assert((ix >= 0) && (ix < shape[axis]));

    // Slicing (1-dim -> 0-dim) doesn't make sense,
    // since our zero-dimensional Arrays are empty.
    assert(ndim > 1);
    
    Array<T> ret;
    ret.ndim = ndim-1;
    ret.aflags = aflags & af_location_flags;
    ret.base = base;
    ret.size = 1;

// Suppress spurious GCC warning in loop that follows.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
    
    for (int i = 0; i < ndim-1; i++) {
	int j = (i < axis) ? i : (i+1);
	ret.shape[i] = shape[j];
	ret.strides[i] = strides[j];
	ret.size *= shape[j];
    }
    
#pragma GCC diagnostic pop

    if (ret.size == 0) {
	ret.data = nullptr;
	return ret;
    }

    ret.data = data + (ix * strides[axis]);
    return ret;
}


template<typename T>
Array<T> Array<T>::slice(int axis, int start, int stop) const
{
    // Currently we allow slices like arr[2:10] but not arr[2:-10] or arr[2:10:2].
    // This would be easy to generalize!
    assert((axis >= 0) && (axis < ndim));
    assert((start >= 0) && (start <= stop) && (stop <= shape[axis]));
    
    Array<T> ret;
    ret.ndim = ndim;
    ret.aflags = aflags & af_location_flags;
    ret.base = base;
    ret.size = 1;

    for (int i = 0; i < ndim; i++) {
	ret.shape[i] = (i == axis) ? (stop-start) : shape[i];
	ret.strides[i] = strides[i];
	ret.size *= ret.shape[i];
    }

    if (ret.size == 0) {
	ret.data = nullptr;
	return ret;
    }

    ret.data = data + (start * strides[axis]);
    return ret;
}


template<typename T>
Array<T> Array<T>::reshape_ref(int ndim_, const ssize_t *shape_) const
{
    assert(ndim_ >= 0);
    assert(ndim_ <= ArrayMaxDim);
	   
    Array<T> ret;
    ret.ndim = ndim_;
    ret.data = this->data;
    ret.size = this->size;
    ret.base = this->base;
    ret.aflags = this->aflags;
    
    for (int d = 0; d < ret.ndim; d++)
	ret.shape[d] = shape_[d];

    // reshape_ref_helper()
    //  - assumes src_ndim, src_shape, src_strides, dst_ndim have been validated by caller
    //  - validates dst_shape
    //  - initializes dst_strides
    //  - throws exception if shapes are incompatible, or src_strides are bad.
    reshape_ref_helper(ndim, shape, strides, ret.ndim, ret.shape, ret.strides);
    
    return ret;
}

template<typename T>
inline Array<T> Array<T>::reshape_ref(const std::vector<ssize_t> &shape) const
{
    return reshape_ref(shape.size(), &shape[0]);
}

template<typename T>
inline Array<T> Array<T>::reshape_ref(std::initializer_list<ssize_t> shape) const
{
    return reshape_ref(shape.size(), shape.begin());
}


template<typename T>
std::vector<ssize_t> Array<T>::ix_start() const
{
    return std::vector<ssize_t> (ndim, 0);
}

template<typename T>
bool Array<T>::ix_valid(const std::vector<ssize_t> &ix) const
{
    // Warning: ix_valid() is not a general-purpose index validator!
    // Only works on the output of ix_start() -> ix_next() -> ...
    return (size > 0) && (ix[0] < shape[0]);
}

template<typename T>
void Array<T>::ix_next(std::vector<ssize_t> &ix) const
{
    for (int d = ndim-1; d >= 1; d--) {
	if (ix[d] < shape[d]-1) {
	    ix[d]++;
	    return;
	}
	ix[d] = 0;
    }
    
    if (ndim > 0)
	ix[0]++;
}


template<typename T>
bool Array<T>::shape_equals(int ndim_, const ssize_t *shape_) const
{
    return shape_eq(this->ndim, this->shape, ndim_, shape_);
}

template<typename T>
bool Array<T>::shape_equals(const std::vector<ssize_t> &shape) const
{
    return shape_eq(this->ndim, this->shape, shape.size(), &shape[0]);
}

template<typename T>
bool Array<T>::shape_equals(std::initializer_list<ssize_t> shape_) const
{
    return shape_eq(this->ndim, this->shape, shape_.size(), shape_.begin());
}

template<typename T> template<typename T2>
bool Array<T>::shape_equals(const Array<T2> &a) const
{
    return shape_eq(this->ndim, this->shape, a.ndim, a.shape);
}

template<typename T> std::string Array<T>::shape_str() const
{
    return ::gputils::shape_str(ndim, shape);
}

template<typename T> std::string Array<T>::stride_str() const
{
    return ::gputils::shape_str(ndim, strides);
}

template<typename T> void Array<T>::check_invariants() const
{
    check_array_invariants(data, ndim, shape, size, strides, aflags);
}
    

} // namespace gputils

#endif // _GPUTILS_ARRAY_HPP
