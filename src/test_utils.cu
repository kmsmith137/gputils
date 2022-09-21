#include <cassert>
#include <complex>
#include <iostream>

#include "../include/gputils/rand_utils.hpp"
#include "../include/gputils/test_utils.hpp"

// is_complex_v<T>, decomplexify_type<T>::type
#include "../include/gputils/complex_type_traits.hpp"

using namespace std;

namespace gputils {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


vector<ssize_t> make_random_strides(int ndim, const ssize_t *shape, int ncontig, int nalign)
{
    assert(ndim <= ArrayMaxDim);
    assert(ncontig >= 0);
    assert(ncontig <= ndim);
    assert(nalign >= 1);

    int nd_strided = ndim - ncontig;
    vector<ssize_t> axis_ordering = rand_permutation(nd_strided);
    
    vector<ssize_t> strides(ndim);
    ssize_t curr_stride = 1;

    for (int d = ndim-1; d >= nd_strided; d--) {
	assert(shape[d] > 0);
	strides[d] = curr_stride;
	curr_stride += (shape[d]-1) * strides[d];
    }

    for (int i = 0; i < nd_strided; i++) {
	int d = axis_ordering[i];
	assert(shape[d] > 0);

	ssize_t smin = (curr_stride + nalign - 1) / nalign;
	ssize_t smax = std::max(smin+1, (2*curr_stride)/nalign);
	strides[d] = nalign * rand_int(smin, smax+1);
	curr_stride += (shape[d]-1) * strides[d];
    }

    return strides;
}


vector<ssize_t> make_random_strides(const vector<ssize_t> &shape, int ncontig, int nalign)
{
    return make_random_strides(shape.size(), &shape[0], ncontig, nalign);
}


// -------------------------------------------------------------------------------------------------


template<typename T>
typename gputils::decomplexify_type<T>::type
assert_arrays_equal(const Array<T> &arr1,
		    const Array<T> &arr2,
		    const string &name1,
		    const string &name2,
		    const vector<string> &axis_names,
		    float epsabs,
		    float epsrel,
		    ssize_t max_display,
		    bool verbose)
{
    using Tr = typename decomplexify_type<T>::type;
    
    assert(arr1.shape_equals(arr2));
    assert(axis_names.size() == arr1.ndim);
    assert(max_display > 0);
    assert(epsabs >= 0.0);
    assert(epsrel >= 0.0);

    Array<T> harr1 = arr1.to_host(false);  // page_locked=false
    Array<T> harr2 = arr2.to_host(false);  // page_locked=false
    int nfail = 0;
    Tr maxdiff = 0;

    for (auto ix = arr1.ix_start(); arr1.ix_valid(ix); arr1.ix_next(ix)) {
	T x = harr1.at(ix);
	T y = harr2.at(ix);

	Tr delta;
	if constexpr (!is_unsigned_v<T>)
	    delta = std::abs(x-y);
	else
	    delta = (x > y) ? (x-y) : (y-x);
	
	Tr thresh = 0;
	if constexpr (!is_integral_v<T>)
	    thresh = epsabs + 0.5*epsrel * (std::abs(x) + std::abs(y));

	maxdiff = max(maxdiff, delta);
	bool failed = (delta > thresh);
	
	if (!failed && !verbose)
	    continue;

	if (failed && (nfail == 0))
	    cout << "\nassert_arrays_equal() failed [shape=" << arr1.shape_str() << "]\n";

	if (failed)
	    nfail++;
	
	if ((nfail >= max_display) || !verbose)
	    continue;
	
	cout << "   ";
	for (int d = 0; d < arr1.ndim; d++)
	    cout << " " << axis_names[d] << "=" << ix[d];

	cout << ": " << name1 << "=" << x << ", " << name2
	     << "=" << y << "  [delta=" << delta << "]";

	if (failed)
	    cout << " FAILED";

	cout << "\n";
    }
    
    if ((nfail > max_display) && !verbose)
	cout << "        [ + " << (nfail-max_display) << " more failures]\n";
    
    if (nfail > 0)
	exit(1);
    
    return maxdiff;
}


#define INSTANTIATE_ASSERT_ARRAYS_EQUAL(T)  \
    template				    \
    gputils::decomplexify_type<T>::type	    \
    assert_arrays_equal(		    \
	const Array<T> &arr1,	            \
	const Array<T> &arr2,		    \
	const string &name1,		    \
	const string &name2,		    \
	const vector<string> &axis_names,   \
	float epsabs,                       \
	float epsrel,                       \
	ssize_t max_display, 	            \
	bool verbose);



INSTANTIATE_ASSERT_ARRAYS_EQUAL(float);
INSTANTIATE_ASSERT_ARRAYS_EQUAL(double);
INSTANTIATE_ASSERT_ARRAYS_EQUAL(int);
INSTANTIATE_ASSERT_ARRAYS_EQUAL(long);
INSTANTIATE_ASSERT_ARRAYS_EQUAL(short);
INSTANTIATE_ASSERT_ARRAYS_EQUAL(char);
INSTANTIATE_ASSERT_ARRAYS_EQUAL(unsigned int);
INSTANTIATE_ASSERT_ARRAYS_EQUAL(unsigned long);
INSTANTIATE_ASSERT_ARRAYS_EQUAL(unsigned short);
INSTANTIATE_ASSERT_ARRAYS_EQUAL(unsigned char);
INSTANTIATE_ASSERT_ARRAYS_EQUAL(complex<float>);
INSTANTIATE_ASSERT_ARRAYS_EQUAL(complex<double>);


}  // namespace gputils
