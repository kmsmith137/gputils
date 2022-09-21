#ifndef _GPUTILS_TEST_UTILS_HPP
#define _GPUTILS_TEST_UTILS_HPP

#include "Array.hpp"

// is_complex_v<T>, decomplexify_type<T>::type
#include "complex_type_traits.hpp"


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// If ncontig > 0, then the last 'ncontig' axes are guaranteed contiguous.
// If nalign > 1, then all strides besides the last 'ncontig' are guaranteed multiples of 'nalign'.

extern std::vector<ssize_t> make_random_strides(int ndim, const ssize_t *shape, int ncontig=0, int nalign=1);
extern std::vector<ssize_t> make_random_strides(const std::vector<ssize_t> &shape, int ncontig=0, int nalign=1);


// -------------------------------------------------------------------------------------------------


// Instantiated for T = float, double, (u)int, (u)long, (u)short, (u)char, complex<float>, complex<double>.
// For non floating point types, the 'epsabs' and 'epsrel' arguments are ignored.
//
// Returns the max difference between arrays. The return value is:
//    T    if instantiated for a non-complex type T
//    R    if instantiated for T= std::complex<R>

template<typename T>
extern typename gputils::decomplexify_type<T>::type
assert_arrays_equal(const Array<T> &arr1,
		    const Array<T> &arr2,
		    const std::string &name1,
		    const std::string &name2,
		    const std::vector<std::string> &axis_names,
		    float epsabs = 3.0e-5,
		    float epsrel = 1.0e-5,
		    ssize_t max_display = 15,
		    bool verbose = false);


}  // namespace test_utils

#endif // _GPUTILS_TEST_UTILS_HPP
