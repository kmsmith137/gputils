#ifndef _GPUTILS_TEST_UTILS_HPP
#define _GPUTILS_TEST_UTILS_HPP

#include "Array.hpp"

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


extern void assert_arrays_equal(const Array<float> &arr1,
				const Array<float> &arr2,
				const std::string &name1,
				const std::string &name2,
				const std::vector<std::string> &axis_names,
				float epsabs = 3.0e-5,
				float epsrel = 1.0e-5,
				ssize_t max_display = 15);
				

}  // namespace test_utils

#endif // _GPUTILS_TEST_UTILS_HPP
