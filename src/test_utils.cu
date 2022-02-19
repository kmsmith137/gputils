#include <cassert>
#include <iostream>
#include "../include/gputils/test_utils.hpp"

using namespace std;

namespace gputils {
#if 0
}  // editor auto-indent
#endif

    
void assert_arrays_equal(const Array<float> &arr1,
			 const Array<float> &arr2,
			 const string &name1,
			 const string &name2,
			 const vector<string> &axis_names,
			 float epsabs,
			 float epsrel,
			 ssize_t max_display)
{
    assert(arr1.shape_equals(arr2));
    assert(axis_names.size() == arr1.ndim);
    assert(max_display > 0);
    assert(epsabs >= 0.0);
    assert(epsrel >= 0.0);

    Array<float> harr1 = arr1.to_host(false);  // page_locked=false
    Array<float> harr2 = arr2.to_host(false);  // page_locked=false
    int nfail = 0;

    for (auto ix = arr1.ix_start(); arr1.ix_valid(ix); arr1.ix_next(ix)) {
	float x = harr1.at(ix);
	float y = harr2.at(ix);
	float delta = std::abs(x-y);
	float thresh = epsabs + 0.5*epsrel * (std::abs(x) + std::abs(y));
	    
	if (delta <= thresh)
	    continue;

	if (nfail == 0)
	    cout << "\nassert_arrays_equal() failed [shape=" << arr1.shape_str() << "]\n";

	if (nfail++ >= max_display)
	    continue;
	
	cout << "   ";
	for (int d = 0; d < arr1.ndim; d++)
	    cout << " " << axis_names[d] << "=" << ix[d];

	cout << ": " << name1 << "=" << x << ", " << name2
	     << "=" << y << "  [delta=" << delta << "]\n";
    }
    
    if (nfail > max_display)
	cout << "        [ + " << (nfail-max_display) << " failures]\n";

    if (nfail > 0)
	exit(1);
}


}  // namespace gputils
