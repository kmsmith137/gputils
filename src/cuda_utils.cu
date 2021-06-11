#include <random>
#include <cstring>
#include <cassert>
#include <sstream>
#include <iostream>

#include "../include/gputils/cuda_utils.hpp"

using namespace std;


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


runtime_error make_cuda_exception(cudaError_t xerr, const char *xstr, const char *file, int line)
{
    stringstream ss;
    
    ss << "CUDA error: " << xstr << " returned " << xerr
       << " (" << cudaGetErrorString(xerr) << ")"
       << " [" << file << ":" << line << "]";

    return runtime_error(ss.str());
}


} // namespace gputils
