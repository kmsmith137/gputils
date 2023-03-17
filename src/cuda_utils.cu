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


double get_sm_cycles_per_second(int device)
{
    // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));

    // prop.clockRate is in kHz
    return 1.0e3 * double(prop.multiProcessorCount) * double(prop.clockRate);
}


} // namespace gputils
