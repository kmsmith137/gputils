#include <cmath>
#include <iostream>

#include "../include/gputils/cuda_utils.hpp"

using namespace std;


static void show_device(int device)
{
    cout << "Device " << device << endl;

    // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));

    cout << "    name = " << prop.name << "\n"
	 << "    compute capability = " << prop.major << "." << prop.minor << "\n"
	 << "    multiProcessorCount = " << prop.multiProcessorCount << "\n"
	 << "    clockRate = " << prop.clockRate << " kHZ  = " << (prop.clockRate / 1.0e6) << " GHz [deprecated]\n"
	 << "    l2CacheSize = " << prop.l2CacheSize << " bytes = " << (prop.l2CacheSize / pow(2,20.)) << " MB\n"
	 << "    totalGlobalMem = " << prop.totalGlobalMem << " bytes = " << (prop.totalGlobalMem / pow(2,30.)) << " GB\n"
	 << "    memoryClockRate = " << prop.memoryClockRate << " kHZ [deprecated]\n"
	 << "    memoryBusWidth = " << prop.memoryBusWidth << " bits\n"
	 << "    implied global memory bandwidth = " << (prop.memoryClockRate * double(prop.memoryBusWidth) / 1.0e6 / 4.) << " GB/s\n"   // empirical!
	 << endl;
}


int main(int argc, char **argv)
{
    int ndevices = -1;
    CUDA_CALL(cudaGetDeviceCount(&ndevices));

    cout << "Number of devices: " << ndevices << endl;

    for (int device = 0; device < ndevices; device++)
	show_device(device);
    
    return 0;
}
