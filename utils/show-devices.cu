#include <iostream>
#include <cassert>

using namespace std;


static void show_device(int device)
{
    cout << "Device " << device << endl;

    // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    assert(err == cudaSuccess);

    cout << "    major = " << prop.major << endl;
    cout << "    minor = " << prop.minor << endl;
    cout << "    multiProcessorCount = " << prop.multiProcessorCount << endl;
    cout << "    clockRate = " << prop.clockRate << endl;
    cout << "    l2CacheSize = " << prop.l2CacheSize << endl;
}


int main(int argc, char **argv)
{
    int ndevices = -1;
    cudaError_t err = cudaGetDeviceCount(&ndevices);
    assert(err == cudaSuccess);

    cout << "Number of devices:" << ndevices << endl;

    for (int device = 0; device < ndevices; device++)
	show_device(device);
    
    return 0;
}
