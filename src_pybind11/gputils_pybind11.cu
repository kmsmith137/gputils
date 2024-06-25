// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments later in this source filE.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_gputils

#include "../include/gputils/pybind11.hpp"
#include "../include/gputils/cuda_utils.hpp"
#include "../include/gputils/test_utils.hpp"
#include <iostream>


using namespace std;
using namespace gputils;

namespace py = pybind11;


// -------------------------------------------------------------------------------------------------


static double _sum(Array<double> &arr)
{
    if (!arr.on_host()) {
	cout << "sum() copying array from GPU to host" << endl;
	arr = arr.to_host();
    }
    
    double sum = 0;
    for (auto ix = arr.ix_start(); arr.ix_valid(ix); arr.ix_next(ix))
	sum += arr.at(ix);
    
    return sum;
}


static void _double(Array<double> &arr)
{
    Array<double> asave;
    bool copy = !arr.on_host();

    if (copy) {
	cout << "double() copying array from GPU to host" << endl;
	asave = arr;
	arr = arr.to_host();
    }
    
    for (auto ix = arr.ix_start(); arr.ix_valid(ix); arr.ix_next(ix))
	arr.at({ix}) *= 2;

    if (copy) {
	cout << "double() copying array from host to GPU" << endl;
	asave.fill(arr);
    }
}    


static Array<double> _arange(long n)
{
    // Note af_verbose here.
    Array<double> ret({n}, af_uhost | af_verbose);
    
    for (long i = 0; i < n; i++)
	ret.at({i}) = i;
    
    return ret;
}


static void _convert_array_from_python(py::object &obj)
{
    // FIXME eventually, this function should work for arbitrary types.
    using T = double;
    
    void *data = nullptr;
    Array<T> value;

    gputils::convert_array_from_python(
        data,                                 // void *&data
	value.ndim,                           // int &ndim
	value.shape,                          // long *shape
	value.strides,                        // long *strides
	value.size,                           // long &size
	gputils::dlpack_type_code<T>::value,  // int dlpack_type_code
	sizeof(T),                            // int itemsize
	value.base,                           // std::shared_ptr<void> &base
	value.aflags,                         // int &aflags
	obj.ptr(),                            // PyObject *src
	false,                                // bool convert
	"convert_array_from_python"           // const char *debug_prefix
    );
}


int get_cuda_device()
{
    int device = -1;
    CUDA_CALL(cudaGetDevice(&device));
    return device;
}


void _launch_busy_wait_kernel(Array<uint> &arr, double a40_sec, long stream_ptr)
{
    cudaStream_t s = reinterpret_cast<cudaStream_t> (stream_ptr);
    gputils::launch_busy_wait_kernel(arr, a40_sec, s);
}


struct Stash
{
    Array<double> x;
    
    Stash(const Array<double> &x_) : x(x_) { }
    ~Stash() { cout << "Stash destructor called!" << endl; }
    
    Array<double> get() { return x; }
    void clear() { x = Array<double> (); }
};


PYBIND11_MODULE(gputils_pybind11, m)  // extension module gets compiled to gputils_pybind11.so
{
    m.doc() = "gputils: a library of low-level utilities for cuda/cupy.";

    // Here is a quick summary of what you should do for import_array():
    //
    //   - In the main pybind11 source file (i.e. this file), before including
    //     any source files, put:
    //
    //       #define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_{modname}
    //
    //     where "modname" is e.g. 'gputils' (a per-extension-module string)
    //
    //   - If there are any other source files in the python extension module
    //     (e.g. gputils/src_pybind11/gputils_pybind11_utils.cu), then those
    //     files should contain (before including any source files):
    //
    //       #define NO_IMPORT_ARRAY
    //       #define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_{modname}
    // 
    // To understand the reasoning behind this, it's easiest to read the source
    // file core/include/numpy/__multiarray_api.h (search for PyArray_API). However,
    // note that this file is procedurally generated during the numpy build process,
    // so it will be in your python site-packages, but not in the numpy source code
    // in the github repo.
    //
    // Note: looks like _import_array() will fail if different numpy versions are
    // found at compile-time versus runtime.

    if (_import_array() < 0) {
	PyErr_Print();
	PyErr_SetString(PyExc_ImportError, "gputils: numpy.core.multiarray failed to import");
	return;
    }

    // ------------------------------  Classes  ------------------------------
    
    const char *baseptr_doc =
	"This helper class is a hack, used interally for converting C++ arrays to python\n"
	"It can't be instantiated or used from python.";

    const char *stash_doc =
	"Helper class intended for testing C++ <-> python array conversion.\n"
	"   s = Stash(numpy_or_cupy_array)     # converts array to C++ and saves it\n"
	"   arr = Stash.get()                  # converts array to python and returns it";

    py::class_<PybindBasePtr>(m, "BasePtr", baseptr_doc);

    py::class_<Stash>(m, "Stash", stash_doc)
	.def(py::init<const Array<double> &>(), py::arg("arr"))
	.def("get", &Stash::get)
        .def("clear", &Stash::clear)
    ;

    // ------------------------------  Functions  ------------------------------
    
    m.def("sum", &_sum,	  
	  "Equivalent to {numpy,cupy}.sum(arr), but uses the python -> C++ converter",
	  py::arg("arr"));
	  
    m.def("double", &_double,
	  "Equivalent to 'arr *= 2', but uses the python -> C++ converter",
	  py::arg("arr"));

    m.def("arange", &_arange,
	  "Equivalent to numpy.arange(n), but uses the C++ -> python converter."
	  " (FIXME CPU-only for now.)",
	  py::arg("n"));

    m.def("convert_array_from_python", &_convert_array_from_python,
	  "Converts Array<double> from python to C++, but with a lot of debug output."
	  " This is intended as a mechanism for tracing/debugging array conversion.",
	  py::arg("arr"));

    m.def("get_cuda_device", &get_cuda_device, "Wraps cudaGetDevice()");

    m.def("_launch_busy_wait_kernel", &_launch_busy_wait_kernel,
	  py::arg("arr"), py::arg("a40_sec"), py::arg("stream_ptr"));
}
