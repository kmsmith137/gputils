// For an explanation of NO_IMPORT_ARRAY + PY_ARRAY_UNIQUE_SYMBOL, see comments in gputils_pybind11.cu.
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_gputils

#include <complex>
#include <iostream>

#include "../include/gputils/Array.hpp"
#include "../include/gputils/string_utils.hpp"  // tuple_str()
#include "../include/gputils/pybind11_utils.hpp"

using namespace std;

namespace gputils {
#if 0
}  // editor auto-indent
#endif


static string py_str(PyObject *x)
{
    // FIXME using pybind11 as a clutch here.
    // I'd prefer to use the python C-api directly, to guarantee that no exception is thrown.
    return string(pybind11::str(x));
}

static string py_type_str(PyObject *x)
{
    PyObject *t = (PyObject *) Py_TYPE(x);
    
    if (!t) {
	PyErr_Clear();
	return "unknown";
    }

    return py_str(t);
}


// -------------------------------------------------------------------------------------------------


static const char *dl_type_code_to_str(int code)
{
    switch (code) {
    case kDLInt:
	return "int";
    case kDLUInt:
	return "uint";
    case kDLFloat:
	return "float";
    case kDLOpaqueHandle:
	return "opaque";
    case kDLBfloat:
	return "bfloat";
    case kDLComplex:
	return "complex";
    case kDLBool:
	return "bool";
    default:
	return "unrecognized";
    }
}


static string dl_type_to_str(DLDataType d)
{
    stringstream ss;
    ss << dl_type_code_to_str(d.code);

    if (d.code == kDLComplex) {
	int n2 = d.bits >> 1;
	ss << n2 << "+" << int(d.bits - n2);
    }
    else if ((d.code != kDLBool) || (d.bits != 1))
	ss << int(d.bits);

    if (d.lanes != 1)
	ss << "x" << int(d.lanes);

    return ss.str();

}


static const char *dl_device_type_to_str(DLDeviceType d)
{
    // Reference: https://dmlc.github.io/dlpack/latest/c_api.html#c.DLDeviceType
    
    switch (d) {
    case kDLCPU:
	return "kDLCPU";
    case kDLCUDA:
	return "kDLCUDA";
    case kDLCUDAHost:
	return "kDLCUDAHost";
    case kDLOpenCL:
	return "kDLOpenCL";
    case kDLVulkan:
	return "DLVulkan";
    case kDLMetal:
	return "kDLMetal";
    case kDLVPI:
	return "kDLVPI";
    case kDLROCM:
	return "kDLROCM";
    case kDLROCMHost:
	return "kDLROCMHost";
    case kDLExtDev:
	return "kDLExtDev";
    case kDLCUDAManaged:
	return "kDLCUDAManaged";
    case kDLOneAPI:
	return "kDLOneAPI";
    case kDLWebGPU:
	return "kDLWebGPU";
    case kDLHexagon:
	return "kDLHexagon";
    case kDLMAIA:
	return "kDLMAIA";
    default:
	return "unrecognized";
    }
}


// Returns 0 on failure.
static int device_type_to_aflags(DLDeviceType d)
{
    switch (d) {
    case kDLCPU:
	return af_uhost;
	
    case kDLCUDAHost:  // Pinned CUDA CPU memory by cudaMallocHost
    case kDLROCMHost:  // Pinned ROCm CPU memory allocated by hipMallocHost
	return af_rhost;
	
    case kDLCUDA:
    // case kDLROCM:
	return af_gpu;

    case kDLCUDAManaged:
	return af_unified;

    default:
	return 0;
    }
}


// -------------------------------------------------------------------------------------------------
//
// Array conversion part 1: python -> C++
// convert_array_from_python() throws a C++ exception on failure.
//
// FIXME this interface could be streamlined by defining Array<void>.
//
// FIXME: at some point I should try to implement "base compression", here and/or in the
// python -> C++ conversion.
//
// FIXME implement 'convert' argument.
//
// FIXME we currently use __dlpack__ for all array conversions. If the array is a numpy array,
// then the conversion can be done more efficiently by calling functions in the numpy C-API.
//
// If the 'debug_prefix' argument is non-NULL, then some debug info will be printed to stdout.
// This feature is wrapped by gputils.convert_array_from_python(). It is intended as a mechanism
// for tracing/debugging array conversion.

__attribute__ ((visibility ("default")))
void convert_array_from_python(
    void *&data, int &ndim, ssize_t *shape, ssize_t *strides, ssize_t &size,
    int dlpack_type_code, int itemsize, std::shared_ptr<void> &base, int &aflags,
    PyObject *src, bool convert, const char *debug_prefix)
{
    DLManagedTensor *mt = nullptr;    
    pybind11::object capsule;   // must hold reference for entire function
    
    if (debug_prefix != nullptr)
	cout << debug_prefix << ": testing for presence of __dlpack__ attr\n";
    
    PyObject *dlp = PyObject_GetAttrString(src, "__dlpack__");
    
    if (dlp) {
	if (debug_prefix != nullptr)
	    cout << debug_prefix << ": __dlpack__ attr found, now calling it with no arguments\n";

	// FIXME the dlpack documentation specifies that __dlpack__() should be called
	// with a keyword argument 'max_version'. However, this didn't seem to be implemented
	// in numpy/cupy (in June 2024). For now, we call __dlpack__() with no args, but
	// I might revisit this in the future.

	PyObject *rp = PyObject_CallNoArgs(dlp);
	capsule = pybind11::reinterpret_steal<pybind11::object> (rp);
	Py_DECREF(dlp);
    }

    if (capsule.ptr()) {
	if (debug_prefix != nullptr)
	    cout << debug_prefix << ": __dlpack__() returned, now testing whether return value is a capsule with name \"dltensor\"\n";
    
	// Okay if this pointer is NULL.
	mt = reinterpret_cast<DLManagedTensor *> (PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    
	if (debug_prefix && mt)
	    cout << debug_prefix << ": successfully extracted pointer from capsule" << endl;
    }
    
    
    PyErr_Clear();  // no-ops if PyErr is not set
    
    if (!mt) {
	stringstream ss;
	bool vflag = capsule.ptr() && PyCapsule_GetPointer(capsule.ptr(), "dltensor_versioned");
	PyErr_Clear();

	if (vflag) {
	    ss << "gputils::convert_array_from_python() received 'dltensor_versioned' object."
	       << " This is a planned dlpack feature which isn't implemented yet (in June 2024) in numpy/cupy."
	       << " Unfortunately some (minor) code changes will be needed in gputils to support it!";
	}
	else {
	    ss << "Couldn't convert python argument(s) to a C++ array."
	       << " You might need to wrap the argument in numpy.asarray(...) or cupy.asarray(...).";
	}
	
	ss << " The offending argument is: " << py_str(src)
	   << " and its type is " << py_type_str(src) << ".";

	throw pybind11::type_error(ss.str());
    }
    
    DLTensor &t = mt->dl_tensor;

    if (debug_prefix != nullptr) {
	cout << debug_prefix << ": dereferencing DLManagedTensor\n"
	     << "   data: " << t.data << "\n"
	     << "   device_type: " << t.device.device_type
	     << " (" << dl_device_type_to_str(t.device.device_type) << ")\n"
	     << "   device_id: " << t.device.device_id << "\n"
	     << "   ndim: " << t.ndim << "\n"
	     << "   dtype_code: " << int(t.dtype.code)
	     << " (" << dl_type_code_to_str(t.dtype.code) << ")\n"
	     << "   dtype_bits: " << int(t.dtype.bits) << "\n"
	     << "   dtype_lanes: " << t.dtype.lanes << "\n"
	     << "   byte_offset: " << t.byte_offset << "\n";
	
	cout << debug_prefix << ": dereferencing shape" << endl;
	cout << "   shape: " << gputils::tuple_str(t.ndim, t.shape, " ") << "\n";
	
	if (t.strides == nullptr)
	    cout << debug_prefix << ": strides pointer is null" << endl;
	else {
	    cout << debug_prefix << ": dereferencing strides" << endl;
	    cout << "   strides: " << gputils::tuple_str(t.ndim, t.strides, " ") << "\n";
	}
    }
    
    ndim = t.ndim;
    data = (char *)t.data + t.byte_offset;
    aflags = device_type_to_aflags(t.device.device_type);
    
    if (aflags == 0) {
	stringstream ss;
	ss << "Couldn't convert python argument to a C++ array."
	   << " The python argument returned DLDeviceType=" << t.device.device_type
	   << " [" << dl_device_type_to_str(t.device.device_type) << "],"
	   << " which we don't currently support."
	   << " The offending argument is: " << py_str(src)
	   << " and its type is " << py_type_str(src) << ".";
	
	throw pybind11::type_error(ss.str());
    }

#if 0
    if (t.device.device_id != 0) {
	stringstream ss;
	ss << "Couldn't convert python argument to a C++ array."
	   << " The python argument returned device_id=" << t.device.device_id
	   << " and we only support device_id=0. (Multi-GPU support is coming soon!)"
	   << " The offending argument is: " << py_str(src)
	   << " its DLDeviceType is " << t.device.device_type << ","
	   << " [" << dl_device_type_to_str(t.device.device_type) << "],"	    
	   << " and its type is " << py_type_str(src) << ".";
	
	throw pybind11::type_error(ss.str());
    }
#endif
    
    if (ndim > gputils::ArrayMaxDim) {
	stringstream ss;
	ss << "Couldn't convert python argument to a C++ array."
	   << " The python argument is an array of dimension " << ndim
	   << ", and gputils::ArrayMaxDim=" << gputils::ArrayMaxDim;

	throw pybind11::type_error(ss.str());
    }

    if ((t.dtype.code != dlpack_type_code) || (t.dtype.bits != (8 * itemsize)) || (t.dtype.lanes != 1)) {
	DLDataType dsrc;
	dsrc.code = dlpack_type_code;
	dsrc.bits = 8 * itemsize;
	dsrc.lanes = 1;
	
	stringstream ss;
	ss << "Couldn't convert python argument to a C++ array: type mismatch."
	   << " The python argument has dtype " << dl_type_to_str(t.dtype) << ","
	   << " and the C++ code expects dtype " << dl_type_to_str(dsrc) << "."
	   << " The offending argument is: " << py_str(src)
	   << " and its type is " << py_type_str(src) << ".";

	throw pybind11::type_error(ss.str());
    }

    size = 1;
    for (int i = ndim-1; i >= 0; i--) {
	// Note: if t.strides==NULL, then array is contiguous.
	shape[i] = t.shape[i];
	strides[i] = t.strides ? t.strides[i] : size;
	size *= shape[i];
    }

    // C++ array holds reference to python object!
    // FIXME could be improved by pointer-chasing to base object.
    base = shared_ptr<void> (src, Py_DecRef);
    Py_INCREF(src);

    // FIXME I think this calls assert() if it fails.
    gputils::check_array_invariants(data, ndim, shape, size, strides, aflags);

    if (debug_prefix != nullptr)
	cout << debug_prefix << ": array converted succesfully" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Array conversion part 2: C++ -> python
// On failure, convert_array_to_python() returns NULL and sets PyErr.
//
// FIXME this interface could be streamlined by defining Array<void>.
//
// FIXME: at some point I should try to implement "base compression", here and/or in the
// python -> C++ conversion.
//
// FIXME implement return value policies (and the 'parent' argument).
//   https://pybind11.readthedocs.io/en/stable/advanced/functions.html#return-value-policies
//   https://github.com/pybind/pybind11/blob/master/include/pybind11/detail/common.h

__attribute__ ((visibility ("default")))
PyObject *convert_array_to_python(
    void *data, int ndim, const ssize_t *shape, const ssize_t *strides,
    int type_num, int itemsize, const shared_ptr<void> &base, int aflags,
    pybind11::return_value_policy policy, pybind11::handle parent)
{
    // FIXME!!
    if (!af_on_host(aflags)) {
	PyErr_SetString(PyExc_TypeError,
			"Currently C++ -> python array conversion"
			" is not implemented for GPU arrays");
	return NULL;
    }
	
    npy_intp npy_shape[gputils::ArrayMaxDim];
    for (int i = 0; i < ndim; i++)
	npy_shape[i] = shape[i];
	
    npy_intp npy_strides[gputils::ArrayMaxDim];
    for (int i = 0; i < ndim; i++)
	npy_strides[i] = strides[i] * itemsize;
    
    // Array creation: https://numpy.org/doc/stable/reference/c-api/array.html#creating-arrays
    // Array flags: https://numpy.org/doc/stable/reference/c-api/array.html#array-flags
    // PyArray_New() is defined in this source file: src/multiarray/ctors.c
    // Flags are defined in this source file: include/numpy/ndarraytypes.h
    
    PyObject *ret = PyArray_New(
	&PyArray_Type,   // PyTypeObject *subtype
	ndim,            // int nd
	npy_shape,       // npy_intp const *dims
	type_num,        // int type_num
	npy_strides,     // npy_intp const *strides
	data,            // void *data
	itemsize,        // int itemsize
        NPY_ARRAY_WRITEABLE,   // int flags
	NULL             // PyObject *obj (extra constructor arg, only used if subtype != &PyArrayType)
    );
    
    if (!ret)
	return NULL;

    // Paranoid!
    int flags = PyArray_FLAGS((PyArrayObject *) ret);
    assert((flags & NPY_ARRAY_OWNDATA) == 0);
    
    // We need a mechanism for keeping a reference to 'base' (a shared_ptr<void>) in the
    // newly constructed numpy array.
    //
    // For now we use a hack (class PybindBasePtr).
    // FIXME defining a PyArray subclass would be cleaner and more efficient.
    //
    // The PybindBasePtr class is a trivial wrapper around shared_ptr<void>, but the
    // wrapper class is exported to python (via a pybind11::class_<> in the top-level
    // gputils extension module). This allows us to set the numpy 'base' member (which
    // must be a python object) to a PybindBasePtr instance.
    //
    // One nuisance issue: here in the C++ code, we need to convert a PybindBasePtr
    // instance to a (PyObject *). Surprisingly, pybind11::cast() doesn't work here,
    // and we need the following mysterious code.
    // (This reference was useful: https://github.com/pybind/pybind11/issues/1176)
    
    PybindBasePtr p(base);
    using caster = pybind11::detail::type_caster_base<PybindBasePtr>;
    pybind11::handle base_ptr = caster::cast(p, pybind11::return_value_policy::copy, pybind11::handle()); 
    
    // PyArray_SetBaseObject() checks whether 'base_ptr' is NULL.
    // PyArray_SetBaseObject() steals the reference to 'base' (even on error).
    // PyArray_SetBaseObject() is defined in this file: numpy/_core/src/multiarray/arrayobject.c
    
    int err = PyArray_SetBaseObject((PyArrayObject *) ret, base_ptr.ptr());

    if (err < 0) {
	Py_XDECREF(ret);
	return NULL;
    }

    return ret;
}


}  // namespace gputils
