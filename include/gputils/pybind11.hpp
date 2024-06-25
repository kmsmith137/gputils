#ifndef _GPUTILS_PYBIND11_HPP
#define _GPUTILS_PYBIND11_HPP

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <pybind11/pybind11.h>

// gputils::Array<T>
#include "Array.hpp"

// convert_array_from_python(), convert_array_to_python()
// array_type_name<T>, npy_type_num<T>, dlpack_type_code<T>
#include "pybind11_utils.hpp"


// type_caster<> converters: these must be available at compile time,
// to any pybind11 extension module which uses gputils::Array<T>.

namespace PYBIND11_NAMESPACE { namespace detail {
#if 0
}} // editor
#endif


template<typename T>
struct type_caster<gputils::Array<T>>
{
    // This macro establishes the name 'Array' in in function signatures,
    // and declares a local variable 'value' of type gputils::Array<T>>.

    PYBIND11_TYPE_CASTER(gputils::Array<T>, gputils::array_type_name<T>::value);
    
    // load(): convert python -> C++.
    // FIXME for now, we ignore the 'convert' argument.

    bool load(handle src, bool convert)
    {
	void *data = nullptr;

	// Throws a C++ exception on failure. (I tried a few ways of reporting
	// failure, including calling PyErr_SetString() and returning false,
	// but I liked throwing a C++ exception best.)

	gputils::convert_array_from_python(
	    data,                                 // void *&data
	    this->value.ndim,                     // int &ndim
	    this->value.shape,                    // long *shape
	    this->value.strides,                  // long *strides
	    this->value.size,                     // long &size
	    gputils::dlpack_type_code<T>::value,  // int dlpack_type_code
	    sizeof(T),                            // int itemsize
	    this->value.base,                     // std::shared_ptr<void> &base
	    this->value.aflags,                   // int &aflags
	    src.ptr(),                            // PyObject *src
	    convert                               // bool convert
	);

	this->value.data = reinterpret_cast<T *> (data);
	return true;
    }
    
    // cast(): convert C++ -> python
    // FIXME for now, we ignore the 'policy' and 'parent' args.

    static handle cast(gputils::Array<T> src, return_value_policy policy, handle parent)
    {
	// On failure, gputils::convert_array_to_python() calls PyErr_SetString()
	// and returns NULL. (I tried a few ways of reporting failure, and I liked
	// this way best.)
	
	return gputils::convert_array_to_python(
	    src.data,                         // void *data
	    src.ndim,                         // int ndim
	    src.shape,                        // const long *shape
	    src.strides,                      // const long *strides
	    gputils::npy_type_num<T>::value,  // int npy_typenum
	    sizeof(T),                        // int itemsize
	    src.base,                         // const shared_ptr<void> &base
	    src.aflags,                       // int aflags
	    policy,                           // pybind11::return_value_policy policy
	    parent                            // pybind11::handle parent
	);
    }
};


}} // namespace PYBIND11_NAMESPACE::detail

#endif  // _GPUTILS_PYBIND11_HPP
