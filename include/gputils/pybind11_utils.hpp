#ifndef _GPUTILS_PYBIND11_UTILS_HPP
#define _GPUTILS_PYBIND11_UTILS_HPP

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/pybind11.h>
#include "dlpack.h"


namespace gputils {
#if 0
}  // editor
#endif


// Convert python -> C++.
// On failure, throws a C++ exception.
//
// If the 'debug_prefix' argument is specified, then some debug info will be printed to stdout.
// This feature is wrapped by gputils.convert_array_from_python(). It is intended as a mechanism
// for tracing/debugging array conversion.

extern void convert_array_from_python(
    void *&data, int &ndim, long *shape, long *strides, long &size,
    int dlpack_type_code, int itemsize, std::shared_ptr<void> &base, int &aflags,
    PyObject *src, bool convert, const char *debug_prefix = nullptr);


// Convert C++ -> python.
// On failure, calls PyErr_SetString() and returns NULL.

extern PyObject *convert_array_to_python(
    void *data, int ndim, const long *shape, const long *strides,
    int type_num, int itemsize, const std::shared_ptr<void> &base, int aflags,
    pybind11::return_value_policy policy, pybind11::handle parent);


// PybindBasePtr: hack for array C++ -> python conversion.
// See comments in gputils/src_pybind11/gputils_pybind11_utils.cu
// Must be visible at compile time in gputils/src_pybind11/gputils_pybind11.cu

struct PybindBasePtr
{
    std::shared_ptr<void> p;
    PybindBasePtr(const std::shared_ptr<void> &p_) : p(p_) { }
};


// I'm pretty sure CUDA guarantees these type sizes, but can't hurt to check with a static_assert().
static_assert(sizeof(float) == 4);
static_assert(sizeof(double) == 8);
static_assert(sizeof(std::complex<float>) == 8);
static_assert(sizeof(std::complex<double>) == 16);
static_assert((sizeof(long) == 8) && (sizeof(ulong) == 8));
static_assert((sizeof(int) == 4) && (sizeof(uint) == 4));
static_assert((sizeof(short) == 2) && (sizeof(ushort) == 2));
static_assert((sizeof(char) == 1) && (sizeof(unsigned char) == 1));

// These array type names will appear in docstrings and error messages.
template<typename> struct array_type_name { };
template<> struct array_type_name<float>                   { static constexpr auto value = pybind11::detail::const_name("Array<float32>"); };
template<> struct array_type_name<double>                  { static constexpr auto value = pybind11::detail::const_name("Array<float64>"); };
template<> struct array_type_name<std::complex<float>>     { static constexpr auto value = pybind11::detail::const_name("Array<complex32+32>"); };
template<> struct array_type_name<std::complex<double>>    { static constexpr auto value = pybind11::detail::const_name("Array<complex64+64>"); };
template<> struct array_type_name<long>                    { static constexpr auto value = pybind11::detail::const_name("Array<int64>"); };
template<> struct array_type_name<int>                     { static constexpr auto value = pybind11::detail::const_name("Array<int32>"); };
template<> struct array_type_name<short>                   { static constexpr auto value = pybind11::detail::const_name("Array<int16>"); };
template<> struct array_type_name<char>                    { static constexpr auto value = pybind11::detail::const_name("Array<int8>"); };
template<> struct array_type_name<ulong>                   { static constexpr auto value = pybind11::detail::const_name("Array<uint64>"); };
template<> struct array_type_name<uint>                    { static constexpr auto value = pybind11::detail::const_name("Array<uint32>"); };
template<> struct array_type_name<ushort>                  { static constexpr auto value = pybind11::detail::const_name("Array<uint16>"); };
template<> struct array_type_name<unsigned char>           { static constexpr auto value = pybind11::detail::const_name("Array<uint8>"); };

// Reference: https://numpy.org/doc/stable/reference/c-api/dtype.html
template<typename T> struct npy_type_num   { };
template<> struct npy_type_num<float>                      { static constexpr int value = NPY_FLOAT; };
template<> struct npy_type_num<double>                     { static constexpr int value = NPY_DOUBLE; };
template<> struct npy_type_num<std::complex<float>>        { static constexpr int value = NPY_CFLOAT; };
template<> struct npy_type_num<std::complex<double>>       { static constexpr int value = NPY_CDOUBLE; };
template<> struct npy_type_num<long>                       { static constexpr int value = NPY_LONG; };
template<> struct npy_type_num<int>                        { static constexpr int value = NPY_INT; };
template<> struct npy_type_num<short>                      { static constexpr int value = NPY_SHORT; };
template<> struct npy_type_num<char>                       { static constexpr int value = NPY_BYTE; };
template<> struct npy_type_num<ulong>                      { static constexpr int value = NPY_ULONG; };
template<> struct npy_type_num<uint>                       { static constexpr int value = NPY_UINT; };
template<> struct npy_type_num<ushort>                     { static constexpr int value = NPY_USHORT; };
template<> struct npy_type_num<unsigned char>              { static constexpr int value = NPY_UBYTE; };

// Reference: https://dmlc.github.io/dlpack/latest/c_api.html#c.DLDataTypeCode
template<typename T> struct dlpack_type_code { };
template<> struct dlpack_type_code<float>                  { static constexpr int value = kDLFloat; };
template<> struct dlpack_type_code<double>                 { static constexpr int value = kDLFloat; };
template<> struct dlpack_type_code<std::complex<float>>    { static constexpr int value = kDLComplex; };
template<> struct dlpack_type_code<std::complex<double>>   { static constexpr int value = kDLComplex; };
template<> struct dlpack_type_code<long>                   { static constexpr int value = kDLInt; };
template<> struct dlpack_type_code<int>                    { static constexpr int value = kDLInt; };
template<> struct dlpack_type_code<short>                  { static constexpr int value = kDLInt; };
template<> struct dlpack_type_code<char>                   { static constexpr int value = kDLInt; };
template<> struct dlpack_type_code<ulong>                  { static constexpr int value = kDLUInt; };
template<> struct dlpack_type_code<uint>                   { static constexpr int value = kDLUInt; };
template<> struct dlpack_type_code<ushort>                 { static constexpr int value = kDLUInt; };
template<> struct dlpack_type_code<unsigned char>          { static constexpr int value = kDLUInt; };


}   // namespace gputils


#endif  // _GPUTILS_PYBIND11_UTILS_HPP
