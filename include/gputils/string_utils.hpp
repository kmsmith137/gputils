#ifndef _GPUTILS_STRING_UTILS_HPP
#define _GPUTILS_STRING_UTILS_HPP

#include <string>
#include <vector>
#include <sstream>
#include <typeinfo>
#include <stdexcept>

#ifdef __GNUG__
#include <cxxabi.h>
#endif

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// -----------------------------------  to_str(), from_str()  --------------------------------------


template<typename T>
static std::string to_str(const T &x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}


template<typename T>
static T from_str(const std::string &s)
{
    std::stringstream ss;
    ss << s;
        
    T ret = 0;
    ss >> ret;
    int f1 = ss.fail();
    
    std::string t;
    ss >> t;
    int f2 = ss.fail();

    if (f1 || !f2) {
	std::stringstream err;
	err << "couldn't convert string \"" << s << "\" to type " << type_name<T>();
	throw std::runtime_error(err.str());
    }

    return ret;
}


// ---------------------------------------   tuple_str()   -----------------------------------------


// Returns a formatted tuple, e.g. "(1,2,3)"
template<typename T>
static std::string tuple_str(int ndim, const T *tuple)
{
    if (ndim == 0)
	return "()";
	
    std::stringstream ss;
    ss << "(" << tuple[0];

    if (ndim == 1) {
	ss << ",)";
	return ss.str();
    }

    for (int d = 1; d < ndim; d++)
	ss << "," << tuple[d];

    ss << ")";
    return ss.str();
}


template<typename T>
static std::string tuple_str(const std::vector<T> &tuple)
{
    return tuple_str(tuple.size(), &tuple[0]);
}


// ---------------------------------------   type_name()   -----------------------------------------


// Returns a string typename, e.g. type_name<int>() -> "int"
template<typename T>
static std::string type_name()
{
    const char *s = typeid(T).name();

#ifdef __GNUG__
    // Reference: https://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
    int status = -1;
    char *t = abi::__cxa_demangle(s, nullptr, nullptr, &status);
    std::string ret((t && !status) ? t : s);
    free(t);
    return ret;
#else
    return std::string(s);
#endif
}


} // namespace gputils

#endif  // _GPUTILS_STRING_UTILS_HPP
