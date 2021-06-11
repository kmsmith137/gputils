#ifndef _GPUTILS_RAND_UTILS_HPP
#define _GPUTILS_RAND_UTILS_HPP

#include <vector>
#include <random>
#include <cassert>
#include <type_traits>

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif

extern std::mt19937 default_rng;


// -------------------------------------------------------------------------------------------------


inline ssize_t rand_int(ssize_t lo, ssize_t hi, std::mt19937 &rng = default_rng)
{
    assert(lo < hi);
    return std::uniform_int_distribution<ssize_t>(lo,hi-1)(rng);   // note hi-1 here!
}


inline float rand_uniform(float lo=0.0, float hi=1.0, std::mt19937 &rng = default_rng)
{
    return std::uniform_real_distribution<float>(lo,hi) (rng);
}


// -------------------------------------------------------------------------------------------------


// Version of randomize() for floating-point types
template<typename T>
inline void randomize_f(T *buf, ssize_t nelts, std::mt19937 &rng = default_rng)
{
    static_assert(std::is_floating_point_v<T>);

    auto dist = std::uniform_real_distribution<T>(-1.0, 1.0);    
    for (ssize_t i = 0; i < nelts; i++)
	buf[i] = dist(rng);
}


// Version of randomize() for integral types
template<typename T>
inline void randomize_i(T *buf, ssize_t nelts, std::mt19937 &rng = default_rng)
{
    static_assert(std::is_integral_v<T>);

    ssize_t nbytes = nelts * sizeof(T);
    ssize_t nints = nbytes / sizeof(int);
    
    for (int i = 0; i < nints; i++)
	((int *)buf)[i] = rng();
    for (int i = nints * sizeof(int); i < nbytes; i++)
	((char *)buf)[i] = rng();
}


// General randomize()
template<typename T>
inline void randomize(T *buf, ssize_t nelts, std::mt19937 &rng = default_rng)
{
    assert(nelts >= 0);
    
    if constexpr (std::is_floating_point_v<T>)
	randomize_f(buf, nelts, rng);
    else {
	static_assert(std::is_integral_v<T>, "randomize() array must be either integral or floating-point type");
	randomize_i(buf, nelts, rng);
    }
}


// -------------------------------------------------------------------------------------------------


template<typename T>
inline void randomly_permute(std::vector<T> &v, std::mt19937 &rng = default_rng)
{
    for (size_t i = 1; i < v.size(); i++) {
	size_t j = rand_int(0, i+1, rng);
	std::swap(v[i], v[j]);
    }
}


// Returns a random permutation of {0,1,...,(n-1)}
inline std::vector<ssize_t> rand_permutation(ssize_t nelts, std::mt19937 &rng = default_rng)
{
    assert(nelts >= 0);
    
    std::vector<ssize_t> v(nelts);
    for (ssize_t i = 0; i < nelts; i++)
	v[i] = i;

    randomly_permute(v, rng);
    return v;
}



} // namespace gputils

#endif // _GPUTILS_RAND_UTILS_HPP
