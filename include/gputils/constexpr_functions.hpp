#ifndef _GPUTILS_CONSTEXPR_FUNCTIONS_HPP
#define _GPUTILS_CONSTEXPR_FUNCTIONS_HPP


constexpr __host__ __device__ bool constexpr_is_divisible(int m, int n)
{
    return (m >= 0) && (n > 0) && ((m % n) == 0);
}

// Frequently used in static_assert().
constexpr __host__ __device__ bool constexpr_is_pow2(int n)
{
    return (n >= 1) && ((n & (n-1)) == 0);
}


constexpr __host__ __device__ int constexpr_ilog2(int n)
{
    // static_assert() is not allowed in constexpr-functions, so
    // caller should call static_assert(constexpr_is_pow2(n));
    
    return (n > 1) ? (constexpr_ilog2(n/2)+1) : 0;
}

#endif // _GPUTILS_CONSTEXPR_FUNCTIONS_HPP
