#ifndef _GPUTILS_HPP
#define _GPUTILS_HPP


// Array class
#include "gputils/Array.hpp"

// CudaStreamPool: run multiple streams with dynamic load-balancing, intended for timing
#include "gputils/CudaStreamPool.hpp"

// constexpr_is_pow2(), constexpr_ilog2()
#include "gputils/constexpr_functions.hpp"

// CUDA_CALL(), CUDA_PEEK(), CudaStreamWrapper
#include "gputils/cuda_utils.hpp"

// af_alloc(), af_copy(), af_clone()
#include "gputils/mem_utils.hpp"

// rand_int(), rand_uniform(), randomize()
#include "gputils/rand_utils.hpp"

// to_str(), from_str(), tuple_str(), type_name()
#include "gputils/string_utils.hpp"

// assert_arrays_equal()
#include "gputils/test_utils.hpp"

// get_time(), time_diff(), time_since()
#include "gputils/time_utils.hpp"


#endif // _GPUTILS_HPP
