#ifndef _GPUTILS_CUDA_UTILS_HPP
#define _GPUTILS_CUDA_UTILS_HPP

#include <vector>
#include <memory>
#include <cassert>
#include <stdexcept>


// Note: CUDA_CALL(), CUDA_PEEK(), and CUDA_CALL_ABORT() are implemented with #define,
// and therefore are outside the gputils namespace.

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------   Macros   ------------------------------------------
//
// CUDA_CALL(f()): wrapper for CUDA API calls which return cudaError_t.
//
// CUDA_PEEK("label"): throws exception if (cudaPeekAtLastError() != cudaSuccess).
// The argument is a string label which appears in the error message.
// Can be used anywhere, but intended to be called immediately after kernel launches.
//
// CUDA_CALL_ABORT(f()): infrequently-used version of CUDA_CALL() which aborts instead
// of throwing an exception (for use in contexts where exception-throwing is not allowed,
// e.g. shared_ptr deleters).
//
// Example:
//
//    CUDA_CALL(cudaMalloc(&ptr, size));
//    mykernel<<B,T>> (ptr);
//    CUDA_PEEK("mykernel launch")


// Branch predictor hint
#ifndef _unlikely
#define _unlikely(cond)  (__builtin_expect(cond,0))
#endif

#define CUDA_CALL(x) _CUDA_CALL(x, __STRING(x), __FILE__, __LINE__)
#define CUDA_PEEK(x) _CUDA_CALL(cudaPeekAtLastError(), x, __FILE__, __LINE__)
#define CUDA_CALL_ABORT(x) _CUDA_CALL_ABORT(x, __STRING(x), __FILE__, __LINE__)

#define _CUDA_CALL(x, xstr, file, line) \
    do { \
	cudaError_t xerr = (x); \
	if (_unlikely(xerr != cudaSuccess)) \
	    throw ::gputils::make_cuda_exception(xerr, xstr, file, line); \
    } while (0)

#define _CUDA_CALL_ABORT(x, xstr, file, line) \
    do { \
	cudaError_t xerr = (x); \
	if (_unlikely(xerr != cudaSuccess)) { \
	    fprintf(stderr, "CUDA call '%s' failed at %s:%d\n", xstr, file, line); \
	    exit(1); \
	} \
    } while (0)

// Helper for CUDA_CALL().
std::runtime_error make_cuda_exception(cudaError_t xerr, const char *xstr, const char *file, int line);


// ------------------------------  RAII wrapper for cudaStream_t  ----------------------------------


struct CudaStreamWrapper {
protected:
    // Reminder: cudaStream_t is a typedef for (CUstream_st *)
    std::shared_ptr<CUstream_st> p;

public:
    CudaStreamWrapper()
    {
	cudaStream_t s;
	CUDA_CALL(cudaStreamCreate(&s));
	this->p = std::shared_ptr<CUstream_st> (s, cudaStreamDestroy);
    }

    // A CudaStreamWrapper can be used anywhere a cudaStream_t can be used
    // (e.g. in a kernel launch, or elsewhere in the CUDA API), via this
    // conversion operator.
    
    operator cudaStream_t() { return p.get(); }
};


} // namespace gputils


#endif // _GPUTILS_CUDA_UTILS_HPP
