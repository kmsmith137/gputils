#ifndef _GPUTILS_MEM_UTILS_HPP
#define _GPUTILS_MEM_UTILS_HPP

#include <string>
#include <memory>
#include <type_traits>
#include "rand_utils.hpp"   // randomize()

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Core functions, defined later in this file ("Implementation" below)


// See below for a complete list of flags.
// Note that default flags=0 allocates uninitialized memory on host.
template<typename T>
inline std::shared_ptr<T> af_alloc(ssize_t nelts, int flags=0);


template<typename T>
inline void af_copy(T *dst, int dst_flags, const T *src, int src_flags, ssize_t nelts);


template<typename T>
inline std::shared_ptr<T> af_clone(int dst_flags, const T *src, int src_flags, ssize_t nelts);


// -------------------------------------------------------------------------------------------------
//
// Flags for use in af_alloc().
// Note: anticipate refining 'af_gpu', to allow multiple devices.
// Note: anticipate refining 'af_unified', to toggle cudaMemAttachHost vs cudaMemAttachGlobal.
// Note: anticipate refining 'af_page_locked' (e.g. cudaHostAllocWriteCombined).


static constexpr int af_gpu = 0x01;                 // allocate on gpu
static constexpr int af_zero = 0x10;                // zero allocated memory
static constexpr int af_guard = 0x20;               // detect buffer overruns when freed (has overhead)
static constexpr int af_random = 0x40;              // randomize allocated memory
static constexpr int af_verbose = 0x080;            // prints verbose messages
static constexpr int af_unified = 0x100;            // allocate unified (host+device) memory
static constexpr int af_page_locked = 0x1000;       // allocate page-locked host memory
static constexpr int af_uninitialized = 0x1000000;  // to catch uninitialized aflags (e.g. in Array)

static constexpr int af_location_flags = af_gpu | af_unified | af_page_locked;
static constexpr int af_initialization_flags = af_zero | af_random;
static constexpr int af_debug_flags = af_guard | af_verbose;
static constexpr int af_all_flags = af_location_flags | af_initialization_flags | af_debug_flags | af_uninitialized;

// Throws exception if aflags are uninitialized or invalid.
extern void check_aflags(int aflags, const char *where = nullptr);

// Utility function for printing flags.
extern std::string aflag_str(int flags);

// Is memory addressable on GPU? On host?
inline bool af_on_gpu(int flags) { return (flags & (af_gpu | af_unified)) != 0; }
inline bool af_on_host(int flags) { return (flags & af_gpu) == 0; }


// -------------------------------------------------------------------------------------------------
//
// Implementation.


// Handles all flags except 'af_random'.
extern std::shared_ptr<void> _af_alloc(ssize_t nbytes, int flags);

// Uses location flags, but ignores initialization and debug flags.
extern void _af_copy(void *dst, int dst_flags, const void *src, int src_flags, ssize_t nbytes);


template<typename T>
inline std::shared_ptr<T> af_alloc(ssize_t nelts, int flags)
{
    // FIXME should have some static_asserts here, to ensure
    // that 'T' doesn't have constructors/destructors.

    assert(nelts >= 0);
    ssize_t nbytes = nelts * sizeof(T);

    // _af_alloc() handles all flags except 'af_random'.
    std::shared_ptr<T> ret = std::reinterpret_pointer_cast<T> (_af_alloc(nbytes, flags));

    if (!(flags & af_random))
	return ret;

    // FIXME should use "if constexpr" for more graceful handling
    // of non-integral and non-floating-point types.
        
    if (!(flags & af_gpu)) {
	randomize(ret.get(), nelts);
	return ret;
    }

    // FIXME slow, memory-intensive way of randomizing array on GPU, by randomizing on CPU
    // and copying. It would be better to launch a kernel to randomize directly on GPU.

    std::shared_ptr<T> host = std::reinterpret_pointer_cast<T> (_af_alloc(nbytes, 0));
    randomize(host.get(), nelts);
    _af_copy(ret.get(), flags, host.get(), 0, nbytes);
    return ret;
}


template<typename T>
inline void af_copy(T *dst, int dst_flags, const T *src, int src_flags, ssize_t nelts)
{
    // FIXME should have some static_asserts here, to ensure
    // that 'T' doesn't have constructors/destructors.

    assert(nelts >= 0);
    ssize_t nbytes = nelts * sizeof(T);
    
    _af_copy(dst, dst_flags, src, src_flags, nbytes);
}


template<typename T>
inline std::shared_ptr<T> af_clone(int dst_flags, const T *src, int src_flags, ssize_t nelts)
{
    dst_flags &= ~af_initialization_flags;
    std::shared_ptr<T> ret = af_alloc<T> (nelts, dst_flags);
    af_copy(ret.get(), dst_flags, src, src_flags, nelts);
}


} // namespace gputils

#endif  // _GPUTILS_MEM_UTILS_HPP
