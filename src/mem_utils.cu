#include <sstream>
#include <iostream>
#include "../include/gputils/mem_utils.hpp"
#include "../include/gputils/cuda_utils.hpp"


using namespace std;

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif

    
// Helper for check_aflags().
inline bool multiple_bits(int x)
{
    return (x & (x-1)) != 0;
}

void check_aflags(int flags, const char *where)
{
    if (!where)
	where = "bristlecone::check_aflags()";
	
    if (_unlikely(flags & ~af_all_flags))
	throw runtime_error(string(where) + ": unrecognized flags were specified");
    if (_unlikely(multiple_bits(flags & af_initialization_flags)))
	throw runtime_error(string(where) + ": can specify at most one of " + aflag_str(af_initialization_flags));
    if (_unlikely(multiple_bits(flags & af_location_flags)))
	throw runtime_error(string(where) + ": can specify at most one of " + aflag_str(af_location_flags));
    if (_unlikely(flags & af_uninitialized))
	throw runtime_error(string(where) + ": flags were uninitialized ('af_uninitialized' bit was set)");
}


// Helper for aflag_str().
inline void _aflag_str(stringstream &ss, int &count, bool pred, const char *name)
{
    if (!pred)
	return;
    if (count > 0)
	ss << " | ";
    ss << name;
    count++;
}

string aflag_str(int flags)
{
    stringstream ss;
    int count = 0;
    
    _aflag_str(ss, count, flags & af_gpu, "af_gpu");
    _aflag_str(ss, count, flags & af_zero, "af_zero");
    _aflag_str(ss, count, flags & af_guard, "af_guard");
    _aflag_str(ss, count, flags & af_random, "af_random");
    _aflag_str(ss, count, flags & af_unified, "af_unified");
    _aflag_str(ss, count, flags & af_verbose, "af_verbose");
    _aflag_str(ss, count, flags & af_page_locked, "af_page_locked");
    _aflag_str(ss, count, flags & af_uninitialized, "af_uninitialized");
    _aflag_str(ss, count, flags & ~af_all_flags, "(unrecognized flags)");

    if (count == 0)
	return "0";
    if (count == 1)
	return ss.str();
    return "(" + ss.str() + ")";
}


// -------------------------------------------------------------------------------------------------


struct alloc_helper {
    static constexpr ssize_t nguard = 4096;

    const ssize_t nbytes;
    const int flags;
    
    char *base = nullptr;
    char *data = nullptr;
    char *gcopy = nullptr;
    

    // Helper for constructor
    void fill(void *dst, const void *src, ssize_t nbytes)
    {
	if (flags & af_gpu)
	    CUDA_CALL(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
	else
	    memcpy(dst, src, nbytes);
    }
    
    
    alloc_helper(ssize_t nbytes_, int flags_) :
	nbytes(nbytes_), flags(flags_)
    {
	check_aflags(flags, "bristlecone::alloc()");
	
	ssize_t g = (flags & af_guard) ? nguard : 0;
	ssize_t nalloc = nbytes + 2*g;

	// Allocate memory
	if (flags & af_gpu)
	    CUDA_CALL(cudaMalloc((void **) &this->base, nalloc));
	else if (flags & af_unified)
	    CUDA_CALL(cudaMallocManaged((void **) &this->base, nalloc, cudaMemAttachGlobal));
	else if (flags & af_page_locked)
	    CUDA_CALL(cudaHostAlloc((void **) &this->base, nalloc, 0));
	else if (posix_memalign((void **) &this->base, 128, nalloc))
	    throw std::runtime_error("bristlecone::alloc(): couldn't allocate " + to_string(nalloc) + " bytes");

	this->data = base + g;

	// Keep this part in sync with "Allocate memory" a few lines above
	if (flags & af_verbose) {
	    if (flags & af_gpu)
		cout << "cudaMalloc";
	    else if (flags & af_unified)
		cout << "cudaMallocManaged";
	    else if (flags & af_page_locked)
		cout << "cudaHostAlloc";
	    else
		cout << "posix_memalign";
	    
	    cout << "(" << nalloc << "): " << ((void *) base);
	    if (base != data)
		cout << ", " << ((void *) data);
	    cout << endl;
	}
	
	assert(base != nullptr);

	if (flags & af_zero) {
	    if (flags & af_gpu)
		CUDA_CALL(cudaMemset(data, 0, nbytes));
	    else
		memset(data, 0, nbytes);
	}

	if (flags & af_guard) {
	    this->gcopy = (char *) malloc(2*nguard);
	    assert(gcopy != nullptr);
	    randomize(gcopy, 2*nguard);
	    
	    fill(base, gcopy, nguard);
	    fill(base + nguard + nbytes, gcopy + nguard, nguard);
	}
    }

    
    // check_guard() is called when the shared_ptr is deleted, in the
    // case where one the flags 'af_guard' is specified.
    //
    // Note: we call assert() instead of throwing exceptions, since
    // shared_ptr deleters aren't supposed to throw exceptions.

    void check_guard()
    {
	assert(flags & af_guard);
	assert(gcopy != nullptr);

	if (flags & af_gpu) {
	    char *p = (char *) malloc(2*nguard);
	    assert(p != nullptr);

	    CUDA_CALL_ABORT(cudaMemcpy(p, base, nguard, cudaMemcpyDeviceToHost));
	    CUDA_CALL_ABORT(cudaMemcpy(p + nguard, base + nguard + nbytes, nguard, cudaMemcpyDeviceToHost));

	    // If this fails, buffer overflow occurred.
	    assert(memcmp(p, gcopy, 2*nguard) == 0);
	    free(p);
	}
	else {
	    // If these fail, buffer overflow occurred.
	    assert(memcmp(base, gcopy, nguard) == 0);
	    assert(memcmp(base + nguard + nbytes, gcopy + nguard, nguard) == 0);
	}	
    }
    

    // operator() is called when the shared_ptr is deleted, in the
    // case where one of the flags (af_guard | af_verbose) is specified.
    //
    // Note: we call assert() instead of throwing exceptions, since
    // shared_ptr deleters aren't supposed to throw exceptions.
    
    void operator()(void *ptr)
    {
	if (flags & af_guard) {
	    check_guard();
	    free(gcopy);
	}

	// Keep this part in sync with "Deallocate memory" just below.
	if (flags & af_verbose) {
	    if (flags & (af_gpu | af_unified))
		cout << "cudaFree";
	    else if (flags & af_page_locked)
		cout << "cudaFreeHost";
	    else
		cout << "free";

	    cout << "(" << ((void *) base) << ")" << endl;
	}

	// Deallocate memory
	if (flags & (af_gpu | af_unified))
	    CUDA_CALL_ABORT(cudaFree(base));
	else if (flags & af_page_locked)
	    CUDA_CALL_ABORT(cudaFreeHost(base));
	else
	    free(base);
    }
};


// Handles all flags except 'af_random'
shared_ptr<void> _af_alloc(ssize_t nbytes, int flags)
{
    alloc_helper h(nbytes, flags);

    // Keep this part in sync with "Deallocate memory"
    // in 'struct alloc_helper' above.
    
    if (flags & (af_guard | af_verbose))
	return shared_ptr<void> (h.data, h);
    else if (flags & (af_gpu | af_unified))
	return shared_ptr<void> (h.data, cudaFree);
    else if (flags & af_page_locked)
	return shared_ptr<void> (h.data, cudaFreeHost);
    else
	return shared_ptr<void> (h.data, free);
}


// -------------------------------------------------------------------------------------------------


// FIXME oops, can get rid of this in favor of cudaMemcpyDefault??
inline cudaMemcpyKind af_copy_kind(int dst_flags, int src_flags)
{
    bool hdst = !af_on_gpu(dst_flags);   // host-only
    bool gdst = !af_on_host(dst_flags);  // gpu-only
    bool hsrc = !af_on_gpu(src_flags);   // host-only
    bool gsrc = !af_on_host(src_flags);  // gpu-only
    
    if (hdst)
	return gsrc ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
    if (gdst)
	return hsrc ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    if (hsrc)
	return cudaMemcpyHostToHost;
    if (gsrc)
	return cudaMemcpyDeviceToDevice;

    // FIXME unified -> unified copy is ambiguous, how should we choose?
    // Maybe it's best to add an argument to specify default 'kind'?
    return cudaMemcpyDeviceToDevice;
}


void _af_copy(void *dst, int dst_flags, const void *src, int src_flags, ssize_t nbytes)
{
    if (nbytes == 0)
	return;
    
    cudaMemcpyKind kind = af_copy_kind(dst_flags, src_flags);
    CUDA_CALL(cudaMemcpy(dst, src, nbytes, kind));
}


}  // namespace gputils
