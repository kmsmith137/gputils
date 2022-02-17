#ifndef _GPUTILS_CUDASTREAMPOOL_HPP
#define _GPUTILS_CUDASTREAMPOOL_HPP

#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <sys/time.h>

#include "cuda_utils.hpp"  // CudaStreamWrapper


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


class CudaStreamPool {
public:
    // callback(pool, stream, istream)
    using callback_t = std::function<void(const CudaStreamPool &, cudaStream_t stream, int)>;

    // If max_callbacks=0, then CudaStreamPool.run() will run forever.
    CudaStreamPool(const callback_t &callback, int max_callbacks=0, int nstreams=2);

    void run();

    // These members are not protected by a lock. We currently assume that:
    //
    //   - when the pool is running, only the manager thread accesses these members
    //      (possibly via callback)
    //
    //   - after the pool is finished, these members are constant.

    int num_callbacks = 0;
    double elapsed_time = 0.0;
    double time_per_callback = 0.0;
    
protected:
    // Constant after construction, not protected by lock.
    const callback_t callback;
    const int nstreams;
    const int max_callbacks;
    std::vector<CudaStreamWrapper> streams;  // length nstreams
    
    std::condition_variable cv;
    mutable std::mutex lock;
    
    struct StreamState {
	int state = 0;   // 0 = initial state, 1 = kernel running, 2 = kernel done, 3 = stream done
	int istream = -1;
	CudaStreamPool *pool = nullptr;
    };

    // Protected by lock
    std::vector<StreamState> sstate;
    bool is_started = false;
    
    static void manager_thread_body(CudaStreamPool *pool);
    static void cuda_callback(void *stream_state);

    // Used internally by manager thread
    void synchronize();
};

    

} // namespace gputils

#endif // _GPUTILS_CUDASTREAMPOOL_HPP
