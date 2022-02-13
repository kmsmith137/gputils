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
    // callback(pool, stream, index) -> doneflag
    using callback_t = std::function<bool(const CudaStreamPool &, cudaStream_t stream, int)>;
    CudaStreamPool(int nstreams, const callback_t &callback);

    void run();

    int get_num_callbacks() const;    
    double get_elapsed_time() const;
    
protected:
    // Constant after construction, not protected by lock
    const int nstreams;
    const callback_t callback;
    std::vector<CudaStreamWrapper> streams;  // length nstreams
    
    std::condition_variable cv;
    mutable std::mutex lock;
    
    struct StreamState {
	int state = 0;   // 0 = kernel not running, 1 = kernel running, 2 = stream done
	int index = -1;
	CudaStreamPool *pool = nullptr;
    };

    // Protected by lock
    std::vector<StreamState> sstate;
    int num_callbacks = 0;
    bool is_running = false;

    // FIXME some day I'll learn how to use std::chrono
    struct timeval start_time;
    
    static void manager_thread_body(CudaStreamPool *pool);
    static void cuda_callback(void *stream_state);
};

    

} // namespace gputils

#endif // _GPUTILS_CUDASTREAMPOOL_HPP
