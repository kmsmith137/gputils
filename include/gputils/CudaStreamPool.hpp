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


// CudaStreamPool: run multiple streams with dynamic load-balancing, intended for timing
//
// Example:
// 
//   int num_callbacks = 100;
//   int num_streams = 2;
//
//   // Callback function is called when kernel(s) finish, and queues new kernel(s).
//   auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
//       {
//           kernel <<<...., stream>>> ();   // queue kernel(s)
//	};
//
//   CudaStreamPool sp(callback, num_callbacks, nstreams, "kernel name");
//
//   // Example timing monitor: suppose each callback uses 200 GB of global memory BW
//   sp.monitor_throughput("Global memory BW (GB/s)", 200.0);
//
//   // Example timing monitor: suppose each callback processes 0.5 sec of real-time data
//   sp.monitor_timing("Real-time fraction", 0.5);
//
//   sp.run();


class CudaStreamPool {
public:
    // callback(pool, stream, istream)
    using callback_t = std::function<void(const CudaStreamPool &, cudaStream_t stream, int)>;

    // If max_callbacks=0, then CudaStreamPool.run() will run forever.
    CudaStreamPool(const callback_t &callback, int max_callbacks=0, int nstreams=2, const std::string &name="CudaStreamPool");

    // Runs stream pool to completion.
    void run();

    // These functions define "timing monitors".
    // To monitor continuously as stream runs, call between constructor and run().
    // To show once at the end, call after run(), then call show_timings().
    void monitor_throughput(const std::string &label = "callbacks/sec", double coeff=1.0);
    void monitor_time(const std::string &label = "seconds/callback", double coeff=1.0);

    // Show all timing monitors (call without lock held).
    void show_timings();
    
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
    const std::string name;
    std::vector<CudaStreamWrapper> streams;  // length nstreams
    
    std::condition_variable cv;
    mutable std::mutex lock;
    
    struct StreamState {
	int state = 0;   // 0 = initial state, 1 = kernel running, 2 = kernel done, 3 = stream done
	int istream = -1;
	CudaStreamPool *pool = nullptr;
    };

    struct TimingMonitor {
	std::string label;
	double coeff;
	bool thrflag;
    };

    // Protected by lock
    std::vector<StreamState> sstate;
    std::vector<TimingMonitor> timing_monitors;
    bool is_started = false;
    bool is_done = false;

    void _add_timing_monitor(const std::string &label, double coeff, bool thrflag);
    
    static void manager_thread_body(CudaStreamPool *pool);
    static void cuda_callback(void *stream_state);

    // Used internally by manager thread
    void synchronize();
};
    

} // namespace gputils

#endif // _GPUTILS_CUDASTREAMPOOL_HPP
