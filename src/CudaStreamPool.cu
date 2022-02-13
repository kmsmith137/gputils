#include <thread>
#include "../include/gputils/cuda_utils.hpp"    // CUDA_CALL(), CudaStreamWrapper
#include "../include/gputils/CudaStreamPool.hpp"

using namespace std;


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


CudaStreamPool::CudaStreamPool(int ns, const callback_t &cb)
    : nstreams(ns), callback(cb)
{
    assert(nstreams > 0);

    // Streams will be created by cudaStreamWrapper constructor.
    this->streams.resize(nstreams);
    this->sstate.resize(nstreams);
    
    for (int i = 0; i < nstreams; i++) {
	this->sstate[i].pool = this;
	this->sstate[i].state = 0;
	this->sstate[i].index = i;
    }
}


void CudaStreamPool::run()
{
    unique_lock ulock(lock);
    if (is_running)
	throw runtime_error("CudaStreamPool::run() called on already-running pool");
    is_running = true;
    ulock.unlock();
    
    std::thread t(manager_thread_body, this);
    t.join();

    ulock.lock();
    assert(is_running);
    is_running = false;
}


int CudaStreamPool::get_num_callbacks() const
{
    lock_guard lg(lock);
    return num_callbacks;
}


double CudaStreamPool::get_elapsed_time() const
{
    struct timeval curr_time;
    
    int err = gettimeofday(&curr_time, NULL);
    assert(err == 0);

    return (curr_time.tv_sec - start_time.tv_sec) + 1.0e-6 * (curr_time.tv_usec - start_time.tv_usec);
}


void CudaStreamPool::manager_thread_body(CudaStreamPool *pool)
{
    int err = gettimeofday(&pool->start_time, NULL);
    assert(err == 0);
    
    unique_lock ulock(pool->lock);

    for (;;) {
	bool did_callback = false;
	bool found_running_stream = false;
	
	for (int i = 0; i < pool->nstreams; i++) {
	    if (pool->sstate[i].state == 0) {
		// Call callback function without holding lock.
		ulock.unlock();
		bool still_running = pool->callback(*pool, pool->streams[i], i);
		did_callback = true;

		// Reacquire lock to set state, before queueing cuda_callback.
		ulock.lock();
		pool->sstate[i].state = still_running ? 1 : 2;

		// Queue cuda_callback without holding lock.
		ulock.unlock();
		if (still_running)
		    CUDA_CALL(cudaLaunchHostFunc(pool->streams[i], cuda_callback, &pool->sstate[i]));

		// Reacquire lock before proceeding with loop.
		ulock.lock();
	    }
	    else if (pool->sstate[i].state == 1)
		found_running_stream = true;
	}

	if (!did_callback && !found_running_stream)
	    return;

	if (!did_callback)
	    pool->cv.wait(ulock);
    }
}


void CudaStreamPool::cuda_callback(void *up)
{
    StreamState *u = reinterpret_cast<StreamState *> (up);
    CudaStreamPool *pool = u->pool;
    int index = u->index;

    assert((index >= 0) && (index < pool->nstreams));
    assert(&pool->sstate[index] == u);

    unique_lock<mutex> ulock(pool->lock);
    pool->cv.notify_all();
    pool->num_callbacks++;
    u->state = 0;
}


}  // namespace gputils
