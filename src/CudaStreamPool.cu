#include <thread>
#include "../include/gputils/cuda_utils.hpp"    // CUDA_CALL(), CudaStreamWrapper
#include "../include/gputils/time_utils.hpp"    // get_time(), time_since()
#include "../include/gputils/CudaStreamPool.hpp"

using namespace std;


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


CudaStreamPool::CudaStreamPool(const callback_t &callback_, int max_callbacks_, int nstreams_)
    : callback(callback_), max_callbacks(max_callbacks_), nstreams(nstreams_)
{
    assert(max_callbacks >= 0);
    assert(nstreams > 0);

    // Streams will be created by cudaStreamWrapper constructor.
    this->streams.resize(nstreams);
    this->sstate.resize(nstreams);
    
    for (int i = 0; i < nstreams; i++) {
	this->sstate[i].pool = this;
	this->sstate[i].state = 0;
	this->sstate[i].istream = i;
    }
}


void CudaStreamPool::run()
{
    unique_lock ulock(lock);
    if (is_started)
	throw runtime_error("CudaStreamPool::run() called twice");
    
    is_started = true;
    ulock.unlock();
    
    std::thread t(manager_thread_body, this);
    t.join();
}

void CudaStreamPool::manager_thread_body(CudaStreamPool *pool)
{
    auto start_time = get_time();
    unique_lock ulock(pool->lock);

    for (;;) {
	bool did_callback = false;

	for (int istream = 0; istream < pool->nstreams; istream++) {
	    // At top of loop, lock is held.
	    StreamState &ss = pool->sstate[istream];

	    if (ss.state == 1)  // kernel running on stream
		continue;

	    if (ss.state == 2) {  // kernel finished
		pool->num_callbacks++;
		pool->elapsed_time = time_since(start_time);
		pool->time_per_callback = pool->elapsed_time / pool->num_callbacks;

		if ((pool->max_callbacks > 0) && (pool->num_callbacks >= pool->max_callbacks)) {
		    ulock.unlock();
		    pool->synchronize();
		    return;   // this is where the manager thread exits!
		}
	    }

	    // Call callback function without holding lock.
	    ulock.unlock();
	    pool->callback(*pool, pool->streams[istream], istream);
	    did_callback = true;

	    // Reacquire lock to set state, before queueing cuda_callback.
	    ulock.lock();
	    ss.state = 1;
	    
	    // Queue cuda_callback without holding lock.
	    ulock.unlock();
	    CUDA_CALL(cudaLaunchHostFunc(pool->streams[istream], cuda_callback, &pool->sstate[istream]));

	    // Reacquire lock before proceeding with loop.
	    ulock.lock();
	}

	if (!did_callback)
	    pool->cv.wait(ulock);
    }
}


void CudaStreamPool::cuda_callback(void *up)
{
    StreamState *u = reinterpret_cast<StreamState *> (up);
    CudaStreamPool *pool = u->pool;
    int istream = u->istream;

    assert((istream >= 0) && (istream < pool->nstreams));
    assert(&pool->sstate[istream] == u);

    unique_lock<mutex> ulock(pool->lock);
    u->state = 2;
    pool->cv.notify_all();
}


void CudaStreamPool::synchronize()
{
    for (int istream = 0; istream < nstreams; istream++)
	CUDA_CALL(cudaStreamSynchronize(streams[istream]));
}



}  // namespace gputils
