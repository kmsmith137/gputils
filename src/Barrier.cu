#include <stdexcept>
#include "../include/gputils/Barrier.hpp"

using namespace std;


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


Barrier::Barrier(int nthreads_) :
    nthreads(nthreads_)
{
    if (nthreads <= 0)
	throw runtime_error("Barrier constructor: expected nthreads > 0");
}


void Barrier::wait()
{
    std::unique_lock ul(lock);
    
    if (aborted)
	throw runtime_error(abort_msg);

    if (nthreads_waiting == nthreads-1) {
	this->nthreads_waiting = 0;
	this->wait_count++;
	ul.unlock();
	cv.notify_all();
	return;
    }
	
    this->nthreads_waiting++;
    
    int wc = this->wait_count;
    cv.wait(ul, [this,wc] { return (this->aborted || (this->wait_count > wc)); });
    
    if (aborted)
	throw runtime_error(abort_msg);
}


void Barrier::abort(const string &msg)
{
    std::unique_lock ul(lock);
    if (aborted)
	return;
    
    this->aborted = true;
    this->abort_msg = msg;
    ul.unlock();
    cv.notify_all();
};


} // namespace gputils
