#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

#include "../include/gputils/Epoll.hpp"

using namespace std;


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


Epoll::Epoll(bool close_on_exec)
{
    int flags = close_on_exec ? EPOLL_CLOEXEC : 0;
    this->epfd = epoll_create1(flags);
    
    if (epfd < 0) {
	stringstream ss;
	ss << "epoll_create(): " << strerror(errno);
	throw runtime_error(ss.str());
    }
}


Epoll::~Epoll()
{
    if (epfd >= 0) {
	close(epfd);
	epfd = -1;
    }
}


void Epoll::add_fd(int fd, struct epoll_event &ev)
{
    int err = epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev);

    if (err < 0) {
	stringstream ss;
	ss << "epoll_ctl(): " << strerror(errno);
	throw runtime_error(ss.str());
    }

    struct epoll_event ev0;
    memset(&ev0, 0, sizeof(ev0));
    this->events.push_back(ev0);
}


int Epoll::wait(int timeout_ms)
{
    if (events.size() == 0)
	throw runtime_error("gputils::Epoll::wait() was called before Epoll::add_fd()");
	    
    int ret = epoll_wait(epfd, &events[0], events.size(), timeout_ms);

    if (ret < 0) {
	stringstream ss;
	ss << "epoll_wait(): " << strerror(errno);
	throw runtime_error(ss.str());
    }

    return ret;
}


} // namespace gputils
