#include "../include/gputils/Epoll.hpp"

#include <cstring>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>


// Branch predictor hint
#ifndef _unlikely
#define _unlikely(cond)  (__builtin_expect(cond,0))
#endif

using namespace std;


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


inline string errstr(const string &func_name)
{
    stringstream ss;
    ss << func_name << "() failed: " << strerror(errno);
    return ss.str();
}


inline string errstr(int fd, const string &func_name)
{
    if (fd < 0) {
	stringstream ss;
	ss << func_name << "() called on uninitalized or closed Epoll instance";
	return ss.str();
    }

    return errstr(func_name);
}


// -------------------------------------------------------------------------------------------------


Epoll::Epoll(bool init_flag, bool close_on_exec)
{
    if (init_flag)
	this->initialize(close_on_exec);
}


void Epoll::initialize(bool close_on_exec)
{
    if (_unlikely(epfd >= 0))
	throw runtime_error("Epoll::initialize() called on already-initialized Epoll instance");

    int flags = close_on_exec ? EPOLL_CLOEXEC : 0;
    this->epfd = epoll_create1(flags);

    if (_unlikely(epfd < 0))
	throw runtime_error(errstr("epoll_create"));
}


void Epoll::close()
{
    if (epfd < 0)
	return;

    int err = ::close(epfd);
    this->epfd = -1;

    if (_unlikely(err < 0))
	cout << errstr("Epoll::close") << endl;
}


void Epoll::add_fd(int fd, struct epoll_event &ev)
{
    int err = epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev);

    if (_unlikely(err < 0))
	throw runtime_error(errstr(epfd, "Epoll::add_fd"));

    struct epoll_event ev0;
    memset(&ev0, 0, sizeof(ev0));
    this->events.push_back(ev0);
}


int Epoll::wait(int timeout_ms)
{
    int ret = epoll_wait(epfd, &events[0], events.size(), timeout_ms);

    if (_unlikely(ret < 0)) {
	if (events.size() == 0)
	    throw runtime_error("gputils::Epoll::wait() was called before Epoll::add_fd()");
	throw runtime_error(errstr(epfd, "Epoll::wait"));
    }

    return ret;
}


} // namespace gputils
