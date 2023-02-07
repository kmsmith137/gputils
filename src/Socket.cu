#include "../include/gputils/Socket.hpp"
#include "../include/gputils/system_utils.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cassert>
#include <sstream>
#include <iostream>
#include <stdexcept>

// Branch predictor hint
#ifndef _unlikely
#define _unlikely(cond)  (__builtin_expect(cond,0))
#endif

using namespace std;


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


inline void check_initialized(const Socket *s, const char *method_name)
{
    if (_unlikely(s->fd < 0)) {
	stringstream ss;
	ss << "Socket::" << method_name << "() called on uninitialized or closed socket";
	throw runtime_error(ss.str());
    }
}


Socket::Socket(int domain, int type, int protocol)
{
    this->fd = socket_x(domain, type, protocol);
}


void Socket::connect(const std::string &ip_addr, short port)
{
    check_initialized(this, "connect");
    connect_x(this->fd, ip_addr, port);
}


void Socket::bind(const std::string &ip_addr, short port)
{
    check_initialized(this, "bind");
    bind_x(this->fd, ip_addr, port);
}


void Socket::listen(int backlog)
{
    check_initialized(this, "listen");
    listen_x(this->fd, backlog);
}


void Socket::close()
{
    if (fd < 0)
	return;

    int err = ::close(fd);
    
    this->fd = -1;
    this->zerocopy = false;

    if (_unlikely(err < 0)) {
	stringstream ss;
	ss << "Socket(): close() failed?! (" << strerror(errno) << ")\n";
	cout << ss.str() << flush;
    }
}

    
ssize_t Socket::read(void *buf, ssize_t count)
{
    assert(count > 0);
    ssize_t nbytes = ::read(this->fd, buf, count);
    
    if (_unlikely(nbytes < 0)) {
	check_initialized(this, "read");
	stringstream ss;
	ss << "Socket::read() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }

    return nbytes;
}


ssize_t Socket::send(const void *buf, ssize_t count, int flags)
{
    if (zerocopy)
	flags |= MSG_ZEROCOPY;

    check_initialized(this, "send");
    return send_x(this->fd, const_cast<void *> (buf), count, flags);
}


Socket Socket::accept()
{
    check_initialized(this, "accept");

    Socket ret;
    ret.fd = accept_x(this->fd);
    return ret;
}


void Socket::getopt(int level, int optname, void *optval, socklen_t *optlen)
{
    check_initialized(this, "getopt");
    getsockopt_x(this->fd, level, optname, optval, optlen);
}


void Socket::setopt(int level, int optname, const void *optval, socklen_t optlen)
{
    check_initialized(this, "setopt");
    setsockopt_x(this->fd, level, optname, optval, optlen);
}


void Socket::set_reuseaddr()
{
    int on = 1;
    this->setopt(SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
}


void Socket::set_nonblocking()
{
    int flags = fcntl(this->fd, F_GETFL);
    
    if (_unlikely(flags < 0)) {
	check_initialized(this, "set_nonblocking");
	stringstream ss;
	ss << "Socket::set_nonblocking(): fcntl(F_GETFL) failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }

    int err = fcntl(this->fd, F_SETFL, flags | O_NONBLOCK);

    if (_unlikely(err < 0)) {
	stringstream ss;
	ss << "Socket::set_nonblocking(): fcntl(F_SETFL) failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }
}


void Socket::set_pacing_rate(double bytes_per_sec)
{
    assert(bytes_per_sec >= 1.0);

    if (_unlikely(bytes_per_sec > 4.0e9)) {
	stringstream ss;
	ss << "Socket::set_pacing_rate(" << bytes_per_sec << "):"
	   << " 'bytes_per_sec' values larger than 4e9 (i.e. 36 Gpbs) are not currently supported!"
	   << " This is because setsockopt(SOL_SOCKET, SO_MAX_PACING_RATE) takes a uint32 argument."
	   << " Suggested workaround: split output across multiple sockets/threads.";
	throw runtime_error(ss.str());
    }

    uint32_t b = uint32_t(bytes_per_sec + 0.5);
    this->setopt(SOL_SOCKET, SO_MAX_PACING_RATE, &b, sizeof(b));
}
 

void Socket::set_zerocopy()
{
      int on = 1;
      this->setopt(SOL_SOCKET, SO_ZEROCOPY, &on, sizeof(on));

      // If the 'zerocopy' flag is set, then MSG_ZEROCOPY will be included in future calls to send().
      this->zerocopy = true;
}


// Move constructor.
Socket::Socket(Socket &&s)
{
    // FIXME figure out how to avoid cut-and-paste between move constructor (here) and move assignment-operator (below).
    // Defining a helper function Socket::_move(Socket &&) didn't work ("an rvalue reference cannot be bound to an lvalue").
    // Maybe I need an explicit call to std::move()?
    
    this->close();
    this->fd = s.fd;
    this->zerocopy = s.zerocopy;
    
    s.fd = -1;
    s.zerocopy = false;
}


// Move assignment-operator.
Socket &Socket::operator=(Socket &&s)
{
    this->close();
    this->fd = s.fd;
    this->zerocopy = s.zerocopy;
    
    s.fd = -1;
    s.zerocopy = false;
    return *this;
}


} // namespace gputils
