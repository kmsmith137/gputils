#include "../include/gputils/Socket.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <arpa/inet.h>

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
	ss << func_name << "() called on uninitalized or closed socket";
	return ss.str();
    }

    return errstr(func_name);
}


// -------------------------------------------------------------------------------------------------


static void inet_pton_x(struct sockaddr_in &saddr, const string &ip_addr, short port)
{
    memset(&saddr, 0, sizeof(saddr));

    saddr.sin_family = AF_INET;
    saddr.sin_port = htons(port);   // note htons() here!
    
    int err = inet_pton(AF_INET, ip_addr.c_str(), &saddr.sin_addr);
    
    if (err < 0) {
	stringstream ss;
	ss << "inet_pton() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }

    if (err == 0) {
	stringstream ss;
	ss << "invalid IPv4 address: '" << ip_addr << "'";
	throw runtime_error(ss.str());
    }
}


// -------------------------------------------------------------------------------------------------


Socket::Socket(int domain, int type, int protocol)
{
    this->fd = socket(domain, type, protocol);

    if (_unlikely(fd < 0))
	throw runtime_error(errstr("socket"));
}


void Socket::connect(const std::string &ip_addr, short port)
{
    struct sockaddr_in saddr;
    inet_pton_x(saddr, ip_addr, port);

    int err = ::connect(this->fd, (const struct sockaddr *) &saddr, sizeof(saddr));

    if (_unlikely(err < 0))
	throw runtime_error(errstr(fd, "Socket::connect"));
}


void Socket::bind(const std::string &ip_addr, short port)
{
    struct sockaddr_in saddr;
    inet_pton_x(saddr, ip_addr, port);

    int err = ::bind(this->fd, (const struct sockaddr *) &saddr, sizeof(saddr));
    
    if (_unlikely(err < 0))
	throw runtime_error(errstr(fd, "Socket::bind"));
}


void Socket::listen(int backlog)
{
    int err = ::listen(fd, backlog);

    if (_unlikely(err < 0))
	throw runtime_error(errstr(fd, "Socket::listen"));
}


void Socket::close()
{
    if (fd < 0)
	return;

    int err = ::close(fd);
    
    this->fd = -1;
    this->zerocopy = false;

    if (_unlikely(err < 0))
	cout << errstr("Socket::close") << endl;
}

    
ssize_t Socket::read(void *buf, ssize_t count)
{
    assert(count > 0);
    ssize_t nbytes = ::read(this->fd, buf, count);

    if (_unlikely(nbytes < 0))
	throw runtime_error(errstr(fd, "Socket::read"));

    assert(nbytes <= count);
    return nbytes;
}


ssize_t Socket::send(const void *buf, ssize_t count, int flags)
{
    if (zerocopy)
	flags |= MSG_ZEROCOPY;

    assert(count > 0);
    ssize_t nbytes = ::send(this->fd, buf, count, flags);

    if (_unlikely(nbytes < 0))
	throw runtime_error(errstr(fd, "Socket::send"));

    // Can send() return zero? If so, then this next line needs removal or rethinking.
    if (_unlikely(nbytes == 0))
	throw runtime_error("Socket::send() returned zero?!");

    assert(nbytes <= count);
    return nbytes;
}


Socket Socket::accept()
{
    // FIXME currently throwing away sender's IP address
    sockaddr_in saddr_throwaway;
    socklen_t saddr_len = sizeof(saddr_throwaway);

    Socket ret;
    ret.fd = ::accept(fd, (struct sockaddr *) &saddr_throwaway, &saddr_len);

    if (_unlikely(ret.fd < 0))
	throw runtime_error(errstr(fd, "Socket::accept"));

    return ret;
}


void Socket::getopt(int level, int optname, void *optval, socklen_t *optlen)
{
    assert(optval != nullptr);
    assert(optlen != nullptr);

    int err = getsockopt(fd, level, optname, optval, optlen);
    
    if (_unlikely((err < 0)))
	throw runtime_error(errstr(fd, "getsockopt"));
}


void Socket::setopt(int level, int optname, const void *optval, socklen_t optlen)
{
    assert(optval != nullptr);
	
    int err = setsockopt(fd, level, optname, optval, optlen);

    if (_unlikely(err < 0))
	throw runtime_error(errstr(fd, "setsockopt"));
}


void Socket::set_reuseaddr()
{
    int on = 1;
    int err = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    if (err < 0)
	throw runtime_error(errstr(fd, "Socket::set_reuseaddr"));
}


void Socket::set_nonblocking()
{
    int flags = fcntl(this->fd, F_GETFL);
    
    if (_unlikely(flags < 0))
	throw runtime_error(errstr(fd, "Socket::set_nonblocking: F_GETFL fcntl"));

    int err = fcntl(this->fd, F_SETFL, flags | O_NONBLOCK);

    if (_unlikely(err < 0))
	throw runtime_error(errstr(fd, "Socket::set_nonblocking: F_SETFL fcntl"));
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
    int err = setsockopt(fd, SOL_SOCKET, SO_MAX_PACING_RATE, &b, sizeof(b));

    if (_unlikely(err < 0))
	throw runtime_error(errstr(fd, "Socket::set_pacing_rate"));
}
 

void Socket::set_zerocopy()
{
      int on = 1;
      int err = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

      if (err < 0)
	  throw runtime_error(errstr(fd, "Socket::set_zerocopy"));

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
