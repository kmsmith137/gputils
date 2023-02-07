#ifndef _GPUTILS_SOCKET_HPP
#define _GPUTILS_SOCKET_HPP

#include <string>
#include <sys/socket.h>  // socklen_t

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// RAII wrapper for network socket
struct Socket
{
    int fd = -1;
    bool zerocopy = false;  // set by set_zerocopy(), supplies MSG_ZEROCOPY on future calls to send().

    Socket() { }
    Socket(int domain, int type, int protocol=0);

    void connect(const std::string &ip_addr, short port);
    void bind(const std::string &ip_addr, short port);
    void listen(int backlog=128);
    void close();

    // Reminder: read() returns zero if connection ended, or if socket is nonblocking and no data is ready.
    ssize_t read(void *buf, ssize_t maxbytes);
    ssize_t send(const void *buf, ssize_t count, int flags=0);

    // FIXME in current API, sender's IP address is thrown away!
    Socket accept();

    // General wrappers for getsockopt(), setsockopt()
    void getopt(int level, int optname, void *optval, socklen_t *optlen);
    void setopt(int level, int optname, const void *optval, socklen_t optlen);

    // Specific options
    void set_reuseaddr();    // setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    void set_nonblocking();  // fcntl(O_NONBLOCK)
    void set_pacing_rate(double bytes_per_sec);  // setsockopt(SOL_SOCKET, SO_MAX_PACING_RATE)
    void set_zerocopy();     // setsockopt(SOL_SOCKET, SO_ZEROCOPY) + (MSG_ZEROCOPY on future send() calls)
    
    // Socket is noncopyable but moveable (can always do shared_ptr<Socket> to avoid copies).
    
    Socket(const Socket &) = delete;
    Socket &operator=(const Socket &) = delete;
    
    Socket(Socket &&s);
    Socket &operator=(Socket &&s);
};


}  // namespace gputils

#endif // _GPUTILS_SOCKET_HPP
