#ifndef _GPUTILS_SOCKET_HPP
#define _GPUTILS_SOCKET_HPP

#include <string>
#include <sys/socket.h>  // socklen_t

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// RAII wrapper for network socket.
// See extended comment at end of file for "cheat sheet".
struct Socket
{
    int fd = -1;
    bool zerocopy = false;   // set by set_zerocopy(), supplies MSG_ZEROCOPY on future calls to send().
    bool connreset = false;  // set by send() if 

    // For TCP, use (domain,type) = (PF_INET,SOCK_STREAM). See "cheat sheet" below.
    Socket(int domain, int type, int protocol=0);
    Socket() { }
    
    ~Socket() { this->close(); }
    
    void connect(const std::string &ip_addr, short port);
    void bind(const std::string &ip_addr, short port);
    void listen(int backlog=128);
    void close();

    // Reminder: read() returns zero if connection ended, or if socket is nonblocking and no data is ready.
    ssize_t read(void *buf, ssize_t maxbytes);

    // If receiver closes connection, then send() returns zero and sets Socket::connreset = true.
    // If send() is called subsequently (with Socket::connreset == true), then an exception is thrown.
    // This provides a mechanism for the sender to detect a closed connection.
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


// Socketio "cheat sheet"
//
// Sending TCP data:
//
//   Socket s(PF_INET, SOCK_STREAM);
//   s.connect("127.0.0.0", 1370);      // (dst IP address, port)
//   s.get_zerocopy();                  // optional
//   s.set_pacing_rate(bytes_per_sec);  // optional
//   ssize_t nbytes_sent = s.send(buf, maxbytes);
//
// Receiving TCP data from single connection:
//
//   Socket s(PF_INET, SOCK_STREAM);
//   s.set_reuseaddr();
//   s.bind("127.0.0.0", 1370);   // (dst IP address, port)
//   s.listen();
//
//   Socket sd = s.accept();
//   ssize_t nbytes_received = s.read(buf, maxbytes);


}  // namespace gputils

#endif // _GPUTILS_SOCKET_HPP
