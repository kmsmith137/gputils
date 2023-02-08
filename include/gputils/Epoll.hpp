#ifndef _GPUTILS_EPOLL_HPP
#define _GPUTILS_EPOLL_HPP

#include <vector>
#include <sys/epoll.h>

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// Reminder: 'struct epoll_event' looks like this
//
//   struct epoll_event {
//      uint32_t events;        // bitmask, see below
//      union {                 // union, not struct!!
//           void        *ptr;
//           int          fd;
//           uint32_t     u32;
//           uint64_t     u64;
//      } data;
//   };
//
// Here is a partial list of bits in epoll_event::events (see 'man epoll' for complete list):
//
//   EPOLLIN         fd is ready for read()
//   EPOLLOUT        fd is ready for write()
//   EPOLLRDHUP      peer closed connection, or shut down writing half of connection
//   EPOLLHUP        "hangup", i.e. peer closed connection (what is difference versus EPOLLRDHUP)?
//   EPOLLPRI        "exceptional condition" on fd (see POLLPRI in 'man poll')
//   EPOLLERR        fd has error (also reported for the write end of a pipe when the read end has been closed)
//   EPOLLET         requests edge-triggered notification (see 'man epoll')
//   EPOLLONESHOT    requests one-shot notification (see 'man epoll')
//   EPOLLWAKEUP     ensure system does not "suspend" or "hibernate" while event is being processed (see 'man epoll')
//   EPOLLEXCLUSIVE  sets an exclusive wakeup mode for the epoll file descriptor (see 'man epoll')
//
// Epoll always waits on (EPOLLERR | EPOLLHUP), i.e. no need to include these flags in Epoll::add_fd().


// RAII wrapper for epoll file descriptor
struct Epoll
{
    int epfd = -1;

    // Events returned by wait() are stored here.
    // Note that wait() returns the number of events (which is <= events.size()).
    std::vector<epoll_event> events;

    // If constructor is called with initialize=false, then Epoll::initialize() must be called later.
    Epoll(bool initialize=true, bool close_on_exec=false);
    ~Epoll() { this->close(); }

    void add_fd(int fd, struct epoll_event &ev);
    // To add later: modify_fd(fd,ev), delete_fd(fd).

    // Returns number of events (or zero, if timeout expires).
    // Negative timeout means "blocking". Zero timeout is nonblocking.
    int wait(int timeout_ms=-1);

    void initialize(bool close_on_exec=false);
    void close();
	
    // The Epoll class is noncopyable, but if copy semantics are needed, you can do
    //   shared_ptr<Epoll> ep = make_shared<Epoll> ();
    
    Epoll(const Epoll &) = delete;
    Epoll &operator=(const Epoll &) = delete;
};


}  // namespace gputils

#endif // _GPUTILS_EPOLL_HPP
