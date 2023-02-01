#ifndef _GPUTILS_SYSTEM_UTILS_HPP
#define _GPUTILS_SYSTEM_UTILS_HPP

#include <vector>
#include <string>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Error-checked (*_x) versions of linux/posix functions, which throw verbose exceptions on failure.


extern void mkdir_x(const char *path, int mode=0755);
extern void mkdir_x(const std::string &path, int mode=0755);

extern void mlockall_x(int flags = MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT);

extern void *mmap_x(void *addr, ssize_t length, int prot, int flags, int fd, off_t offset);
extern void munmap_x(void *addr, ssize_t length);

// Reminder: returns number of bytes read (must be <= count), returns zero on EOF.
extern ssize_t read_x(int fd, void *buf, ssize_t count);


// -------------------------------------------------------------------------------------------------
//
// Socket API.
// Some these functions assume IPv4 for now -- I may add IPv6 counterparts later.
// I can never remember the socket API, so the following comments are notes to myself.
//
// Accepting a TCP connection:
//
//   int fdlisten = socket_x(PF_INET, SOCK_STREAM);
//   bind_x(fdlisten, "10.0.0.1", 1370);
//   listen_x(fdlisten);
//
//   int fd = accept_x(fdlisten);
//   fcntl_x(fd, F_SETFL, O_NDELAY);
//
//
// SO_REUSEADDR option on listening socket is usually a good idea:
//   int on = 1;
//   setsockopt_x(fdlisten, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
//
// Sending TCP data:
//
//   int fd = socket_x(PF_INET, SOCK_STREAM);

extern int socket_x(int domain, int type, int protocol=0);
extern int accept_x(int sockfd, sockaddr_in *addr = nullptr);
extern void listen_x(int sockfd, int backlog=128);
extern void bind_x(int sockfd, const struct sockaddr_in &saddr);
extern void bind_x(int sockfd, const std::string &ip_addr, short port);
extern void connect_x(int sockfd, const struct sockaddr_in &saddr);
extern void connect_x(int sockfd, const std::string &ip_addr, short port);
extern void getsockopt_x(int sockfd, int level, int optname, void *optval, socklen_t *optlen);
extern void setsockopt_x(int sockfd, int level, int optname, const void *optval, socklen_t optlen);
extern void inet_pton_x(struct sockaddr_in &saddr, const std::string &ip_addr, short port);

// Reminder: returns number of bytes written (must be <= count), returns zero on EOF.
extern ssize_t send_x(int sockfd, void *buf, ssize_t count, int flags=0);


// pin_thread_to_vcpus(vcpu_list)
//
// The 'vcpu_list' argument is a list of integer vCPUs, where I'm defining a vCPU
// to be the scheduling unit in pthread_setaffinity_np() or sched_setaffinity().
//
// If hyperthreading is disabled, then there should be one vCPU per core.
// If hyperthreading is enabled, then there should be two vCPUs per core
// (empirically, always with vCPU indices 2*n and 2*n+1?)
//
// I think that the number of vCPUs and their location in the NUMA hierarchy
// always follows the output of 'lscpu -ae', but AFAIK this isn't stated anywhere.
//
// If 'vcpu_list' is an empty vector, then pin_thread_to_vcpus() is a no-op.

extern void pin_thread_to_vcpus(const std::vector<int> &vcpu_list);


} // namespace gputils

#endif  // _GPUTILS_SYSTEM_UTILS_HPP
