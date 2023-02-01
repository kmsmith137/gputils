#ifndef _GPUTILS_SYSTEM_UTILS_HPP
#define _GPUTILS_SYSTEM_UTILS_HPP

#include <vector>
#include <string>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// Error-checked (*_x) versions of linux/posix functions, which throw verbose
// exceptions on failure.


extern void mkdir_x(const char *path, int mode=0755);
extern void mkdir_x(const std::string &path, int mode=0755);

extern void mlockall_x(int flags = MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT);

// Reminder: reads up to 'count' bytes, returns zero on EOF.
extern ssize_t read_x(int fd, void *buf, size_t count);


// Socket API cheat sheet: accepting a TCP connection
//
//   int on = 1;
//   int fdlisten = socket_x(PF_INET, SOCK_STREAM);
//   setsockopt_x(fdlisten, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
//   bind_socket(fdlisten, "10.0.0.1", 1370);
//   listen_x(fdlisten);
//
//   int fd = accept_x(fdlisten);
//   fcntl_x(fd, F_SETFL, O_NDELAY);


extern int socket_x(int domain, int type, int protocol=0);
extern int accept_x(int sockfd, sockaddr_in *addr = nullptr);
extern void listen_x(int sockfd, int backlog=128);
extern void getsockopt_x(int sockfd, int level, int optname, void *optval, socklen_t *optlen);
extern void setsockopt_x(int sockfd, int level, int optname, const void *optval, socklen_t optlen);


// bind_socket(): wrapper for inet_pton(), followed by bind().
// IPv4 assumed for now! (I may add an ip=4 optional argument later).
extern void bind_socket(int sockfd, const std::string &ip_addr, short port);


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
