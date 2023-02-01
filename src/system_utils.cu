#include <thread>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <sched.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <arpa/inet.h>

#include "../include/gputils/system_utils.hpp"

using namespace std;


namespace gputils {
#if 0
};  // pacify emacs c-mode!
#endif


void mkdir_x(const char *path, int mode)
{
    int err = mkdir(path, mode);
    
    if (err < 0) {
	stringstream ss;
	ss << "mkdir('" << path << "') failed: " << strerror(errno);  // FIXME should show mode
	throw runtime_error(ss.str());
    }
}


void mkdir_x(const std::string &path, int mode)
{
    mkdir_x(path.c_str(), mode);
}


void mlockall_x(int flags)
{
    // FIXME low-priority things to add:
    //   - test that 'flags' is a subset of (MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT)
    //   - if mlockall() fails, then exception text should pretty-print flags
    
    int err = mlockall(flags);
    
    if (err < 0) {
	stringstream ss;
	ss << "mlockall() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }
}


void *mmap_x(void *addr, ssize_t length, int prot, int flags, int fd, off_t offset)
{
    assert(length > 0);
    void *ret = mmap(addr, length, prot, flags, fd, offset);

    if (ret == MAP_FAILED) {
	stringstream ss;
	ss << "mmap() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }

    assert(ret != nullptr);  // paranoid
    return ret;
}

	     
void munmap_x(void *addr, ssize_t length)
{
    assert(length > 0);
    
    int err = munmap(addr, length);

    if (err < 0) {
	stringstream ss;
	ss << "mmap() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }
}


ssize_t read_x(int fd, void *buf, ssize_t count)
{
    assert(count > 0);
    ssize_t nbytes = read(fd, buf, count);

    if (nbytes < 0) {
	stringstream ss;
	ss << "read() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }

    // Returns 0 on EOF.
    return nbytes;
}


// -------------------------------------------------------------------------------------------------


int socket_x(int domain, int type, int protocol)
{
    int fd = socket(domain, type, protocol);

    if (fd < 0) {
	stringstream ss;
	ss << "socket() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }

    return fd;
}


int accept_x(int sockfd, sockaddr_in *addr)
{
    sockaddr_in saddr;
    addr = addr ? &saddr : addr;

    socklen_t addrlen = sizeof(sockaddr_in);
    int fd = accept(sockfd, (struct sockaddr *) addr, &addrlen);

    if (fd < 0) {
	stringstream ss;
	ss << "accept() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }    

    return fd;
}


void listen_x(int sockfd, int backlog)
{
    int err = listen(sockfd, backlog);

    if (err < 0) {
	stringstream ss;
	ss << "listen() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }    
}


void bind_x(int sockfd, const struct sockaddr_in &saddr)
{
    int err = bind(sockfd, (const struct sockaddr *) &saddr, sizeof(saddr));
    
    if (err < 0) {
	stringstream ss;
	ss << "bind() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }
}


void bind_x(int sockfd, const string &ip_addr, short port)
{
    struct sockaddr_in saddr;
    inet_pton_x(saddr, ip_addr, port);
    bind_x(sockfd, saddr);
}


void connect_x(int sockfd, const struct sockaddr_in &saddr)
{
    int err = connect(sockfd, (const struct sockaddr *) &saddr, sizeof(saddr));
    
    if (err < 0) {
	stringstream ss;
	ss << "connect() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }
}
    
void connect_x(int sockfd, const string &ip_addr, short port)
{
    struct sockaddr_in saddr;
    inet_pton_x(saddr, ip_addr, port);
    connect_x(sockfd, saddr);
}


void getsockopt_x(int sockfd, int level, int optname, void *optval, socklen_t *optlen)
{
    assert(optval != nullptr);
    assert(optlen != nullptr);

    int err = getsockopt(sockfd, level, optname, optval, optlen);
    
    if (err < 0) {
	stringstream ss;
	ss << "getsockopt() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }
}


void setsockopt_x(int sockfd, int level, int optname, const void *optval, socklen_t optlen)
{
    assert(optval != nullptr);

    int err = setsockopt(sockfd, level, optname, optval, optlen);

    if (err < 0) {
	stringstream ss;
	ss << "setsockopt() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }
}


void inet_pton_x(struct sockaddr_in &saddr, const string &ip_addr, short port)
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


ssize_t send_x(int sockfd, void *buf, ssize_t count, int flags)
{
    assert(count > 0);
    ssize_t nbytes = send(sockfd, buf, count, flags);

    if (nbytes < 0) {
	stringstream ss;
	ss << "send() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }

    // Can send() return zero? If so, then this next line needs removal or rethinking.
    if (nbytes == 0)
	throw runtime_error("send() returned zero?!");
    
    return nbytes;
}


// -------------------------------------------------------------------------------------------------
//
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


void pin_thread_to_vcpus(const vector<int> &vcpu_list)
{
    if (vcpu_list.size() == 0)
	return;

    // I wanted to argument-check 'vcpu_list', by comparing with the number of VCPUs available.
    //
    // FIXME Calling std::thread::hardware_concurrency() doesn't seem quite right, but doing the "right"
    // thing seems nontrivial. According to 'man sched_setaffinity()':
    //
    //     "There are various ways of determining the number of CPUs available on the system, including:
    //      inspecting the contents of /proc/cpuinfo; using sysconf(3) to obtain the values of the
    //      _SC_NPROCESSORS_CONF and _SC_NPROCESSORS_ONLN parameters; and inspecting the list of CPU
    //      directories under /sys/devices/system/cpu/."
    
    int num_vcpus = std::thread::hardware_concurrency();

    cpu_set_t cs;
    CPU_ZERO(&cs);

    for (int vcpu: vcpu_list) {
	if ((vcpu < 0) || (vcpu >= num_vcpus)) {
	    stringstream ss;
	    ss << "gputils: pin_thread_to_vcpus: vcpu=" << vcpu
	       << " is out of range (num_vcpus=" << num_vcpus <<  to_string(num_vcpus) + ")";
	    throw runtime_error(ss.str());
	}
	CPU_SET(vcpu, &cs);
    }

    // Note: pthread_self() always succeeds, no need to check its return value.
    int err = pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    
    if (err) {
	// If pthread_setaffinity_np() fails, then according to its manpage,
	// it returns an error code, rather than setting 'errno'.

	stringstream ss;
	ss << "pthread_affinity_np() failed: " << strerror(err);
	throw runtime_error(ss.str());
    }
}


}  // namespace gputils
