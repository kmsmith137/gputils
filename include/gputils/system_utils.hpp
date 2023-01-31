#ifndef _GPUTILS_SYSTEM_UTILS_HPP
#define _GPUTILS_SYSTEM_UTILS_HPP

#include <vector>
#include <sys/mman.h>   // mlock flags

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// Error-checked (*_ec) versions of linux/posix functions, which throw verbose
// exceptions on failure.


extern void mkdir_ec(const char *path, int mode=0755);
extern void mkdir_ec(const std::string &path, int mode=0755);

extern void mlockall_ec(int flags = MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT);


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
