#ifndef _GPUTILS_FILE_HPP
#define _GPUTILS_FILE_HPP

#include <string>
#include <sys/types.h>  // ssize_t

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// RAII wrapper for unix file descriptor
struct File
{
    std::string filename;
    int fd = -1;
    
    // Constructor opens file.
    // Suggest oflags = O_RDONLY for reading, and (O_WRONLY | O_CREAT | O_TRUNC) for writing.
    
    File(const std::string &filename, int oflags, int mode=0644);
    ~File();

    void write(const void *p, ssize_t nbytes);
    
    // The File class is noncopyable, but if copy semantics are needed, you can do
    //   shared_ptr<File> fp = make_shared<File> (filename, oflags, mode);
    
    File(const File &) = delete;
    File &operator=(const File &) = delete;
};


}  // namespace gputils

#endif // _GPUTILS_FILE_HPP
