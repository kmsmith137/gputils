#include <cassert>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>

#include "../include/gputils/File.hpp"

using namespace std;


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif

    
File::File(const string &filename_, int oflags, int mode)
    : filename(filename_)
{
    fd = open(filename.c_str(), oflags, mode);
    
    if (fd < 0) {
	// FIXME exception text should show 'oflags' and 'mode'.
	stringstream ss;
	ss << filename << ": open() failed: " << strerror(errno);
	throw runtime_error(ss.str());
    }
}

File::~File()
{
    if (fd >= 0) {
	close(fd);
	fd = -1;
    }
}


void File::write(const void *p, ssize_t nbytes)
{
    if (nbytes == 0)
	return;
    
    assert(p != nullptr);
    assert(nbytes > 0);
    assert(fd >= 0);

    // C++ doesn't alllow '+=' on a (const void *).
    const char *pc = reinterpret_cast<const char *> (p);
	
    while (nbytes > 0) {
	ssize_t n = ::write(fd, pc, nbytes);
	
	if (n < 0) {
	    stringstream ss;
	    ss << filename << ": write() failed: " << strerror(errno);
	    throw runtime_error(ss.str());
	}
	
	if (n == 0) {
	    // Just being paranoid -- I don't think this can actually happen.
	    stringstream ss;
	    ss << filename << ": write() returned zero?!";
	    throw runtime_error(ss.str());
	}
	
	pc += n;
	nbytes -= n;
    }
}


} // namespace gputils
