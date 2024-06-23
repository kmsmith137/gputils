import numpy
import cupy

# FIXME this makes dir(gputils) look weird, since it consists entirely of ad hoc functions
# for testing. I'll probably clean this up when there's more python functionality in gputils.

from .gputils_pybind11 import *


def launch_busy_wait_kernel(arr, a40_seconds):
    """
    Launches a "busy wait" kernel with one threadblock and 32 threads.
    Useful for testing stream/device synchronization.

    The 'arr' argument is a caller-allocated length-32 uint32 array.
    The 'a40_seconds' arg determines the amount of work done by the kernel,
    normalized to "seconds on an NVIDIA A40".
    """
    
    gputils_pybind11._launch_busy_wait_kernel(arr, a40_seconds, cupy.cuda.get_current_stream().ptr)
