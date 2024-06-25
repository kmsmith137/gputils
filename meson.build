project(
    'gputils',
    'cuda',
     default_options: ['python.install_env=auto','optimization=3','debug=false','b_ndebug=false','cpp_std=c++17']
)

header_files_toplevel = [
    'include/gputils.hpp'
]

header_files_gputils_subdir = [
    'include/gputils/Array.hpp',
    'include/gputils/Barrier.hpp',
    'include/gputils/CpuThreadPool.hpp',
    'include/gputils/CudaStreamPool.hpp',
    'include/gputils/ThreadSafeRingBuffer.hpp',
    'include/gputils/complex_type_traits.hpp',
    'include/gputils/constexpr_functions.hpp',
    'include/gputils/cuda_utils.hpp',
    'include/gputils/device_mma.hpp',
    'include/gputils/mem_utils.hpp',
    'include/gputils/memcpy_kernels.hpp',
    'include/gputils/rand_utils.hpp',
    'include/gputils/string_utils.hpp',
    'include/gputils/system_utils.hpp',
    'include/gputils/test_utils.hpp',
    'include/gputils/time_utils.hpp',
    'include/gputils/xassert.hpp',
    'include/gputils/dlpack.h',
    'include/gputils/pybind11.hpp',
    'include/gputils/pybind11_utils.hpp'
]

libgputils_source_files = [
    'src_lib/Array.cu',
    'src_lib/Barrier.cu',
    'src_lib/CpuThreadPool.cu',
    'src_lib/CudaStreamPool.cu',
    'src_lib/cuda_utils.cu',
    'src_lib/mem_utils.cu',
    'src_lib/memcpy_kernels.cu',
    'src_lib/rand_utils.cu',
    'src_lib/string_utils.cu',
    'src_lib/system_utils.cu',
    'src_lib/test_utils.cu'
]

pyext_source_files = [
    'src_pybind11/gputils_pybind11.cu',
    'src_pybind11/pybind11_utils.cu'
]

python_files_toplevel = [
    'src_python/gputils/__init__.py'
]    

local_executables = [
    'reverse-engineer-mma',
    'scratch',
    'show-devices',
    'test-array',
    'test-memcpy-kernels',
    'test-sparse-mma',
    'time-atomic-add',
    'time-fma',
    'time-l2-cache',
    'time-local-transpose',
    'time-memcpy-kernels',
    'time-shared-memory',
    'time-tensor-cores',
    'time-warp-shuffle'
]


################################################################################


cuda_args = ['-std=c++17', '-m64']
cuda_args += ['-gencode','arch=compute_80,code=sm_80']
cuda_args += ['-gencode','arch=compute_86,code=sm_86']
cuda_args += ['-gencode','arch=compute_89,code=sm_89']
link_args = ['-m64']

py3 = import('python').find_installation('python3', pure: false)
# message('py3.path = ', py3.path())
# message('py3.install_dir = ', py3.get_install_dir())

# Show sys.path for debugging
# ret = run_command(py3.path(), '-c', 'import sys; print(sys.path)', check: true)
# message('Debug: py3.sys.path = ', ret.stdout().strip())


################################################################################


pybind11_ret = run_command(py3.path(), '-m', 'pybind11', '--includes', check: true)
pybind11_ret = pybind11_ret.stdout().strip()
# message('Debug: Output of python -m pybind11 --includes: ', pybind11_ret)

pybind11_include_dirs = [ ]
foreach include_dir: pybind11_ret.split()
    include_dir = include_dir.strip().strip('-I')
    pybind11_include_dirs += include_dir
endforeach

# message('Debug: pybind11_include_dirs = ', pybind11_include_dirs)
pybind11_dep = declare_dependency(include_directories: pybind11_include_dirs)


################################################################################


numpy_incdir = run_command(py3.path(), '-c', 'import numpy; print(numpy.get_include())', check: true)
numpy_incdir = numpy_incdir.stdout().strip()

message('Debug: numpy.get_include() = ', numpy_incdir)
numpy_dep = declare_dependency(include_directories: numpy_incdir)


################################################################################


gputils_lib = library(
    'gputils',
    libgputils_source_files,
    cuda_args: cuda_args,
    link_args: link_args,
    install: true,
    install_dir: py3.get_install_dir() / 'gputils'
)

py3.extension_module(
     'gputils_pybind11',   # compiled extension will be named gputils_pybind11.so
     sources: pyext_source_files,
     dependencies : [pybind11_dep, numpy_dep],
     cuda_args: cuda_args,
     link_args: link_args,
     link_language : 'cuda',
     link_with: gputils_lib,  # link libgputils.so to gputils_pybind11.so
     override_options: ['cpp_rtti=true'],
     subdir: 'gputils',
     install: true
)

py3.install_sources(
    python_files_toplevel,
    subdir: 'gputils'
)

py3.install_sources(
    header_files_toplevel,
    subdir: 'gputils/include'
)

py3.install_sources(
    header_files_gputils_subdir,
    subdir: 'gputils/include/gputils'
)

foreach x: local_executables
    executable(
        x,
        'src_bin/' + x + '.cu',
        cuda_args: cuda_args,
	link_args: link_args,
	link_with: gputils_lib
    )
endforeach
