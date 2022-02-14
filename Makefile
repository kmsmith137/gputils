# FIXME hardcoded -arch=sm_86 here. What is best practice?
NVCC=nvcc -std=c++17 -arch=sm_86 -m64 -O3 --compiler-options -Wall,-fPIC
SHELL := /bin/bash

.DEFAULT_GOAL: all
.PHONY: all clean .FORCE

HFILES = \
  include/gputils/Array.hpp \
  include/gputils/CudaStreamPool.hpp \
  include/gputils/constexpr_functions.hpp \
  include/gputils/cuda_utils.hpp \
  include/gputils/mem_utils.hpp \
  include/gputils/rand_utils.hpp \
  include/gputils/string_utils.hpp

OFILES = \
  src/Array.o \
  src/CudaStreamPool.o \
  src/cuda_utils.o \
  src/mem_utils.o \
  src/rand_utils.o \

LIBFILES = \
  lib/libgputils.a \
  lib/libgputils.so

XFILES = \
  benchmarks/l2-cache-bandwidth \
  benchmarks/mma-int4 \
  reverse_engineering/reveng-mma-int4 \
  tests/reverse-engineer-fragments \
  tests/test-array

SRCDIRS = \
  include \
  include/gputils \
  include/cuda_kernels \
  src \
  tests

all: $(LIBFILES) $(XFILES)

# Not part of 'make all', needs explicit 'make source_files.txt'
source_files.txt: .FORCE
	rm -f source_files.txt
	shopt -s nullglob && for d in $(SRCDIRS); do for f in $$d/*.cu $$d/*.hpp $$d/*.cuh; do echo $$f; done; done >$@

clean:
	rm -f $(XFILES) $(LIBFILES) source_files.txt *~
	rmdir lib
	shopt -s nullglob && for d in $(SRCDIRS); do rm -f $$d/*~ $$d/*.o; done

%.o: %.cu $(HFILES)
	$(NVCC) -c -o $@ $<

lib/libgputils.so: $(OFILES)
	@mkdir -p lib
	rm -f $@
	$(NVCC) -shared -o $@ $^

lib/libgputils.a: $(OFILES)
	@mkdir -p lib
	rm -f $@
	ar rcs $@ $^

benchmarks/l2-cache-bandwidth: benchmarks/l2-cache-bandwidth.o lib/libgputils.a
	$(NVCC) -o $@ $^

benchmarks/mma-int4: benchmarks/mma-int4.o lib/libgputils.a
	$(NVCC) -o $@ $^

reverse_engineering/reveng-mma-int4: reverse_engineering/reveng-mma-int4.o lib/libgputils.a
	$(NVCC) -o $@ $^

tests/reverse-engineer-fragments: tests/reverse-engineer-fragments.o lib/libgputils.a
	$(NVCC) -o $@ $^

tests/test-array: tests/test-array.o lib/libgputils.a
	$(NVCC) -o $@ $^
