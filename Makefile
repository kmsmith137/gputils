# FIXME hardcoded -arch=sm_86 here. What is best practice?
NVCC=nvcc -std=c++17 -arch=sm_86 -m64 -O3 --compiler-options -Wall,-fPIC
SHELL := /bin/bash

.DEFAULT_GOAL: all
.PHONY: all clean .FORCE

HFILES = \
  include/gputils/Array.hpp \
  include/gputils/CpuThreadPool.hpp \
  include/gputils/CudaStreamPool.hpp \
  include/gputils/complex_type_traits.hpp \
  include/gputils/constexpr_functions.hpp \
  include/gputils/cuda_utils.hpp \
  include/gputils/device_mma.hpp \
  include/gputils/mem_utils.hpp \
  include/gputils/rand_utils.hpp \
  include/gputils/string_utils.hpp \
  include/gputils/test_utils.hpp \
  include/gputils/time_utils.hpp

OFILES = \
  src/Array.o \
  src/CpuThreadPool.o \
  src/CudaStreamPool.o \
  src/cuda_utils.o \
  src/mem_utils.o \
  src/rand_utils.o \
  src/test_utils.o

LIBFILES = \
  lib/libgputils.a \
  lib/libgputils.so

XFILES = \
  benchmarks/fma \
  benchmarks/l2-cache-bandwidth \
  benchmarks/local-transpose \
  benchmarks/tensor-cores \
  benchmarks/warp-shuffle \
  loose_ends/bit-mapping \
  loose_ends/scratch \
  reverse_engineering/reverse-engineer-mma \
  tests/test-array \
  tests/test-sparse-mma \
  utils/show-devices

SRCDIRS = \
  include \
  include/gputils \
  benchmarks \
  reverse_engineering \
  src \
  tests

all: $(LIBFILES) $(XFILES)

# Not part of 'make all', needs explicit 'make source_files.txt'
source_files.txt: .FORCE
	rm -f source_files.txt
	shopt -s nullglob && for d in $(SRCDIRS); do for f in $$d/*.cu $$d/*.hpp $$d/*.cuh; do echo $$f; done; done >$@

clean:
	rm -f $(XFILES) $(LIBFILES) source_files.txt *~
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

# Special rule for procedurally generating 'device_mma.hpp'
include/gputils/device_mma.hpp: generate_device_mma_hpp.py
	python3 $^ >$@

benchmarks/fma: benchmarks/fma.o lib/libgputils.a
	$(NVCC) -o $@ $^

benchmarks/l2-cache-bandwidth: benchmarks/l2-cache-bandwidth.o lib/libgputils.a
	$(NVCC) -o $@ $^

benchmarks/local-transpose: benchmarks/local-transpose.o lib/libgputils.a
	$(NVCC) -o $@ $^

benchmarks/tensor-cores: benchmarks/tensor-cores.o lib/libgputils.a
	$(NVCC) -o $@ $^

benchmarks/warp-shuffle: benchmarks/warp-shuffle.o lib/libgputils.a
	$(NVCC) -o $@ $^

loose_ends/bit-mapping: loose_ends/bit-mapping.o lib/libgputils.a
	$(NVCC) -o $@ $^

loose_ends/scratch: loose_ends/scratch.o lib/libgputils.a
	$(NVCC) -o $@ $^

reverse_engineering/reverse-engineer-mma: reverse_engineering/reverse-engineer-mma.o lib/libgputils.a
	$(NVCC) -o $@ $^

tests/test-array: tests/test-array.o lib/libgputils.a
	$(NVCC) -o $@ $^

tests/test-sparse-mma: tests/test-sparse-mma.o lib/libgputils.a
	$(NVCC) -o $@ $^

utils/show-devices: utils/show-devices.o
	$(NVCC) -o $@ $^

INSTALL_DIR ?= /usr/local

install: $(LIBFILES)
	mkdir -p $(INSTALL_DIR)/include
	mkdir -p $(INSTALL_DIR)/lib
	cp -rv lib $(INSTALL_DIR)/
	cp -rv include $(INSTALL_DIR)/
