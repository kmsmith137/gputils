NVCC=nvcc -std=c++17 -arch=sm_75 -m64 -O3 --compiler-options -Wall
SHELL := /bin/bash

.DEFAULT_GOAL: all
.PHONY: all clean .FORCE

HFILES = \
  include/gputils/Array.hpp \
  include/gputils/cuda_utils.hpp \
  include/gputils/mem_utils.hpp \
  include/gputils/rand_utils.hpp \
  include/gputils/string_utils.hpp

OFILES = \
  src/Array.o \
  src/cuda_utils.o \
  src/mem_utils.o \
  src/rand_utils.o \

XFILES = \
  tests/test-array

SRCDIRS = \
  include \
  include/gputils \
  include/cuda_kernels \
  src \
  tests

all: $(XFILES)

# Not part of 'make all', needs explicit 'make source_files.txt'
source_files.txt: .FORCE
	rm -f source_files.txt
	shopt -s nullglob && for d in $(SRCDIRS); do for f in $$d/*.cu $$d/*.hpp $$d/*.cuh; do echo $$f; done; done >$@

clean:
	rm -f $(XFILES) source_files.txt *~
	shopt -s nullglob && for d in $(SRCDIRS); do rm -f $$d/*~ $$d/*.o; done

%.o: %.cu $(HFILES)
	$(NVCC) -c -o $@ $<

tests/test-array: tests/test-array.o $(OFILES)
	$(NVCC) -o $@ $^
