# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

INCLUDES :=
ALL_CCFLAGS := -m64    --threads 0 --std=c++11
ALL_LDFLAGS := -m64 

# GENCODE_FLAGS := -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
GENCODE_FLAGS := 

all: vector_mult

vector_mult.o: vector_mult.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

vector_mult: vector_mult.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: vector_mult
	./vector_mult

clean:
	rm -f vector_mult vector_mult.o
