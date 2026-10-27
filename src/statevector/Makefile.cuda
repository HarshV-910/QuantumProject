CXX ?= g++
NVCC ?= nvcc
PYTHON ?= python3
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_75

PYBIND_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")

TARGET := ../../backends/_statevector_cuda$(EXT_SUFFIX)
OBJ := statevector_cuda.o

.PHONY: all clean

all: $(TARGET)

$(OBJ): statevector_cuda.cu statevector_cuda.hpp
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) \
		-Xcompiler=-fPIC -I$(CUDA_HOME)/include \
		-c statevector_cuda.cu -o $(OBJ)

$(TARGET): $(OBJ) bindings_cuda.cpp statevector_cuda.hpp
	$(CXX) -O3 -std=c++17 -shared -fPIC \
		$(PYBIND_INCLUDES) -I$(CUDA_HOME)/include \
		bindings_cuda.cpp $(OBJ) \
		-L$(CUDA_HOME)/lib64 -lcudart -o $(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)
