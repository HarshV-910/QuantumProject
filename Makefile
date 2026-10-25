
PYTHON ?= python3
PIP := $(PYTHON) -m pip
NVCC ?= nvcc
CXX ?= g++
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_75

BACKENDS := stabilizer mps dd

export CUDA_HOME
export PATH := $(CUDA_HOME)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_HOME)/lib64:$(CUDA_HOME)/targets/x86_64-linux/lib:$(LD_LIBRARY_PATH)

.PHONY: all setup check check-python check-cuda cpu cuda statevector other-backends test test-cpu clean

# Install dependencies, then build all available backends.
all: setup check cpu cuda other-backends

# Install dependencies using the selected Python interpreter.
setup:
	$(PIP) install numpy pybind11 qiskit qiskit-aer

check: check-python

check-python:
	$(PYTHON) -c "import numpy, pybind11, qiskit, qiskit_aer; print('Python dependencies OK')"

# Report CUDA compiler availability without making CPU-only systems fail.
check-cuda:
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(NVCC) --version; \
	else \
		echo "WARNING: nvcc not found; CUDA build will be skipped"; \
	fi

# Build CPU extension for the active Python interpreter.
cpu:
	$(MAKE) -C src/statevector -f Makefile PYTHON="$(PYTHON)" CXX="$(CXX)"

# Build CUDA extension only when nvcc is available.
cuda:
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		echo "Building CUDA backend"; \
		$(MAKE) -C src/statevector -f Makefile.cuda \
			PYTHON="$(PYTHON)" CXX="$(CXX)" NVCC="$(NVCC)" \
			CUDA_HOME="$(CUDA_HOME)" CUDA_ARCH="$(CUDA_ARCH)"; \
	else \
		echo "Skipping CUDA backend: nvcc is unavailable"; \
	fi

# Build both CPU and CUDA implementations where supported.
statevector: cpu cuda

# Build teammate backends where their Makefiles exist.
other-backends:
	@set -eu; \
	for backend in $(BACKENDS); do \
		dir="src/$$backend"; \
		if [ ! -d "$$dir" ]; then \
			echo "ERROR: missing $$dir"; exit 1; \
		fi; \
		if [ -f "$$dir/Makefile" ]; then \
			$(MAKE) -C "$$dir" PYTHON="$(PYTHON)" \
				CXX="$(CXX)" NVCC="$(NVCC)"; \
		else \
			has_sources=0; \
			for ext in cpp cc cu; do \
				if find "$$dir" -type f -name "*.$$ext" -print -quit | grep -q .; then \
					has_sources=1; \
				fi; \
			done; \
			if [ "$$has_sources" -eq 1 ]; then \
				echo "ERROR: $$dir has native sources but no Makefile"; \
				exit 1; \
			else \
				echo "NOTE: $$dir has no native sources to compile"; \
			fi; \
		fi; \
	done

# Run correctness tests against the CPU implementation.
test: test-cpu

test-cpu:
	STATEVECTOR_IMPL=cpu $(PYTHON) harness.py --backend statevector --n-max 20

# Remove generated build artifacts.
clean:
	$(MAKE) -C src/statevector -f Makefile clean
	$(MAKE) -C src/statevector -f Makefile.cuda clean
	@set -eu; \
	for backend in $(BACKENDS); do \
		if [ -f "src/$$backend/Makefile" ]; then \
			$(MAKE) -C "src/$$backend" clean; \
		fi; \
	done
