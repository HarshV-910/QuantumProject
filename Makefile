# COL8422 Quantum Computing Systems — Part 1
# Root Makefile: MPS, Statevector, Stabilizer, and DD backends

PYTHON ?= python3
CXX ?= g++
NVCC ?= nvcc
PIP := $(PYTHON) -m pip

CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fPIC
LDFLAGS ?= -shared

# ============================================================
# Default target
# ============================================================

.PHONY: all setup check
.PHONY: mps stabilizer statevector dd cpu cuda
.PHONY: test test-mps test-stabilizer test-statevector test-dd
.PHONY: clean clean-mps clean-stabilizer clean-statevector clean-dd
.PHONY: clean-statevector

all: mps stabilizer statevector dd

# ============================================================
# Environment setup and dependency checks
# ============================================================

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install pybind11

check:
	$(PYTHON) -c "import numpy, qiskit, qiskit_aer; print('Python dependencies OK')"

# ============================================================
# MPS backend — Member C
# ============================================================

MPS_DIR := src/mps
MPS_SRC := $(MPS_DIR)/mps_kernels.cpp
MPS_LIB := $(MPS_DIR)/libmps_kernels.so

mps: $(MPS_LIB)

$(MPS_LIB): $(MPS_SRC)
	$(CXX) $(CXXFLAGS) -shared -o $@ $<

clean-mps:
	rm -f $(MPS_LIB) $(MPS_DIR)/*.o

# ============================================================
# Stabilizer backend — Member B
# ============================================================

STAB_SRC := src/stabilizer/stabilizer.cpp
STAB_LIB := src/stabilizer/libstabilizer.so

stabilizer: $(STAB_LIB)

$(STAB_LIB): $(STAB_SRC)
	$(CXX) $(CXXFLAGS) -shared -o $@ $<

clean-stabilizer:
	rm -f $(STAB_LIB)

# ============================================================
# Statevector backend — Member A
# ============================================================

cpu:
	$(MAKE) -C src/statevector -f Makefile \
		PYTHON="$(PYTHON)" CXX="$(CXX)"

cuda:
	@if [ -f src/statevector/Makefile.cuda ] && \
	   command -v $(NVCC) >/dev/null 2>&1; then \
		$(MAKE) -C src/statevector -f Makefile.cuda \
			PYTHON="$(PYTHON)" CXX="$(CXX)" NVCC="$(NVCC)"; \
	else \
		echo "CUDA build skipped: Makefile.cuda or nvcc unavailable"; \
	fi

statevector: cpu cuda

clean-statevector:
	@if [ -f src/statevector/Makefile ]; then \
		$(MAKE) -C src/statevector -f Makefile clean; \
	fi
	@if [ -f src/statevector/Makefile.cuda ] && \
	   command -v $(NVCC) >/dev/null 2>&1; then \
		$(MAKE) -C src/statevector -f Makefile.cuda clean; \
	fi

# ============================================================
# Decision Diagram (DD) backend — Member D
# ============================================================

DD_SRC := src/dd/dd_core.cpp
DD_LIB := backends/libddcore.so
DD_CXXFLAGS ?= -O1 -std=c++17 -Wall -Wextra -fPIC

dd: $(DD_LIB)

$(DD_LIB): $(DD_SRC)
	$(CXX) $(DD_CXXFLAGS) $(LDFLAGS) -o $@ $<

clean-dd:
	rm -f $(DD_LIB)

# ============================================================
# Correctness tests
# ============================================================

test: test-mps test-stabilizer test-statevector test-dd

test-mps:
	$(PYTHON) harness.py --backend mps

test-stabilizer:
	$(PYTHON) harness.py --backend stabilizer

test-statevector:
	$(PYTHON) harness.py --backend statevector

test-dd: $(DD_LIB)
	$(PYTHON) -m unittest tests_dd.py -v
	$(PYTHON) harness.py --backend dd

# ============================================================
# Cleanup
# ============================================================

clean: clean-mps clean-stabilizer clean-statevector clean-dd
