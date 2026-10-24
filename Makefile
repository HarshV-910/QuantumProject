CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fPIC
MPS_DIR := src/mps
MPS_SRC := $(MPS_DIR)/mps_kernels.cpp
MPS_LIB := $(MPS_DIR)/libmps_kernels.so
OTHER_BACKENDS := statevector stabilizer dd

.PHONY: all mps other-backends clean
all: mps other-backends

mps: $(MPS_LIB)

$(MPS_LIB): $(MPS_SRC)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(MPS_SRC)

other-backends:
	@for backend in $(OTHER_BACKENDS); do \
	  if [ -f src/$$backend/Makefile ]; then \
	    echo "Building $$backend backend..."; \
	    $(MAKE) -C src/$$backend; \
	  fi; \
	done

clean:
	rm -f $(MPS_LIB) $(MPS_DIR)/*.o
	@for backend in $(OTHER_BACKENDS); do \
	  if [ -f src/$$backend/Makefile ]; then \
	    $(MAKE) -C src/$$backend clean || exit $$?; \
	  fi; \
	done
