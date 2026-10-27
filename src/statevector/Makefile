CXX ?= g++
PYTHON ?= python3

PYBIND_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")
PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

TARGET := ../../backends/_statevector$(EXT_SUFFIX)
SOURCES := statevector.cpp bindings.cpp
HEADERS := statevector.hpp

CXXFLAGS ?= -O3
CXXFLAGS += -shared -std=c++17 -fPIC

.PHONY: all clean FORCE

all: $(TARGET)

FORCE:

$(TARGET): $(SOURCES) $(HEADERS) FORCE
	@mkdir -p ../../backends
	@echo "Building CPU backend with Python $(PYTHON_VERSION)"
	$(CXX) $(CXXFLAGS) $(PYBIND_INCLUDES) $(SOURCES) -o $(TARGET)

clean:
	rm -f ../../backends/_statevector*.so
