
CXX = g++
PYTHON = python

PYBIND_INCLUDES = $(shell $(PYTHON) -m pybind11 --includes)
PY_EXT_SUFFIX = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.pyd')")

TARGET = ../../backends/_statevector$(PY_EXT_SUFFIX)
SOURCES = statevector.cpp bindings.cpp

all: $(TARGET)

$(TARGET): $(SOURCES) statevector.hpp
	$(CXX) -O3 -shared -std=c++17 -fPIC $(PYBIND_INCLUDES) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
