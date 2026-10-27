CXX ?= g++
CXXFLAGS ?= -O1 -std=c++17 -Wall -Wextra -fPIC
LDFLAGS ?= -shared

.PHONY: all clean test
all: backends/libddcore.so

backends/libddcore.so: src/dd/dd_core.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@

clean:
	rm -f backends/libddcore.so

test: all
	python -m unittest tests_dd.py -v
	python harness.py --backend dd
