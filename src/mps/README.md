# MPS C++ kernels

`mps_kernels.cpp` provides two C ABI functions called by `backends/mps.py` via
Python's `ctypes`:

- `mps_apply_1q`: applies a 2x2 complex gate to a rank-3 MPS tensor.
- `mps_apply_adjacent_2q`: contracts two neighbouring MPS tensors and applies a
  4x4 complex gate, producing a merged rank-4 tensor.

The Python layer reshapes the merged tensor into a matrix, calls NumPy SVD, and
stores the resulting two tensors. Tensor and gate arrays use contiguous complex128
storage, and the C++ implementation uses row-major indexing.
