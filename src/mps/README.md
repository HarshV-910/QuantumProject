# Matrix Product State (MPS) Backend

## 1. Overview

This directory contains the C++ implementation of the Matrix Product State (MPS) backend for the COL8422 Quantum Computing Systems project.

The MPS backend represents a quantum state using a chain of tensors instead of storing the complete state vector explicitly. This representation can reduce memory requirements for quantum circuits whose states have relatively low entanglement.

The implementation uses Python for the simulator interface and tensor operations, with C++ kernels for selected performance-critical operations.

### Main objectives

- Implement quantum-state simulation using an MPS representation.
- Support single-qubit and two-qubit quantum gates.
- Support gates acting on non-adjacent qubits.
- Support measurement and sampling operations.
- Validate correctness against the project's provided test circuits.
- Benchmark the backend on the 1D QAOA circuit.

## 2. File Structure

```text
part1/
├── backends/
│   └── mps.py
├── src/
│   └── mps/
│       ├── mps_kernels.cpp
│       ├── libmps_kernels.so
│       └── README.md
├── harness.py
├── benchmark.py
├── circuits.py
├── simulator.py
└── Makefile
```

### `backends/mps.py` — Python MPS backend

This file implements the MPS simulator and connects it to the project's common simulator interface.

Its responsibilities include:

- Representing and updating the quantum state using MPS tensors.
- Applying supported single-qubit gates.
- Applying two-qubit gates by combining the relevant tensors and decomposing the result using singular value decomposition (SVD).
- Handling non-adjacent two-qubit operations through SWAP-based routing.
- Supporting measurement and sampling.
- Providing state-related functionality required by the simulator and correctness tests.

The Python implementation uses NumPy for numerical operations and `ctypes` to call the compiled C++ library.

### `src/mps/mps_kernels.cpp` — C++ kernels

This file contains the C++ implementation of selected computational kernels.

The kernels support operations on MPS tensor data, including single-qubit updates and adjacent two-qubit gate application.

The goal is to move selected performance-critical operations into compiled C++ code while keeping the simulator interface and other tensor operations in Python.

### `src/mps/libmps_kernels.so` — Compiled library

This is the shared library generated from `mps_kernels.cpp`.

The Python backend loads the library at runtime through `ctypes`.

**This file is a build artifact:** it should be regenerated from the C++ source when necessary rather than treated as the source implementation.

### Root `Makefile` — Build entry point

The root Makefile contains the build rule for the MPS C++ library.

Run the build command from the project's root directory:

```bash
make mps
```

To force recompilation of the MPS library:

```bash
make -B mps
```

### `harness.py` — Correctness testing

The project's correctness harness tests the MPS backend using the supplied test circuits.

Run:

```bash
python harness.py --backend mps
```

The tests cover GHZ, QFT, and Bernstein–Vazirani (BV) circuits at several qubit counts.

### `benchmark.py` — Performance evaluation

The project's benchmark script evaluates backend performance on supported circuits.

The MPS QAOA benchmark command is:

```bash
python benchmark.py --backend mps --circuit qaoa_1d --n-max 40 --plot
```

This evaluates the 1D QAOA circuit over the benchmark's configured range up to 40 qubits, subject to its execution budget and simulator limitations.

### Other project files

- `circuits.py`: Defines the circuits used by the project.
- `simulator.py`: Provides the project's common simulation interface and orchestration.
- `requirements.txt`: Lists the Python dependencies required by the project.

These files are part of the shared project infrastructure and are not specific to the MPS implementation.

## 3. Implementation Approach

### 3.1 MPS representation

An MPS represents a multi-qubit quantum state as a sequence of connected tensors. The dimensions connecting adjacent tensors are called bond dimensions.

Unlike a full statevector representation, which stores \(2^n\) complex amplitudes for \(n\) qubits, an MPS stores tensors whose sizes depend on the bond dimensions.

For suitable quantum states and circuits, this can substantially reduce memory requirements.

### 3.2 Single-qubit gates

Single-qubit operations update the tensor associated with the target qubit. The implementation uses the Python backend and the C++ kernels for the corresponding tensor operations.

### 3.3 Two-qubit gates

For supported two-qubit operations, the implementation combines the relevant tensor data, applies the gate, and uses SVD to decompose the resulting tensor into MPS form.

The SVD step determines the resulting bond dimension. The implementation uses a numerical cutoff to control the treatment of small singular values.

### 3.4 Non-adjacent qubits

MPS tensors are arranged in a chain. When a two-qubit gate acts on non-adjacent qubits, the implementation uses SWAP-based routing to bring the target qubits together, applies the gate, and handles the required qubit ordering.

### 3.5 Measurement and sampling

The backend supports measurement and sampling operations required by the project's simulator interface and correctness tests.

### 3.6 Statevector reconstruction

The backend also provides statevector-related functionality for operations such as correctness validation. Reconstructing a complete statevector requires exponential memory in the number of qubits, so this functionality should not be confused with the memory-efficient MPS representation itself.

## 4. Build and Run Instructions

Run these commands from the project root directory.

### Step 1: Activate the Python environment

```bash
cd "$HOME/Downloads/QT project/part1"
source .venv/bin/activate
```

If the virtual environment has already been activated, this step can be skipped.

### Step 2: Build the MPS backend

```bash
make -B mps
```

The `-B` option forces the MPS build rule to run.

### Step 3: Run correctness tests

```bash
python harness.py --backend mps
```

Check that the GHZ, QFT, and BV tests pass.

### Step 4: Run the QAOA benchmark

```bash
python benchmark.py --backend mps --circuit qaoa_1d --n-max 40 --plot
```

Inspect the generated benchmark results and plot. The benchmark may also attempt a Qiskit comparison, and some comparison runs may fail because of memory limits.

## 5. Local Validation Results

In the local development environment, the MPS backend passed all 15 correctness tests in the provided harness.

| Circuit | Tested qubit counts | Result |
|---|---|---|
| GHZ | 4, 8, 12, 16, 20 | 5/5 passed |
| QFT | 4, 8, 12, 16, 20 | 5/5 passed |
| BV | 4, 8, 12, 16, 20 | 5/5 passed |
| **Total** | | **15/15 passed** |

The QFT and GHZ tests reported fidelity 1.0 at each tested size. The BV tests produced the expected bit string in all 100 shots for each tested size.

The local QAOA benchmark also ran through 40 qubits in the displayed output. Some Qiskit statevector comparison runs reported insufficient memory.

These are local development results, not final graded performance measurements. The course guide specifies that final performance results must be generated on the course evaluation JupyterHub.

## 6. Limitations and Considerations

- MPS efficiency depends on the circuit's entanglement and the bond dimensions required to represent its state accurately.
- SVD and tensor operations introduce computational overhead.
- SWAP-based routing adds operations for gates acting on non-adjacent qubits.
- Reconstructing the full statevector can require exponential memory.
- Local benchmark timings depend on the development machine and should not be directly treated as official evaluation results.
- The compiled shared library must be available and compatible with the local runtime.

## 7. Summary

The MPS contribution consists of a Python simulator backend, C++ computational kernels, and a build rule for generating the shared library. The implementation is validated locally using the project's GHZ, QFT, and BV correctness tests and can be benchmarked using the supplied 1D QAOA benchmark command.

Final performance evaluation and submission should follow the course project's instructions and use the designated evaluation environment.