# COL8422 Project Part 1 — Design Journal

## Project and Responsibility

- **Member:** C — Matrix Product State (MPS) backend
- **Backend class:** `MPSSimulator`
- **Primary implementation files:** `backends/mps.py`, `src/mps/mps_kernels.cpp`
- **Benchmark utility:** `benchmark_mps_qaoa_p.py`
- **Foundational reference:** G. Vidal, “Efficient Classical Simulation of Slightly Entangled Quantum Computations,” *Physical Review Letters*, vol. 91, 147902, 2003.

This journal records implementation milestones, design decisions, validation, experiments, and remaining work using the available Git commit history and recorded test results.

## 19 October 2026 — Project Baseline

The initial project structure was committed, including the supplied simulator interface, correctness harness, benchmark utilities, and placeholder backend files.

The MPS backend was initially a skeleton. The implementation work that followed focused on providing the required simulator operations while retaining the common `QuantumSimulator` interface.

**Status:** Starter baseline established; MPS implementation not yet complete.

## 21 October 2026 — MPS Gate Implementation

**Commit:** `d593b30` — `backend/gates implementation`

The first substantial MPS implementation was added to `backends/mps.py`.

### Design and implementation

- Represented the quantum state as a chain of rank-three tensors with dimensions `(left bond, physical bit, right bond)`.
- Initialized the system in the computational-basis state \(|0\rangle^{\otimes n}\), using bond dimension one.
- Added single-qubit gate matrices for the supported fixed gates and parameterized `U` gate.
- Implemented single-qubit operations using tensor–matrix contraction.
- Added two-qubit gate matrices for CNOT, CZ, SWAP, and controlled-U.
- Implemented adjacent two-site gate application by contracting tensors, applying the gate, and splitting the result using NumPy SVD.
- Added SWAP routing for gates acting on non-adjacent sites, with the routing reversed after the operation.

The SVD implementation removes singular values that are indistinguishable from zero at floating-point precision. It does not impose a fixed maximum bond dimension.

**Validation status:** The commit establishes the gate implementation. This commit alone does not establish that the complete correctness harness passed.

## 22 October 2026 — Measurement and State Recovery

**Commit:** `83df37c` — `backend/measurement and recovery state vector method added`

The backend was extended beyond gate application.

### Added functionality

- Calculated projected norms to determine computational-basis measurement probabilities.
- Implemented measurement, state collapse, and normalization of the post-measurement state.
- Implemented sampling across independent shots while preserving the original state after sampling.
- Added Pauli-string expectation-value calculation using tensor contractions.
- Added the `n_qubits` property.
- Added statevector reconstruction by contracting the MPS tensors, supporting correctness checks on small test systems.

Statevector reconstruction is intended for verification and inspection. It materializes the full \(2^n\)-amplitude vector and is therefore not representative of the memory advantage of MPS simulation during ordinary gate application.

**Validation status:** The commit documents the addition of these capabilities. The available commit summary does not independently establish a complete test result for each operation.

## 22 October 2026 — C++ Kernel and Build Integration

**Commit:** `843ff7a` — `mps_kernels and makefile added`

The initial C++ kernel implementation and project-root Makefile were introduced.

### Implementation

- Added a C ABI function for single-qubit tensor updates.
- Added a C ABI function that contracts adjacent MPS tensors and applies a two-qubit gate.
- Used contiguous complex128 tensor and gate data, with row-major indexing in C++.
- Added `src/mps/README.md` to document the kernel interface and the division of responsibilities between C++ and Python.
- Added a root Makefile target to build the MPS shared library and invoke other backend builds where their Makefiles exist.

The architecture retained NumPy SVD in Python, while moving the tensor-update loops into C++.

### Integration consideration

The initial root Makefile conditionally invokes other backend builds only when their backend-specific Makefiles exist. This behavior needs to be checked during group integration to ensure that the final `make` command builds all required backends and does not silently skip missing targets.

**Validation status:** The source and build files were committed. This milestone alone does not establish that the final integrated project builds successfully for every backend.

## 24 October 2026 — C++ Kernel Integration and Corrections

**Commit:** `d80e76c` — `Corrected C++ MPS kernels and QAOA benchmarks`

The Python MPS backend was updated to use the compiled C++ kernels through `ctypes`.

### Changes made

- Loaded the compiled MPS shared library from `src/mps/libmps_kernels.so`.
- Declared the C++ function argument types and return types.
- Changed single-qubit operations from NumPy tensor contraction to the compiled C++ kernel.
- Changed adjacent two-qubit operations to use the C++ contraction-and-gate kernel.
- Retained NumPy SVD for splitting the merged tensor into two MPS tensors.
- Added checks for nonzero C++ status codes and raised errors when kernel calls failed.
- Removed an unnecessary `#pragma once` directive from the C++ source.
- Added `benchmark_mps_qaoa_p.py` to sweep one-dimensional QAOA depths \(p=1,\ldots,5\).
- Added the corresponding local sweep results in `results/mps_qaoa_p_sweep.csv`.
- Updated `.gitignore` to exclude generated development artifacts.

The resulting design uses C++ for the tensor-update loops and Python/NumPy for higher-level simulator operations and SVD.

### Validation and limitations

The commit title records that the C++ kernels were corrected, but the available diff does not establish a specific underlying defect or justify attributing the correction to a particular bug. No more specific failure explanation is recorded here.

The benchmark utility records median runtime and Python-tracked peak memory. The latter is measured using `tracemalloc` and does not represent total process memory.

## 24 October 2026 — Local QAOA Depth Sweep

The local QAOA depth sweep covered qubit counts from 4 to 40 and depths \(p=1,\ldots,5\).

- **Output:** `results/mps_qaoa_p_sweep.csv`
- **Recorded cases:** 70
- **Recorded status:** All cases marked `OK`

### Results at 40 qubits

| Depth \(p\) | Median runtime | Python-tracked peak memory |
|---:|---:|---:|
| 1 | 0.003035 s | 0.143 MB |
| 2 | 0.005783 s | 0.154 MB |
| 3 | 0.009460 s | 0.200 MB |
| 4 | 0.016430 s | 0.374 MB |
| 5 | 0.042550 s | 1.168 MB |

The recorded measurements show increasing runtime and Python-tracked memory with increasing depth for these tested cases.

These are local development measurements, not the final graded performance results. The recorded memory values should not be interpreted as total process memory.

## 24 October 2026 — Correctness and Baseline Recording

**Commit:** `484de39` — `MPS correctness passed`

A baseline QAOA CSV was committed at `results/mps_qaoa_baseline.csv`. Its Aer timing column contains `nan`, so it does not provide a usable baseline comparison against Qiskit Aer.

The current local correctness-harness run produced the following result:

- Command: `python harness.py --backend mps`
- GHZ: 5 of 5 tests passed, at \(n=4,8,12,16,20\).
- QFT: 5 of 5 tests passed, at \(n=4,8,12,16,20\).
- BV: 5 of 5 tests passed, at \(n=4,8,12,16,20\).
- **Total: 15/15 tests passed.**
- Reported GHZ and QFT fidelity: 1.0 for all tested sizes.
- BV tests produced the expected output in all 100 shots per test.

These results establish the recorded local correctness status. The current test output does not, by itself, prove that the same full test suite was run at the time of the earlier implementation commits.

## 24 October 2026 — Qiskit Aer Comparison Experiments

Local comparisons with Qiskit Aer statevector simulation were attempted using the available benchmark tooling and a separate warmed comparison script.

The recorded comparisons showed substantial timing variability, including unusually high Aer runtimes for some sizes. The supplied benchmark also generates QAOA parameters independently for the student backend and Aer, so those measurements do not establish a controlled, parameter-matched comparison.

The separate diagnostic script used fixed parameters per qubit count, one warm-up, and five timed repetitions at \(p=2\). Its timing variability means the output remains exploratory.

### Decision

- Do not claim a definitive speedup or crossover from the existing local comparisons.
- Keep the comparison CSVs for diagnostic purposes, clearly labelled as preliminary.
- Use the course-specified JupyterHub evaluation machine for the final graded performance measurements.
- Record the final environment, methodology, scaling curves, and any failed or resource-limited cases.

## Complexity Hypothesis and Limitations

For maximum bond dimension \(\chi\), MPS storage is approximately \(O(n\chi^2)\) when bond dimensions are comparable. A single-qubit gate typically requires \(O(\chi^2)\) work, while a two-site update and SVD typically require \(O(\chi^3)\) work for uniform bond dimensions.

These costs depend on bond dimensions, which can grow with entanglement. Non-adjacent gates may introduce SWAP overhead. MPS can therefore be efficient for suitable low-entanglement circuits, but it does not guarantee efficient simulation for arbitrary circuits.

The hypothesis for the final QAOA evaluation is that the MPS representation may be advantageous while bond dimensions remain moderate. The final measurements must determine whether this expectation holds for the tested circuits and sizes.

## Remaining Work

1. Run the required MPS performance benchmark on the course-specified JupyterHub evaluation machine.
2. Collect the final scaling results and compare them with Qiskit Aer using the prescribed methodology.
3. Generate the report figures from the final accepted benchmark data.
4. Verify the integrated project build and correctness tests for all four backends with the group.
5. Incorporate this MPS contribution into the shared report and keep the final report within the specification's page limit.

## Reference

G. Vidal, “Efficient Classical Simulation of Slightly Entangled Quantum Computations,” *Physical Review Letters*, vol. 91, 147902, 2003.