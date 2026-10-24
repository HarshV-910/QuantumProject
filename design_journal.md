# COL8422 Project Part 1 — Design Journal

## Project and responsibility
- **Member:** C — Matrix Product State (MPS) backend
- **Backend class:** `MPSSimulator`
- **Foundational reference:** Vidal, “Efficient Classical Simulation of Slightly Entangled Quantum Computations” (2003). Add the exact citation format requested by the course.

## 2026-10-21 — Correctness baseline and C++ kernel integration

### Goal
Build a working MPS backend behind the supplied `QuantumSimulator` interface and pass the instructor-provided correctness harness.

### Representation
Each site is stored as a rank-3 complex tensor with dimensions `(left bond, physical bit, right bond)`. The physical index has dimension 2. Bond dimensions represent the Schmidt-rank capacity across cuts in the chain.

### Gate application
- A one-qubit gate contracts its 2×2 matrix with the physical index of one tensor.
- A two-qubit gate is moved to adjacent sites using SWAP operations where needed.
- The two-site tensors are merged, the gate is applied, and NumPy SVD splits the merged tensor back into two sites.
- The C++ kernels perform the tensor-update loops; NumPy performs SVD.

### Correctness checks
- Command: `python3 harness.py --backend mps`
- Record actual result: **15/15 tests passed** (GHZ, QFT, BV at n = 4, 8, 12, 16, 20).
- Do not claim performance success from correctness tests alone.

### Known limitations / next steps
- Benchmark QAOA 1D for depths p = 1..5 and increasing n.
- Run the supplied `benchmark.py` on the course evaluation machine and retain its CSV/plot.
- Compare with Qiskit Aer when available.
- Document bond-dimension growth and discuss the limits of MPS for high-entanglement circuits.

## MPS QAOA Depth Sweep — 24 October 2026

### Experimental setup

- Backend: Matrix Product State (MPS)
- Circuit: shallow 1D QAOA
- Qubit counts: 4–40
- QAOA depths: p = 1, 2, 3, 4, 5
- Measurement: median runtime reported by the benchmark helper
- Output: `results/mps_qaoa_p_sweep.csv`
- Status: all reported benchmark cases completed successfully

### Results at 40 qubits

| QAOA depth | Median runtime | Python-tracked peak memory |
|---:|---:|---:|
| 1 | 0.003035 s | 0.143 MB |
| 2 | 0.005783 s | 0.154 MB |
| 3 | 0.009460 s | 0.200 MB |
| 4 | 0.016430 s | 0.374 MB |
| 5 | 0.042550 s | 1.168 MB |

### Observations

1. Runtime increases with QAOA depth.
2. The largest tested case, n = 40 and p = 5, completed in approximately 42.55 ms on the current machine.
3. The Python-tracked memory metric also increased with circuit depth.
4. These measurements are preliminary results from the current development environment.
5. The memory metric comes from Python `tracemalloc`; it does not represent total process memory.
6. A Qiskit Aer comparison has not yet been completed, so no relative speedup is claimed.
7. Final graded performance measurements must be collected on the course evaluation machine.

### Next steps

- Compare against Qiskit Aer statevector simulation in the evaluation environment.
- Collect total process-memory measurements if required.
- Preserve the CSV output and record the hardware/software environment for reproducibility.
