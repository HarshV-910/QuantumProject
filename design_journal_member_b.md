# Design Journal: Stabilizer / Tableau Backend

**Name:** Gaurav  **Entry number:** 2026MCS2245
**Course:** COL8422 Quantum Computing Systems, Project Part 1 (Member B)
**Foundational paper:** Aaronson & Gottesman, *Improved Simulation of Stabilizer Circuits*, PRA 2004
**Files:** `backends/stabilizer.py`, `src/stabilizer/stabilizer.cpp`, `tests/test_stabilizer.py`, `scripts/bench_stabilizer_clifford.py`, the `stabilizer` target in the `Makefile`

**LLM usage:** I used Claude (Anthropic's AI assistant) to explain the paper, to draft the implementation, tests and benchmark script, and to review the code against the project spec. I worked through the algorithm by hand (tableau updates for H, S and CNOT, and both measurement cases on a Bell pair). I built and ran all tests on my own machine and made the design decisions recorded below.

---

## Week of Oct 19 – 25: algorithm and first implementation

**Oct 23: understanding the paper**
- The state is a tableau of 2n+1 rows: n destabilizers, n stabilizers and one scratch row. Each row is a Pauli string stored as x bits, z bits and a sign bit. That's about 4n² bits instead of 2ⁿ amplitudes.
- A gate is an O(n) update of every row. Measurement is O(n²): destabilizer i anticommutes only with stabilizer i, so the stabilizers that multiply to ±Z_a can be read directly from the x_ia bits, with no O(n³) Gaussian elimination.
- The paper's page-4 description of the bit encoding contradicts its initial tableau and Section V. I used (x,z) = (1,0) X, (1,1) Y, (0,1) Z.

**Oct 23: design decisions**
- **Two kernels with the same methods.** A NumPy reference written first to check correctness, and a bit-packed C++ kernel as the real implementation.
- **Row-major layout, 64 qubits per `uint64_t`.** Rowsum, the cost driver in measurement, becomes word-wise XOR plus `popcount` for the phase. Column-major (as in Stim) would speed up gates but slow down rowsum.
- **ctypes instead of pybind11.** Only `g++` is needed on the evaluation machine. Functions are exported with `extern "C"`.
- **Randomness comes from Python.** Measurement coins come from a single seeded NumPy generator, so runs are reproducible and both kernels agree.
- **Extra gates as direct tableau rules.** X, Y, Z, CZ and SWAP. T, U and CU raise `UnsupportedOperationError`; unknown names raise `ValueError`.

**Oct 23: sampling and expectation values**
- *Dead end:* measuring a copy once per shot costs O(shots · n³).
- *Replacement:* the outcome distribution is uniform over v₀ ⊕ rowspace(stabilizer X-block). One measurement pass gives v₀, GF(2) elimination gives a basis of k rows, and each shot is v₀ XOR a random combination of them. Cost: O(n³) once, plus O(shots · k · n).
- `expectation(P)` is exact. It returns 0 if P anticommutes with any stabilizer; otherwise it returns ±1, the sign of the product of stabilizers that equals P, built in the scratch row.

**Oct 23: verification**
- Compared against an independent dense statevector on 300 random Clifford circuits (n ≤ 5) with mid-circuit measurements: all 4ⁿ Pauli expectations (92,232 checks), measurement probabilities and sample statistics. All passed.
- The C++ and NumPy tableaux are bit-identical at n = 63, 64, 65, 130 and 200, across the 64-bit word boundaries.
- Harness: 10/10 (QFT is skipped for this backend).
- Reproduced the paper's Figure 2: at β = 1.2, time per measurement roughly quadruples when n doubles (0.014 → 0.043 → 0.18 ms for n = 400 → 800 → 1600); at β = 0.6 it grows roughly linearly.

---

## Week of Oct 26 – Nov 1: spec compliance, testing and benchmarking

**Oct 26: changes after checking the project spec**
- **Removed the silent fallback.** My first version switched to NumPy quietly if the C++ library was missing. The spec makes C/C++ kernels mandatory and forbids silent fallbacks. A missing library now raises `ImportError` ("run make"), which the harness reports cleanly. NumPy runs only on request (`STAB_KERNEL=numpy`), as a test oracle.
- **Moved sampling into C++.** The elimination and sample generation now happen in `Tableau::sample`. Python only supplies random bits and counts the results.
- **Root Makefile.** Added `all` and `clean` targets, so a plain `make` builds every backend.
- **Decided not to support T gates.** §5.4 assigns near-Clifford circuits with k T gates to this backend, but §4 requires non-Clifford gates on the stabilizer backend to raise `UnsupportedOperationError`. I followed §4. An extension (paper Sec. VII-C) would cost about O(16ᵏ · n) and would be a large amount of extra code. As a result, `benchmark.py --circuit near_clifford`, which always inserts T gates, cannot run. My assigned-class curve uses k = 0.

**Oct 26: unit tests (§8.1)**
- `tests/test_stabilizer.py` has 9 tests: GHZ for n = 2…20; QFT correctly rejected; random Clifford circuits against the dense reference at n ≤ 5 (all Paulis) and n = 20; `near_clifford` with k = 0 matching the reference and k = 5, 10, 20 raising; BV; C++/NumPy agreement; error handling.
- *Dead end:* the first version took about 2 minutes. After trimming the n = 20 checks it runs in about 13 s.
- Verified with Python 3.10, NumPy 1.26 and GCC 12, the evaluation toolchain: tests 9/9, harness 10/10, no compiler warnings.

**Oct 26: benchmarking (preliminary run, development machine)**
- `benchmark.py` cannot produce a k = 0 curve, so I wrote `scripts/bench_stabilizer_clifford.py`. It takes the gate list from the course's own `near_clifford(t_count=0)` and times both my backend and Qiskit Aer statevector.
- `benchmark.py` records a tiny Qiskit time when Qiskit runs out of memory (it times a failed run). My script checks `result.success` and marks those rows as failed instead.

| n | Gates | Stabilizer (s) | Qiskit Aer SV (s) |
|---:|---:|---:|---:|
| 4 | 100 | 0.00013 | 0.0036 |
| 20 | 1,680 | 0.0018 | 0.24 |
| 24 | 2,400 | 0.0034 | 4.9 |
| 1,453 | 8,450,648 | 52 | out of memory |

- **Crossover:** at the smallest n. Total time grows as O(n³), because there are about 4n² gates at O(n) each. On BV, a run takes about 0.1 ms for every n up to 40.

**Oct 26 – 27: building on my own machine**
- `make` is not available in Windows PowerShell, so I set up WSL (Ubuntu) with `build-essential` and a Python virtual environment. My backend builds there, the unit tests pass 9/9 and the harness passes 10/10.
- Git warned "LF will be replaced by CRLF" because Git for Windows converts line endings (`core.autocrlf`). This does not affect the code. The repository stores LF, which is what the Linux evaluation machine uses.

**Complexity prediction (§8.5)**
On Clifford circuits, the tableau beats Qiskit Aer statevector from the smallest n:
- **Per gate:** Θ(n) against Θ(2ⁿ).
- **Memory:** about 4n² bits against 16 · 2ⁿ bytes, so Qiskit cannot go beyond n ≈ 31 within 48 GB.
- **Maximum n:** on `near_clifford` with k = 0 the total time is Θ(n³), giving a maximum n of a few thousand within 120 s.
- **Where it loses:** any T, U or CU gate is refused. With the Sec. VII-C extension, the cost would cross Qiskit's 2ⁿ at about k ≈ n/4.
