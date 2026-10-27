# COL8422 Design Journal — Member A: CUDA Statevector Backend

**Student:** Chaitanya Bhardwaj  
**Course:** COL8422 — Quantum Computing Systems and Architecture  
**Contribution:** Native C++/CUDA statevector simulator and Python integration

## Entry 1 — Statevector backend and Python interface

**Goal:** Implement the statevector backend behind the project simulator interface.

- Implemented a Python adapter in `backends/statevector.py` that delegates state operations to a native extension.
- The interface supports initialization, single-qubit gates (`H`, `X`, `Y`, `Z`, `S`, `T`, `U`), two-qubit gates (`CNOT`, `CZ`, `SWAP`, `CU`), measurement, sampling, and Pauli-string expectation values.
- Added validation for initialization, qubit indices, supported operations, and parameterized gates.
- The state is exposed through the `statevector` property. The native object owns the underlying state rather than a separate Python `_state` array.

**Decision and rationale:** Kept the required Python interface while performing statevector operations in native code.

## Entry 2 — CPU/CUDA backend selection and integration debugging

**Problem:** The Python adapter initially imported the CPU extension directly, which meant setting `STATEVECTOR_IMPL=cuda` would not necessarily select the CUDA implementation.

**Action:** Updated the adapter to select the CUDA extension when `STATEVECTOR_IMPL=cuda` and retain CPU-extension selection for comparison.

**Lesson:** Benchmark results are meaningful only when the intended backend is actually loaded. Backend selection should be explicit and checked before interpreting timing data.

## Entry 3 — Updating local tests to the public interface

**Problem:** The local tests accessed `sim._state`, but the current adapter does not expose that private field. The state is managed by the native object and made available through `sim.statevector`.

**Action:** Updated the tests to use `sim.statevector`; the sampling test copies that property before sampling and checks that sampling does not change the state.

**Result:** The updated Member A basic tests passed, including initialization, H, X, CNOT, expectation values, and sampling.

**Decision and rationale:** Adapted the tests to the simulator’s public interface rather than adding a redundant `_state` attribute solely to satisfy outdated tests.

## Entry 4 — Direct comparison with Qiskit Aer

**Goal:** Verify that the student simulator and Qiskit Aer execute the same seeded random-universal circuit.

- Built matching circuits using the same seed, angle-generation order, `5 * n` layers, and alternating brick-wall CNOT pattern.
- Retrieved the statevectors from the student backend and Qiskit Aer.
- Compared states using fidelity:

  `F = |<psi_Aer | psi_student>|^2`

- The four-qubit comparison returned fidelity `1.000000000000`.

**Why fidelity:** Statevector fidelity is insensitive to an overall global phase, so it is a suitable comparison metric for quantum states.

**Follow-up:** Additional random-circuit checks were run at 8, 12, and 16 qubits with a required threshold of `0.999`. Retain the actual notebook output as supporting evidence rather than copying unrecorded values into this journal.

## Entry 5 — Course correctness harness

**Command used:**

```bash
STATEVECTOR_IMPL=cuda python harness.py --backend statevector --n-max 20
```

**Observed result:** The course correctness harness passed all 15 cases.

| Circuit | Tested qubit counts | Result |
|---|---|---|
| GHZ | 4, 8, 12, 16, 20 | 5/5 passed; reported fidelity 1.000000 |
| QFT | 4, 8, 12, 16, 20 | 5/5 passed; reported fidelity 1.000000 |
| BV | 4, 8, 12, 16, 20 | 5/5 passed; expected bitstring observed in 100 samples per case |
| **Total** | **15 cases** | **15/15 passed** |

**Conclusion:** The statevector backend passed the required harness cases for the three listed circuit families through 20 qubits. This is evidence for those tests, not a proof of correctness for every possible circuit.

## Entry 6 — CUDA versus Qiskit Aer performance experiment

**Goal:** Measure end-to-end runtime for seeded random-universal circuits.

**Method recorded in `compare_aer.py`:**

- Qubit counts: 4, 6, 8, 10, 12, 14, 16, 18, and 20.
- Circuit depth: `5 * n` layers.
- Aer receives an initial execution before measured repetitions.
- Three repetitions are timed and the median is reported.
- Student timing includes simulator construction, initialization, and circuit execution. Aer timing includes backend execution and result construction.
- These are script-level timings, not isolated native gate-kernel timings.

### Recorded Colab measurements

| Qubits | Student CUDA (s) | Qiskit Aer (s) | Aer speedup |
|---:|---:|---:|---:|
| 4 | 0.005099 | 0.005614 | 1.10× |
| 6 | 0.009977 | 0.012318 | 1.23× |
| 8 | 0.017388 | 0.018589 | 1.07× |
| 10 | 0.028008 | 0.031544 | 1.13× |
| 12 | 0.026694 | 0.042959 | 1.61× |
| 14 | 0.040073 | 0.141774 | 3.54× |
| 16 | 0.077582 | 0.375652 | 4.84× |
| 18 | 0.168901 | 1.735397 | 10.27× |
| 20 | 0.539250 | 8.484346 | 15.73× |

Speedup is `Aer runtime / student runtime`. In this recorded run, the student backend was faster at all tested sizes; the largest measured speedup was approximately `15.73×` at 20 qubits.

**Interpretation and limitation:** The smallest measured size was four qubits, and the student backend was already faster there. These measurements therefore do not identify an exact crossover point below four qubits. Earlier runs produced different timings, so runtime values are dependent on the execution environment and run conditions.

**Important:** These measurements were collected in Colab. They must be labelled preliminary, not official evaluation results. The project’s specified evaluation environment is an NVIDIA A40 GPU with CUDA 11.5. Final performance claims require measurements on that environment.

## Entry 7 — Complexity analysis and report preparation

For an `n`-qubit statevector, the simulator stores `2^n` complex amplitudes. With complex128 values, the statevector alone requires approximately `16 * 2^n` bytes; at 20 qubits this is 16 MiB, excluding temporary buffers and runtime overhead.

- Statevector storage: `O(2^n)`.
- One-qubit gate application: `O(2^n)`.
- Two-qubit gate application: `O(2^n)`.
- Circuit with `G` gates: `O(G * 2^n)` time under the standard statevector model.
