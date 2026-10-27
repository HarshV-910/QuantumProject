# Weighted decision diagram backend — current implementation status

`backends/dd.py` is now a genuine weighted binary decision-diagram implementation, not the previous dense NumPy fallback.

## Data structure
- An edge is `(complex_weight, node_id)`.
- Each internal node stores its qubit level and low/high weighted edges.
- The terminal node represents scalar 1; a zero edge is represented by weight 0.
- `make_node` normalizes by the first nonzero child weight and interns the normalized node in a unique table.
- Gate application recurses through the DD and reconstructs only changed nodes. Single-qubit gates use recursive linear combinations of edges; two-qubit gates are decomposed into 2x2 blocks acting on the lower qubit and combine those subgraphs.
- Measurement, sampling, and Pauli expectation values operate on the DD representation without expanding it to a dense vector.
- `statevector` is a debug-only compatibility property for the provided harness. Do not use it in performance measurements or claim it is part of the common API.

## Tests
Run:
```bash
python -m unittest tests_dd.py
python harness.py --backend dd
```

## Important remaining project requirement
The course specification explicitly requires C++ backend kernels and a Makefile that builds with plain `make`. This implementation is a DD algorithm in Python and is a meaningful correctness/algorithm milestone, but it is not yet fully compliant with that language/build requirement. The next engineering step is to port the node manager, unique table, canonicalization, recursive addition, and recursive gate kernels to C++, then expose the C++ core to Python (for example through pybind11 or a small C ABI/ctypes wrapper). Keep the API in `backends/dd.py` unchanged while replacing its internal manager.

## Complexity and interpretation
The DD's storage and runtime depend on the number of unique normalized nodes reached by the circuit, not directly on `2**n`. For BV, the current implementation's small observed runtime growth is consistent with exploitable structure. These are local measurements and must be compared against Qiskit Aer on the same machine with the same workload before claiming a crossover.
