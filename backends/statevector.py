# COL8422 — Statevector backend
# Member A: implement StatevectorSimulator here.
#
# Your C++ kernel goes in src/statevector/.
# Expose it via pybind11 or ctypes, then call it from this class.
#
# Do NOT rename the class — the harness looks for 'StatevectorSimulator' exactly.

import sys
from pathlib import Path

# Adjust this path if your compiled extension lands elsewhere
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'statevector'))

from simulator import QuantumSimulator, UnsupportedOperationError  # noqa: F401


class StatevectorSimulator(QuantumSimulator):

    def initialize(self, n: int) -> None:
        raise NotImplementedError

    def apply_1q_gate(self, gate: str, qubit: int, **params) -> None:
        raise NotImplementedError

    def apply_2q_gate(self, gate: str, ctrl: int, tgt: int, **params) -> None:
        raise NotImplementedError

    def measure(self, qubit: int) -> int:
        raise NotImplementedError

    def sample(self, shots: int) -> dict:
        raise NotImplementedError

    def expectation(self, pauli_string: str) -> float:
        raise NotImplementedError

    @property
    def n_qubits(self) -> int:
        raise NotImplementedError
