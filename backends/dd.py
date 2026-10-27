# COL8422 — Decision Diagram backend
# Member D: implement DDSimulator here.
#
# Your C++ kernel goes in src/dd/.
# Expose it via pybind11 or ctypes, then call it from this class.
#
# Key concept: the DAG shares nodes for repeated sub-structure in the
# state vector. Structured circuits (BV oracles, arithmetic circuits)
# expose this regularity; random universal circuits defeat compression.
# Do NOT rename the class — the harness looks for 'DDSimulator' exactly.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'dd'))

from simulator import QuantumSimulator, UnsupportedOperationError  # noqa: F401


class DDSimulator(QuantumSimulator):

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
