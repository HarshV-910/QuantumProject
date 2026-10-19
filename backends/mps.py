# COL8422 — MPS / Tensor Network backend
# Member C: implement MPSSimulator here.
#
# Your C++ kernel goes in src/mps/.
# Expose it via pybind11 or ctypes, then call it from this class.
#
# Key concept: bond dimension grows with entanglement. SVD truncation is
# the main approximation knob. Low-entanglement circuits (shallow QAOA 1D)
# are where this backend should outperform the statevector baseline.
# Do NOT rename the class — the harness looks for 'MPSSimulator' exactly.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'mps'))

from simulator import QuantumSimulator, UnsupportedOperationError  # noqa: F401


class MPSSimulator(QuantumSimulator):

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
