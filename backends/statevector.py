# COL8422 — Statevector backend
# Member A: implement StatevectorSimulator here.
#
# Your C++ kernel goes in src/statevector/.
# Expose it via pybind11 or ctypes, then call it from this class.
#
# Do NOT rename the class — the harness looks for 'StatevectorSimulator' exactly.

# import sys
# from pathlib import Path


import numpy as np

from simulator import QuantumSimulator, UnsupportedOperationError
import os

if os.environ.get("STATEVECTOR_IMPL", "cuda").lower() == "cuda":
    from backends._statevector_cuda import CudaStateVector as StateVector
else:
    from backends._statevector import StateVector


class StatevectorSimulator(QuantumSimulator):
    """CUDA-backed statevector simulator."""

    def __init__(self):
        self._cpp = StateVector()
        self.t_n_qubits = 0

    def initialize(self, n: int) -> None:
        if (
            not isinstance(n, (int, np.integer))
            or isinstance(n, bool)
        ):
            raise TypeError("Number of qubits must be an integer.")

        if n < 1:
            raise ValueError("Number of qubits must be at least 1.")

        self._cpp.initialize(int(n))
        self.t_n_qubits = int(n)

    def f_check_initialization(self):
        if self.t_n_qubits == 0:
            raise RuntimeError("Simulator is not initialized.")

    def f_validate_qubit(self, q):
        self.f_check_initialization()

        if (
            not isinstance(q, (int, np.integer))
            or isinstance(q, bool)
        ):
            raise ValueError("Qubit index must be an integer.")

        if q < 0 or q >= self.t_n_qubits:
            raise ValueError(f"Qubit index out of range: {q}")

    @staticmethod
    def _validate_u(params):
        required = ("theta", "phi", "lam")

        if not all(k in params for k in required):
            raise ValueError(
                "U/CU gate requires theta, phi, and lam."
            )

        for key in required:
            try:
                value = float(params[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{key} must be a real number."
                ) from exc

            if not np.isfinite(value):
                raise ValueError(
                    f"{key} must be finite."
                )

    def apply_1q_gate(
        self, gate: str, qubit: int, **params
    ) -> None:
        self.f_validate_qubit(qubit)

        if gate not in {"H", "X", "Y", "Z", "S", "T", "U"}:
            raise UnsupportedOperationError(
                f"Unsupported single-qubit gate: {gate}"
            )

        if gate == "U":
            self._validate_u(params)

        self._cpp.apply_1q_gate(
            gate,
            int(qubit),
            float(params.get("theta", 0.0)),
            float(params.get("phi", 0.0)),
            float(params.get("lam", 0.0)),
        )

    def apply_2q_gate(
        self, gate: str, ctrl: int, tgt: int, **params
    ) -> None:
        self.f_validate_qubit(ctrl)
        self.f_validate_qubit(tgt)

        if ctrl == tgt:
            raise ValueError(
                "Control and target qubits must be different."
            )

        if gate not in {"CNOT", "CZ", "SWAP", "CU"}:
            raise UnsupportedOperationError(
                f"Unsupported two-qubit gate: {gate}"
            )

        if gate == "CU":
            self._validate_u(params)

        self._cpp.apply_2q_gate(
            gate,
            int(ctrl),
            int(tgt),
            float(params.get("theta", 0.0)),
            float(params.get("phi", 0.0)),
            float(params.get("lam", 0.0)),
        )

    def measure(self, qubit: int) -> int:
        self.f_validate_qubit(qubit)
        return int(self._cpp.measure(int(qubit)))

    def sample(self, shots: int) -> dict[str, int]:
        self.f_check_initialization()
        counts = {}

        if hasattr(self._cpp, "sample"):
            for bits in self._cpp.sample(int(shots)):
                counts[bits] = counts.get(bits, 0) + 1
        else:
            state = np.asarray(self.statevector, dtype=np.complex128)
            probabilities = np.abs(state) ** 2
            probabilities /= probabilities.sum()

            indices = np.random.choice(
                len(probabilities), size=int(shots), p=probabilities
            )

            for index in indices:
                bits = format(int(index), f"0{self.t_n_qubits}b")[::-1]
                counts[bits] = counts.get(bits, 0) + 1

        return counts

    def expectation(self, pauli_string: str) -> float:
        self.f_check_initialization()

        if not isinstance(pauli_string, str):
            raise ValueError("Pauli string must be a string.")

        if len(pauli_string) != self.t_n_qubits:
            raise ValueError(
                "Pauli string length must match number of "
                f"qubits ({self.t_n_qubits})."
            )

        if any(p not in "IXYZ" for p in pauli_string):
            raise ValueError(
                "Pauli string can only contain I, X, Y, Z."
            )

        return float(self._cpp.expectation(pauli_string))

    @property
    def n_qubits(self) -> int:
        self.f_check_initialization()
        return self.t_n_qubits

    @property
    def statevector(self) -> np.ndarray:
        self.f_check_initialization()

        return np.asarray(
            self._cpp.statevector(),
            dtype=np.complex128,
        ).copy()
