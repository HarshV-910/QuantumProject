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
from backends._statevector import StateVector

# Adjust this path if your compiled extension lands elsewhere
# sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'statevector'))

from simulator import QuantumSimulator, UnsupportedOperationError  # noqa: F401


class StatevectorSimulator(QuantumSimulator):
    def __init__(self):
        self.t_n_qubits = 0
        self._state = None
        self._cpp=StateVector()

    def initialize(self, n: int) -> None:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
            raise TypeError("Number of qubits must be an integer.")

        if n<1:
            raise ValueError("Number of qubits must be at least 1.")
        self._cpp.initialize(int(n))
        self.t_n_qubits = int(n)
        self._state = None

    #helper functions
    def f_check_initialization(self):
        if self.t_n_qubits == 0:
            raise RuntimeError("Simulator is not initialized")

    def f_current_state(self):
        self.f_check_initialization()
        return np.asarray(self._cpp.statevector(), dtype=np.complex128)

    def f_validate_qubit(self, qubit):
        self.f_check_initialization()
        if not isinstance(qubit, (int, np.integer)):
            raise ValueError("Qubit index must be an integer.")
        if qubit < 0 or qubit >=self.t_n_qubits:
            raise ValueError(f"Qubit index out of range. Must be between 0 and {self.t_n_qubits-1}.")

    @staticmethod
    def f_unitary_matrix(theta, phi, lam):
        return np.array([[np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
                         [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]], dtype=np.complex128)

    def f_single_gate_matrix(self, gate, params):
        sqrt_2 = np.sqrt(2)
        gates = {
            "H":  np.array([[1, 1], [1, -1]], dtype=np.complex128) / sqrt_2,
            "X":  np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "Y":  np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "Z":  np.array([[1, 0], [0, -1]], dtype=np.complex128),
            "S":  np.array([[1, 0], [0, 1j]], dtype=np.complex128),
            "T":  np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
        }

        if gate=="U":
            required_params = ['theta', 'phi', 'lam']
            if not all(param in params for param in required_params):
                raise ValueError(f"Missing parameters for U gate. Required: {required_params}")
            return self.f_unitary_matrix(params['theta'], params['phi'], params['lam'])

        if gate not in gates:
            raise UnsupportedOperationError(f"Unsupported gate: {gate}")

        return gates[gate]
        

    def apply_1q_gate(self, gate: str, qubit: int, **params) -> None:
        self.f_validate_qubit(qubit)

        # Preserve Python-side gate and parameter validation.
        self.f_single_gate_matrix(gate, params)

        self._cpp.apply_1q_gate(
            gate,
            int(qubit),
            float(params.get("theta", 0.0)),
            float(params.get("phi", 0.0)),
            float(params.get("lam", 0.0)),
        )

    def apply_2q_gate(self, gate: str, ctrl: int, tgt: int, **params) -> None:
        self.f_validate_qubit(ctrl)
        self.f_validate_qubit(tgt)

        if ctrl == tgt:
            raise ValueError("Control and target qubits must be different.")

        supported_gates = {"CNOT", "CZ", "SWAP", "CU"}
        if gate not in supported_gates:
            raise UnsupportedOperationError(f"Unsupported 2-qubit gate: {gate}")

        if gate == "CU":
            self.f_single_gate_matrix("U", params)

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

    def sample(self, shots: int) -> dict:
        self.f_check_initialization()
        if not isinstance(shots, (int, np.integer)) or isinstance(shots, bool):
            raise TypeError("Number of shots must be an integer.")
        if shots<1:
            raise ValueError("Number of shots must be at least 1.")
        state = self.f_current_state()
        probs = np.abs(state)**2
        probs /= probs.sum()
        outcomes = np.random.choice(len(state), size=shots, p=probs)
        counts = {}
        for outcome in outcomes:
            bitstring = "".join(
                str((int(outcome) >> q) & 1)
                for q in range(self.t_n_qubits)
            )
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def expectation(self, pauli_string: str) -> float:
        self.f_check_initialization()
        if not isinstance(pauli_string, str):
            raise ValueError("Pauli string must be a string.")
        if len(pauli_string) != self.t_n_qubits:
            raise ValueError(f"Pauli string length must match number of qubits ({self.t_n_qubits}).")
        if any(p not in "IXYZ" for p in pauli_string):
            raise ValueError("Pauli string can only contain 'I', 'X', 'Y', 'Z'.")
        state = self.f_current_state()
        result=0.0+0.0j
        for i, amplitude in enumerate(state):
            target = i
            phase = 1.0 + 0.0j

            for char_index, pauli in enumerate(pauli_string):
                q = self.t_n_qubits - 1 - char_index
                bit = (i >> q) & 1

                if pauli == "I":
                    continue

                elif pauli == "X":
                    target ^= 1 << q

                elif pauli == "Y":
                    target ^= 1 << q
                    phase *= 1j if bit == 0 else -1j

                elif pauli == "Z":
                    if bit == 1:
                        phase *= -1

            result += (
                np.conjugate(state[target])
                * phase
                * amplitude
            )

        if abs(result.imag) > 1e-10:
            raise ArithmeticError(
                "Expectation has a significant imaginary component."
            )

        return float(result.real)

    @property
    def n_qubits(self) -> int:
        self.f_check_initialization()
        return self.t_n_qubits

    @property
    def statevector(self) -> np.ndarray:
        return self.f_current_state().copy()