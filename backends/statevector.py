# COL8422 — Statevector backend
# Member A: implement StatevectorSimulator here.
#
# Your C++ kernel goes in src/statevector/.
# Expose it via pybind11 or ctypes, then call it from this class.
#
# Do NOT rename the class — the harness looks for 'StatevectorSimulator' exactly.

import sys
from pathlib import Path
import numpy as np

# Adjust this path if your compiled extension lands elsewhere
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'statevector'))

from simulator import QuantumSimulator, UnsupportedOperationError  # noqa: F401


class StatevectorSimulator(QuantumSimulator):
    def __init__(self):
        self.t_n_qubits = 0
        self._state = None

    def initialize(self, n: int) -> None:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
            raise TypeError("Number of qubits must be an integer.")

        if n<1:
            raise ValueError("Number of qubits must be at least 1.")
        self.t_n_qubits=n
        self._state=np.zeros((1<<self.t_n_qubits), dtype=np.complex128)
        #To simulate the |0> state as the starting state
        self._state[0]=1.0 + 0.0j

    #helper functions
    def f_check_initialization(self):
        if self._state is None:
            raise RuntimeError("Simulator is not initialized")

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
        U = self.f_single_gate_matrix(gate, params)

        mask = 1 << qubit
        indices = np.arange(len(self._state))
        zero_indices = indices[(indices & mask) == 0]
        one_indices = zero_indices | mask

        a0 = self._state[zero_indices].copy()
        a1 = self._state[one_indices].copy()

        self._state[zero_indices] = (
            U[0, 0] * a0 + U[0, 1] * a1
        )
        self._state[one_indices] = (
            U[1, 0] * a0 + U[1, 1] * a1
        )

    def apply_2q_gate(self, gate: str, ctrl: int, tgt: int, **params) -> None:
        self.f_validate_qubit(ctrl)
        self.f_validate_qubit(tgt)
        if ctrl == tgt:
            raise ValueError("Control and target qubits must be different.")
        ctrl_mask = 1<<ctrl
        tgt_mask = 1<<tgt
        if gate == "CNOT":
            for i in range(len(self._state)):
                if (i&ctrl_mask) and not(i&tgt_mask):
                    j=(i|tgt_mask)
                    self._state[i], self._state[j] = self._state[j], self._state[i]
        elif gate == "CZ":
            for i in range(len(self._state)):
                if (i&ctrl_mask) and (i&tgt_mask):
                    self._state[i] *= -1
        elif gate == "SWAP":
            for i in range(len(self._state)):
                if not (i & ctrl_mask) and (i & tgt_mask):
                    j = i ^ ctrl_mask ^ tgt_mask
                    self._state[i], self._state[j] = (
                        self._state[j], self._state[i]
                    )
        elif gate == "CU":
            U=self.f_single_gate_matrix("U", params)
            for i in range(len(self._state)):
                if(i&ctrl_mask) and not (i&tgt_mask):
                    j=(i|tgt_mask)
                    original_amp=self._state[i]
                    paired_amp=self._state[j]
                    self._state[i]=U[0,0]*original_amp + U[0,1]*paired_amp
                    self._state[j]=U[1,0]*original_amp + U[1,1]*paired_amp
        else:
            raise UnsupportedOperationError(f"Unsupported 2-qubit gate: {gate}")


    def measure(self, qubit: int) -> int:
        self.f_validate_qubit(qubit)
        mask=(1<<qubit)
        indices = np.arange(len(self._state))
        probs = np.abs(self._state)**2
        p1=float(probs[(indices&mask)!=0].sum())
        p1=min(1.0, max(0.0, p1))
        p0=1.0-p1
        outcome = int(np.random.random()<p1)
        selected_prob = p1 if outcome else p0
        if selected_prob<=0.0:
            raise RuntimeError("Measurement probability is zero, cannot collapse state.")
        keep = ((indices&mask)!=0) == bool(outcome)
        self._state[~keep] = 0.0
        self._state /= np.sqrt(selected_prob)

        return outcome

    def sample(self, shots: int) -> dict:
        self.f_check_initialization()
        if not isinstance(shots, (int, np.integer)) or isinstance(shots, bool):
            raise TypeError("Number of shots must be an integer.")
        if shots<1:
            raise ValueError("Number of shots must be at least 1.")
        probs = np.abs(self._state)**2
        probs /= probs.sum()
        outcomes = np.random.choice(len(self._state), size=shots, p=probs)
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
        result=0.0+0.0j
        for i, amplitude in enumerate(self._state):
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
                np.conjugate(self._state[target])
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
        self.f_check_initialization()
        return self._state.copy()