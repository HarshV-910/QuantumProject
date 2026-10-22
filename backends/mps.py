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


import math,numpy as np
from typing import Any
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'mps'))
from simulator import QuantumSimulator, UnsupportedOperationError  # noqa: F401

_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

class MPSSimulator(QuantumSimulator):
    # for simulation using small tensors
    def __init__(self)  -> None:
        self._n=0
        self._tensors: list[np.ndarray] = []
        self._rng = np.random.default_rng() # random number generator

    def initialize(self, n: int) -> None: # create |00000000...0> for bond dim=1
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        self._n = n
        self._tensors = []
        for _ in range(n):
            tensor = np.zeros((1, 2, 1), dtype=np.complex128) # (block,row,col)
            tensor[0, 0, 0] = 1.0  # no entanglement
            self._tensors.append(tensor)

    # --------------- single qubit gate matrix ------------------
    
    def _require_initialized(self) -> None:
        if self._n <= 0 or len(self._tensors) != self._n:
            raise RuntimeError("Call initialize(n) before using the simulator")

    def _check_qubit(self, qubit: int) -> None:
            self._require_initialized()
            if not isinstance(qubit, int) or not 0 <= qubit < self._n:
                raise ValueError(f"qubit index must be in [0, {self._n})")
            
    @staticmethod
    def _single_gate_matrix(gate: str, params: dict[str, Any]) -> np.ndarray:  # X,Y,Z,S,T,I,H,U:random rotation
        s2 = 1.0 / math.sqrt(2.0)
        fixed = {
            "I": np.eye(2, dtype=np.complex128),
            "H": np.array([[s2, s2], [s2, -s2]], dtype=np.complex128),
            "X": _PAULI["X"],
            "Y": _PAULI["Y"],
            "Z": _PAULI["Z"],
            "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
            "T": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128),
        }
        if gate in fixed:
            return fixed[gate]
        if gate != "U":
            raise UnsupportedOperationError(f"MPS backend does not support 1-qubit gate {gate!r}")
        try:
            theta, phi, lam = float(params["theta"]), float(params["phi"]), float(params["lam"])
        except KeyError as exc:
            raise ValueError("U gate requires theta, phi, and lam") from exc
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        return np.array(
            [[c, -np.exp(1j * lam) * s],
             [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
            dtype=np.complex128,
        )
    
    def apply_1q_gate(self, gate: str, qubit: int, **params) -> None:
        '''Tensor-matrix contraction (einsum)'''
        self._check_qubit(qubit)
        matrix = self._single_gate_matrix(gate, params)

        # tensor[left, input_bit, right] -> tensor[left, output_bit, right]
        self._tensors[qubit] = np.einsum("oi,lir->lor", matrix, self._tensors[qubit], optimize=True)

# --------------- 2 qubit gate matrix ------------------

    @staticmethod
    def _swap_matrix() -> np.ndarray:  # swapping for routing
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],dtype=np.complex128,)

    @classmethod
    def _two_gate_matrix(cls, gate: str, params: dict[str, Any]) -> np.ndarray: # CNOT and CZ and any CU:random rotation
        if gate == "CNOT":
            return np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=np.complex128,
            )
        if gate == "CZ":
            return np.diag([1, 1, 1, -1]).astype(np.complex128)
        if gate == "SWAP":
            return cls._swap_matrix()
        if gate == "CU": # random rotation gate
            try:
                unitary = cls._single_gate_matrix("U", params)
            except (KeyError, ValueError) as exc:
                raise ValueError("CU gate requires theta, phi, and lam") from exc
            matrix = np.eye(4, dtype=np.complex128)
            matrix[2:4, 2:4] = unitary
            return matrix
        raise UnsupportedOperationError(f"MPS backend does not support 2-qubit gate {gate!r}")

    @staticmethod
    def _reverse_two_qubit_order(matrix: np.ndarray) -> np.ndarray:  # exchange target and control qubits
        # Re-index |ctrl,tgt> as |left,right> when the control is the right site.
        tensor = matrix.reshape(2, 2, 2, 2)
        return tensor.transpose(1, 0, 3, 2).reshape(4, 4)

    def _apply_adjacent_gate(self, left: int, right: int, matrix: np.ndarray) -> None: # join tensors left and right and apply gate
        """Apply a 4x4 gate to neighbouring MPS tensors using an exact SVD split."""
        a = self._tensors[left]
        b = self._tensors[right]

        # Join the two tensors across their shared virtual bond: theta[l, q_left, q_right, r]
        theta = np.einsum("lpm,mqr->lpqr", a, b, optimize=True)
        gate_tensor = matrix.reshape(2, 2, 2, 2)  # out-left, out-right, in-left, in-right
        theta = np.einsum("uvij,lijr->luvr", gate_tensor, theta, optimize=True)

        left_dim, right_dim = theta.shape[0], theta.shape[3]
        matrix_view = theta.reshape(left_dim * 2, 2 * right_dim)
        u, singular_values, vh = np.linalg.svd(matrix_view, full_matrices=False)

        # Remove singular values that are zero up to floating-point precision.
        if singular_values.size:
            cutoff = np.finfo(np.float64).eps * max(matrix_view.shape) * singular_values[0]
            keep = singular_values > cutoff
            if not np.any(keep):
                keep[0] = True
            u = u[:, keep]
            singular_values = singular_values[keep]
            vh = vh[keep, :]
        bond = len(singular_values)
        new_left = u.reshape(left_dim, 2, bond)
        new_right = (singular_values[:, None] * vh).reshape(bond, 2, right_dim)
        self._tensors[left] = new_left
        self._tensors[right] = new_right

    def apply_2q_gate(self, gate: str, ctrl: int, tgt: int, **params) -> None:
        '''SWAPs to bring adjacent -> contract-> Gate -> SVD split'''
        self._check_qubit(ctrl)
        self._check_qubit(tgt)
        if ctrl == tgt:
            raise ValueError("ctrl and tgt must refer to different qubits")
        matrix = self._two_gate_matrix(gate, params)

        # swapping for physical SWAP gates then undo that after getting original qubit ordering
        left, right = sorted((ctrl, tgt))
        for q in range(right - 1, left, -1):
            self._apply_adjacent_gate(q, q + 1, self._swap_matrix())

        # reverse order to make from (target,control) to (control,target) if not. now it will be like left=control and left+1=target
        if ctrl == left:
            local_matrix = matrix
        else:
            local_matrix = self._reverse_two_qubit_order(matrix)

        self._apply_adjacent_gate(left, left + 1, local_matrix)

        for q in range(left + 1, right): # revert swaping operation
            self._apply_adjacent_gate(q, q + 1, self._swap_matrix())


    # --------------- measurement ------------------

    def _projected_norm_squared(self, qubit: int, bit: int) -> float: # used for measurement
        """Contract the norm network while fixing one qubit to 0 or 1."""
        env = np.ones((1, 1), dtype=np.complex128)
        for index, tensor in enumerate(self._tensors):
            allowed = (bit,) if index == qubit else (0, 1)
            next_env = np.zeros((tensor.shape[2], tensor.shape[2]), dtype=np.complex128)
            for physical in allowed:
                local = tensor[:, physical, :]
                next_env += np.einsum(
                    "ab,ai,bj->ij", env, local.conj(), local, optimize=True
                )
            env = next_env
        return max(0.0, float(env[0, 0].real))

    def measure(self, qubit: int) -> int: # measurement of P(|0>) and P(|1>) and Project physical index -> renormalize
        """Measure one qubit, collapse the MPS, and return its bit."""
        self._check_qubit(qubit)
        p0 = self._projected_norm_squared(qubit, 0)
        p1 = self._projected_norm_squared(qubit, 1)
        total = p0 + p1
        if total <= 0.0:
            raise RuntimeError("Cannot measure a zero-norm quantum state")
        p0 /= total
        outcome = 0 if self._rng.random() < p0 else 1
        probability = p0 if outcome == 0 else 1.0 - p0
        if probability <= 1e-15:
            outcome = 1 - outcome
            probability = p0 if outcome == 0 else 1.0 - p0
        tensor = self._tensors[qubit].copy()
        tensor[:, 1 - outcome, :] = 0.0
        # Normalise the post-measurement state.  Scaling one tensor scales the
        # entire MPS by the same factor.
        tensor /= math.sqrt(probability)
        self._tensors[qubit] = tensor
        return outcome

    def sample(self, shots: int) -> dict:
        """Return measurement counts without changing the original state."""
        if not isinstance(shots, int) or shots < 0:
            raise ValueError("shots must be a non-negative integer")
        counts: dict[str, int] = {}
        original = self._tensors
        try:
            for _ in range(shots):
                # Copy the small tensor list so each shot begins from the same
                # pre-measurement state and collapses independently.
                self._tensors = [tensor.copy() for tensor in original]
                bits = [str(self.measure(q)) for q in range(self._n)]
                bitstring = "".join(bits)
                counts[bitstring] = counts.get(bitstring, 0) + 1
        finally:
            self._tensors = original
        return counts

    
    # --------------- statevector & expectation ------------------

    def expectation(self, pauli_string: str) -> float:
        """Compute <psi|P|psi> by contracting the MPS from left to right. ex: XZI"""
        self._require_initialized()
        if len(pauli_string) != self._n or any(ch not in _PAULI for ch in pauli_string):
            raise ValueError(f"pauli_string must have length {self._n} and contain only I, X, Y, Z")
        env = np.ones((1, 1), dtype=np.complex128) # env contains the overlap accumulated up to the previous bond.
        for tensor, symbol in zip(self._tensors, pauli_string):
            operator = _PAULI[symbol]
            env = np.einsum(
                "ab,asr,st,btd->rd",
                env,
                tensor.conj(),
                operator,
                tensor,
                optimize=True,
            )
        value = env[0, 0]
        return float(value.real) # Pauli string is Hermitian, so its expectation is real.

    @property
    def n_qubits(self) -> int: # how many qubits are there
        return self._n

    @property
    def statevector(self) -> np.ndarray: # from n tensors and virtual bond(tensor dot), reconstruct(recover) 2^n statevector
        """Materialize amplitudes for the instructor harness on small tests.

        This is deliberately a property, not a get_statevector() method in the
        common simulator API.  It should not be used inside the MPS hot path.
        Tensor axes are ordered by qubit index, matching sample()'s bit strings.
        """
        self._require_initialized()
        state = self._tensors[0][0, :, :]
        for tensor in self._tensors[1:]:
            state = np.tensordot(state, tensor, axes=([-1], [0]))
        return np.asarray(state.reshape(-1), dtype=np.complex128)


    