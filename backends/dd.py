"""Python interface to the C++17 weighted decision-diagram simulator.

Build the native core with ``make`` from the project directory.  The state is
kept in the C++ DD manager; Python only validates arguments and converts results.
"""
from __future__ import annotations
import ctypes
import math
from pathlib import Path
from typing import Any
import numpy as np
from simulator import QuantumSimulator, UnsupportedOperationError

_LIB_PATH = Path(__file__).with_name("libddcore.so")
_LIB = None
_LOAD_ERROR = None
try:
    _LIB = ctypes.CDLL(str(_LIB_PATH))
    _LIB.dd_create.argtypes = [ctypes.c_int]
    _LIB.dd_create.restype = ctypes.c_void_p
    _LIB.dd_destroy.argtypes = [ctypes.c_void_p]
    _LIB.dd_error.argtypes = []
    _LIB.dd_error.restype = ctypes.c_char_p
    _LIB.dd_apply_1q.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    _LIB.dd_apply_1q.restype = ctypes.c_int
    _LIB.dd_apply_2q.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    _LIB.dd_apply_2q.restype = ctypes.c_int
    _LIB.dd_measure.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _LIB.dd_measure.restype = ctypes.c_int
    _LIB.dd_sample.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    _LIB.dd_sample.restype = ctypes.c_int
    _LIB.dd_expectation.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    _LIB.dd_expectation.restype = ctypes.c_double
    _LIB.dd_node_count.argtypes = [ctypes.c_void_p]
    _LIB.dd_node_count.restype = ctypes.c_int
    _LIB.dd_nqubits.argtypes = [ctypes.c_void_p]
    _LIB.dd_nqubits.restype = ctypes.c_int
    _LIB.dd_amplitude.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    _LIB.dd_amplitude.restype = ctypes.c_int
except OSError as exc:
    _LOAD_ERROR = exc

_GATE1 = {"H": 0, "X": 1, "Y": 2, "Z": 3, "S": 4, "T": 5, "U": 6}
_GATE2 = {"CNOT": 0, "CZ": 1, "SWAP": 2, "CU": 3}

class DDSimulator(QuantumSimulator):
    """Simulator whose quantum state and gate kernels live in C++ weighted DDs."""
    def __init__(self) -> None:
        if _LIB is None:
            raise RuntimeError(f"C++ DD library not built or failed to load: {_LOAD_ERROR}. Run `make` in the part1 directory.")
        self._handle = None
        self._n = 0
        self._rng = np.random.default_rng()

    def close(self) -> None:
        """Release the native manager explicitly (useful for long-lived processes)."""
        if self._handle and _LIB is not None:
            _LIB.dd_destroy(self._handle)
            self._handle = None
            self._n = 0

    @staticmethod
    def _error() -> str:
        raw = _LIB.dd_error()
        return raw.decode("utf-8", errors="replace") if raw else "unknown C++ DD error"

    def _check(self, status: int) -> None:
        if status != 0:
            raise RuntimeError(self._error())

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def node_count(self) -> int:
        return 0 if self._handle is None else int(_LIB.dd_node_count(self._handle))

    @property
    def statevector(self) -> np.ndarray:
        """Debug/harness accessor. Expands amplitudes; gate kernels never do this."""
        self._require_init()
        out = np.empty(1 << self._n, dtype=np.complex128)
        re, im = ctypes.c_double(), ctypes.c_double()
        for i in range(out.size):
            self._check(_LIB.dd_amplitude(self._handle, i, ctypes.byref(re), ctypes.byref(im)))
            out[i] = complex(re.value, im.value)
        return out

    def _require_init(self):
        if self._handle is None:
            raise RuntimeError("Call initialize(n) before simulating")

    def initialize(self, n: int) -> None:
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer")
        new_handle = _LIB.dd_create(n)
        if not new_handle:
            raise ValueError(self._error())
        if self._handle:
            _LIB.dd_destroy(self._handle)
        self._handle, self._n = new_handle, n

    def _check_qubit(self, q: int) -> None:
        if not isinstance(q, int) or q < 0 or q >= self._n:
            raise ValueError(f"qubit index {q} outside [0, {self._n})")

    @staticmethod
    def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -np.exp(1j * lam) * s],
                         [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=np.complex128)

    @staticmethod
    def _params(params: dict[str, Any], gate: str) -> tuple[float, float, float]:
        if gate not in ("U", "CU"):
            return 0.0, 0.0, 0.0
        try:
            return float(params["theta"]), float(params["phi"]), float(params["lam"])
        except KeyError as exc:
            raise ValueError(f"{gate} requires theta, phi, and lam") from exc

    def apply_1q_gate(self, gate: str, qubit: int, **params: Any) -> None:
        self._require_init(); self._check_qubit(qubit)
        if gate not in _GATE1:
            raise UnsupportedOperationError(f"Unsupported 1-qubit gate: {gate}")
        th, ph, lam = self._params(params, gate)
        self._check(_LIB.dd_apply_1q(self._handle, _GATE1[gate], qubit, th, ph, lam))

    def apply_2q_gate(self, gate: str, ctrl: int, tgt: int, **params: Any) -> None:
        self._require_init(); self._check_qubit(ctrl); self._check_qubit(tgt)
        if ctrl == tgt:
            raise ValueError("control and target must be different qubits")
        if gate not in _GATE2:
            raise UnsupportedOperationError(f"Unsupported 2-qubit gate: {gate}")
        th, ph, lam = self._params(params, gate)
        self._check(_LIB.dd_apply_2q(self._handle, _GATE2[gate], ctrl, tgt, th, ph, lam))

    def measure(self, qubit: int) -> int:
        self._require_init(); self._check_qubit(qubit)
        result = int(_LIB.dd_measure(self._handle, qubit))
        if result < 0:
            raise RuntimeError(self._error())
        return result

    def sample(self, shots: int) -> dict[str, int]:
        self._require_init()
        if not isinstance(shots, int) or shots < 0:
            raise ValueError("shots must be non-negative")
        if shots == 0:
            return {}
        raw = (ctypes.c_int * (shots * self._n))()
        self._check(_LIB.dd_sample(self._handle, shots, raw))
        counts: dict[str, int] = {}
        for s in range(shots):
            # Required interface: qubit 0 is the leftmost character.
            key = "".join(str(raw[s * self._n + q]) for q in range(self._n))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def expectation(self, pauli_string: str) -> float:
        self._require_init()
        if len(pauli_string) != self._n or any(c not in "IXYZ" for c in pauli_string):
            raise ValueError("pauli_string must have length n_qubits and use only I,X,Y,Z")
        result = float(_LIB.dd_expectation(self._handle, pauli_string.encode("ascii")))
        if math.isnan(result):
            raise RuntimeError(self._error())
        return result
