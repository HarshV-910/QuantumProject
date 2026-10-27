# COL8422 — Stabilizer backend
# Member B: StabilizerSimulator
#
# Implements the Aaronson–Gottesman tableau algorithm
#   S. Aaronson & D. Gottesman, "Improved Simulation of Stabilizer Circuits",
#   Phys. Rev. A 70, 052328 (2004), Section III.
#
# Two interchangeable tableau kernels with the same method names:
#   * _CppTableau   — bit-packed C++ kernel in src/stabilizer/ (loaded via ctypes).
#                     This is THE kernel; it is required (build it with `make`).
#   * _NumpyTableau — pure NumPy reference version of the same algorithm, used
#                     ONLY when explicitly requested with STAB_KERNEL=numpy
#                     (unit tests cross-check the C++ kernel against it).
# There is no silent fallback: if the C++ library is missing, the backend
# refuses to load with an ImportError that says to run `make`.
#
# Non-Clifford gates (T, U, CU) raise UnsupportedOperationError.
# Do NOT rename the class — the harness looks for 'StabilizerSimulator' exactly.

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'stabilizer'))

from simulator import QuantumSimulator, UnsupportedOperationError  # noqa: E402


# ============================================================================
# Kernel 1: pure NumPy tableau (reference implementation, easy to read)
# ============================================================================

class _NumpyTableau:
    """(2n+1) x (2n+1) tableau; row 2n is scratch space.

    x[i, j], z[i, j] are booleans; (x,z)=(1,0) X, (1,1) Y, (0,1) Z, (0,0) I.
    r[i] = 1 means generator i carries a minus sign.
    Rows 0..n-1 destabilizers, rows n..2n-1 stabilizers.
    """

    def __init__(self, n: int, _empty: bool = False):
        self.n = n
        if _empty:
            return
        self.x = np.zeros((2 * n + 1, n), dtype=bool)
        self.z = np.zeros((2 * n + 1, n), dtype=bool)
        self.r = np.zeros(2 * n + 1, dtype=np.uint8)
        idx = np.arange(n)
        self.x[idx, idx] = True          # destabilizer i = X_i
        self.z[n + idx, idx] = True      # stabilizer   i = Z_i

    def copy(self) -> "_NumpyTableau":
        t = _NumpyTableau(self.n, _empty=True)
        t.x, t.z, t.r = self.x.copy(), self.z.copy(), self.r.copy()
        return t

    # ---- gates (all vectorised over the 2n rows => O(n)) --------------------

    def h(self, a: int) -> None:
        R = slice(0, 2 * self.n)
        xa, za = self.x[R, a].copy(), self.z[R, a].copy()
        self.r[R] ^= (xa & za).astype(np.uint8)
        self.x[R, a], self.z[R, a] = za, xa

    def s(self, a: int) -> None:
        R = slice(0, 2 * self.n)
        xa, za = self.x[R, a], self.z[R, a]
        self.r[R] ^= (xa & za).astype(np.uint8)
        self.z[R, a] = za ^ xa

    def px(self, a: int) -> None:
        R = slice(0, 2 * self.n)
        self.r[R] ^= self.z[R, a].astype(np.uint8)

    def pz(self, a: int) -> None:
        R = slice(0, 2 * self.n)
        self.r[R] ^= self.x[R, a].astype(np.uint8)

    def py(self, a: int) -> None:
        R = slice(0, 2 * self.n)
        self.r[R] ^= (self.x[R, a] ^ self.z[R, a]).astype(np.uint8)

    def cnot(self, a: int, b: int) -> None:
        R = slice(0, 2 * self.n)
        xa, za = self.x[R, a], self.z[R, a]
        xb, zb = self.x[R, b], self.z[R, b]
        self.r[R] ^= (xa & zb & ~(xb ^ za)).astype(np.uint8)
        self.x[R, b] = xb ^ xa
        self.z[R, a] = za ^ zb

    def cz(self, a: int, b: int) -> None:
        R = slice(0, 2 * self.n)
        xa, za = self.x[R, a], self.z[R, a]
        xb, zb = self.x[R, b], self.z[R, b]
        self.r[R] ^= (xa & xb & (za ^ zb)).astype(np.uint8)
        self.z[R, a] = za ^ xb
        self.z[R, b] = zb ^ xa

    def swap(self, a: int, b: int) -> None:
        R = slice(0, 2 * self.n)
        self.x[R, [a, b]] = self.x[R, [b, a]]
        self.z[R, [a, b]] = self.z[R, [b, a]]

    # ---- rowsum -------------------------------------------------------------

    @staticmethod
    def _g_sum(x1, z1, x2, z2):
        """sum_j g(x1_j, z1_j, x2_j, z2_j) along the last axis (broadcasts)."""
        X1, Y1, Z1 = x1 & ~z1, x1 & z1, ~x1 & z1
        X2, Y2, Z2 = x2 & ~z2, x2 & z2, ~x2 & z2
        plus = (X1 & Y2) | (Y1 & Z2) | (Z1 & X2)
        minus = (X1 & Z2) | (Y1 & X2) | (Z1 & Y2)
        return plus.sum(axis=-1, dtype=np.int64) - minus.sum(axis=-1, dtype=np.int64)

    def rowsum(self, h, i: int) -> None:
        """R_h <- R_i * R_h.  `h` may be an int or an index array (vectorised)."""
        g = self._g_sum(self.x[i], self.z[i], self.x[h], self.z[h])
        total = (2 * self.r[h].astype(np.int64) + 2 * int(self.r[i]) + g) % 4
        self.r[h] = (total == 2).astype(np.uint8)
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]

    # ---- measurement --------------------------------------------------------

    def _find_p(self, a: int) -> int:
        n = self.n
        hits = np.flatnonzero(self.x[n:2 * n, a])
        return int(hits[0]) + n if hits.size else -1

    def is_random(self, a: int) -> bool:
        return self._find_p(a) >= 0

    def measure(self, a: int, rand_bit: int) -> int:
        n = self.n
        p = self._find_p(a)
        if p >= 0:                                     # Case I: random
            rows = np.flatnonzero(self.x[:2 * n, a])
            rows = rows[(rows != p) & (rows != p - n)]
            if rows.size:
                self.rowsum(rows, p)                   # all at once
            self.x[p - n], self.z[p - n], self.r[p - n] = self.x[p], self.z[p], self.r[p]
            self.x[p] = False
            self.z[p] = False
            self.z[p, a] = True
            self.r[p] = rand_bit & 1
            return rand_bit & 1
        s = 2 * n                                      # Case II: deterministic
        self.x[s] = False
        self.z[s] = False
        self.r[s] = 0
        for i in np.flatnonzero(self.x[:n, a]):
            self.rowsum(s, int(i) + n)
        return int(self.r[s])

    # ---- expectation ---------------------------------------------------------

    def expectation(self, px: np.ndarray, pz: np.ndarray) -> int:
        n = self.n
        px, pz = px.astype(bool), pz.astype(bool)
        symp = ((self.x[:2 * n] & pz) ^ (self.z[:2 * n] & px)).sum(axis=1) & 1
        if symp[n:].any():
            return 0
        s = 2 * n
        self.x[s] = False
        self.z[s] = False
        self.r[s] = 0
        for h in np.flatnonzero(symp[:n]):
            self.rowsum(s, int(h) + n)
        return -1 if self.r[s] else 1

    def dump(self):
        n = self.n
        return (self.x[:2 * n].astype(np.uint8), self.z[:2 * n].astype(np.uint8),
                self.r[:2 * n].copy())

    # ---- sampling (same algorithm as Tableau::sample in C++) ----------------

    def sample_bits(self, shots: int, rand: np.ndarray) -> np.ndarray:
        """shots x n array of outcomes; rand is shots x n random 0/1 bytes."""
        n = self.n
        work = self.copy()
        v0 = np.array([work.measure(q, 0) for q in range(n)], dtype=np.uint8)
        basis = _gf2_row_basis(self.x[n:2 * n].astype(np.uint8))   # k x n
        k = basis.shape[0]
        if k == 0:
            return np.tile(v0, (shots, 1))
        coeffs = rand[:, :k].astype(np.int64)
        return (((coeffs @ basis.astype(np.int64)) & 1).astype(np.uint8)) ^ v0


# ============================================================================
# Kernel 2: C++ tableau via ctypes (THE kernel)
# ============================================================================

_LIB = None


def _load_lib():
    """Load src/stabilizer/libstabilizer.so once; return None if unavailable."""
    global _LIB
    if _LIB is not None:
        return _LIB or None
    here = Path(__file__).resolve().parent.parent / "src" / "stabilizer"
    for name in ("libstabilizer.so", "libstabilizer.dylib", "stabilizer.dll"):
        path = here / name
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
            except OSError:
                continue
            vp, ci, u8p = ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)
            lib.stab_new.restype = vp
            lib.stab_new.argtypes = [ci]
            lib.stab_copy.restype = vp
            lib.stab_copy.argtypes = [vp]
            lib.stab_free.restype = None
            lib.stab_free.argtypes = [vp]
            for fn in ("stab_h", "stab_s", "stab_x", "stab_y", "stab_z"):
                getattr(lib, fn).restype = None
                getattr(lib, fn).argtypes = [vp, ci]
            for fn in ("stab_cnot", "stab_cz", "stab_swap"):
                getattr(lib, fn).restype = None
                getattr(lib, fn).argtypes = [vp, ci, ci]
            lib.stab_measure.restype = ci
            lib.stab_measure.argtypes = [vp, ci, ci]
            lib.stab_is_random.restype = ci
            lib.stab_is_random.argtypes = [vp, ci]
            lib.stab_expectation.restype = ci
            lib.stab_expectation.argtypes = [vp, u8p, u8p]
            lib.stab_dump.restype = None
            lib.stab_dump.argtypes = [vp, u8p, u8p, u8p]
            lib.stab_sample.restype = None
            lib.stab_sample.argtypes = [vp, ci, u8p, u8p]
            _LIB = lib
            return lib
    _LIB = False
    return None


class _CppTableau:
    """Thin ctypes wrapper; owns one C++ Tableau object."""

    def __init__(self, n: int, lib, _handle=None):
        self.n = n
        self._lib = lib
        self._h = _handle if _handle is not None else lib.stab_new(n)

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.stab_free(h)
            self._h = None

    def copy(self) -> "_CppTableau":
        return _CppTableau(self.n, self._lib, self._lib.stab_copy(self._h))

    def h(self, a): self._lib.stab_h(self._h, a)
    def s(self, a): self._lib.stab_s(self._h, a)
    def px(self, a): self._lib.stab_x(self._h, a)
    def py(self, a): self._lib.stab_y(self._h, a)
    def pz(self, a): self._lib.stab_z(self._h, a)
    def cnot(self, a, b): self._lib.stab_cnot(self._h, a, b)
    def cz(self, a, b): self._lib.stab_cz(self._h, a, b)
    def swap(self, a, b): self._lib.stab_swap(self._h, a, b)

    def is_random(self, a) -> bool:
        return bool(self._lib.stab_is_random(self._h, a))

    def measure(self, a, rand_bit) -> int:
        return int(self._lib.stab_measure(self._h, a, int(rand_bit) & 1))

    def expectation(self, px: np.ndarray, pz: np.ndarray) -> int:
        px = np.ascontiguousarray(px, dtype=np.uint8)
        pz = np.ascontiguousarray(pz, dtype=np.uint8)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        return int(self._lib.stab_expectation(self._h, px.ctypes.data_as(u8p),
                                              pz.ctypes.data_as(u8p)))

    def dump(self):
        n = self.n
        xs = np.zeros((2 * n, n), dtype=np.uint8)
        zs = np.zeros((2 * n, n), dtype=np.uint8)
        rs = np.zeros(2 * n, dtype=np.uint8)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        self._lib.stab_dump(self._h, xs.ctypes.data_as(u8p), zs.ctypes.data_as(u8p),
                            rs.ctypes.data_as(u8p))
        return xs, zs, rs

    def sample_bits(self, shots: int, rand: np.ndarray) -> np.ndarray:
        """shots x n array of outcomes, computed in C++ (Tableau::sample)."""
        rand = np.ascontiguousarray(rand, dtype=np.uint8)
        out = np.empty((shots, self.n), dtype=np.uint8)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        self._lib.stab_sample(self._h, shots, rand.ctypes.data_as(u8p), out.ctypes.data_as(u8p))
        return out


def _gf2_row_basis(M: np.ndarray) -> np.ndarray:
    """Basis (as rows) of the GF(2) row space of M.  Used by the NumPy reference
    kernel only; the C++ kernel does the same elimination in Tableau::sample."""
    A = (M & 1).astype(np.uint8)
    rows, cols = A.shape
    rank = 0
    for c in range(cols):
        if rank == rows:
            break
        pivots = np.flatnonzero(A[rank:, c])
        if pivots.size == 0:
            continue
        piv = rank + int(pivots[0])
        if piv != rank:
            A[[rank, piv]] = A[[piv, rank]]
        others = np.flatnonzero(A[:, c])
        others = others[others != rank]
        A[others] ^= A[rank]
        rank += 1
    return A[:rank]


# ============================================================================
# The backend class used by the harness
# ============================================================================

_PAULI_BITS = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
_SUPPORTED_1Q = {"H", "S", "X", "Y", "Z"}
_NON_CLIFFORD_1Q = {"T", "U"}
_SUPPORTED_2Q = {"CNOT", "CZ", "SWAP"}
_NON_CLIFFORD_2Q = {"CU"}


class StabilizerSimulator(QuantumSimulator):
    """Clifford-circuit simulator using the Aaronson–Gottesman tableau.

    Memory: 2n(2n+1) bits.  Gates: O(n).  Measurement: O(n^2) worst case.
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._tab = None
        self._rng = np.random.default_rng()
        if os.environ.get("STAB_KERNEL", "").lower() == "numpy":
            self._lib = None                 # explicit request: reference kernel
        else:
            self._lib = _load_lib()
            if self._lib is None:
                raise ImportError(
                    "Stabilizer C++ kernel not found: expected "
                    "src/stabilizer/libstabilizer.so. Run `make` in the project root. "
                    "(Set STAB_KERNEL=numpy only to run the NumPy reference kernel for testing.)")
        self.kernel = "cpp" if self._lib is not None else "numpy"

    # ------------------------------------------------------------------ setup

    def seed(self, seed: int | None) -> None:
        """Optional: make measurement / sampling randomness reproducible."""
        self._rng = np.random.default_rng(seed)

    def initialize(self, n: int) -> None:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        n = int(n)
        self._n = n
        if self._lib is not None:
            self._tab = _CppTableau(n, self._lib)
        else:
            self._tab = _NumpyTableau(n)

    @property
    def n_qubits(self) -> int:
        return self._n

    # ------------------------------------------------------------------ checks

    def _require_init(self) -> None:
        if self._tab is None:
            raise RuntimeError("Simulator not initialised: call initialize(n) first.")

    def _check_qubit(self, q) -> int:
        if not isinstance(q, (int, np.integer)) or isinstance(q, bool):
            raise ValueError(f"qubit index must be an int, got {q!r}")
        q = int(q)
        if not 0 <= q < self._n:
            raise ValueError(f"qubit index {q} out of range for {self._n} qubits")
        return q

    # ------------------------------------------------------------------ gates

    def apply_1q_gate(self, gate: str, qubit: int, **params) -> None:
        self._require_init()
        if gate in _NON_CLIFFORD_1Q:
            raise UnsupportedOperationError(
                f"Gate '{gate}' is not a Clifford gate; the stabilizer backend "
                f"supports only {sorted(_SUPPORTED_1Q)} (1q) and {sorted(_SUPPORTED_2Q)} (2q).")
        if gate not in _SUPPORTED_1Q:
            raise ValueError(f"Unknown single-qubit gate '{gate}'")
        a = self._check_qubit(qubit)
        t = self._tab
        if gate == "H":
            t.h(a)
        elif gate == "S":
            t.s(a)
        elif gate == "X":
            t.px(a)
        elif gate == "Y":
            t.py(a)
        else:  # "Z"
            t.pz(a)

    def apply_2q_gate(self, gate: str, ctrl: int, tgt: int, **params) -> None:
        self._require_init()
        if gate in _NON_CLIFFORD_2Q:
            raise UnsupportedOperationError(
                f"Gate '{gate}' is not a Clifford gate; the stabilizer backend "
                f"supports only {sorted(_SUPPORTED_2Q)} as two-qubit gates.")
        if gate not in _SUPPORTED_2Q:
            raise ValueError(f"Unknown two-qubit gate '{gate}'")
        a, b = self._check_qubit(ctrl), self._check_qubit(tgt)
        if a == b:
            raise ValueError("control and target must be different qubits")
        t = self._tab
        if gate == "CNOT":
            t.cnot(a, b)
        elif gate == "CZ":
            t.cz(a, b)
        else:  # "SWAP"
            t.swap(a, b)

    # ------------------------------------------------------------ measurement

    def measure(self, qubit: int) -> int:
        """Z-basis measurement of one qubit; collapses the state."""
        self._require_init()
        a = self._check_qubit(qubit)
        return self._tab.measure(a, int(self._rng.integers(0, 2)))

    def sample(self, shots: int) -> dict:
        """Sample all qubits `shots` times without disturbing the state.

        A stabilizer state's computational-basis distribution is UNIFORM over
        an affine subspace  v0 + rowspace(X_stab),  where X_stab is the n x n
        X-block of the stabilizer generators and v0 is any one possible outcome.
        So: get v0 by measuring every qubit on a COPY of the tableau, reduce
        X_stab to a basis of k rows, and draw  v0 XOR (random combination).
        Cost: O(n^3) once + O(shots * k * n), instead of O(shots * n^3).
        The kernel (C++ Tableau::sample) does steps 1-3; Python only supplies
        the random bits and counts the results.
        """
        self._require_init()
        if not isinstance(shots, (int, np.integer)) or isinstance(shots, bool) or shots < 0:
            raise ValueError(f"shots must be a non-negative integer, got {shots!r}")
        shots = int(shots)
        if shots == 0:
            return {}
        n = self._n

        # Random coin flips (one row of n per shot; the kernel uses the first k).
        rand = self._rng.integers(0, 2, size=(shots, n), dtype=np.uint8)
        outcomes = self._tab.sample_bits(shots, rand)          # shots x n of 0/1

        # Count distinct bit-strings (qubit 0 = leftmost character).
        uniq, counts = np.unique(outcomes, axis=0, return_counts=True)
        chars = np.where(uniq == 1, "1", "0")
        return {"".join(row): int(c) for row, c in zip(chars, counts)}

    # ------------------------------------------------------------ observables

    def expectation(self, pauli_string: str) -> float:
        """<psi|P|psi> for a Pauli string; always exactly +1, -1 or 0."""
        self._require_init()
        if not isinstance(pauli_string, str) or len(pauli_string) != self._n:
            raise ValueError(
                f"pauli_string must be a str of length {self._n}, got {pauli_string!r}")
        try:
            bits = [_PAULI_BITS[c] for c in pauli_string]
        except KeyError as exc:
            raise ValueError(f"invalid Pauli character {exc.args[0]!r}; use I, X, Y, Z") from None
        px = np.array([b[0] for b in bits], dtype=np.uint8)
        pz = np.array([b[1] for b in bits], dtype=np.uint8)
        return float(self._tab.expectation(px, pz))

    # ------------------------------------------------------------ debugging

    def _tableau_str(self) -> str:
        """Human-readable tableau (destabilizers, then stabilizers)."""
        self._require_init()
        xs, zs, rs = self._tab.dump()
        letters = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}
        lines = []
        for i in range(2 * self._n):
            if i == self._n:
                lines.append("-" * (self._n + 1))
            sign = "-" if rs[i] else "+"
            lines.append(sign + "".join(letters[(int(xs[i, j]), int(zs[i, j]))]
                                        for j in range(self._n)))
        return "\n".join(lines)