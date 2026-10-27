"""Reduced weighted decision-diagram state simulator.

This implementation represents a state vector as a canonical, weighted binary
decision diagram. It does not materialize a dense statevector during gate
application, measurement, sampling, or expectation evaluation.

Representation:
  Edge = (complex weight, node id)
  Internal node = (qubit level, low edge, high edge)
  terminal node represents scalar 1
Node normalization extracts the first nonzero outgoing edge weight and uses a
unique table to merge equivalent normalized nodes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from simulator import QuantumSimulator, UnsupportedOperationError

_EPS = 1e-13
_KEY_DIGITS = 12
_ZERO_NODE = 0
_Edge = tuple[complex, int]


def _clean(z: complex) -> complex:
    z = complex(z)
    if abs(z.real) < _EPS:
        z = complex(0.0, z.imag)
    if abs(z.imag) < _EPS:
        z = complex(z.real, 0.0)
    return z


def _qcomplex(z: complex) -> tuple[float, float]:
    z = _clean(z)
    return (round(z.real, _KEY_DIGITS), round(z.imag, _KEY_DIGITS))


@dataclass(frozen=True)
class _Node:
    level: int
    low: _Edge
    high: _Edge


class _WeightedDD:
    """Canonical weighted DD manager for a fixed number of qubits."""

    def __init__(self, n: int):
        self.n = n
        # Node 0 is the terminal at level -1.
        self.nodes: list[_Node | None] = [None]
        self.unique: dict[tuple, int] = {}
        self._add_cache: dict[tuple, _Edge] = {}
        self._oneq_cache: dict[tuple, _Edge] = {}
        self._norm_cache: dict[int, float] = {}
        self.root: _Edge = (1.0 + 0.0j, _ZERO_NODE)

    def clear_caches(self) -> None:
        self._add_cache.clear()
        self._oneq_cache.clear()
        self._norm_cache.clear()

    def _node_key(self, level: int, low: _Edge, high: _Edge) -> tuple:
        return (level, low[1], _qcomplex(low[0]), high[1], _qcomplex(high[0]))

    def make_node(self, level: int, low: _Edge, high: _Edge) -> _Edge:
        """Return canonical edge for a node with given child edges."""
        low = (_clean(low[0]), low[1])
        high = (_clean(high[0]), high[1])
        if abs(low[0]) < _EPS:
            low = (0.0 + 0.0j, _ZERO_NODE)
        if abs(high[0]) < _EPS:
            high = (0.0 + 0.0j, _ZERO_NODE)
        if abs(low[0]) < _EPS and abs(high[0]) < _EPS:
            return (0.0 + 0.0j, _ZERO_NODE)

        # Normalize by first nonzero outgoing weight. The external edge
        # carries the removed common factor.
        pivot = low[0] if abs(low[0]) >= _EPS else high[0]
        norm_low = (low[0] / pivot, low[1]) if abs(low[0]) >= _EPS else (0.0 + 0.0j, _ZERO_NODE)
        norm_high = (high[0] / pivot, high[1]) if abs(high[0]) >= _EPS else (0.0 + 0.0j, _ZERO_NODE)
        norm_low = (_clean(norm_low[0]), norm_low[1])
        norm_high = (_clean(norm_high[0]), norm_high[1])
        key = self._node_key(level, norm_low, norm_high)
        node_id = self.unique.get(key)
        if node_id is None:
            node_id = len(self.nodes)
            self.nodes.append(_Node(level, norm_low, norm_high))
            self.unique[key] = node_id
        return (_clean(pivot), node_id)

    def edge_children(self, edge: _Edge) -> tuple[_Edge, _Edge]:
        weight, node_id = edge
        if abs(weight) < _EPS:
            z = (0.0 + 0.0j, _ZERO_NODE)
            return z, z
        if node_id == _ZERO_NODE:
            # A nonzero terminal has no children; this is only used at level -1.
            return (weight, node_id), (0.0 + 0.0j, _ZERO_NODE)
        node = self.nodes[node_id]
        assert node is not None
        return ((weight * node.low[0], node.low[1]),
                (weight * node.high[0], node.high[1]))

    def add(self, a: _Edge, b: _Edge) -> _Edge:
        """Canonical sum of two represented sub-vectors at the same level."""
        if abs(a[0]) < _EPS:
            return b
        if abs(b[0]) < _EPS:
            return a
        if a[1] == b[1]:
            combined = _clean(a[0] + b[0])
            if abs(combined) < _EPS:
                return (0.0 + 0.0j, _ZERO_NODE)
            return (combined, a[1])

        na, nb = a[1], b[1]
        if na == _ZERO_NODE and nb == _ZERO_NODE:
            return (_clean(a[0] + b[0]), _ZERO_NODE)
        # Terminal and internal nodes only occur at matching levels in our
        # fixed-level representation, except for a zero edge.
        key = (na, _qcomplex(a[0]), nb, _qcomplex(b[0]))
        cached = self._add_cache.get(key)
        if cached is not None:
            return cached

        if na == _ZERO_NODE:
            node_b = self.nodes[nb]
            assert node_b is not None
            low_b = (b[0] * node_b.low[0], node_b.low[1])
            high_b = (b[0] * node_b.high[0], node_b.high[1])
            low = self.add((0j, 0), low_b)
            high = self.add((0j, 0), high_b)
            result = self.make_node(node_b.level, low, high)
        elif nb == _ZERO_NODE:
            node_a = self.nodes[na]
            assert node_a is not None
            low_a = (a[0] * node_a.low[0], node_a.low[1])
            high_a = (a[0] * node_a.high[0], node_a.high[1])
            result = self.make_node(node_a.level, low_a, high_a)
        else:
            node_a, node_b = self.nodes[na], self.nodes[nb]
            assert node_a is not None and node_b is not None
            if node_a.level != node_b.level:
                raise RuntimeError("DD addition encountered mismatched levels")
            low_a = (a[0] * node_a.low[0], node_a.low[1])
            high_a = (a[0] * node_a.high[0], node_a.high[1])
            low_b = (b[0] * node_b.low[0], node_b.low[1])
            high_b = (b[0] * node_b.high[0], node_b.high[1])
            result = self.make_node(node_a.level, self.add(low_a, low_b), self.add(high_a, high_b))
        self._add_cache[key] = result
        return result

    def scale(self, edge: _Edge, scalar: complex) -> _Edge:
        w = _clean(edge[0] * scalar)
        if abs(w) < _EPS:
            return (0j, _ZERO_NODE)
        return (w, edge[1])

    def apply_1q(self, edge: _Edge, q: int, matrix: np.ndarray) -> _Edge:
        """Apply a 2x2 matrix at qubit q without expanding the represented vector."""
        if abs(edge[0]) < _EPS:
            return (0j, _ZERO_NODE)
        node_id = edge[1]
        if node_id == _ZERO_NODE:
            raise RuntimeError("single-qubit gate reached terminal too early")
        node = self.nodes[node_id]
        assert node is not None
        key = (node_id, _qcomplex(edge[0]), q, tuple(_qcomplex(complex(x)) for x in matrix.flat))
        cached = self._oneq_cache.get(key)
        if cached is not None:
            return cached
        low = (edge[0] * node.low[0], node.low[1])
        high = (edge[0] * node.high[0], node.high[1])
        if node.level > q:
            result = self.make_node(
                node.level,
                self.apply_1q(low, q, matrix),
                self.apply_1q(high, q, matrix),
            )
        elif node.level == q:
            out_low = self.add(self.scale(low, matrix[0, 0]), self.scale(high, matrix[0, 1]))
            out_high = self.add(self.scale(low, matrix[1, 0]), self.scale(high, matrix[1, 1]))
            result = self.make_node(q, out_low, out_high)
        else:
            raise RuntimeError(f"qubit level {q} missing from DD path at level {node.level}")
        self._oneq_cache[key] = result
        return result

    def norm2(self, edge: _Edge) -> float:
        if abs(edge[0]) < _EPS:
            return 0.0
        if edge[1] == _ZERO_NODE:
            return float(abs(edge[0]) ** 2)
        node = self.nodes[edge[1]]
        assert node is not None
        # The node's children are normalized edges; recursively account for
        # the weight on this incoming edge.
        low = (edge[0] * node.low[0], node.low[1])
        high = (edge[0] * node.high[0], node.high[1])
        return self.norm2(low) + self.norm2(high)

    def probability(self, edge: _Edge, q: int, outcome: int) -> float:
        if abs(edge[0]) < _EPS:
            return 0.0
        if edge[1] == _ZERO_NODE:
            return 0.0
        node = self.nodes[edge[1]]
        assert node is not None
        low = (edge[0] * node.low[0], node.low[1])
        high = (edge[0] * node.high[0], node.high[1])
        if node.level == q:
            return self.norm2(high if outcome else low)
        if node.level < q:
            raise RuntimeError("measurement qubit level missing from DD")
        return self.probability(low, q, outcome) + self.probability(high, q, outcome)

    def project(self, edge: _Edge, q: int, outcome: int) -> _Edge:
        if abs(edge[0]) < _EPS:
            return (0j, _ZERO_NODE)
        node = self.nodes[edge[1]]
        if node is None:
            raise RuntimeError("measurement qubit level missing from DD")
        low = (edge[0] * node.low[0], node.low[1])
        high = (edge[0] * node.high[0], node.high[1])
        if node.level > q:
            return self.make_node(node.level, self.project(low, q, outcome), self.project(high, q, outcome))
        if node.level == q:
            zero = (0j, _ZERO_NODE)
            return self.make_node(q, zero if outcome else low, high if outcome else zero)
        raise RuntimeError("measurement qubit level missing from DD")

    def amplitude(self, edge: _Edge, index: int) -> complex:
        w, node_id = edge
        if abs(w) < _EPS:
            return 0j
        for level in range(self.n - 1, -1, -1):
            if node_id == _ZERO_NODE:
                return 0j
            node = self.nodes[node_id]
            assert node is not None
            bit = (index >> level) & 1
            child = node.high if bit else node.low
            w *= child[0]
            node_id = child[1]
        return w if node_id == _ZERO_NODE else 0j

    def to_statevector(self, edge: _Edge) -> np.ndarray:
        # Only for harness diagnostics/debugging; never used by gate kernels.
        out = np.empty(1 << self.n, dtype=np.complex128)
        for i in range(out.size):
            out[i] = self.amplitude(edge, i)
        return out

    def inner(self, a: _Edge, b: _Edge) -> complex:
        if abs(a[0]) < _EPS or abs(b[0]) < _EPS:
            return 0j
        if a[1] == _ZERO_NODE and b[1] == _ZERO_NODE:
            return np.conjugate(a[0]) * b[0]
        na, nb = self.nodes[a[1]], self.nodes[b[1]]
        if na is None or nb is None or na.level != nb.level:
            raise RuntimeError("inner product encountered mismatched DD levels")
        al = (a[0] * na.low[0], na.low[1])
        ah = (a[0] * na.high[0], na.high[1])
        bl = (b[0] * nb.low[0], nb.low[1])
        bh = (b[0] * nb.high[0], nb.high[1])
        return self.inner(al, bl) + self.inner(ah, bh)


class DDSimulator(QuantumSimulator):
    def __init__(self) -> None:
        self._n = 0
        self._dd: _WeightedDD | None = None
        self._rng = np.random.default_rng()

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def statevector(self) -> np.ndarray:
        """Debug/harness compatibility accessor; not part of common interface."""
        assert self._dd is not None
        return self._dd.to_statevector(self._dd.root)

    @property
    def node_count(self) -> int:
        """Number of allocated DD nodes (including nodes no longer reachable)."""
        return 0 if self._dd is None else len(self._dd.nodes) - 1

    def initialize(self, n: int) -> None:
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        self._n = n
        self._dd = _WeightedDD(n)
        edge: _Edge = (1.0 + 0.0j, _ZERO_NODE)
        zero: _Edge = (0j, _ZERO_NODE)
        # |0...0> as a chain whose high edge is zero at every level.
        for q in range(n):
            edge = self._dd.make_node(q, edge, zero)
        self._dd.root = edge

    def _manager(self) -> _WeightedDD:
        if self._dd is None:
            raise RuntimeError("Call initialize(n) before simulating")
        return self._dd

    def _check_qubit(self, q: int) -> None:
        if not isinstance(q, int) or q < 0 or q >= self._n:
            raise ValueError(f"qubit index {q} outside [0, {self._n})")

    @staticmethod
    def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ], dtype=np.complex128)

    @staticmethod
    def _gate_matrix(gate: str, params: dict[str, Any] | None = None) -> np.ndarray:
        r = 1 / math.sqrt(2)
        mats = {
            "H": [[r, r], [r, -r]],
            "X": [[0, 1], [1, 0]],
            "Y": [[0, -1j], [1j, 0]],
            "Z": [[1, 0], [0, -1]],
            "S": [[1, 0], [0, 1j]],
            "T": [[1, 0], [0, np.exp(1j * math.pi / 4)]],
        }
        if gate == "U":
            p = params or {}
            try:
                return DDSimulator._u3(float(p["theta"]), float(p["phi"]), float(p["lam"]))
            except KeyError as exc:
                raise ValueError("U requires theta, phi, and lam") from exc
        if gate not in mats:
            raise UnsupportedOperationError(f"Unsupported 1-qubit gate: {gate}")
        return np.asarray(mats[gate], dtype=np.complex128)

    def apply_1q_gate(self, gate: str, qubit: int, **params: Any) -> None:
        self._check_qubit(qubit)
        dd = self._manager()
        matrix = self._gate_matrix(gate, params)
        dd.root = dd.apply_1q(dd.root, qubit, matrix)
        dd.clear_caches()

    def _twoq_matrix(self, gate: str, params: dict[str, Any]) -> np.ndarray:
        if gate == "CNOT":
            # Matrix basis index is (ctrl_bit << 1) | tgt_bit.
            m = np.zeros((4, 4), dtype=np.complex128)
            for c in (0, 1):
                for t in (0, 1):
                    inp = 2*c + t
                    out = 2*c + (t ^ c)
                    m[out, inp] = 1
            return m
        if gate == "CZ":
            return np.diag([1, 1, 1, -1]).astype(np.complex128)
        if gate == "SWAP":
            return np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex128)
        if gate == "CU":
            u = self._u3(float(params["theta"]), float(params["phi"]), float(params["lam"]))
            m = np.eye(4, dtype=np.complex128)
            m[2:4, 2:4] = u
            return m
        raise UnsupportedOperationError(f"Unsupported 2-qubit gate: {gate}")

    def _apply_2q_matrix(self, matrix: np.ndarray, ctrl: int, tgt: int) -> None:
        dd = self._manager()
        hi, lo = max(ctrl, tgt), min(ctrl, tgt)
        # Map (ctrl,tgt) bit labels to (high-level bit, low-level bit).
        def basis_index(hi_bit: int, lo_bit: int) -> int:
            bits = {hi: hi_bit, lo: lo_bit}
            return 2 * bits[ctrl] + bits[tgt]

        # At level hi, each branch is a vector over all lower qubits. For each
        # output high bit, apply the appropriate 2x2 block to low bit to each
        # input high branch, then add the resulting sub-vectors.
        root = dd.root

        def recurse(edge: _Edge) -> _Edge:
            if abs(edge[0]) < _EPS:
                return (0j, _ZERO_NODE)
            node = dd.nodes[edge[1]]
            if node is None:
                raise RuntimeError("two-qubit gate reached terminal too early")
            low_edge = (edge[0] * node.low[0], node.low[1])
            high_edge = (edge[0] * node.high[0], node.high[1])
            if node.level > hi:
                return dd.make_node(node.level, recurse(low_edge), recurse(high_edge))
            if node.level != hi:
                raise RuntimeError("two-qubit gate level missing in DD")
            in_hi = [low_edge, high_edge]
            out_hi: list[_Edge] = []
            for out_bit in (0, 1):
                accum: _Edge = (0j, _ZERO_NODE)
                for in_bit in (0, 1):
                    # Matrix coefficients form a 2x2 operator on the lower
                    # target/control qubit for each fixed high-bit transition.
                    block = np.empty((2, 2), dtype=np.complex128)
                    for out_lo in (0, 1):
                        for in_lo in (0, 1):
                            block[out_lo, in_lo] = matrix[
                                basis_index(out_bit, out_lo),
                                basis_index(in_bit, in_lo)
                            ]
                    transformed = dd.apply_1q(in_hi[in_bit], lo, block)
                    accum = dd.add(accum, transformed)
                out_hi.append(accum)
            return dd.make_node(hi, out_hi[0], out_hi[1])

        dd.root = recurse(root)
        dd.clear_caches()

    def apply_2q_gate(self, gate: str, ctrl: int, tgt: int, **params: Any) -> None:
        self._check_qubit(ctrl)
        self._check_qubit(tgt)
        if ctrl == tgt:
            raise ValueError("control and target must be different qubits")
        matrix = self._twoq_matrix(gate, params)
        self._apply_2q_matrix(matrix, ctrl, tgt)

    def measure(self, qubit: int) -> int:
        self._check_qubit(qubit)
        dd = self._manager()
        p1 = dd.probability(dd.root, qubit, 1)
        p1 = min(1.0, max(0.0, p1))
        outcome = int(self._rng.random() < p1)
        p = p1 if outcome else 1.0 - p1
        if p < _EPS:
            outcome = 1 - outcome
            p = p1 if outcome else 1.0 - p1
        dd.root = dd.project(dd.root, qubit, outcome)
        dd.root = dd.scale(dd.root, 1.0 / math.sqrt(p))
        dd.clear_caches()
        return outcome

    def sample(self, shots: int) -> dict[str, int]:
        if shots < 0:
            raise ValueError("shots must be non-negative")
        dd = self._manager()
        counts: dict[str, int] = {}
        for _ in range(shots):
            edge = dd.root
            bits = [0] * self._n
            for q in range(self._n - 1, -1, -1):
                p1 = dd.probability(edge, q, 1)
                p1 = min(1.0, max(0.0, p1))
                b = int(self._rng.random() < p1)
                p = p1 if b else 1.0 - p1
                if p < _EPS:
                    b = 1 - b
                    p = p1 if b else 1.0 - p1
                edge = dd.project(edge, q, b)
                edge = dd.scale(edge, 1.0 / math.sqrt(p))
                bits[q] = b
            key = "".join(str(b) for b in bits)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def expectation(self, pauli_string: str) -> float:
        if len(pauli_string) != self._n or any(c not in "IXYZ" for c in pauli_string):
            raise ValueError("pauli_string must have length n_qubits and use only I,X,Y,Z")
        dd = self._manager()
        transformed = dd.root
        mats = {
            "I": np.eye(2, dtype=np.complex128),
            "X": np.array([[0,1],[1,0]], dtype=np.complex128),
            "Y": np.array([[0,-1j],[1j,0]], dtype=np.complex128),
            "Z": np.array([[1,0],[0,-1]], dtype=np.complex128),
        }
        for q, p in enumerate(pauli_string):
            if p != "I":
                transformed = dd.apply_1q(transformed, q, mats[p])
        value = dd.inner(dd.root, transformed)
        return float(value.real)
