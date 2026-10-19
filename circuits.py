# COL8422 Quantum Computing Systems — Project Harness
# Standard circuit generators and analytical reference statevectors.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulator import QuantumSimulator

__all__ = [
    "ghz",
    "ghz_statevector",
    "qft",
    "qft_statevector",
    "bv_circuit",
    "bv_expected_outcome",
    "qaoa_1d",
    "near_clifford",
    "random_universal",
]


# ---------------------------------------------------------------------------
# GHZ
# ---------------------------------------------------------------------------

def ghz(sim: "QuantumSimulator", n: int) -> None:
    """Apply the n-qubit GHZ circuit to *sim*.

    Prepares the state (|0…0⟩ + |1…1⟩) / √2 starting from |0…0⟩.
    The simulator must already be initialised with *n* qubits.

    Parameters
    ----------
    sim:
        An initialised :class:`~simulator.QuantumSimulator` instance.
    n:
        Number of qubits.
    """
    sim.apply_1q_gate("H", 0)
    for i in range(1, n):
        sim.apply_2q_gate("CNOT", 0, i)


def ghz_statevector(n: int) -> np.ndarray:
    """Return the exact statevector for the n-qubit GHZ state.

    |GHZ_n⟩ = (|0…0⟩ + |1…1⟩) / √2

    Parameters
    ----------
    n:
        Number of qubits.

    Returns
    -------
    np.ndarray
        Complex128 array of length 2**n with amplitudes 1/√2 at indices 0
        and 2**n - 1, and zero elsewhere.
    """
    sv = np.zeros(2**n, dtype=np.complex128)
    sv[0] = 1.0 / math.sqrt(2)
    sv[-1] = 1.0 / math.sqrt(2)
    return sv


# ---------------------------------------------------------------------------
# QFT
# ---------------------------------------------------------------------------

def qft(sim: "QuantumSimulator", n: int) -> None:
    """Apply the n-qubit Quantum Fourier Transform circuit to *sim*.

    Uses the standard textbook decomposition: for each qubit *k* from 0 to
    n-1, apply H then controlled-phase (CU with phi=0, lam=0, theta=0 acting
    as a phase) rotations from higher-indexed qubits, followed by bit-reversal
    SWAP network.

    The controlled-R_k gate is implemented as CU(theta=0, phi=0, lam=π/2^{k-1})
    which gives the diagonal matrix diag(1, 1, 1, e^{iπ/2^{k-1}}) — i.e. a
    phase of e^{iπ/2^{k-1}} on |11⟩.

    Parameters
    ----------
    sim:
        An initialised :class:`~simulator.QuantumSimulator` instance.
    n:
        Number of qubits.
    """
    for k in range(n):
        sim.apply_1q_gate("H", k)
        for m in range(k + 1, n):
            angle = math.pi / (2 ** (m - k))
            # CU with theta=0, phi=0, lam=angle implements the controlled phase R_{m-k+1}
            sim.apply_2q_gate("CU", m, k, theta=0.0, phi=0.0, lam=angle)
    # Bit-reversal swap network
    for i in range(n // 2):
        sim.apply_2q_gate("SWAP", i, n - 1 - i)


def qft_statevector(n: int, input_state: int = 0) -> np.ndarray:
    """Return the exact statevector after applying QFT to a computational basis state.

    QFT|j⟩ = (1/√N) ∑_{k=0}^{N-1} e^{2πi jk/N} |k⟩,  where N = 2**n.

    Parameters
    ----------
    n:
        Number of qubits.
    input_state:
        Index *j* of the computational basis state to transform.
        Must satisfy 0 ≤ input_state < 2**n.

    Returns
    -------
    np.ndarray
        Complex128 array of length 2**n.
    """
    N = 2**n
    if not (0 <= input_state < N):
        raise ValueError(f"input_state {input_state} out of range [0, {N})")
    j = input_state
    k = np.arange(N, dtype=np.float64)
    phases = np.exp(2j * math.pi * j * k / N)
    return phases / math.sqrt(N)


# ---------------------------------------------------------------------------
# Bernstein-Vazirani
# ---------------------------------------------------------------------------

def bv_circuit(sim: "QuantumSimulator", n: int, s: int) -> None:
    """Apply the Bernstein-Vazirani circuit for hidden string *s* to *sim*.

    The circuit queries the oracle f(x) = s·x (mod 2) and produces the
    hidden string *s* as the measurement outcome with probability 1.

    Circuit structure (n data qubits + 1 ancilla at index n):
      1. Initialise ancilla to |−⟩ = H|1⟩.
      2. Apply H to all data qubits (|0…0⟩ → |+…+⟩).
      3. Oracle: for each bit i of s, apply CNOT(ctrl=i, tgt=n).
      4. Apply H to all data qubits.

    The simulator must be initialised with n + 1 qubits before calling this
    function.

    Parameters
    ----------
    sim:
        Initialised simulator with n+1 qubits.
    n:
        Number of data qubits (hidden string has n bits).
    s:
        Hidden string encoded as a non-negative integer; bit i of s
        (0-indexed from LSB) corresponds to data qubit i.
    """
    ancilla = n
    # Ancilla → |1⟩ → |−⟩
    sim.apply_1q_gate("X", ancilla)
    sim.apply_1q_gate("H", ancilla)
    # Data qubits → |+⟩
    for i in range(n):
        sim.apply_1q_gate("H", i)
    # Oracle
    for i in range(n):
        if (s >> i) & 1:
            sim.apply_2q_gate("CNOT", i, ancilla)
    # Hadamard to read out s
    for i in range(n):
        sim.apply_1q_gate("H", i)


def bv_expected_outcome(n: int, s: int) -> str:
    """Return the expected measurement outcome bit-string for BV with hidden string *s*.

    Parameters
    ----------
    n:
        Number of data qubits.
    s:
        Hidden string as an integer.

    Returns
    -------
    str
        Zero-padded binary string of length n representing *s*, with qubit 0
        at position 0 (LSB first).
    """
    return format(s, f"0{n}b")[::-1]  # LSB at index 0, matching qubit ordering


# ---------------------------------------------------------------------------
# QAOA on 1D chain MaxCut
# ---------------------------------------------------------------------------

def qaoa_1d(
    sim: "QuantumSimulator",
    n: int,
    p: int,
    gammas: list[float],
    betas: list[float],
) -> None:
    """Apply a QAOA circuit for MaxCut on a 1D chain (path graph) to *sim*.

    The cost Hamiltonian is H_C = ∑_{i=0}^{n-2} (I - Z_i Z_{i+1}) / 2.
    The mixer is H_B = ∑_i X_i.

    Circuit layout:
      1. Prepare |+⟩^⊗n (H on each qubit).
      2. For each of p layers:
           a. Cost layer U_C(γ): for each edge (i, i+1), apply
              RZZ(γ) = CNOT · RZ(2γ) · CNOT (implements e^{-iγ Z_i Z_{i+1}}).
           b. Mixer layer U_B(β): apply RX(2β) = U(2β, -π/2, π/2) on each qubit.

    The simulator must be initialised with *n* qubits before calling.

    Parameters
    ----------
    sim:
        Initialised simulator.
    n:
        Number of qubits / graph vertices.
    p:
        Number of QAOA layers.
    gammas:
        List of p cost-layer angles.
    betas:
        List of p mixer-layer angles.
    """
    if len(gammas) != p or len(betas) != p:
        raise ValueError(f"Expected {p} gammas and {p} betas, got {len(gammas)} and {len(betas)}.")

    # Initial state |+⟩^n
    for i in range(n):
        sim.apply_1q_gate("H", i)

    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        # Cost layer: RZZ(2*gamma) on each edge
        for i in range(n - 1):
            sim.apply_2q_gate("CNOT", i, i + 1)
            sim.apply_1q_gate("U", i + 1, theta=2 * gamma, phi=0.0, lam=0.0)
            sim.apply_2q_gate("CNOT", i, i + 1)

        # Mixer layer: RX(2*beta) on each qubit
        # RX(θ) = U(θ, -π/2, π/2)
        for i in range(n):
            sim.apply_1q_gate("U", i, theta=2 * beta, phi=-math.pi / 2, lam=math.pi / 2)


# ---------------------------------------------------------------------------
# Near-Clifford circuit
# ---------------------------------------------------------------------------

def near_clifford(
    sim: "QuantumSimulator",
    n: int,
    rng: np.random.Generator,
    t_count: int,
) -> None:
    """Apply a random Clifford circuit with *t_count* randomly inserted T gates.

    The circuit is constructed as:
      1. A random Clifford-like base: alternating layers of random single-qubit
         Clifford gates (H, S, X, Z chosen uniformly) and CNOT ladders on
         random neighbouring pairs.
      2. *t_count* T gates inserted at random (qubit, layer-slot) positions.

    This produces a circuit that is efficiently simulable for t_count = 0
    (stabilizer) and increasingly classically hard as t_count grows.

    The simulator must be initialised with *n* qubits before calling.

    Parameters
    ----------
    sim:
        Initialised simulator.
    n:
        Number of qubits.
    rng:
        A :class:`numpy.random.Generator` for reproducibility.
    t_count:
        Number of T gates to inject into the Clifford skeleton.
    """
    clifford_1q = ["H", "S", "X", "Z"]
    depth = max(4 * n, 20)

    # Decide T gate injection sites: (qubit, step) pairs
    t_positions: set[tuple[int, int]] = set()
    while len(t_positions) < min(t_count, n * depth):
        q = int(rng.integers(0, n))
        d = int(rng.integers(0, depth))
        t_positions.add((q, d))

    for step in range(depth):
        # Single-qubit layer
        for q in range(n):
            if (q, step) in t_positions:
                sim.apply_1q_gate("T", q)
            else:
                gate = clifford_1q[int(rng.integers(0, len(clifford_1q)))]
                sim.apply_1q_gate(gate, q)
        # Random CNOT on a random neighbouring pair
        if n >= 2:
            ctrl = int(rng.integers(0, n - 1))
            sim.apply_2q_gate("CNOT", ctrl, ctrl + 1)


# ---------------------------------------------------------------------------
# Random universal circuit
# ---------------------------------------------------------------------------

def random_universal(
    sim: "QuantumSimulator",
    n: int,
    rng: np.random.Generator,
    depth_factor: int = 10,
) -> None:
    """Apply a Haar-random circuit of U gates and a CNOT ladder to *sim*.

    The circuit consists of ``depth_factor * n`` layers, each containing:
      - A U3(θ, φ, λ) gate on every qubit with angles drawn from the Haar
        measure (θ ~ arccos(uniform(-1,1)), φ,λ ~ uniform(0, 2π)).
      - A CNOT ladder: CNOT(0→1), CNOT(2→3), … on even-offset pairs, and on
        odd steps CNOT(1→2), CNOT(3→4), … (brick-wall pattern).

    The simulator must be initialised with *n* qubits before calling.

    Parameters
    ----------
    sim:
        Initialised simulator.
    n:
        Number of qubits.
    rng:
        A :class:`numpy.random.Generator` for reproducibility.
    depth_factor:
        Total number of layers = depth_factor * n.
    """
    total_layers = depth_factor * n

    for layer in range(total_layers):
        # Haar-random single-qubit unitaries
        for q in range(n):
            # θ = arccos(u) where u ~ Uniform(-1,1) gives correct Haar measure on SU(2)
            u = rng.uniform(-1.0, 1.0)
            theta = float(np.arccos(u))
            phi = float(rng.uniform(0.0, 2 * math.pi))
            lam = float(rng.uniform(0.0, 2 * math.pi))
            sim.apply_1q_gate("U", q, theta=theta, phi=phi, lam=lam)

        # Brick-wall CNOT layer
        offset = layer % 2
        for ctrl in range(offset, n - 1, 2):
            sim.apply_2q_gate("CNOT", ctrl, ctrl + 1)
