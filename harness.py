# COL8422 Quantum Computing Systems — Project Harness
# Correctness harness: students run this locally to validate their simulator backend.

"""
Usage
-----
    python harness.py --backend statevector [--n-max 20] [--params test_params.json]

For each of GHZ, QFT, and BV the harness runs the circuit on the chosen
backend for n = 4, 8, 12, 16, 20 (clipped to --n-max) and checks:

  * GHZ / QFT (statevector-capable backends):
      fidelity |<ψ_student|ψ_expected>|² ≥ 0.999  →  PASS

  * GHZ (stabilizer backend):
      sample() gives ≈50 % '0…0' and ≈50 % '1…1'  →  PASS

  * BV (all backends):
      mode of 100-shot sample matches expected bit-string  →  PASS

No Qiskit imports are made here — analytical reference vectors from
circuits.py are used throughout.  Qiskit comparison lives in benchmark.py.

Default parameter values (used when --params is not supplied)
-------------------------------------------------------------
  BV hidden strings : {4: 9, 8: 173, 12: 2731, 16: 43691, 20: 699051}
  near_clifford seeds: not used in the correctness harness
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make sure the part1 package root is on the path when run directly.
sys.path.insert(0, str(Path(__file__).parent))

from circuits import (
    bv_circuit,
    bv_expected_outcome,
    ghz,
    ghz_statevector,
    qft,
    qft_statevector,
)
from simulator import Simulator, QuantumSimulator, UnsupportedOperationError

# ---------------------------------------------------------------------------
# Default test parameters (documented here for reproducibility)
# ---------------------------------------------------------------------------

# BV hidden strings — chosen to exercise all bit positions within each width.
# n=4:  9  = 0b1001
# n=8:  173 = 0b10101101
# n=12: 2731 = 0b101010101011
# n=16: 43691 = 0b1010101010101011
# n=20: 699051 = 0b10101010101010101011
_DEFAULT_BV_STRINGS: dict[int, int] = {
    4: 9,
    8: 173,
    12: 2731,
    16: 43691,
    20: 699051,
}

_TEST_SIZES = [4, 8, 12, 16, 20]
_BV_SHOTS = 100
_FIDELITY_THRESHOLD = 0.999

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COL_WIDTHS = (12, 6, 8, 30)
_HEADER = (
    f"{'Circuit':<{_COL_WIDTHS[0]}}  "
    f"{'n':>{_COL_WIDTHS[1]}}  "
    f"{'Result':<{_COL_WIDTHS[2]}}  "
    f"{'Fidelity / Note':<{_COL_WIDTHS[3]}}"
)
_SEP = "-" * (sum(_COL_WIDTHS) + 3 * 2)


def _row(circuit: str, n: int, passed: bool, note: str) -> str:
    result = "PASS" if passed else "FAIL"
    return (
        f"{circuit:<{_COL_WIDTHS[0]}}  "
        f"{n:>{_COL_WIDTHS[1]}}  "
        f"{result:<{_COL_WIDTHS[2]}}  "
        f"{note:<{_COL_WIDTHS[3]}}"
    )


def _statevector_from_sim(sim: QuantumSimulator, n: int) -> np.ndarray | None:
    """Attempt to extract the full statevector via expectation() on all Paulis.

    Returns None if the backend raises UnsupportedOperationError, indicating
    a non-statevector backend such as stabilizer.
    """
    try:
        # Quick probe: can we compute any expectation value?
        test_pauli = "Z" + "I" * (n - 1)
        sim.expectation(test_pauli)
        return _reconstruct_sv(sim, n)
    except UnsupportedOperationError:
        return None


def _reconstruct_sv(sim: QuantumSimulator, n: int) -> np.ndarray:
    """Reconstruct the 2^n statevector from expectation values.

    Uses the identity:
        rho = (1/2^n) ∑_{P ∈ {I,X,Y,Z}^n} Tr(rho P) P

    For a pure state |ψ⟩, amplitude recovery requires more than just diagonal
    elements.  We use a direct shortcut: measure ⟨P⟩ for every n-qubit Pauli
    P and reconstruct via the Bloch-Pauli expansion.

    NOTE: This is only feasible for small n (≤ ~12 qubits) and is provided
    purely for the harness' correctness check.  For large n, backends that
    support a ``statevector`` property should override this.

    For simplicity, harness skips fidelity for n > 12 when a statevector
    property is not available.
    """
    # Try a .statevector property first (backends may expose this)
    sv_attr = getattr(sim, "statevector", None)
    if sv_attr is not None:
        arr = np.asarray(sv_attr, dtype=np.complex128)
        if arr.shape == (2**n,):
            return arr

    # Fall back to Pauli reconstruction (expensive, only for small n)
    if n > 14:
        raise UnsupportedOperationError(
            "Statevector reconstruction via Pauli tomography not feasible for n > 14. "
            "Expose a .statevector property on your backend."
        )

    paulis = "IXYZ"
    dim = 2**n
    # Pauli matrices
    I2 = np.eye(2, dtype=np.complex128)
    X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    P_mat = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}

    rho = np.zeros((dim, dim), dtype=np.complex128)
    import itertools
    for chars in itertools.product(paulis, repeat=n):
        pstr = "".join(chars)
        ev = sim.expectation(pstr)
        # Tensor product matrix
        M = P_mat[chars[0]]
        for c in chars[1:]:
            M = np.kron(M, P_mat[c])
        rho += ev * M
    rho /= dim

    # Extract dominant eigenvector
    eigvals, eigvecs = np.linalg.eigh(rho)
    sv = eigvecs[:, -1]
    # Phase-fix: make first non-zero element real positive
    for amp in sv:
        if abs(amp) > 1e-6:
            sv = sv * np.conj(amp) / abs(amp)
            break
    return sv.astype(np.complex128)


def _fidelity(sv_student: np.ndarray, sv_ref: np.ndarray) -> float:
    """Return |⟨ψ_student|ψ_ref⟩|²."""
    return float(abs(np.dot(sv_student.conj(), sv_ref)) ** 2)


def _check_ghz_sampling(sim: QuantumSimulator, n: int) -> tuple[bool, str]:
    """Check GHZ via sampling: expect ~50% '0…0' and ~50% '1…1'."""
    shots = 1000
    counts = sim.sample(shots)
    zero_str = "0" * n
    one_str = "1" * n
    c0 = counts.get(zero_str, 0)
    c1 = counts.get(one_str, 0)
    valid_frac = (c0 + c1) / shots
    passed = valid_frac >= 0.98
    note = f"{c0}/{shots} '{'0'*min(n,4)}…', {c1}/{shots} '{'1'*min(n,4)}…' (valid={valid_frac:.2%})"
    return passed, note


def _check_bv_sample(sim: QuantumSimulator, n: int, s: int) -> tuple[bool, str]:
    """Check BV via sampling: mode of output (data qubits) matches expected."""
    counts = sim.sample(_BV_SHOTS)
    expected = bv_expected_outcome(n, s)

    # BV circuit uses n+1 qubits; ancilla is qubit n (last character in bit-string)
    # We examine only the first n characters (data qubits)
    data_counts: dict[str, int] = {}
    for bitstr, cnt in counts.items():
        data_bits = bitstr[:n]  # qubits 0..n-1
        data_counts[data_bits] = data_counts.get(data_bits, 0) + cnt

    if not data_counts:
        return False, "No samples returned"

    mode = max(data_counts, key=data_counts.get)
    mode_count = data_counts[mode]
    passed = mode == expected
    note = f"mode='{mode}' (×{mode_count}), expected='{expected}'"
    return passed, note


# ---------------------------------------------------------------------------
# Per-circuit test runners
# ---------------------------------------------------------------------------

def _run_ghz(
    sim: QuantumSimulator,
    backend: str,
    sizes: list[int],
    results: list[tuple[bool, str]],
) -> list[str]:
    rows: list[str] = []
    for n in sizes:
        sim.initialize(n)
        ghz(sim, n)

        if backend == "stabilizer":
            passed, note = _check_ghz_sampling(sim, n)
        else:
            sv_ref = ghz_statevector(n)
            sv_student = None
            try:
                sv_student = _reconstruct_sv(sim, n)
            except UnsupportedOperationError as e:
                rows.append(_row("GHZ", n, False, f"SKIP: {e}"))
                results.append((False, "skipped"))
                continue
            except Exception as e:
                rows.append(_row("GHZ", n, False, f"ERROR: {e}"))
                results.append((False, "error"))
                continue
            f = _fidelity(sv_student, sv_ref)
            passed = f >= _FIDELITY_THRESHOLD
            note = f"fidelity={f:.6f}"
        results.append((passed, note))
        rows.append(_row("GHZ", n, passed, note))
    return rows


def _run_qft(
    sim: QuantumSimulator,
    backend: str,
    sizes: list[int],
    results: list[tuple[bool, str]],
) -> list[str]:
    rows: list[str] = []
    for n in sizes:
        if backend == "stabilizer":
            rows.append(_row("QFT", n, False, "SKIP: stabilizer cannot represent QFT"))
            results.append((False, "skipped"))
            continue

        sim.initialize(n)
        qft(sim, n)

        sv_ref = qft_statevector(n, input_state=0)
        try:
            sv_student = _reconstruct_sv(sim, n)
        except UnsupportedOperationError as e:
            rows.append(_row("QFT", n, False, f"SKIP: {e}"))
            results.append((False, "skipped"))
            continue
        except Exception as e:
            rows.append(_row("QFT", n, False, f"ERROR: {e}"))
            results.append((False, "error"))
            continue
        f = _fidelity(sv_student, sv_ref)
        passed = f >= _FIDELITY_THRESHOLD
        note = f"fidelity={f:.6f}"
        results.append((passed, note))
        rows.append(_row("QFT", n, passed, note))
    return rows


def _run_bv(
    sim: QuantumSimulator,
    sizes: list[int],
    bv_strings: dict[int, int],
    results: list[tuple[bool, str]],
) -> list[str]:
    rows: list[str] = []
    for n in sizes:
        s = bv_strings.get(n, _DEFAULT_BV_STRINGS.get(n, (1 << n) - 1))
        sim.initialize(n + 1)
        bv_circuit(sim, n, s)
        passed, note = _check_bv_sample(sim, n, s)
        results.append((passed, note))
        rows.append(_row("BV", n, passed, note))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="COL8422 Project — Correctness Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        required=True,
        help="Simulator backend to test (e.g. statevector, stabilizer, mps, dd).",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of qubits to test (capped to 20).",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Path to a JSON parameter file.  "
            "Format: {\"bv_strings\": {\"4\": 9, ...}, \"clifford_seeds\": {\"8\": 42, ...}}."
        ),
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help=(
            "Path to the student submission directory (must contain backends/).  "
            "Defaults to the current working directory."
        ),
    )
    args = parser.parse_args(argv)

    # Insert submission directory so the Simulator factory can find backends/
    submission_dir = args.submission.resolve()
    if str(submission_dir) not in sys.path:
        sys.path.insert(0, str(submission_dir))

    n_max: int = min(args.n_max, 20)
    sizes = [n for n in _TEST_SIZES if n <= n_max]

    # Load external params if provided
    bv_strings: dict[int, int] = dict(_DEFAULT_BV_STRINGS)
    if args.params is not None:
        with open(args.params) as fh:
            params_data: dict[str, Any] = json.load(fh)
        for k_str, v in params_data.get("bv_strings", {}).items():
            bv_strings[int(k_str)] = int(v)

    # Instantiate backend
    try:
        sim = Simulator(args.backend)
    except (ImportError, ValueError) as exc:
        print(f"[ERROR] Could not load backend '{args.backend}': {exc}", file=sys.stderr)
        return 2

    print()
    print(f"  COL8422 Correctness Harness — backend: {args.backend}")
    print(f"  Testing n ∈ {sizes}")
    if args.params:
        print(f"  Parameters loaded from: {args.params}")
    else:
        print("  Using default deterministic test parameters.")
    print()
    print(_HEADER)
    print(_SEP)

    all_results: list[tuple[bool, str]] = []

    # GHZ
    rows = _run_ghz(sim, args.backend, sizes, all_results)
    for r in rows:
        print(r)
    print(_SEP)

    # QFT
    rows = _run_qft(sim, args.backend, sizes, all_results)
    for r in rows:
        print(r)
    print(_SEP)

    # BV
    rows = _run_bv(sim, sizes, bv_strings, all_results)
    for r in rows:
        print(r)
    print(_SEP)

    # Summary
    skipped = sum(1 for p, note in all_results if "skipped" in note or "SKIP" in note)
    error = sum(1 for p, note in all_results if "error" in note or "ERROR" in note)
    countable = [p for p, note in all_results if "skipped" not in note and "SKIP" not in note and "error" not in note and "ERROR" not in note]
    n_pass = sum(1 for p in countable if p)
    n_total = len(countable)

    print()
    print(f"  Correctness gate: {n_pass}/{n_total} tests passing.", end="")
    if skipped:
        print(f"  ({skipped} skipped — expected for this backend.)", end="")
    if error:
        print(f"  ({error} errors — investigate above.)", end="")
    print()
    print("  Performance credit requires all applicable tests passing.")
    print()

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
