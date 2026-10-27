# COL8422 Quantum Computing Systems — Project Harness
# Timing and scaling benchmark; produces CSV results and optional plots.

"""
Usage
-----
    python benchmark.py --backend statevector --circuit qaoa_1d \\
        [--n-min 4] [--n-max 40] [--budget 120] [--output results.csv] [--plot]

Measures median wall-clock time and peak memory (tracemalloc) across 3 runs
for each (backend, circuit, n).  If Qiskit is installed, a matching Qiskit
Aer statevector simulation is run as a baseline.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import math
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from circuits import (
    bv_circuit,
    ghz,
    near_clifford,
    qaoa_1d,
    qft,
    random_universal,
)
from simulator import Simulator, QuantumSimulator

# ---------------------------------------------------------------------------
# Circuit registry
# ---------------------------------------------------------------------------

_CIRCUIT_CHOICES = ["ghz", "qft", "bv", "qaoa_1d", "near_clifford", "random_universal"]


def _make_circuit_fn(circuit_name: str, n: int, rng: np.random.Generator) -> Callable[[QuantumSimulator], None]:
    """Return a zero-argument callable that, given a sim, runs the circuit."""

    if circuit_name == "ghz":
        def run(sim: QuantumSimulator) -> None:
            sim.initialize(n)
            ghz(sim, n)
        return run

    elif circuit_name == "qft":
        def run(sim: QuantumSimulator) -> None:
            sim.initialize(n)
            qft(sim, n)
        return run

    elif circuit_name == "bv":
        s = int(rng.integers(1, 2**n))
        def run(sim: QuantumSimulator) -> None:
            sim.initialize(n + 1)
            bv_circuit(sim, n, s)
        return run

    elif circuit_name == "qaoa_1d":
        p = 2
        gammas = list(rng.uniform(0, math.pi, p))
        betas = list(rng.uniform(0, math.pi / 2, p))
        def run(sim: QuantumSimulator) -> None:
            sim.initialize(n)
            qaoa_1d(sim, n, p, gammas, betas)
        return run

    elif circuit_name == "near_clifford":
        t_count = max(1, n // 4)
        child_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        def run(sim: QuantumSimulator) -> None:
            sim.initialize(n)
            near_clifford(sim, n, child_rng, t_count)
        return run

    elif circuit_name == "random_universal":
        child_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        def run(sim: QuantumSimulator) -> None:
            sim.initialize(n)
            random_universal(sim, n, child_rng, depth_factor=5)
        return run

    else:
        raise ValueError(f"Unknown circuit: {circuit_name!r}")


# ---------------------------------------------------------------------------
# Measurement utilities
# ---------------------------------------------------------------------------

def _time_and_memory(fn: Callable[[], None], n_reps: int = 3) -> tuple[float, float]:
    """Run fn n_reps times; return (median_wall_s, peak_mem_mb).

    Memory is measured on the first run only to keep overhead low.
    """
    times: list[float] = []

    # First run: also measure memory
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    times.append(t1 - t0)

    # Remaining runs: time only
    for _ in range(n_reps - 1):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    median_t = times[len(times) // 2]
    peak_mb = peak_bytes / (1024 * 1024)
    return median_t, peak_mb


# ---------------------------------------------------------------------------
# Qiskit baseline
# ---------------------------------------------------------------------------

def _qiskit_available() -> bool:
    return importlib.util.find_spec("qiskit") is not None and importlib.util.find_spec("qiskit_aer") is not None


def _run_qiskit_circuit(circuit_name: str, n: int, rng: np.random.Generator) -> float | None:
    """Run a matching circuit via Qiskit Aer and return wall-clock seconds.

    Returns None if Qiskit is unavailable or the circuit cannot be represented.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
    except ImportError:
        return None

    try:
        qc = QuantumCircuit(n)

        if circuit_name == "ghz":
            qc.h(0)
            for i in range(1, n):
                qc.cx(0, i)

        elif circuit_name == "qft":
            for k in range(n):
                qc.h(k)
                for m in range(k + 1, n):
                    angle = math.pi / (2 ** (m - k))
                    qc.cp(angle, m, k)
            for i in range(n // 2):
                qc.swap(i, n - 1 - i)

        elif circuit_name == "bv":
            s = int(rng.integers(1, 2**n))
            qc = QuantumCircuit(n + 1)
            qc.x(n)
            qc.h(n)
            qc.h(list(range(n)))
            for i in range(n):
                if (s >> i) & 1:
                    qc.cx(i, n)
            qc.h(list(range(n)))

        elif circuit_name == "qaoa_1d":
            p = 2
            gammas = list(rng.uniform(0, math.pi, p))
            betas = list(rng.uniform(0, math.pi / 2, p))
            qc.h(list(range(n)))
            for layer in range(p):
                for i in range(n - 1):
                    qc.cx(i, i + 1)
                    qc.rz(2 * gammas[layer], i + 1)
                    qc.cx(i, i + 1)
                for i in range(n):
                    qc.rx(2 * betas[layer], i)

        elif circuit_name in ("near_clifford", "random_universal"):
            # Skip — these are random circuits without a canonical Qiskit equivalent
            return None

        else:
            return None

        qc.save_statevector()
        backend = AerSimulator(method="statevector")
        t0 = time.perf_counter()
        backend.run(qc).result()
        return time.perf_counter() - t0

    except Exception:
        return None


# ---------------------------------------------------------------------------
# N-step generator
# ---------------------------------------------------------------------------

def _n_range(n_min: int, n_max: int) -> list[int]:
    """Generate qubit counts: step 2 up to 20, step 4 beyond."""
    ns = []
    n = n_min
    while n <= n_max:
        ns.append(n)
        if n < 20:
            n += 2
        else:
            n += 4
    return ns


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_COL_W = (14, 16, 6, 12, 12, 18)
_HEADER_ROW = (
    f"{'backend':<{_COL_W[0]}}  "
    f"{'circuit':<{_COL_W[1]}}  "
    f"{'n':>{_COL_W[2]}}  "
    f"{'time(s)':>{_COL_W[3]}}  "
    f"{'mem(MB)':>{_COL_W[4]}}  "
    f"{'qiskit_time(s)':>{_COL_W[5]}}"
)
_SEP = "-" * (sum(_COL_W) + 5 * 2)


def _table_row(backend: str, circuit: str, n: int, t: float, mem: float, qt: float | None) -> str:
    qt_str = f"{qt:.4f}" if qt is not None else "N/A"
    return (
        f"{backend:<{_COL_W[0]}}  "
        f"{circuit:<{_COL_W[1]}}  "
        f"{n:>{_COL_W[6] if False else _COL_W[2]}}  "
        f"{t:>{_COL_W[3]}.4f}  "
        f"{mem:>{_COL_W[4]}.2f}  "
        f"{qt_str:>{_COL_W[5]}}"
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(
    records: list[dict],
    backend: str,
    circuit_name: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available — skipping plot.", file=sys.stderr)
        return

    ns = [r["n"] for r in records]
    times = [r["time_s"] for r in records]
    qt_pairs = [(r["n"], r["qiskit_time_s"]) for r in records if not math.isnan(r["qiskit_time_s"])]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(ns, times, marker="o", label=f"{backend} (student)")
    if qt_pairs:
        qns, qts = zip(*qt_pairs)
        ax.semilogy(qns, qts, marker="s", linestyle="--", label="Qiskit Aer")

    ax.set_xlabel("Number of qubits (n)")
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.set_title(f"COL8422 Benchmark — {circuit_name} — {backend}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="COL8422 Project — Timing & Scaling Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", required=True, help="Simulator backend name.")
    parser.add_argument(
        "--circuit",
        required=True,
        choices=_CIRCUIT_CHOICES,
        help="Circuit to benchmark.",
    )
    parser.add_argument("--n-min", type=int, default=4, metavar="N", help="Minimum qubit count.")
    parser.add_argument("--n-max", type=int, default=40, metavar="N", help="Maximum qubit count.")
    parser.add_argument(
        "--budget",
        type=float,
        default=120.0,
        metavar="SEC",
        help="Stop if median run time exceeds this many seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.csv"),
        metavar="FILE",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a time-vs-n matplotlib plot.",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="Path to the student submission directory (must contain backends/). Defaults to CWD.",
    )
    args = parser.parse_args(argv)

    # Insert submission directory so the Simulator factory can find backends/
    submission_dir = args.submission.resolve()
    if str(submission_dir) not in sys.path:
        sys.path.insert(0, str(submission_dir))

    rng = np.random.default_rng(2024_10_01)  # fixed seed for reproducibility

    # Validate backend
    try:
        sim = Simulator(args.backend)
    except (ImportError, ValueError) as exc:
        print(f"[ERROR] Could not load backend '{args.backend}': {exc}", file=sys.stderr)
        return 2

    qiskit_ok = _qiskit_available()
    if not qiskit_ok:
        print("[WARNING] Qiskit/Aer not found — qiskit baseline will be skipped.", file=sys.stderr)

    ns = _n_range(args.n_min, args.n_max)

    print()
    print(f"  COL8422 Benchmark — backend={args.backend}  circuit={args.circuit}")
    print(f"  n range: {ns[0]}..{ns[-1]}  budget={args.budget}s  reps=3")
    print()
    print(_HEADER_ROW)
    print(_SEP)

    records: list[dict] = []
    csv_rows: list[dict] = []

    for n in ns:
        circuit_fn = _make_circuit_fn(args.circuit, n, rng)

        def run_student() -> None:
            circuit_fn(sim)

        try:
            t_med, mem_mb = _time_and_memory(run_student)
        except Exception as exc:
            print(f"  [ERROR] n={n}: {exc}", file=sys.stderr)
            continue

        # Qiskit baseline
        qt: float | None = None
        if qiskit_ok:
            qt = _run_qiskit_circuit(args.circuit, n, rng)

        qt_val = float("nan") if qt is None else qt
        print(_table_row(args.backend, args.circuit, n, t_med, mem_mb, qt))

        records.append({"n": n, "time_s": t_med, "mem_mb": mem_mb, "qiskit_time_s": qt_val})
        csv_rows.append({
            "backend": args.backend,
            "circuit": args.circuit,
            "n": n,
            "time_s": t_med,
            "mem_mb": mem_mb,
            "qiskit_time_s": qt_val,
        })

        if t_med > args.budget:
            print()
            print(f"  [STOP] n={n}: median time {t_med:.2f}s exceeds budget {args.budget}s.")
            break

    print(_SEP)

    # Write CSV
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["backend", "circuit", "n", "time_s", "mem_mb", "qiskit_time_s"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  Results written to: {args.output}")

    # Optional plot
    if args.plot and records:
        plot_path = Path(f"results_{args.backend}_{args.circuit}.png")
        _plot(records, args.backend, args.circuit, plot_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
