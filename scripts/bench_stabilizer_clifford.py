"""Scaling curve for the stabilizer backend on its assigned class (random Clifford).

Why this script exists
----------------------
The spec (Sec. 5.4) assigns the stabilizer backend "Random Clifford + k T-gates"
with k = 0, 5, 10, 20.  The provided benchmark.py always inserts at least one
T gate (t_count = max(1, n // 4)), and our backend correctly raises
UnsupportedOperationError for T, so benchmark.py cannot produce the k = 0 curve.
This script uses the course's own generator, circuits.near_clifford(...,
t_count=k), and times:
  * our StabilizerSimulator  (initialize + circuit [+ optional sampling])
  * Qiskit Aer statevector    (same gate list, same way benchmark.py times it)
For k > 0 our backend raises UnsupportedOperationError at the first T gate;
that is recorded in the CSV as status "unsupported" (not as a time).

Usage (from the project root, after `make`):
    python scripts/bench_stabilizer_clifford.py --k 0 --n-max 1000 --plot
    python scripts/bench_stabilizer_clifford.py --k 5 --n-max 30      # unsupported rows + Qiskit only
Output: results/stabilizer_near_clifford_k<K>.csv  (+ .png with --plot)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from circuits import near_clifford  # noqa: E402
from simulator import Simulator, UnsupportedOperationError  # noqa: E402


class _Recorder:
    """Captures the gate list produced by circuits.near_clifford."""

    def __init__(self):
        self.ops = []

    def apply_1q_gate(self, gate, qubit, **kw):
        self.ops.append((gate, qubit))

    def apply_2q_gate(self, gate, ctrl, tgt, **kw):
        self.ops.append((gate, ctrl, tgt))


def _ops(n: int, k: int, seed: int):
    rec = _Recorder()
    near_clifford(rec, n, np.random.default_rng(seed), t_count=k)
    return rec.ops


def _time_ours(sim, n, ops, shots, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sim.initialize(n)
        for op in ops:
            if len(op) == 2:
                sim.apply_1q_gate(op[0], op[1])
            else:
                sim.apply_2q_gate(op[0], op[1], op[2])
        if shots:
            sim.sample(shots)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def _time_qiskit(n, ops):
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
    except ImportError:
        return None, "qiskit not installed"
    qc = QuantumCircuit(n)
    gate_map = {"H": qc.h, "S": qc.s, "X": qc.x, "Y": qc.y, "Z": qc.z, "T": qc.t}
    for op in ops:
        if len(op) == 2:
            gate_map[op[0]](op[1])
        else:
            {"CNOT": qc.cx, "CZ": qc.cz, "SWAP": qc.swap}[op[0]](op[1], op[2])
    qc.save_statevector()
    backend = AerSimulator(method="statevector")
    t0 = time.perf_counter()
    result = backend.run(qc).result()
    dt = time.perf_counter() - t0
    if not result.success:   # e.g. out of memory: do NOT report this as a time
        return None, "qiskit failed (memory)"
    return dt, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=0, help="number of T gates (spec: 0, 5, 10, 20)")
    ap.add_argument("--n-min", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=1000)
    ap.add_argument("--shots", type=int, default=0, help="also time sample(shots) (0 = gates only, like benchmark.py)")
    ap.add_argument("--qiskit-n-max", type=int, default=30, help="skip Qiskit above this n (2^n memory)")
    ap.add_argument("--budget", type=float, default=120.0, help="stop when our median time exceeds this (s)")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    out = args.output or ROOT / "results" / f"stabilizer_near_clifford_k{args.k}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sim = Simulator("stabilizer")

    ns, n = [], args.n_min
    while n <= args.n_max:
        ns.append(n)
        n = n + 2 if n < 20 else (n + 4 if n < 40 else int(math.ceil(n * 1.25)))

    rows = []
    print(f"{'n':>6} {'gates':>8} {'ours(s)':>12} {'qiskit(s)':>12}  status")
    for n in ns:
        ops = _ops(n, args.k, seed=1000 + n)
        try:
            t_ours, status = _time_ours(sim, n, ops, args.shots), "ok"
        except UnsupportedOperationError:
            t_ours, status = float("nan"), "unsupported"
        t_q, q_status = (None, "skipped") if n > args.qiskit_n_max else _time_qiskit(n, ops)
        t_q = float("nan") if t_q is None else t_q
        rows.append({"backend": "stabilizer", "circuit": f"near_clifford_k{args.k}", "n": n,
                     "gates": len(ops), "time_s": t_ours, "status": status,
                     "qiskit_time_s": t_q, "qiskit_status": q_status})
        print(f"{n:>6} {len(ops):>8} {t_ours:>12.5f} {t_q:>12.5f}  {status} / {q_status}")
        if status == "ok" and t_ours > args.budget:
            print(f"[STOP] n={n}: {t_ours:.1f}s exceeds budget")
            break

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ok = [r for r in rows if r["status"] == "ok"]
        if ok:
            ax.loglog([r["n"] for r in ok], [r["time_s"] for r in ok], "o-", label="stabilizer (ours)")
        q = [r for r in rows if r["qiskit_status"] == "ok"]
        if q:
            ax.loglog([r["n"] for r in q], [r["qiskit_time_s"] for r in q], "s--", label="Qiskit Aer statevector")
        ax.set_xlabel("n (qubits)")
        ax.set_ylabel("wall-clock time (s)")
        ax.set_title(f"Random Clifford, k = {args.k} T gates")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out.with_suffix(".png"), dpi=150)
        print(f"plot: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
