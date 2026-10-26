import csv
import math
import statistics
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from backends.mps import MPSSimulator
from circuits import qaoa_1d

OUTPUT = Path("results/mps_aer_warmed_comparison.csv")
REPEATS = 5
SEED = 20261024
SIZES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24]


def make_parameters(n):
    rng = np.random.default_rng(SEED + n)
    p = 2
    gammas = list(rng.uniform(0, math.pi, p))
    betas = list(rng.uniform(0, math.pi / 2, p))
    return p, gammas, betas


def build_qiskit_qaoa(n, p, gammas, betas):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for i in range(n - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * gammas[layer], i + 1)
            qc.cx(i, i + 1)
        for i in range(n):
            qc.rx(2 * betas[layer], i)
    qc.save_statevector()
    return qc


def run_mps(n, p, gammas, betas):
    sim = MPSSimulator()
    sim.initialize(n)
    qaoa_1d(sim, n, p, gammas, betas)


def median_runtime(fn):
    fn()  # warm-up
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), samples


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    print("MPS vs Qiskit Aer — fixed QAOA parameters, p=2")
    print(f"Warm-up: 1 run per backend; timed repetitions: {REPEATS}")
    print(f"{'n':>4} {'MPS ms':>12} {'Aer ms':>12} {'Status':>20}")

    aer = AerSimulator(method="statevector")

    for n in SIZES:
        p, gammas, betas = make_parameters(n)
        qc = build_qiskit_qaoa(n, p, gammas, betas)

        mps_time = None
        aer_time = None
        mps_status = "OK"
        aer_status = "OK"

        try:
            mps_time, _ = median_runtime(
                lambda: run_mps(n, p, gammas, betas)
            )
        except Exception as exc:
            mps_status = f"FAILED:{type(exc).__name__}"

        try:
            # Warm-up and timed runs all require a successful result.
            def run_aer():
                result = aer.run(qc).result()
                if not result.success:
                    raise RuntimeError("Aer result reported failure")
                result.get_statevector()

            aer_time, _ = median_runtime(run_aer)
        except Exception as exc:
            aer_status = f"FAILED:{type(exc).__name__}"

        mps_ms = "" if mps_time is None else mps_time * 1000
        aer_ms = "" if aer_time is None else aer_time * 1000

        if mps_time is not None and aer_time is not None:
            ratio = aer_time / mps_time
            comparison = "MPS faster" if ratio > 1 else "Aer faster/equal"
            status = f"OK:{comparison}"
            print(f"{n:4d} {mps_ms:12.3f} {aer_ms:12.3f} {status:>20}")
        else:
            status = f"MPS={mps_status}; Aer={aer_status}"
            print(f"{n:4d} {str(mps_ms):>12} {str(aer_ms):>12} {status:>20}")

        rows.append({
            "n": n,
            "p": p,
            "mps_median_ms": mps_ms,
            "aer_median_ms": aer_ms,
            "mps_status": mps_status,
            "aer_status": aer_status,
            "status": status,
        })

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n", "p", "mps_median_ms", "aer_median_ms",
                "mps_status", "aer_status", "status"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
