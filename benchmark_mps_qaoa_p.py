#!/usr/bin/env python3
"""Sweep the MPS backend over 1D QAOA depth p=1..5.

This is an additional project utility. It does not modify the instructor's
benchmark.py. Run from the project root after `make`:
    python3 benchmark_mps_qaoa_p.py --n-max 40 --output results/mps_qaoa_p_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circuits import qaoa_1d
from simulator import Simulator


def n_values(n_min: int, n_max: int) -> list[int]:
    values = []
    n = n_min
    while n <= n_max:
        values.append(n)
        n += 2 if n < 20 else 4
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-min', type=int, default=4)
    parser.add_argument('--n-max', type=int, default=40)
    parser.add_argument('--budget', type=float, default=120.0,
                        help='Stop testing larger n for a given p after a run exceeds this many seconds.')
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('results/mps_qaoa_p_sweep.csv'))
    args = parser.parse_args()
    if args.n_min < 2 or args.n_max < args.n_min or args.reps < 1:
        parser.error('Use n-min >= 2, n-max >= n-min, and reps >= 1.')

    rng = np.random.default_rng(20261021)
    sim = Simulator('mps')
    rows: list[dict[str, object]] = []
    print('MPS 1D QAOA depth sweep (p = 1..5)')
    print(f"{'p':>2} {'n':>3} {'median_s':>12} {'peak_python_MB':>16} {'status':>10}")
    print('-' * 50)

    for p in range(1, 6):
        stopped = False
        for n in n_values(args.n_min, args.n_max):
            gammas = rng.uniform(0.0, math.pi, p).tolist()
            betas = rng.uniform(0.0, math.pi / 2, p).tolist()
            durations = []
            peak_mb = 0.0
            try:
                for rep in range(args.reps):
                    sim.initialize(n)
                    if rep == 0:
                        tracemalloc.start()
                    start = time.perf_counter()
                    qaoa_1d(sim, n, p, gammas, betas)
                    elapsed = time.perf_counter() - start
                    durations.append(elapsed)
                    if rep == 0:
                        _, peak_bytes = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        peak_mb = peak_bytes / (1024 * 1024)
                median_s = statistics.median(durations)
                status = 'OK' if median_s <= args.budget else 'OVER_BUDGET'
                print(f'{p:>2} {n:>3} {median_s:>12.6f} {peak_mb:>16.3f} {status:>10}')
                rows.append({'backend': 'mps', 'circuit': 'qaoa_1d', 'p': p, 'n': n,
                             'median_time_s': median_s, 'peak_python_mb': peak_mb,
                             'repetitions': args.reps, 'status': status})
                if median_s > args.budget:
                    stopped = True
                    break
            except Exception as exc:
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                print(f'ERROR p={p}, n={n}: {exc}', file=sys.stderr)
                rows.append({'backend': 'mps', 'circuit': 'qaoa_1d', 'p': p, 'n': n,
                             'median_time_s': '', 'peak_python_mb': '',
                             'repetitions': args.reps, 'status': f'ERROR: {exc}'})
                stopped = True
                break
        if stopped:
            print(f'  Stopped increasing n for p={p}; continuing to the next p.')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['backend', 'circuit', 'p', 'n', 'median_time_s',
                                                'peak_python_mb', 'repetitions', 'status'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nCSV saved to: {args.output}')
    print('Note: tracemalloc reports Python-tracked allocations, not total process/RSS memory.')
    print('Run graded performance measurements on the course evaluation machine.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
