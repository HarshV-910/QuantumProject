# Benchmark result notes

The CSVs in this directory are development measurements collected on the lab machine, not the graded evaluation machine.

- `mps_qaoa_p_sweep.csv`: exploratory MPS QAOA depth sweep for p=1..5.
- `mps_qaoa_baseline.csv`: official harness MPS-only baseline before Qiskit Aer was installed.
- `mps_qaoa_with_aer.csv`: official harness run after Aer was installed. Some Aer executions failed due to memory limits; numeric times at those rows must not be treated as successful measurements.
- `mps_aer_warmed_comparison.csv`: exploratory warmed comparison for QAOA p=2, with fixed parameters per qubit count. Timing variability was substantial; do not use it to claim a final speedup or crossover.

For the graded scaling curve, rerun the supplied `benchmark.py` on the course evaluation machine and retain the CSV and plot. Report failed/out-of-memory cases separately and only calculate crossover from successful, comparable measurements.
