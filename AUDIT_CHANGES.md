# Audit changes in this snapshot

Added:
- requirements.txt
- results/README.md
- MPS_INTEGRATION_HANDOFF.md

Updated:
- .gitignore: ignores virtual environments, caches, compiled artifacts, and root-level generated plots/results.
- design_journal.md: fixed foundational reference citation and documented preliminary Aer results honestly, without claiming final speedup.
- compare_mps_aer.py: preserves MPS and Aer statuses independently so one backend's failure cannot overwrite the other backend's status.
- src/mps/mps_kernels.cpp: removed duplicate include.

Preserved unchanged:
- Instructor-provided simulator.py, circuits.py, harness.py, benchmark.py.
- Backend class names and the current MPS interface.

Validation performed on this snapshot:
- `make mps` succeeded.
- `python3 harness.py --backend mps`: 15/15 passed.

Not claimed complete:
- Full plain `make` integration across all four backends; other backend build files are not all present in the uploaded snapshot.
- Group report.pdf (6–8 pages).
- All four backends' correctness tests and CSVs.
- Evaluation-machine benchmark/scaling curve.
