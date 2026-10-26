# Member C (MPS) integration handoff

## Owned files
Member C should primarily own:
- `backends/mps.py`
- `src/mps/mps_kernels.cpp`
- `src/mps/README.md`
- `src/mps/Makefile` (if the team adopts per-backend Makefiles)
- MPS-specific benchmark helper(s), MPS CSVs, and MPS entries in `design_journal.md`

## Shared files — coordinate before editing
- Root `Makefile`: shared integration file. Assign one integrator to merge all four backend build targets. Do not have all members independently rewrite it.
- `requirements.txt`: shared dependency list; add dependencies in one coordinated edit.
- `README.md` and `design_journal.md`: coordinate section ownership to avoid overlapping edits.
- `simulator.py`, `circuits.py`, `harness.py`, `benchmark.py`: instructor-provided files; do not modify.

## Current integration gaps visible in the supplied current-code archive
- `src/statevector/Makefile` exists, but `src/stabilizer/` and `src/dd/` currently contain only `.gitkeep` in the C++ source directories; their backend owners must add their implementation/build files.
- The current root Makefile only invokes other backend builds when their subdirectory Makefiles exist, so plain `make` can silently skip backends without a Makefile. Before final submission, the integration owner must make the root `make` build all four compiled artifacts and fail clearly if a required backend target is missing.
- The uploaded archive includes local `.venv`, `__pycache__`, `.so`, `.o`, and a root-level generated plot. These are not source deliverables and should not be included in a clean source submission. Rebuild `.so` files using `make` on the target environment.
- The project specification requires `report.pdf` (6–8 pages), all four backend correctness runs, benchmark CSVs for all backends, and an evaluation-machine scaling curve. Those group-wide items are not complete in the uploaded current archive.
