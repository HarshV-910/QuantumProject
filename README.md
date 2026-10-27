# COL8422 Quantum Computing Systems — Project Part 1
## Multi-Backend Quantum Circuit Simulator

**Deadline:** Oct 28, 2026 &nbsp;|&nbsp; **Submission:** Course portal &nbsp;|&nbsp; **Teams:** 3–4 students

---

## What you received

| File | Status | Description |
|------|--------|-------------|
| `simulator.py` | **Do not modify** | Common `QuantumSimulator` ABC and `Simulator()` factory |
| `circuits.py` | **Do not modify** | Circuit generators (GHZ, QFT, BV, QAOA-1D, …) |
| `harness.py` | **Do not modify** | Correctness checker — run this to test your backends |
| `benchmark.py` | **Do not modify** | Timing and scaling harness |
| `backends/__init__.py` | **Do not modify** | Package marker — must exist |
| `backends/statevector.py` | **Your code (Member A)** | Fill in `StatevectorSimulator` |
| `backends/stabilizer.py` | **Your code (Member B)** | Fill in `StabilizerSimulator` |
| `backends/mps.py` | **Your code (Member C)** | Fill in `MPSSimulator` |
| `backends/dd.py` | **Your code (Member D)** | Fill in `DDSimulator` |

---

## Getting started

### Step 1 — Rename your submission directory

Rename `part1/` to `group<N>/` where N is your assigned group number:
```
mv part1 group3      # example for group 3
```

### Step 2 — Create your source directories

```
mkdir -p src/statevector src/stabilizer src/mps src/dd
```

Your C++ kernel for each backend goes in the corresponding `src/<backend>/` directory.
A `Makefile` in the project root must compile everything with a plain `make`.

### Step 3 — Start with the statevector backend (all groups)

Before splitting work, everyone should understand the statevector baseline.
Member A leads; the rest read along.

1. Read Häner & Steiger, SC 2017 (the foundational paper for Member A).
2. Implement `StatevectorSimulator` in `backends/statevector.py`.
   Start with a pure-Python/NumPy version to verify correctness,
   then move the hot path to C++.
3. Run the harness:
   ```
   python harness.py --backend statevector
   ```
   You should see 15/15 tests passing before moving to your own backend.

### Step 4 — Each member implements their assigned backend

Each member:
1. Reads their foundational paper (see `project_spec.pdf`, Section 2).
2. Implements their backend in `backends/<name>.py` + `src/<name>/`.
3. Runs the harness for their backend:
   ```
   python harness.py --backend stabilizer   # Member B
   python harness.py --backend mps          # Member C
   python harness.py --backend dd           # Member D
   ```

### Step 5 — Integration

All four backends run behind the common interface. The instructor will run:
```
python harness.py --backend statevector --submission group<N>/
python harness.py --backend stabilizer  --submission group<N>/
python harness.py --backend mps         --submission group<N>/
python harness.py --backend dd          --submission group<N>/
```
A backend that cannot be loaded (wrong class name, missing `__init__.py`,
broken `Makefile`) receives **zero marks** for that component.

### Step 6 — Benchmarking (on the evaluation machine)

```
python benchmark.py --backend statevector --circuit ghz      --n-max 32 --plot
python benchmark.py --backend mps         --circuit qaoa_1d  --n-max 40 --plot
python benchmark.py --backend stabilizer  --circuit bv       --n-max 40 --plot
python benchmark.py --backend dd          --circuit bv       --n-max 40 --plot
```

Graded benchmark numbers must come from the evaluation machine (JupyterHub).
Numbers from your laptop are not accepted for performance scoring.

---

## Directory structure — strictly enforced

```
group<N>/
├── simulator.py            (provided)
├── circuits.py             (provided)
├── harness.py              (provided)
├── benchmark.py            (provided)
├── backends/
│   ├── __init__.py         (provided — do not delete)
│   ├── statevector.py      StatevectorSimulator  ← Member A
│   ├── stabilizer.py       StabilizerSimulator   ← Member B
│   ├── mps.py              MPSSimulator          ← Member C
│   └── dd.py               DDSimulator           ← Member D
├── src/
│   ├── statevector/        C++ source
│   ├── stabilizer/         C++ source
│   ├── mps/                C++ source
│   └── dd/                 C++ source
├── Makefile                must compile with: make
├── report.pdf
├── design_journal.md
└── results/
    └── *.csv
```

**Class names are enforced.** The harness imports each module and looks for the
class name listed above. Any other name is treated as not found.

---

## Common mistakes (read before you start)

| Mistake | Consequence |
|---------|-------------|
| Renaming the simulator class | Harness cannot load it → zero marks for that backend |
| Forgetting `backends/__init__.py` | `No module named 'backends'` error |
| Hardcoded paths in `Makefile` or Python | Compiles on your laptop, fails on eval machine |
| Submitting a single large git commit | Commit history will be checked during oral defense |
| `get_statevector()` as a required interface method | It is not in the common interface — do not add it |
| Silent fallback on unsupported gates | Must raise `UnsupportedOperationError`, not silently pass |

---

## Checklist before submission

- [ ] `python harness.py --backend statevector` exits 0
- [ ] `python harness.py --backend stabilizer` exits 0
- [ ] `python harness.py --backend mps` exits 0
- [ ] `python harness.py --backend dd` exits 0
- [ ] `make` builds cleanly on the evaluation machine (test via JupyterHub)
- [ ] Benchmark CSVs for all four backends are in `results/`
- [ ] `design_journal.md` has entries for each week
- [ ] `report.pdf` is present and under 8 pages
- [ ] Git commit history has at least one commit per 2–3 days

---

## Quick reference

```bash
# Correctness (run often during development)
python harness.py --backend <name>

# Full benchmark with plot
python benchmark.py --backend <name> --circuit <circuit> --n-max 40 --plot

# Available circuits: ghz, qft, bv, qaoa_1d, near_clifford, random_universal
# Available backends: statevector, stabilizer, mps, dd
```
