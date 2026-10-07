# Reproducibility guide

This document separates fast verification from manuscript-grade Monte Carlo reproduction.

## 1. Environment

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`
- A single worker for the clearest deterministic execution order

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Verify the implementation

```bash
python tests/run_tests.py
```

Expected release result: `TOTAL: 62 passed, 0 failed`.

The tests cover dose and signal fields, drag direction and scaling, latch-up permanence, communication range and channel gates, neighborhood-local information flow, centralized failure behavior, hot-standby promotion, connectivity diagnostics, and deterministic RNG behavior.

## 3. Fast smoke campaigns

These commands test the end-to-end pipeline with small trial counts. They are not manuscript evidence.

```bash
python scripts/run_benchmarks.py --campaign baseline --trials 20 --workers 1
python scripts/run_comm_fragility.py --mode smoke --workers 1
python scripts/run_hot_standby.py --mode smoke --workers 1
```

The communication-fragility and hot-standby smoke modes intentionally do not replace the archived 500-trial result CSVs.

## 4. Manuscript-grade campaigns

Run from the repository root:

```bash
python scripts/run_benchmarks.py --campaign all --trials 500 --workers 1
python scripts/run_comm_fragility.py --mode full --workers 1
python scripts/run_hot_standby.py --mode full --workers 1
```

Campaign outputs are written beneath `results/`. Exact wall-clock time depends on the machine and worker count.

## 5. Figure regeneration

```bash
python scripts/make_figures.py
python scripts/make_comm_fragility_figure.py
```

Vector manuscript assets are stored in `figures/`. PNG files in `docs/assets/` are presentation previews only.

## 6. Seed policy

Every Monte Carlo condition has a human-readable condition identity. The runner hashes the condition identity together with the trial index using SHA-256 to derive independent random-number streams. This avoids Python's process-randomized `hash()` and makes campaign conditions reproducible across processes.

Changing a condition identity intentionally creates a different stream. Therefore, compare reproduced output only when the condition labels, parameters, trial count, and code version match.

## 7. Statistical reporting

- Primary outcome: localization success within the stated success radius.
- Uncertainty: Wilson 95% interval for binomial success rates.
- Trial count: 500 independent trials per reported campaign condition.
- Non-trivial success: success after excluding trials initialized inside the success radius.
- Full grids are archived even when the six-page paper reports only a compact subset.

## 8. Publication boundary

The IEEE article and its Version of Record are not distributed from this
repository. Access the publication through its official IEEE Xplore record.
This repository reproduces the computational experiments, aggregate evidence,
and independently generated figures.

## 9. Reproducibility boundary

Reproduction means recovering the released computational results under the released model. It does not validate the model against an operating pressure vessel, prove universal superiority of decentralized control, or substitute for hardware testing.
