# This repository accompanies *Decentralized Swarm Resilience Under Stochastic Gamma Noise: A Computational Model for Pressure Vessel Inspection*.

- Development repository: https://github.com/MangoMuffinLover/decentralized-m-pso-radiation
- Archived release v2.1.0: https://doi.org/10.5281/zenodo.22166217

## Scope

The code implements a reduced-order computational model.  It is not a
hardware-in-the-loop or validated operating-reactor experiment.  Quantitative
results are scoped to the stated parameter grids and Monte Carlo protocol.

The `paper/` directory contains the canonical camera-ready source.  Compile
`paper/main_final_submission.tex` before publishing a paper PDF; this release
does not include a stale pre-revision PDF.

## Quick start

Install the packages in `requirements.txt`, then run:

```text
python tests/run_tests.py
python scripts/run_benchmarks.py
python scripts/run_comm_fragility.py --mode full --workers 1
python scripts/run_hot_standby.py --mode full --workers 1
python scripts/make_comm_fragility_figure.py
```

## Reproducibility policy

Monte Carlo conditions use a SHA-256 hash of condition identity and trial index
to derive independent random-number streams.  The archived communication-
fragility and hot-standby summaries in `results/` use 500 trials per condition.
The hardened-baseline study tests one state-replicating backup across the
pre-specified missed-heartbeat timeouts `h={1,3,5}`. It does not claim to
exhaust multi-standby or quorum-based centralized designs or serve as a
numerical MARL comparison.

## Layout

- `pso_nuclear_sim/`: model and controller implementation.
- `scripts/`: campaign and figure-generation scripts.
- `tests/`: regression and model tests.
- `results/`: archived 500-trial summary data.
- `figures/`: paper figure assets.
- `paper/`: final manuscript source, figures, bibliography, and checklist.

The simulator source, campaign scripts, and tests are released under the MIT
License; the manuscript and figure assets remain subject to their separate
publication rights. Before a public release, verify the author metadata and
remove any private data or credentials.
