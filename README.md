# Decentralized swarm resilience under stochastic gamma noise

[![IECON 2026](https://img.shields.io/badge/IEEE%20IECON-2026-00629B)](https://www.iecon2026.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22166217.svg)](https://doi.org/10.5281/zenodo.22166217)
[![Tests](https://github.com/MangoMuffinLover/decentralized-m-pso-radiation/actions/workflows/tests.yml/badge.svg)](https://github.com/MangoMuffinLover/decentralized-m-pso-radiation/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/Code%20license-MIT-yellow.svg)](LICENSE)

**Official code, data, and reproducibility package for the IEEE IECON 2026 paper**

> **Decentralized Swarm Resilience Under Stochastic Gamma Noise: A Computational Model for Pressure Vessel Inspection**  
> Dhyan Ishwar Shaji - Independent Researcher, Manama, Kingdom of Bahrain

This work is published through IEEE IECON 2026 and indexed in IEEE Xplore. The IEEE article should be accessed through its official IEEE Xplore record. The immutable software and evidence release is archived on [Zenodo](https://doi.org/10.5281/zenodo.22166217).

[Faculty and reviewer guide](docs/FOR_REVIEWERS.md) · [Reproduce the study](docs/REPRODUCIBILITY.md) · [Inspect the evidence](docs/RESULTS.md) · [Model scope](docs/MODEL_SCOPE.md) · [Paper-to-code map](docs/PAPER_CODE_MAP.md)

![Communication-fragility sensitivity under the Extreme profile](docs/assets/communication_fragility.png)

## Research question

Can a swarm continue localizing an inspection target when turbulent flow, spatially heterogeneous gamma radiation, sensor corruption, packet loss, and cumulative-dose transceiver latch-up act together?

This repository evaluates a neighborhood-restricted Modified Particle Swarm Optimization (M-PSO) controller in a reduced-order pressurized-water-reactor inspection model. It compares decentralized coordination with static-leader, leader-re-election, canonical-PSO, and single-hot-standby alternatives.

## Contributions

- A **physically coupled fault-generation model** connecting turbulent drag, a non-homogeneous dose field, dose-dependent sensing and communication degradation, and permanent cumulative-dose latch-up.
- A **per-agent, range-limited communication graph** without a privileged global information channel for decentralized M-PSO.
- A **500-trial Monte Carlo evaluation** using SHA-256 condition/trial seeding and Wilson 95% confidence intervals.
- A **seven-cell ablation study** together with swarm-size, communication-range, packet-loss-sensitivity, and latch-up-threshold sweeps.
- A **state-replicating hot-standby experiment** examining whether successful coordinator failover resolves a shared communication failure domain.
- A tested and documented implementation connecting the manuscript equations, experiments, and archived numerical evidence.

## Headline evidence

Under the manuscript's modeled **Extreme** profile:

| Architecture | Success | Wilson 95% CI | Interpretation |
|---|---:|---:|---|
| Decentralized M-PSO | **84.0%** | [80.5, 87.0] | Neighborhood-local coordination |
| Centralized, static leader | 3.0% | [1.8, 4.9] | Freezes permanently after leader latch-up |
| Centralized, re-election | 1.8% | [0.9, 3.4] | Replacement still depends on degraded communication |
| Canonical PSO | 29.6% | [25.8, 33.7] | Global-information reference without the complete degradation model |

In a separate hardened-baseline campaign, the single hot standby completed failover in **78.6-81.2%** of trials but achieved only **0.6-1.6%** localization success.

The bounded conclusion is not that every centralized architecture fails. Within the released model, changing coordinator identity does not restore communication edges lost through the same range- and radiation-gated information pathway.

See [RESULTS.md](docs/RESULTS.md) for campaign identities, statistical denominators, independent replication results, and interpretation boundaries.

## Quick start

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/MangoMuffinLover/decentralized-m-pso-radiation.git
cd decentralized-m-pso-radiation
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the deterministic verification suite:

```bash
python tests/run_tests.py
```

Expected release result:

```text
TOTAL: 62 passed, 0 failed
```

Run small end-to-end diagnostics:

```bash
python scripts/run_benchmarks.py --campaign baseline --trials 20 --workers 1
python scripts/run_comm_fragility.py --mode smoke --workers 1
python scripts/run_hot_standby.py --mode smoke --workers 1
```

Reduced-trial smoke runs verify the pipeline but must not be cited as manuscript evidence.

## Full reproduction

```bash
python scripts/run_benchmarks.py --campaign all --trials 500 --workers 1
python scripts/run_comm_fragility.py --mode full --workers 1
python scripts/run_hot_standby.py --mode full --workers 1
python scripts/make_figures.py
python scripts/make_comm_fragility_figure.py
```

Full campaigns create or replace CSV files under `results/`. Preserve the archived release files before rerunning if you need a byte-for-byte copy of the released evidence.

Complete instructions are provided in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository structure

```text
pso_nuclear_sim/   model, degradation channels, controllers, and runner
scripts/           benchmark, sensitivity, hot-standby, and figure campaigns
tests/             deterministic physics, communication, and regression checks
results/           archived 500-trial aggregate evidence and frozen protocols
figures/           independently generated vector figures
docs/assets/       GitHub-compatible figure previews
docs/              reviewer guide, results, scope, reproducibility, and code map
logs/              retained execution records for the hot-standby campaign
```

## Scientific scope

This is a **simulation study**, not a hardware-in-the-loop demonstration or a validated operating-reactor deployment model.

The simulator is two-dimensional, uses a reduced-order one-seventh-power-law flow approximation, and does not include thermal-hydraulic coupling. Results are restricted to the stated model, parameter grids, seed policy, and implemented comparator architectures.

The experiments do not establish that:

- decentralized control is universally superior;
- every centralized fault-tolerance design fails;
- the simulator quantitatively predicts an operating reactor;
- the proposed controller is ready for deployment; or
- the reported comparisons substitute for hardware validation.

Read [MODEL_SCOPE.md](docs/MODEL_SCOPE.md) before extending or citing the conclusions.

## Reproducibility policy

Each Monte Carlo condition uses a human-readable condition identity. That identity and the trial index are hashed with SHA-256 to derive independent random-number streams.

Changing a condition identity intentionally changes its stream. Reproduced results should therefore be compared only when the code version, condition identity, trial count, parameters, and seed policy match.

The archived communication-fragility and hot-standby summaries use 500 independent trials per condition and Wilson 95% confidence intervals.

## Citation

The immutable research-software release is Zenodo version **2.1.0**:

```bibtex
@software{shaji_2026_m_pso,
  author    = {Dhyan Ishwar Shaji},
  title     = {Decentralized Swarm Resilience Under Stochastic Gamma Noise: Reproducibility Package},
  year      = {2026},
  version   = {2.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22166217},
  url       = {https://doi.org/10.5281/zenodo.22166217}
}
```

Machine-readable citation metadata are available in [CITATION.cff](CITATION.cff).

The GitHub `v2.1.1` release is a presentation and documentation refinement of the archived Zenodo v2.1.0 scientific package. It does not replace or silently alter the archived Monte Carlo evidence.

## Copyright and licensing

The simulator and supporting source code in `pso_nuclear_sim/`, `scripts/`, and `tests/` are released under the [MIT License](LICENSE).

The IEEE article itself is not distributed from this repository. Please access the publication through its official IEEE Xplore record.

The independently generated figures and aggregate result summaries are included to support transparent inspection and computational reproduction. Their inclusion does not grant rights to reproduce the IEEE Version of Record.

## Contact

For research questions, replication reports, or prospective collaboration, open a GitHub issue or contact **Dhyan Ishwar Shaji** using the address listed in the IEEE publication.
