# Decentralized swarm resilience under stochastic gamma noise

[![IECON 2026](https://img.shields.io/badge/IECON-2026-00629B)](https://www.iecon2026.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22166217.svg)](https://doi.org/10.5281/zenodo.22166217)
[![Tests](https://github.com/MangoMuffinLover/decentralized-m-pso-radiation/actions/workflows/tests.yml/badge.svg)](https://github.com/MangoMuffinLover/decentralized-m-pso-radiation/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/Code%20license-MIT-yellow.svg)](LICENSE)

**Official code, data, and reproducibility package for the accepted IEEE IECON 2026 paper**

> **Decentralized Swarm Resilience Under Stochastic Gamma Noise: A Computational Model for Pressure Vessel Inspection**  
> Dhyan Ishwar Shaji - Independent Researcher, Manama, Kingdom of Bahrain

[Read the camera-ready manuscript](paper/IECON2026_camera_ready.pdf) · [Faculty/reviewer guide](docs/FOR_REVIEWERS.md) · [Archived release](https://doi.org/10.5281/zenodo.22166217) · [Reproduce the study](docs/REPRODUCIBILITY.md) · [Inspect the evidence](docs/RESULTS.md)

![Communication-fragility sensitivity under the Extreme profile](docs/assets/communication_fragility.png)

## Research question

Can a swarm continue localizing an inspection target when turbulent flow, spatially heterogeneous gamma radiation, sensor corruption, packet loss, and cumulative-dose transceiver latch-up act together?

This repository evaluates a neighborhood-restricted Modified Particle Swarm Optimization (M-PSO) controller in a reduced-order pressurized-water-reactor inspection model. It compares decentralized coordination with static-leader, leader-re-election, canonical-PSO, and single-hot-standby alternatives.

## What the study contributes

- A **physically coupled fault-generation model** connecting turbulent drag, a non-homogeneous dose field, dose-dependent sensing and communication degradation, and permanent cumulative-dose latch-up.
- A **per-agent, range-limited communication graph** with no privileged global information channel for decentralized M-PSO.
- A **500-trial Monte Carlo evaluation** with collision-resistant SHA-256 condition/trial seeding and Wilson 95% confidence intervals.
- A **seven-cell ablation study** plus swarm-size, communication-range, packet-loss-sensitivity, and latch-up-threshold sweeps.
- A **state-replicating hot-standby experiment** showing that successful failover alone does not remove a shared communication failure domain.

## Headline evidence

Under the manuscript's modeled **Extreme** profile:

| Architecture | Success | Wilson 95% CI | Interpretation |
|---|---:|---:|---|
| Decentralized M-PSO | **84.0%** | [80.5, 87.0] | Neighborhood-local coordination |
| Centralized, static leader | 3.0% | [1.8, 4.9] | Permanent freeze after leader latch-up |
| Centralized, re-election | 1.8% | [0.9, 3.4] | Replacement still depends on degraded communication |

In the separate hardened-baseline campaign, a single hot standby completed failover in **78.6-81.2%** of trials, yet achieved only **0.6-1.6%** success. The bounded conclusion is not that every centralized design fails; it is that replacing the coordinator does not restore communication edges lost through the same range- and radiation-gated information path.

See [RESULTS.md](docs/RESULTS.md) for campaign identities, denominators, and the distinction between manuscript and independent replication runs.

## Quick start

Python 3.10+ is recommended.

```bash
git clone https://github.com/MangoMuffinLover/decentralized-m-pso-radiation.git
cd decentralized-m-pso-radiation
python -m venv .venv
```

Activate the environment, then install and verify:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tests/run_tests.py
python scripts/run_benchmarks.py --campaign baseline --trials 20 --workers 1
python scripts/run_comm_fragility.py --mode smoke --workers 1
python scripts/run_hot_standby.py --mode smoke --workers 1
```

The test suite currently contains **62 deterministic model and regression tests**. Reduced-trial smoke runs are diagnostics only and must not be cited as paper evidence.

## Full reproduction

```bash
python scripts/run_benchmarks.py --campaign all --trials 500 --workers 1
python scripts/run_comm_fragility.py --mode full --workers 1
python scripts/run_hot_standby.py --mode full --workers 1
python scripts/make_figures.py
python scripts/make_comm_fragility_figure.py
```

Full campaigns overwrite or create CSVs in `results/`. Preserve the archived files before rerunning if you want a byte-for-byte record of the released evidence. Detailed commands, outputs, and expected behavior are in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository map

```text
pso_nuclear_sim/   model, degradation channels, controllers, and runner
scripts/           benchmark, sensitivity, hot-standby, and figure campaigns
tests/             deterministic physics, graph, and regression checks
results/           archived 500-trial aggregate evidence and protocol
figures/           vector figure assets used by the manuscript
paper/             camera-ready PDF, LaTeX source, bibliography, and audit notes
docs/              research overview, results guide, scope, and code map
logs/              retained execution records for the hot-standby campaign
```

## Scientific scope

This is a **simulation study**, not a hardware-in-the-loop demonstration or a validated operating-reactor deployment model. The simulator is two-dimensional, uses a reduced-order one-seventh-power-law flow approximation, and does not include thermal-hydraulic coupling. Claims are restricted to the stated model, parameter grids, and comparator implementations. Read [MODEL_SCOPE.md](docs/MODEL_SCOPE.md) before reusing the results.

## Citation

If you use the simulator or archived data, cite the DOI release:

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

Machine-readable citation metadata are provided in [CITATION.cff](CITATION.cff).

## License and contact

The simulator and supporting code in `pso_nuclear_sim/`, `scripts/`, and `tests/` are available under the [MIT License](LICENSE). The manuscript and publication figures are retained for scholarly reading and reproducibility and are not covered by that software license.

For research questions, replication reports, or prospective collaboration, open a GitHub issue or contact **Dhyan Ishwar Shaji** at the address listed in the manuscript.
