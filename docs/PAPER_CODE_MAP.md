# Paper-to-code map

| Manuscript element | Implementation or evidence |
|---|---|
| Eqs. 2-4: swarm update and neighborhood-local best | `pso_nuclear_sim/agents.py`, `pso_nuclear_sim/baselines.py` |
| Eqs. 5-6: turbulent velocity and drag | `pso_nuclear_sim/environment.py` |
| Eqs. 7-8: dose field and sensor degradation | `pso_nuclear_sim/environment.py` |
| Eqs. 9-11: packet loss, accumulated dose, latch-up | `pso_nuclear_sim/agents.py`, `pso_nuclear_sim/runner.py` |
| Table I: configuration | `pso_nuclear_sim/config.py` |
| Table II: architecture comparison | `scripts/run_benchmarks.py --campaign baseline` |
| Table III: seven-cell ablation | `scripts/run_benchmarks.py --campaign ablation` |
| Table IV: swarm-size sensitivity | `scripts/run_benchmarks.py --campaign sweep-n` |
| Communication-range sweep | `scripts/run_benchmarks.py --campaign sweep-rcomm` |
| Figure 3 / Table V: beta and latch-up sweeps | `scripts/run_comm_fragility.py`, `results/table_*_sweep.csv` |
| Table VI: single-hot-standby study | `scripts/run_hot_standby.py`, `results/hot_standby_extreme.csv` |
| Wilson intervals and campaign aggregation | `pso_nuclear_sim/runner.py` |
| Physics and communication checks | `tests/test_physics.py`, `tests/test_comm_graph.py` |
| Sensitivity/diagnostic regression checks | `tests/test_comm_fragility.py` |

The manuscript's code-and-data statement points to the Zenodo snapshot. This GitHub repository is the readable development surface; the DOI release is the immutable citation target.

