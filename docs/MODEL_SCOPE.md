# Model scope and responsible interpretation

## What is modeled

- Two-dimensional axial/wall-normal agent motion.
- A one-seventh-power-law mean flow profile with stochastic fluctuation.
- Quadratic hydrodynamic drag.
- A spatially non-homogeneous background-plus-inverse-square dose field.
- Dose-dependent sensor noise and transient packet loss.
- Permanent per-agent cumulative-dose latch-up.
- Range-limited, per-agent communication state.
- Decentralized M-PSO, canonical PSO, static-leader, re-election, and single-hot-standby comparators.

## What is not modeled

- Three-dimensional geometry or CFD-resolved turbulence.
- Thermal-hydraulic coupling.
- Hardware-in-the-loop, radiation-source, or operating-reactor validation.
- Learned MARL baselines under exactly matched observations, training budget, and degradation shifts.
- Multi-standby, quorum, relay-assisted, or independent hardened side-channel centralized systems.
- Exhaustive controller-weight or control-policy optimization.

## Claims supported by the experiments

The experiments support a conditional statement: under the released reduced-order model and parameter grids, neighborhood-local decentralized coordination is substantially more resilient than the evaluated centralized aggregation designs when range and radiation jointly degrade the information graph.

## Claims not supported

The results do not establish that decentralized control is universally superior, that all centralized fault-tolerance designs fail, that the simulator predicts an operating plant quantitatively, or that the controller is ready for deployment.

This distinction is intentional. The repository is designed to make both the positive evidence and its validation boundary inspectable.

