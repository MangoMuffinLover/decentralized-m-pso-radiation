"""
pso_nuclear_sim
=================
Reference implementation accompanying "Decentralized Swarm Resilience Under
Stochastic Gamma Noise: A Computational Model for Pressure Vessel
Inspection".

A reduced-order Monte Carlo simulator of a decentralized, communication-
degradation-tolerant swarm (Modified Particle Swarm Optimization, M-PSO)
performing defect localization inside a turbulent, radiologically hazardous
PWR coolant loop, alongside static-leader, re-election, state-replicating
hot-standby, and canonical-PSO benchmark architectures.

Quick start
-----------
>>> from pso_nuclear_sim import DEFAULT_CONFIG, PROFILES, run_condition
>>> result = run_condition(
...     "demo|Extreme|Decentralized M-PSO",
...     DEFAULT_CONFIG,
...     "Decentralized M-PSO",
...     PROFILES["Extreme"],
...     n_trials=20,
... )
>>> round(result.success_rate_pct, 1)  # doctest: +SKIP
84.0

See README.md for the full architecture overview, the mathematical
formulation summary, and benchmark replication instructions.
"""

from .agents import AgentSwarm, drag_force, fraction_isolated
from .baselines import (
    ARCHITECTURE_REGISTRY,
    CanonicalPSO,
    CentralizedHotStandby,
    CentralizedHotStandbyH1,
    CentralizedHotStandbyH3,
    CentralizedHotStandbyH5,
    CentralizedReElection,
    CentralizedStaticLeader,
    DecentralizedMPSO,
)
from .config import (
    AblationFlags,
    AgentPhysicalConfig,
    AlgorithmConfig,
    DEFAULT_CONFIG,
    PipeGeometry,
    PROFILES,
    RadiationConfig,
    RadiationProfile,
    SimulationConfig,
)
from .runner import ConditionResult, TrialResult, run_condition, run_single_trial, wilson_ci

__version__ = "2.1.0"

__all__ = [
    "__version__",
    # config
    "PipeGeometry",
    "AgentPhysicalConfig",
    "AlgorithmConfig",
    "RadiationConfig",
    "RadiationProfile",
    "AblationFlags",
    "SimulationConfig",
    "PROFILES",
    "DEFAULT_CONFIG",
    # agents
    "AgentSwarm",
    "drag_force",
    "fraction_isolated",
    # baselines
    "ARCHITECTURE_REGISTRY",
    "DecentralizedMPSO",
    "CentralizedStaticLeader",
    "CentralizedReElection",
    "CentralizedHotStandby",
    "CentralizedHotStandbyH1",
    "CentralizedHotStandbyH3",
    "CentralizedHotStandbyH5",
    "CanonicalPSO",
    # runner
    "run_condition",
    "run_single_trial",
    "wilson_ci",
    "ConditionResult",
    "TrialResult",
]
