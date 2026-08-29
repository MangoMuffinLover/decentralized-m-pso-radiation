"""
pso_nuclear_sim.config
=======================
Dataclass-based configuration objects for the decentralized swarm resilience
simulator. All physical, algorithmic, and radiological constants used in the
paper (Table I: "Physical and Computational Simulation Configuration
Parameters") are defined here in one place, with no magic numbers scattered
across the rest of the package.

Every dataclass is frozen (immutable) so that a ``SimulationConfig`` instance
can be safely shared across parallel Monte Carlo trials without risk of
accidental mutation between threads/processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "PipeGeometry",
    "AgentPhysicalConfig",
    "AlgorithmConfig",
    "RadiationConfig",
    "RadiationProfile",
    "AblationFlags",
    "SimulationConfig",
    "PROFILES",
    "DEFAULT_CONFIG",
]


@dataclass(frozen=True)
class PipeGeometry:
    """2D PWR coolant-pipe cross-section geometry (Table I, Environment /
    Hydrodynamic Constants)."""

    length: float = 10.0                       # L (m)
    radius: float = 1.0                        # R (m)
    fissure_position: Tuple[float, float] = (8.0, 0.9)  # x_f (m)

    @property
    def x_f(self) -> np.ndarray:
        return np.array(self.fissure_position, dtype=float)

    @property
    def domain_area(self) -> float:
        """Omega_pipe = [0, L] x [-R, R]."""
        return self.length * 2.0 * self.radius


@dataclass(frozen=True)
class AgentPhysicalConfig:
    """Per-agent physical properties driving hydrodynamic drag (Eq. 6)."""

    mass_kg: float = 0.05                      # m
    coolant_density_kgm3: float = 1000.0       # rho
    drag_coefficient: float = 0.47             # C_d
    frontal_area_m2: float = 1.26e-3           # A
    max_fluid_velocity_ms: float = 1.5         # v_max
    turbulence_std_ms: float = 0.1             # sigma_turb
    velocity_clamp_ms: float = 3.0             # v_clamp


@dataclass(frozen=True)
class AlgorithmConfig:
    """M-PSO algorithmic hyper-parameters (Table I, Algorithmic Parameters)."""

    n_agents: int = 25                         # N
    timestep_s: float = 0.1                    # dt
    max_steps: int = 150                       # t_max
    inertia_weight: float = 0.6                # w
    cognitive_weight: float = 1.2              # c_1
    social_weight: float = 1.2                 # c_2
    comm_range_m: float = 2.0                  # r_comm
    success_threshold_m: float = 0.3           # scoring radius


@dataclass(frozen=True)
class RadiationConfig:
    """Radiological hazard constants shared across all profiles
    (Table I, Radiological Hazard Constants)."""

    drop_sensitivity: float = 0.15             # beta (s/Gy)
    sensor_noise_proportionality: float = 0.08  # sigma_gamma
    cumulative_dose_fail_threshold_gy: float = 300.0  # Lambda_fail


@dataclass(frozen=True)
class RadiationProfile:
    """A single (D_bg, K) operating point of the non-homogeneous gamma dose
    rate field D(x) = D_bg + K / (||x - x_f||^2 + 1.0) (Eq. 7)."""

    name: str
    d_bg: float                                 # ambient dose rate (Gy/s)
    k: float                                     # source intensity


# Table I, "Profile Dose Params. (Ctrl/Mod/Extr)": (D_bg, K) triples, in the
# fixed Control -> Moderate -> Extreme order used everywhere in the paper.
PROFILES: Dict[str, RadiationProfile] = {
    "Control": RadiationProfile("Control", d_bg=0.0, k=0.0),
    "Moderate": RadiationProfile("Moderate", d_bg=0.5, k=10.0),
    "Extreme": RadiationProfile("Extreme", d_bg=2.0, k=100.0),
}


@dataclass(frozen=True)
class AblationFlags:
    """Mechanism on/off switches used by the seven-cell ablation
    (Table III) and by the Sec. IV/comm-graph discussion of the explicit
    chi_i(t) social-term gate. Defaults reproduce the full M-PSO model.

    use_drag:
        Include the hydrodynamic drag term (F_d/m)*dt in the velocity
        update (Eq. 2). False reproduces ablation cell B / D's "no drag"
        condition depending on architecture.
    use_comms_degradation:
        Master switch for *all* radiologically-driven communication
        failure: both the transient Bernoulli channel drop (Eq. 9) and
        the permanent cumulative-dose latch-up (Eq. 11). False means
        chi_i(t) == 1 for every agent, every step (ablation cell C).
    use_sensor_noise:
        Include the Gaussian sensor-corruption term epsilon_gamma in
        Eq. 8. False isolates the pure packet-loss channel (cell G).
    use_packet_loss:
        Include the transient Bernoulli P_drop(x)-driven channel drop.
        Independent of the *permanent* latch-up, which is still governed
        solely by use_comms_degradation. False isolates the pure
        sensor-noise channel (cell F).
    use_explicit_chi_gate:
        Whether the M-PSO social term (Eq. 2) is *additionally*
        multiplied by chi_i(t), on top of the fact that an isolated
        agent's neighborhood N_i(t) already collapses to {i}. False
        relies on neighborhood self-collapse alone (cell E; see
        Sec. II-A.1 / "Dynamic Communication Graph").

    Note: leader re-election is deliberately *not* a flag here. It is a
    distinct coordination strategy, not a physics ablation, so it is
    selected by choosing ``CentralizedReElection`` from
    ``baselines.ARCHITECTURE_REGISTRY`` rather than by toggling a boolean
    on an otherwise-static-leader run. Folding it into this dataclass was
    exactly the kind of hidden architecture/physics coupling that caused
    the drag/Canonical-PSO bug documented in ``baselines.CanonicalPSO``
    and ``runner.run_single_trial`` -- see README "Known Issues Found
    During This Refactor".
    """

    use_drag: bool = True
    use_comms_degradation: bool = True
    use_sensor_noise: bool = True
    use_packet_loss: bool = True
    use_explicit_chi_gate: bool = True


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level configuration bundling all of the above, plus Monte Carlo
    campaign parameters. This is the single object threaded through
    ``environment``, ``agents``, ``baselines``, and ``runner``."""

    pipe: PipeGeometry = field(default_factory=PipeGeometry)
    agent: AgentPhysicalConfig = field(default_factory=AgentPhysicalConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    radiation: RadiationConfig = field(default_factory=RadiationConfig)
    n_monte_carlo: int = 500
    master_entropy: int = 20260722  # fixed root entropy for SHA-256 trial seeding

    def with_overrides(self, **kwargs) -> "SimulationConfig":
        """Return a new SimulationConfig with top-level fields overridden,
        e.g. ``config.with_overrides(n_monte_carlo=50)`` for a fast smoke
        test. Nested dataclasses (pipe/agent/algorithm/radiation) must be
        replaced wholesale if they need to change."""
        import dataclasses

        return dataclasses.replace(self, **kwargs)


DEFAULT_CONFIG = SimulationConfig()
