"""
pso_nuclear_sim.environment
=============================
The physical and radiological environment: turbulent hydrodynamic flow field
(Sec. II-B), non-homogeneous gamma dose-rate field D(x) and its two
degradation phenomena (Sec. II-C), and the per-agent cumulative-dose
latch-up model (Eq. 10-11).

This module is stateless with respect to agent identity except where noted
(``RadiationState`` tracks each agent's running dose and latch status across
timesteps, since Eq. 10 is a recurrence). Everything else is a pure function
of position and configuration, which keeps the physics independently
testable (see tests/test_physics.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AgentPhysicalConfig, PipeGeometry, RadiationConfig, RadiationProfile

__all__ = [
    "dose_rate",
    "true_signal",
    "fluid_velocity",
    "sensor_noise_std",
    "drop_probability",
    "RadiationState",
]


def dose_rate(
    positions: np.ndarray, profile: RadiationProfile, pipe: PipeGeometry
) -> np.ndarray:
    """Non-homogeneous gamma dose-rate field, Eq. 7:

        D(x) = D_bg + K / (||x - x_f||^2 + 1.0)

    Parameters
    ----------
    positions : (N, 2) array of agent positions.
    profile : the active RadiationProfile (Control/Moderate/Extreme).
    pipe : pipe geometry, providing x_f.

    Returns
    -------
    (N,) array of dose rates in Gy/s, one per agent.
    """
    dist_sq = np.sum((positions - pipe.x_f) ** 2, axis=-1)
    return profile.d_bg + profile.k / (dist_sq + 1.0)


def true_signal(positions: np.ndarray, pipe: PipeGeometry) -> np.ndarray:
    """True, noise-free proximity signal S(x) used as the PSO fitness
    function f(.) throughout: peaked at x_f, decaying with squared distance.
    Not itself part of the radiation model, but shares the same
    inverse-square falloff shape and the same fissure location.
    """
    dist_sq = np.sum((positions - pipe.x_f) ** 2, axis=-1)
    return 1.0 / (dist_sq + 1.0)


def fluid_velocity(
    positions: np.ndarray,
    rng: np.random.Generator,
    pipe: PipeGeometry,
    agent_cfg: AgentPhysicalConfig,
) -> np.ndarray:
    """Turbulent flow field, Eq. 5:

        v_fluid(y) = v_max * (1 - |y|/R)^(1/7) * i_hat + v_turb,
        v_turb ~ N(0, sigma_turb^2)

    One-seventh power-law profile plus an isotropic stochastic fluctuation
    term representing the Reynolds-decomposed turbulent fluctuation.
    """
    y = positions[:, 1]
    frac = np.clip(1.0 - np.abs(y) / pipe.radius, 0.0, None)
    vx = agent_cfg.max_fluid_velocity_ms * frac ** (1.0 / 7.0)
    v_turb = rng.normal(0.0, agent_cfg.turbulence_std_ms, size=positions.shape)
    v_fluid = np.zeros_like(positions)
    v_fluid[:, 0] = vx
    return v_fluid + v_turb


def sensor_noise_std(dose_now: np.ndarray, radiation_cfg: RadiationConfig) -> np.ndarray:
    """Standard deviation of the Gaussian sensor-corruption term in Eq. 8:
    sigma_gamma * sqrt(D(x)), i.e. noise *variance* scales linearly with
    dose rate."""
    return radiation_cfg.sensor_noise_proportionality * np.sqrt(np.maximum(dose_now, 0.0))


def drop_probability(dose_now: np.ndarray, radiation_cfg: RadiationConfig) -> np.ndarray:
    """Transient communication-drop probability, Eq. 9:

        P_drop(x) = 1 - exp(-beta * D(x))
    """
    return 1.0 - np.exp(-radiation_cfg.drop_sensitivity * dose_now)


@dataclass
class RadiationState:
    """Stateful per-agent cumulative dose and permanent latch-up tracker
    (Eq. 10-11). One instance per trial; ``update`` is called once per
    timestep with each agent's *true* physical dose rate.

        Lambda_i(t) = Lambda_i(t-1) + D(x_i(t)) * dt,  Lambda_i(0) = 0
        Lambda_i(t) >= Lambda_fail  =>  chi_i(t') = 0  for all t' >= t

    Latch-up is permanent and, once ``gated`` is True for an agent, stays
    True for the remainder of the trial regardless of subsequent position.
    """

    n_agents: int
    lambda_fail: float
    cumulative_dose: np.ndarray = field(init=False)
    latched: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.cumulative_dose = np.zeros(self.n_agents)
        self.latched = np.zeros(self.n_agents, dtype=bool)

    def update(self, true_dose_rate: np.ndarray, dt: float, gated: bool) -> np.ndarray:
        """Advance the cumulative-dose state by one timestep using the
        *true* physical dose rate (always accumulated, regardless of
        whether the comms-degradation ablation switch is on) and, if
        ``gated`` is True, latch any agent that has just crossed
        Lambda_fail. Returns the boolean "newly latched this step" mask.
        """
        self.cumulative_dose += true_dose_rate * dt
        newly_latched = np.zeros(self.n_agents, dtype=bool)
        if gated:
            newly_latched = (self.cumulative_dose >= self.lambda_fail) & (~self.latched)
            self.latched[newly_latched] = True
        return newly_latched

    @property
    def total_dose(self) -> float:
        return float(np.sum(self.cumulative_dose))
