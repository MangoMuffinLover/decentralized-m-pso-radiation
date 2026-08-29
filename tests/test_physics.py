"""
tests/test_physics.py
========================
Unit tests for the pure physics functions in ``pso_nuclear_sim.environment``
and the drag model in ``pso_nuclear_sim.agents``: turbulent velocity
profile, drag force, and radiation dose-rate calculation.

These are deliberately independent of the Monte Carlo campaign machinery
(runner.py) so a physics regression is caught at the level of a single
equation, not buried inside an aggregate success-rate drift.
"""

import numpy as np
import pytest

from pso_nuclear_sim.agents import drag_force
from pso_nuclear_sim.config import (
    AgentPhysicalConfig,
    PipeGeometry,
    RadiationConfig,
    RadiationProfile,
)
from pso_nuclear_sim.environment import (
    RadiationState,
    dose_rate,
    drop_probability,
    fluid_velocity,
    sensor_noise_std,
    true_signal,
)


@pytest.fixture
def pipe() -> PipeGeometry:
    return PipeGeometry()


@pytest.fixture
def agent_cfg() -> AgentPhysicalConfig:
    return AgentPhysicalConfig()


@pytest.fixture
def radiation_cfg() -> RadiationConfig:
    return RadiationConfig()


# --------------------------------------------------------------------------
# Dose-rate field D(x), Eq. 7
# --------------------------------------------------------------------------

class TestDoseRate:
    def test_control_profile_is_zero_everywhere(self, pipe):
        """Control: D_bg = K = 0, so D(x) == 0 for any position."""
        profile = RadiationProfile("Control", d_bg=0.0, k=0.0)
        positions = np.array([[0.0, 0.0], [5.0, 0.5], [8.0, 0.9]])
        assert np.allclose(dose_rate(positions, profile, pipe), 0.0)

    def test_dose_peaks_at_fissure(self, pipe):
        """D(x) should be strictly maximized at x == x_f for K > 0."""
        profile = RadiationProfile("Extreme", d_bg=2.0, k=100.0)
        at_fissure = dose_rate(pipe.x_f[None, :], profile, pipe)[0]
        away = dose_rate(np.array([[0.0, 0.0]]), profile, pipe)[0]
        assert at_fissure > away

    def test_dose_matches_closed_form(self, pipe):
        """D(x) = D_bg + K / (||x - x_f||^2 + 1.0), checked at an
        arbitrary offset from the fissure."""
        profile = RadiationProfile("Moderate", d_bg=0.5, k=10.0)
        offset = np.array([[pipe.x_f[0] + 1.0, pipe.x_f[1] + 0.0]])
        expected = 0.5 + 10.0 / (1.0 ** 2 + 1.0)
        assert np.isclose(dose_rate(offset, profile, pipe)[0], expected)

    def test_no_singularity_at_fissure(self, pipe):
        """The +1.0 regularizer must prevent D(x) from blowing up exactly
        at x_f, even for a large source intensity K."""
        profile = RadiationProfile("Extreme", d_bg=2.0, k=1e6)
        value = dose_rate(pipe.x_f[None, :], profile, pipe)[0]
        assert np.isfinite(value)
        assert np.isclose(value, 2.0 + 1e6 / 1.0)

    def test_vectorized_over_many_agents(self, pipe):
        """dose_rate must broadcast correctly over an (N, 2) position array."""
        profile = RadiationProfile("Extreme", d_bg=2.0, k=100.0)
        rng = np.random.default_rng(0)
        positions = rng.uniform([0, -1], [10, 1], size=(37, 2))
        out = dose_rate(positions, profile, pipe)
        assert out.shape == (37,)
        assert np.all(out >= profile.d_bg)  # dose rate never below ambient


# --------------------------------------------------------------------------
# True signal S(x) -- shares the fissure location but is a distinct field
# --------------------------------------------------------------------------

class TestTrueSignal:
    def test_signal_peaks_at_fissure(self, pipe):
        at_fissure = true_signal(pipe.x_f[None, :], pipe)[0]
        away = true_signal(np.array([[0.0, 0.0]]), pipe)[0]
        assert at_fissure > away
        assert np.isclose(at_fissure, 1.0)  # 1 / (0 + 1) == 1

    def test_signal_decays_monotonically_with_distance(self, pipe):
        offsets = np.array([0.0, 1.0, 2.0, 5.0])
        positions = pipe.x_f[None, :] + np.stack([offsets, np.zeros_like(offsets)], axis=1)
        signal = true_signal(positions, pipe)
        assert np.all(np.diff(signal) < 0)  # strictly decreasing


# --------------------------------------------------------------------------
# Sensor noise and communication-drop probability, Eq. 8-9
# --------------------------------------------------------------------------

class TestDegradationChannels:
    def test_sensor_noise_std_scales_with_sqrt_dose(self, radiation_cfg):
        dose = np.array([0.0, 1.0, 4.0, 100.0])
        std = sensor_noise_std(dose, radiation_cfg)
        expected = radiation_cfg.sensor_noise_proportionality * np.sqrt(dose)
        assert np.allclose(std, expected)

    def test_sensor_noise_zero_at_zero_dose(self, radiation_cfg):
        assert sensor_noise_std(np.array([0.0]), radiation_cfg)[0] == 0.0

    def test_drop_probability_bounds(self, radiation_cfg):
        """P_drop(x) = 1 - exp(-beta*D(x)) must lie in [0, 1) for finite
        dose, approaching 1 as dose grows and 0 as dose -> 0. (Note:
        dose values large enough to underflow exp(-beta*D) to exactly 0.0
        will saturate p to exactly 1.0 -- that is correct floating-point
        behavior, not a bug, so this test deliberately stays below that
        regime to check the open interval.)"""
        dose = np.array([0.0, 1.0, 10.0, 50.0])
        p = drop_probability(dose, radiation_cfg)
        assert p[0] == 0.0
        assert np.all((p >= 0.0) & (p < 1.0))
        assert p[-1] > p[-2] > p[-3]  # monotonically increasing with dose

    def test_drop_probability_matches_closed_form(self, radiation_cfg):
        dose = np.array([5.0])
        expected = 1.0 - np.exp(-radiation_cfg.drop_sensitivity * 5.0)
        assert np.isclose(drop_probability(dose, radiation_cfg)[0], expected)


# --------------------------------------------------------------------------
# Turbulent flow field, Eq. 5
# --------------------------------------------------------------------------

class TestFluidVelocity:
    def test_no_slip_at_wall(self, pipe, agent_cfg):
        """v_fluid's mean axial component must vanish at y = +/- R (the
        one-seventh power-law profile's no-slip boundary condition).
        Checked via a zero-turbulence config to isolate the mean profile."""
        agent_cfg_no_turb = AgentPhysicalConfig(turbulence_std_ms=0.0)
        rng = np.random.default_rng(0)
        at_wall = np.array([[5.0, pipe.radius], [5.0, -pipe.radius]])
        v = fluid_velocity(at_wall, rng, pipe, agent_cfg_no_turb)
        assert np.allclose(v[:, 0], 0.0, atol=1e-9)

    def test_max_at_centerline(self, pipe, agent_cfg):
        agent_cfg_no_turb = AgentPhysicalConfig(turbulence_std_ms=0.0)
        rng = np.random.default_rng(0)
        centerline = np.array([[5.0, 0.0]])
        v = fluid_velocity(centerline, rng, pipe, agent_cfg_no_turb)
        assert np.isclose(v[0, 0], agent_cfg.max_fluid_velocity_ms)

    def test_turbulent_fluctuation_adds_noise(self, pipe, agent_cfg):
        """With turbulence enabled, repeated draws at the same position
        should differ (stochastic forcing), unlike the zero-turbulence
        case above."""
        rng = np.random.default_rng(1)
        pos = np.array([[5.0, 0.0]] * 2)
        v = fluid_velocity(pos, rng, pipe, agent_cfg)
        assert not np.allclose(v[0], v[1])


# --------------------------------------------------------------------------
# Drag force, Eq. 6
# --------------------------------------------------------------------------

class TestDragForce:
    def test_zero_relative_velocity_gives_zero_force(self, agent_cfg):
        v_agent = np.array([[1.0, 0.5]])
        v_fluid = np.array([[1.0, 0.5]])
        f_d = drag_force(v_agent, v_fluid, agent_cfg)
        assert np.allclose(f_d, 0.0)

    def test_drag_opposes_relative_velocity(self, agent_cfg):
        """Drag must point opposite to v_rel = v_agent - v_fluid."""
        v_agent = np.array([[1.0, 0.0]])
        v_fluid = np.array([[0.0, 0.0]])
        f_d = drag_force(v_agent, v_fluid, agent_cfg)
        assert f_d[0, 0] < 0.0  # opposes +x relative motion
        assert np.isclose(f_d[0, 1], 0.0)

    def test_drag_magnitude_is_quadratic_in_relative_speed(self, agent_cfg):
        """||F_d|| = 0.5*rho*C_d*A*||v_rel||^2 -- doubling relative speed
        should quadruple the force magnitude."""
        v_fluid = np.array([[0.0, 0.0], [0.0, 0.0]])
        v_agent = np.array([[1.0, 0.0], [2.0, 0.0]])
        f_d = drag_force(v_agent, v_fluid, agent_cfg)
        mag = np.linalg.norm(f_d, axis=-1)
        assert np.isclose(mag[1] / mag[0], 4.0, rtol=1e-6)

    def test_vectorized_over_many_agents(self, agent_cfg):
        rng = np.random.default_rng(0)
        v_agent = rng.normal(size=(50, 2))
        v_fluid = rng.normal(size=(50, 2))
        f_d = drag_force(v_agent, v_fluid, agent_cfg)
        assert f_d.shape == (50, 2)
        assert np.all(np.isfinite(f_d))


# --------------------------------------------------------------------------
# Cumulative dose state and permanent latch-up, Eq. 10-11
# --------------------------------------------------------------------------

class TestRadiationState:
    def test_dose_accumulates_linearly_with_time(self):
        state = RadiationState(n_agents=3, lambda_fail=1e9)
        dose_rate_const = np.array([1.0, 2.0, 3.0])
        for _ in range(10):
            state.update(dose_rate_const, dt=0.1, gated=True)
        assert np.allclose(state.cumulative_dose, dose_rate_const * 0.1 * 10)

    def test_latch_up_triggers_exactly_at_threshold(self):
        state = RadiationState(n_agents=1, lambda_fail=10.0)
        # 10 steps of dt=1.0, dose_rate=1.0 -> cumulative dose hits 10.0 on step 10
        for step in range(10):
            newly = state.update(np.array([1.0]), dt=1.0, gated=True)
            if step < 9:
                assert not state.latched[0]
        assert state.latched[0]

    def test_latch_up_is_permanent(self):
        """Once latched, an agent must stay latched even if its
        subsequent dose rate (and hence position) would not re-trigger."""
        state = RadiationState(n_agents=1, lambda_fail=1.0)
        state.update(np.array([2.0]), dt=1.0, gated=True)
        assert state.latched[0]
        state.update(np.array([0.0]), dt=1.0, gated=True)
        assert state.latched[0]  # still latched

    def test_ungated_update_never_latches(self):
        """gated=False (ablation cell C: use_comms_degradation=False) must
        still accumulate true physical dose for bookkeeping, but never
        set the latch, regardless of how far past threshold it goes."""
        state = RadiationState(n_agents=1, lambda_fail=1.0)
        for _ in range(100):
            state.update(np.array([10.0]), dt=1.0, gated=False)
        assert not state.latched[0]
        assert state.total_dose > 100.0  # dose still bookkept

    def test_total_dose_sums_across_agents(self):
        state = RadiationState(n_agents=3, lambda_fail=1e9)
        state.update(np.array([1.0, 2.0, 3.0]), dt=1.0, gated=True)
        assert np.isclose(state.total_dose, 6.0)


# --------------------------------------------------------------------------
# RNG call order inside AgentSwarm.velocity_update (agents.py)
#
# This is load-bearing for bit-exact reproducibility against the reference
# implementation (README "Verification against the original implementation"):
# r1 and r2 must be drawn *before* the turbulent-flow draw used by drag,
# because both share one numpy.random.Generator stream per trial. Getting
# this backwards doesn't break any individual equation -- it silently
# desynchronizes every subsequent draw for the rest of the trial.
# --------------------------------------------------------------------------

class _RecordingRNG:
    """Wraps a real Generator, logging (method, size) for every draw so a
    test can assert on call order without depending on exact numeric
    coincidences."""

    def __init__(self, rng):
        self._rng = rng
        self.calls = []

    def uniform(self, *args, **kwargs):
        self.calls.append(("uniform", kwargs.get("size", args[-1] if args else None)))
        return self._rng.uniform(*args, **kwargs)

    def normal(self, *args, **kwargs):
        self.calls.append(("normal", kwargs.get("size", args[-1] if args else None)))
        return self._rng.normal(*args, **kwargs)


def test_rng_call_order():
    """velocity_update with drag enabled must draw r1 (uniform), then r2
    (uniform), then the turbulent-flow fluctuation (normal) -- in that
    exact order -- matching the original reference implementation."""
    from pso_nuclear_sim.agents import AgentSwarm
    from pso_nuclear_sim.config import AgentPhysicalConfig, AlgorithmConfig, PipeGeometry

    n = 4
    swarm = AgentSwarm(
        n_agents=n,
        positions=np.zeros((n, 2)),
        velocities=np.zeros((n, 2)),
        personal_best=np.ones((n, 2)),
        personal_best_fitness=np.zeros(n),
    )
    recorder = _RecordingRNG(np.random.default_rng(0))
    g_i = np.ones((n, 2))
    social_gate = np.ones(n)

    swarm.velocity_update(
        g_i, social_gate, recorder, AlgorithmConfig(), AgentPhysicalConfig(), PipeGeometry(),
        use_drag=True,
    )

    assert len(recorder.calls) == 3
    assert recorder.calls[0][0] == "uniform"  # r1
    assert recorder.calls[1][0] == "uniform"  # r2
    assert recorder.calls[2][0] == "normal"   # v_turb inside fluid_velocity (drag)


def test_rng_call_order_no_drag_skips_the_normal_draw():
    """With use_drag=False, only r1 and r2 should be drawn -- no
    turbulent-flow normal draw at all."""
    from pso_nuclear_sim.agents import AgentSwarm
    from pso_nuclear_sim.config import AgentPhysicalConfig, AlgorithmConfig, PipeGeometry

    n = 4
    swarm = AgentSwarm(
        n_agents=n,
        positions=np.zeros((n, 2)),
        velocities=np.zeros((n, 2)),
        personal_best=np.ones((n, 2)),
        personal_best_fitness=np.zeros(n),
    )
    recorder = _RecordingRNG(np.random.default_rng(0))
    g_i = np.ones((n, 2))
    social_gate = np.ones(n)

    swarm.velocity_update(
        g_i, social_gate, recorder, AlgorithmConfig(), AgentPhysicalConfig(), PipeGeometry(),
        use_drag=False,
    )

    assert len(recorder.calls) == 2
    assert [c[0] for c in recorder.calls] == ["uniform", "uniform"]
