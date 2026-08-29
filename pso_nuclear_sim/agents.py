"""
pso_nuclear_sim.agents
========================
Particle (agent) state, hydrodynamic drag force, the dynamic communication
graph (range- and channel-state-gated), neighborhood-restricted local-best
selection, and the M-PSO velocity/position update itself (Sec. II-A).

The ``AgentSwarm`` class owns all per-agent arrays (position, velocity,
personal best, fitness, channel state) for a single trial and exposes one
method per phase of the per-timestep loop in Fig. 2 / Algorithm 1:

    swarm.step_physical(...)      # Phase 1: drag, dose, sensor noise
    swarm.channel_state(...)      # chi_i(t)
    swarm.build_graph(chi, r_comm)  # Phase 2: G(t)
    swarm.local_best(edge)        # g_i(t), decentralized only
    swarm.velocity_update(...)    # Phase 3: Eq. 2 + integration

This keeps ``runner.run_single_trial`` a thin orchestration loop rather than
a monolithic function, and makes each physical mechanism unit-testable in
isolation (see tests/test_comm_graph.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AgentPhysicalConfig, AlgorithmConfig, PipeGeometry, RadiationConfig
from .environment import drop_probability, fluid_velocity

__all__ = ["drag_force", "AgentSwarm", "fraction_isolated"]


def fraction_isolated(edge: np.ndarray) -> float:
    """Fraction of agents whose active neighborhood N_i(t) has collapsed to
    just themselves (Sec. II-A.2: "Ni(t) may ... in the limit, reduce to
    full isolation"), given a communication-graph adjacency matrix ``edge``
    as returned by ``AgentSwarm.build_graph`` (self-loops always present on
    the diagonal, Eq. 3).

    This is a *physical-layer* diagnostic -- it depends only on inter-agent
    range and each agent's own channel-state chi_i(t), not on which
    coordination architecture is consuming the graph. It is therefore
    meaningful to compute identically for Decentralized M-PSO and any
    centralized architecture, as a communication-fragility metric
    independent of the coordination algorithm itself (added for the
    communication-fragility robustness study; see
    ``runner.run_single_trial``'s ``record_connectivity`` flag).
    """
    n = edge.shape[0]
    off_diagonal_degree = edge.sum(axis=1) - 1  # subtract the guaranteed self-loop
    isolated = off_diagonal_degree <= 0
    return float(np.mean(isolated)) if n > 0 else float("nan")


def drag_force(
    v_agent: np.ndarray, v_fluid: np.ndarray, agent_cfg: AgentPhysicalConfig
) -> np.ndarray:
    """Quadratic bluff-body drag, Eq. 6:

        F_d = -0.5 * rho * C_d * A * ||v_rel||^2 * u_hat_d

    where v_rel = v_agent - v_fluid and u_hat_d opposes v_rel. Agents with
    (numerically) zero relative velocity experience zero drag rather than a
    division-by-zero unit vector.
    """
    v_rel = v_agent - v_fluid
    speed = np.linalg.norm(v_rel, axis=-1, keepdims=True)
    speed_safe = np.where(speed < 1e-9, 1.0, speed)
    u_hat = v_rel / speed_safe
    magnitude = 0.5 * agent_cfg.coolant_density_kgm3 * agent_cfg.drag_coefficient \
        * agent_cfg.frontal_area_m2 * speed ** 2
    f_d = -magnitude * u_hat
    f_d[speed.flatten() < 1e-9] = 0.0
    return f_d


@dataclass
class AgentSwarm:
    """Mutable per-trial swarm state for N agents in 2D."""

    n_agents: int
    positions: np.ndarray
    velocities: np.ndarray
    personal_best: np.ndarray
    personal_best_fitness: np.ndarray

    @classmethod
    def initialize(cls, n_agents: int, pipe: PipeGeometry, rng: np.random.Generator) -> "AgentSwarm":
        """Draw initial positions uniformly over Omega_pipe = [0,L] x [-R,R]
        (Algorithm 1, line 1), with zero initial velocity and personal best
        equal to the spawn position."""
        x = rng.uniform([0.0, -pipe.radius], [pipe.length, pipe.radius], size=(n_agents, 2))
        v = np.zeros((n_agents, 2))
        return cls(
            n_agents=n_agents,
            positions=x,
            velocities=v,
            personal_best=x.copy(),
            personal_best_fitness=np.full(n_agents, -np.inf),
        )

    def update_personal_best(self, fitness: np.ndarray) -> None:
        """Maximization convention throughout (Sec. II-A.3): larger fitness
        is a stronger, more target-proximal signal."""
        improved = fitness > self.personal_best_fitness
        self.personal_best[improved] = self.positions[improved]
        self.personal_best_fitness[improved] = fitness[improved]

    def channel_state(
        self,
        true_dose_rate: np.ndarray,
        latched: np.ndarray,
        rng: np.random.Generator,
        radiation_cfg: RadiationConfig,
        use_comms_degradation: bool,
        use_packet_loss: bool,
    ) -> np.ndarray:
        """Per-agent binary channel-state variable chi_i(t) (Sec. II-A.1):
        chi_k(t) ~ Bernoulli(1 - P_drop(x_k(t))), with permanently-latched
        agents forced to chi=0. If ``use_comms_degradation`` is False this
        is the ablation-cell-C condition: chi == 1 unconditionally (a
        fully-connected, comms-failure-immune topology).
        """
        if use_comms_degradation and use_packet_loss:
            p_drop = drop_probability(true_dose_rate, radiation_cfg)
            chi = (rng.uniform(size=self.n_agents) > p_drop).astype(float)
        else:
            chi = np.ones(self.n_agents)
        if use_comms_degradation:
            chi = chi.copy()
            chi[latched] = 0.0
        return chi

    def build_graph(self, chi: np.ndarray, r_comm: float) -> np.ndarray:
        """Dynamic communication graph G(t), Eq. 3: an edge (j, i) requires
        joint range and hardware operability at both endpoints. Every agent
        has a self-loop (i is always its own neighbor)."""
        diffs = self.positions[:, None, :] - self.positions[None, :, :]
        dist_mat = np.linalg.norm(diffs, axis=-1)
        in_range = dist_mat <= r_comm
        both_operational = np.outer(chi, chi) > 0
        edge = in_range & both_operational
        np.fill_diagonal(edge, True)
        return edge

    def local_best(self, edge: np.ndarray) -> np.ndarray:
        """Neighborhood-restricted local best g_i(t), Eq. 4: for each agent
        i, the personal best of the highest-fitness agent in its currently
        active neighborhood N_i(t) = {j : (j,i) in E(t)} U {i}."""
        g_i = np.empty((self.n_agents, 2))
        for i in range(self.n_agents):
            neighbors = np.where(edge[i])[0]
            best_j = neighbors[np.argmax(self.personal_best_fitness[neighbors])]
            g_i[i] = self.personal_best[best_j]
        return g_i

    def velocity_update(
        self,
        g_i: np.ndarray,
        social_gate: np.ndarray,
        rng: np.random.Generator,
        algo_cfg: AlgorithmConfig,
        agent_cfg: AgentPhysicalConfig,
        pipe: PipeGeometry,
        use_drag: bool,
    ) -> None:
        """M-PSO velocity update, Eq. 2:

            v_i(t+1) = w*v_i(t) + c1*r1*(p_i(t) - x_i(t))
                       + chi_i(t)*c2*r2*(g_i(t) - x_i(t)) + (F_d/m)*dt

        followed by velocity clamping and forward-Euler position
        integration, with position clipped to the pipe cross-section.
        ``social_gate`` carries the (possibly all-ones) chi_i(t)-style
        multiplier on the social term.

        RNG call order matters for exact reproducibility against the
        original monolithic simulator and must stay r1, r2, *then* (if
        ``use_drag``) the turbulent-flow draw inside ``fluid_velocity``:
        drag uses the *pre-update* velocity v(t) as its relative-velocity
        reference (physically correct: drag opposes the agent's actual
        current motion, not its not-yet-applied next-step velocity), so
        it is computed from ``old_v`` captured before ``self.velocities``
        is overwritten below, but the RNG draw for it happens after r1/r2
        regardless of ``use_drag`` -- swapping this order silently
        desynchronizes every downstream trial from the reference
        implementation even though each individual physics equation is
        unchanged. See tests/test_physics.py::test_rng_call_order.
        """
        old_v = self.velocities
        r1 = rng.uniform(0, 1, size=(self.n_agents, 1))
        r2 = rng.uniform(0, 1, size=(self.n_agents, 1))
        cognitive = algo_cfg.cognitive_weight * r1 * (self.personal_best - self.positions)
        social = social_gate[:, None] * algo_cfg.social_weight * r2 * (g_i - self.positions)

        v = algo_cfg.inertia_weight * old_v + cognitive + social
        if use_drag:
            v_fluid = fluid_velocity(self.positions, rng, pipe, agent_cfg)
            f_d = drag_force(old_v, v_fluid, agent_cfg)
            v = v + (f_d / agent_cfg.mass_kg) * algo_cfg.timestep_s

        speed = np.linalg.norm(v, axis=-1, keepdims=True)
        scale = np.minimum(1.0, agent_cfg.velocity_clamp_ms / np.where(speed < 1e-12, 1.0, speed))
        v = v * scale

        self.velocities = v
        self.positions = self.positions + v * algo_cfg.timestep_s
        self.positions[:, 1] = np.clip(self.positions[:, 1], -pipe.radius, pipe.radius)
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0.0, pipe.length)
