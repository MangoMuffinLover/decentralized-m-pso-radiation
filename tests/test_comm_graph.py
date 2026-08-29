"""
tests/test_comm_graph.py
===========================
Unit tests for the dynamic communication graph (Sec. II-A.1): range
gating, Bernoulli packet loss / channel-state computation, the explicit
chi_i(t) social-term gate, and the architecture strategies in
``pso_nuclear_sim.baselines`` that consume it.
"""

import numpy as np
import pytest

from pso_nuclear_sim.agents import AgentSwarm
from pso_nuclear_sim.baselines import (
    CanonicalPSO,
    CentralizedHotStandbyH1,
    CentralizedHotStandbyH3,
    CentralizedReElection,
    CentralizedStaticLeader,
    DecentralizedMPSO,
)
from pso_nuclear_sim.config import PipeGeometry, RadiationConfig


def _swarm_at(positions: np.ndarray) -> AgentSwarm:
    """Build a minimal AgentSwarm with the given fixed positions and
    otherwise-default state, for deterministic graph/gate tests."""
    n = positions.shape[0]
    return AgentSwarm(
        n_agents=n,
        positions=positions.astype(float),
        velocities=np.zeros((n, 2)),
        personal_best=positions.astype(float).copy(),
        personal_best_fitness=np.arange(n, dtype=float),  # agent i has fitness i
    )


# --------------------------------------------------------------------------
# Range gating, Eq. 3
# --------------------------------------------------------------------------

class TestRangeGating:
    def test_self_loop_always_present(self):
        """Every agent must always be its own neighbor, even fully
        isolated from everyone else."""
        positions = np.array([[0.0, 0.0], [100.0, 0.0]])  # far apart
        swarm = _swarm_at(positions)
        chi = np.ones(2)
        edge = swarm.build_graph(chi, r_comm=1.0)
        assert edge[0, 0] and edge[1, 1]

    def test_in_range_agents_are_connected(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        swarm = _swarm_at(positions)
        chi = np.ones(3)
        edge = swarm.build_graph(chi, r_comm=2.0)
        assert edge[0, 1] and edge[1, 0]      # 0 <-> 1: distance 1.0, within r_comm=2.0
        assert not edge[0, 2] and not edge[2, 0]  # 0 <-> 2: distance 10.0, out of range

    def test_edge_requires_symmetric_distance_check(self):
        """Distance is symmetric, so range-gating alone must produce a
        symmetric adjacency (before applying chi, which is also symmetric
        via the outer product)."""
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 10, size=(8, 2))
        swarm = _swarm_at(positions)
        chi = np.ones(8)
        edge = swarm.build_graph(chi, r_comm=3.0)
        assert np.array_equal(edge, edge.T)

    def test_shrinking_r_comm_never_increases_connectivity(self):
        rng = np.random.default_rng(1)
        positions = rng.uniform(0, 10, size=(12, 2))
        swarm = _swarm_at(positions)
        chi = np.ones(12)
        edge_wide = swarm.build_graph(chi, r_comm=5.0)
        edge_narrow = swarm.build_graph(chi, r_comm=1.0)
        # Every edge present at r_comm=1.0 must also be present at r_comm=5.0.
        assert np.all(edge_wide[edge_narrow])


# --------------------------------------------------------------------------
# Channel state chi_i(t): Bernoulli packet loss + permanent latch
# --------------------------------------------------------------------------

class TestChannelState:
    def test_disabled_comms_degradation_forces_chi_one(self):
        """Ablation cell C: use_comms_degradation=False must give
        chi == 1 for every agent unconditionally, regardless of dose or
        latch status."""
        positions = np.zeros((5, 2))
        swarm = _swarm_at(positions)
        rng = np.random.default_rng(0)
        latched = np.array([True, False, True, False, False])  # ignored when disabled
        chi = swarm.channel_state(
            true_dose_rate=np.full(5, 1000.0), latched=latched, rng=rng,
            radiation_cfg=RadiationConfig(), use_comms_degradation=False, use_packet_loss=True,
        )
        assert np.all(chi == 1.0)

    def test_latched_agents_always_gated_off(self):
        """When comms degradation is enabled, a latched agent's channel
        must read 0 regardless of the Bernoulli draw."""
        positions = np.zeros((3, 2))
        swarm = _swarm_at(positions)
        rng = np.random.default_rng(0)
        latched = np.array([True, False, False])
        chi = swarm.channel_state(
            true_dose_rate=np.zeros(3),  # zero dose -> P_drop=0 -> would-be-connected agents stay chi=1
            latched=latched, rng=rng, radiation_cfg=RadiationConfig(),
            use_comms_degradation=True, use_packet_loss=True,
        )
        assert chi[0] == 0.0
        assert chi[1] == 1.0 and chi[2] == 1.0

    def test_zero_dose_never_drops_unlatched_agents(self):
        """At D(x)=0, P_drop=0, so an unlatched agent's channel must
        never drop, across many independent draws."""
        positions = np.zeros((1, 2))
        swarm = _swarm_at(positions)
        radiation_cfg = RadiationConfig()
        for trial in range(200):
            rng = np.random.default_rng(trial)
            chi = swarm.channel_state(
                true_dose_rate=np.zeros(1), latched=np.array([False]), rng=rng,
                radiation_cfg=radiation_cfg, use_comms_degradation=True, use_packet_loss=True,
            )
            assert chi[0] == 1.0

    def test_no_packet_loss_flag_disables_only_transient_drop(self):
        """use_packet_loss=False must still force chi=0 for a latched
        agent (permanent failure is independent of the transient
        Bernoulli mechanism) but never drop a non-latched agent."""
        positions = np.zeros((2, 2))
        swarm = _swarm_at(positions)
        rng = np.random.default_rng(0)
        latched = np.array([True, False])
        chi = swarm.channel_state(
            true_dose_rate=np.full(2, 1000.0), latched=latched, rng=rng,
            radiation_cfg=RadiationConfig(), use_comms_degradation=True, use_packet_loss=False,
        )
        assert chi[0] == 0.0  # latched -> always off
        assert chi[1] == 1.0  # non-latched, transient drop disabled -> always on

    def test_high_dose_produces_high_drop_rate_statistically(self):
        """At a high (but finite) dose rate, the empirical drop rate over
        many trials should be close to the closed-form P_drop(x)."""
        radiation_cfg = RadiationConfig()
        dose = 20.0
        n_trials = 2000
        drops = 0
        positions = np.zeros((1, 2))
        for trial in range(n_trials):
            swarm = _swarm_at(positions)
            rng = np.random.default_rng(trial)
            chi = swarm.channel_state(
                true_dose_rate=np.array([dose]), latched=np.array([False]), rng=rng,
                radiation_cfg=radiation_cfg, use_comms_degradation=True, use_packet_loss=True,
            )
            if chi[0] == 0.0:
                drops += 1
        empirical_p_drop = drops / n_trials
        expected_p_drop = 1.0 - np.exp(-radiation_cfg.drop_sensitivity * dose)
        assert abs(empirical_p_drop - expected_p_drop) < 0.03  # generous Monte Carlo tolerance


# --------------------------------------------------------------------------
# Explicit chi_i(t) social-term gate (Sec. II-A.1 / ablation cell E)
# --------------------------------------------------------------------------

class TestExplicitChiGate:
    def test_gate_enabled_matches_chi(self):
        """With use_explicit_chi_gate=True (default), the social gate
        returned by DecentralizedMPSO.step must equal chi exactly."""
        positions = np.array([[0.0, 0.0], [0.5, 0.0], [50.0, 0.0]])
        swarm = _swarm_at(positions)
        chi = np.array([1.0, 0.0, 1.0])
        arch = DecentralizedMPSO.initialize(3)
        _, social_gate = arch.step(
            swarm, chi, latched=np.zeros(3, dtype=bool), r_comm=2.0,
            use_comms_degradation=True, use_explicit_chi_gate=True,
        )
        assert np.array_equal(social_gate, chi)

    def test_gate_disabled_is_all_ones(self):
        """Ablation cell E: use_explicit_chi_gate=False must give an
        all-ones social gate regardless of chi (agents rely on
        neighborhood self-collapse alone)."""
        positions = np.array([[0.0, 0.0], [0.5, 0.0], [50.0, 0.0]])
        swarm = _swarm_at(positions)
        chi = np.array([1.0, 0.0, 1.0])
        arch = DecentralizedMPSO.initialize(3)
        _, social_gate = arch.step(
            swarm, chi, latched=np.zeros(3, dtype=bool), r_comm=2.0,
            use_comms_degradation=True, use_explicit_chi_gate=False,
        )
        assert np.array_equal(social_gate, np.ones(3))

    def test_isolated_agent_self_collapses_regardless_of_gate(self):
        """Even with the explicit gate disabled, an agent with chi=0 and
        no in-range neighbors still only sees itself in N_i(t) -- the
        self-collapse mechanism is structural (Eq. 3/4), not contingent
        on the explicit gate."""
        positions = np.array([[0.0, 0.0], [50.0, 0.0]])  # far apart, so range alone isolates agent 1
        swarm = _swarm_at(positions)
        chi = np.array([1.0, 0.0])
        edge = swarm.build_graph(chi, r_comm=2.0)
        assert edge[1].sum() == 1 and edge[1, 1]  # agent 1 only ever sees itself
        g_i = swarm.local_best(edge)
        assert np.allclose(g_i[1], swarm.personal_best[1])


# --------------------------------------------------------------------------
# Architecture strategies consuming the graph / channel state
# --------------------------------------------------------------------------

class TestArchitectureStrategies:
    def test_decentralized_local_best_restricted_to_neighborhood(self):
        """Agent 0 and 2 are in range of each other but agent 1 is not in
        range of either; agent 2 has the highest fitness overall but
        agent 0's g_i must reflect only its own neighborhood."""
        positions = np.array([[0.0, 0.0], [100.0, 0.0], [1.0, 0.0]])
        swarm = _swarm_at(positions)  # fitness: agent0=0, agent1=1, agent2=2
        chi = np.ones(3)
        arch = DecentralizedMPSO.initialize(3)
        g_i, _ = arch.step(swarm, chi, latched=np.zeros(3, dtype=bool), r_comm=2.0,
                            use_comms_degradation=True, use_explicit_chi_gate=True)
        # Agent 0's neighborhood is {0, 2} (agent 1 is far away) -> best is agent 2.
        assert np.allclose(g_i[0], swarm.personal_best[2])
        # Agent 1 is isolated -> its own best.
        assert np.allclose(g_i[1], swarm.personal_best[1])

    def test_static_leader_freezes_when_leader_channel_drops(self):
        """CentralizedStaticLeader must not update g_best while the
        leader's own channel is down (chi[leader]=0), even if a
        higher-fitness agent has since come within range."""
        positions = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 0.0]])
        swarm = _swarm_at(positions)  # fitness: 0, 1, 2 -- agent 2 is best but starts far away
        arch = CentralizedStaticLeader.initialize(3)  # leader_idx=0 by default

        # First step: only the leader itself is in range (r_comm=1.0) -> adopts its own fitness (0.0).
        chi = np.array([1.0, 1.0, 1.0])
        arch.step(swarm, chi, latched=np.zeros(3, dtype=bool), r_comm=1.0,
                  use_comms_degradation=True, use_explicit_chi_gate=True)
        assert np.isclose(arch.g_best_fitness, 0.0)

        # Second step: agent 2 (fitness 2.0, the best) is now in range, but the
        # leader's own channel has dropped -- global best must NOT update.
        swarm.positions[2] = np.array([0.0, 0.0])  # now within range of the leader
        chi = np.array([0.0, 1.0, 1.0])
        arch.step(swarm, chi, latched=np.zeros(3, dtype=bool), r_comm=1.0,
                  use_comms_degradation=True, use_explicit_chi_gate=True)
        assert np.isclose(arch.g_best_fitness, 0.0)  # unchanged, frozen

    def test_reelection_replaces_inoperative_leader(self):
        """CentralizedReElection must switch leader_idx away from a
        latched incumbent to the highest-fitness operational agent."""
        positions = np.zeros((3, 2))
        swarm = _swarm_at(positions)  # fitness: 0, 1, 2
        arch = CentralizedReElection.initialize(3)
        arch.leader_idx = 0

        latched = np.array([True, False, False])  # leader (0) is latched
        chi = np.array([0.0, 1.0, 1.0])  # latched agent's chi is 0, per latch semantics
        arch.step(swarm, chi, latched=latched, r_comm=100.0,
                  use_comms_degradation=True, use_explicit_chi_gate=True)
        assert arch.leader_idx == 2  # highest-fitness operational candidate

    def test_reelection_leaves_leader_unchanged_if_nobody_operational(self):
        """If every agent is inoperative, the election cannot complete --
        leader_idx must stay unchanged rather than raising or picking
        arbitrarily (mirrors the paper's claim that re-election itself
        requires reliable communication)."""
        positions = np.zeros((2, 2))
        swarm = _swarm_at(positions)
        arch = CentralizedReElection.initialize(2)
        arch.leader_idx = 0
        latched = np.array([True, True])
        chi = np.array([0.0, 0.0])
        arch.step(swarm, chi, latched=latched, r_comm=100.0,
                  use_comms_degradation=True, use_explicit_chi_gate=True)
        assert arch.leader_idx == 0  # unchanged

    def test_hot_standby_selects_nearest_agent_and_replicates_state(self):
        """The standby is the closest available agent at initialization,
        then receives the active leader's best state only over a successful
        in-range heartbeat."""
        positions = np.array([[0.0, 0.0], [8.0, 0.0], [0.5, 0.0]])
        swarm = _swarm_at(positions)  # agent 2 is both closest and fittest in range
        arch = CentralizedHotStandbyH3.initialize(3)
        arch.step(
            swarm, chi=np.ones(3), latched=np.zeros(3, dtype=bool), r_comm=1.0,
            use_comms_degradation=True, use_explicit_chi_gate=True,
        )
        assert arch.standby_idx == 2
        assert arch.missed_heartbeats == 0
        assert np.isclose(arch.standby_g_best_fitness, arch.g_best_fitness)
        assert np.allclose(arch.standby_g_best_global, arch.g_best_global)

    def test_hot_standby_promotes_only_after_predefined_timeout(self):
        """A three-step timeout prevents one transient missed heartbeat
        from immediately switching coordinators; the third missed heartbeat
        promotes an operational standby using its replicated state."""
        positions = np.array([[0.0, 0.0], [0.5, 0.0], [8.0, 0.0]])
        swarm = _swarm_at(positions)
        arch = CentralizedHotStandbyH3.initialize(3)
        healthy = np.ones(3)
        latched = np.zeros(3, dtype=bool)
        arch.step(swarm, healthy, latched, r_comm=1.0,
                  use_comms_degradation=True, use_explicit_chi_gate=True)
        standby = arch.standby_idx
        active_down = np.array([0.0, 1.0, 1.0])
        arch.step(swarm, active_down, latched, r_comm=1.0,
                  use_comms_degradation=True, use_explicit_chi_gate=True)
        assert arch.leader_idx == 0 and not arch.failover_completed
        arch.step(swarm, active_down, latched, r_comm=1.0,
                  use_comms_degradation=True, use_explicit_chi_gate=True)
        assert arch.leader_idx == 0 and not arch.failover_completed
        arch.step(swarm, active_down, latched, r_comm=1.0,
                  use_comms_degradation=True, use_explicit_chi_gate=True)
        assert arch.leader_idx == standby
        assert arch.failover_completed and arch.standby_idx is None

    def test_hot_standby_h1_promotes_after_one_missed_heartbeat(self):
        positions = np.array([[0.0, 0.0], [0.5, 0.0]])
        swarm = _swarm_at(positions)
        arch = CentralizedHotStandbyH1.initialize(2)
        latched = np.zeros(2, dtype=bool)
        arch.step(swarm, np.ones(2), latched, r_comm=1.0,
                  use_comms_degradation=True, use_explicit_chi_gate=True)
        arch.step(swarm, np.array([0.0, 1.0]), latched, r_comm=1.0,
                  use_comms_degradation=True, use_explicit_chi_gate=True)
        assert arch.leader_idx == 1
        assert arch.failover_completed

    def test_canonical_pso_ignores_chi_entirely(self):
        """CanonicalPSO must select the globally best agent regardless of
        chi (it has no communication-degradation model)."""
        positions = np.zeros((3, 2))
        swarm = _swarm_at(positions)  # fitness: 0, 1, 2 -- agent 2 is best
        arch = CanonicalPSO.initialize(3)
        chi = np.array([0.0, 0.0, 0.0])  # would fully disconnect any comm-aware architecture
        arch.step(swarm, chi, latched=np.zeros(3, dtype=bool), r_comm=0.001,
                  use_comms_degradation=True, use_explicit_chi_gate=True)
        assert np.isclose(arch.g_best_fitness, 2.0)
        assert not arch.uses_comm_model
