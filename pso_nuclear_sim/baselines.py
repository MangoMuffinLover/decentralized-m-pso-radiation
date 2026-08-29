"""
pso_nuclear_sim.baselines
============================
The benchmarked architectures, implemented as a small strategy hierarchy
sharing one interface: given the current swarm state and channel state
chi_i(t), produce this step's neighborhood/global attractor g_i(t) and the
social-term gate multiplier.

    Architecture.step(swarm, chi, latched, r_comm) -> (g_i, social_gate)

* DecentralizedMPSO        -- Sec. II-A: neighborhood-restricted g_i(t)
                               (Eq. 4), explicit chi-gate optional (Eq. 2).
* CentralizedStaticLeader  -- Sec. III-A: single leader broadcasts g_best,
                               freezes permanently on leader latch-up.
* CentralizedReElection    -- identical, except an inoperative leader is
                               replaced by the highest-fitness operational,
                               non-latched agent each step.
* CanonicalPSO             -- fully-connected, no drag, no communication-
                               degradation model (Eq. 1, unmodified).
* CentralizedHotStandby    -- static leader with one state-replicating hot
                               standby and an explicit missed-heartbeat
                               failover rule (post-review robustness check).

Centralized/Canonical architectures carry small pieces of state that persist
across timesteps within a trial (the current leader index, and the
best-so-far global attractor), so each is a stateful object constructed once
per trial via ``initialize`` and stepped once per timestep.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .agents import AgentSwarm

__all__ = [
    "Architecture",
    "DecentralizedMPSO",
    "CentralizedStaticLeader",
    "CentralizedReElection",
    "CentralizedHotStandby",
    "CentralizedHotStandbyH1",
    "CentralizedHotStandbyH3",
    "CentralizedHotStandbyH5",
    "CanonicalPSO",
    "ARCHITECTURE_REGISTRY",
]


class Architecture(ABC):
    """Common interface for all four benchmarked coordination strategies."""

    #: Human-readable name, matching the paper's Table II/III labels.
    name: str

    #: Whether this architecture participates in the communication-
    #: degradation model at all (drag/dose/chi are still always computed
    #: for bookkeeping, but Canonical PSO's *coordination* ignores chi).
    uses_comm_model: bool

    @classmethod
    @abstractmethod
    def initialize(cls, n_agents: int) -> "Architecture":
        ...

    @abstractmethod
    def step(
        self,
        swarm: AgentSwarm,
        chi: np.ndarray,
        latched: np.ndarray,
        r_comm: float,
        use_comms_degradation: bool,
        use_explicit_chi_gate: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (g_i, social_gate) for this timestep, each shape (N,) or
        (N, 2) as appropriate."""
        ...


@dataclass
class DecentralizedMPSO(Architecture):
    """Neighborhood-restricted decentralized coordination (this paper's
    proposed method). No persistent state beyond the swarm itself."""

    name: str = field(default="Decentralized M-PSO", init=False)
    uses_comm_model: bool = field(default=True, init=False)

    @classmethod
    def initialize(cls, n_agents: int) -> "DecentralizedMPSO":
        return cls()

    def step(
        self,
        swarm: AgentSwarm,
        chi: np.ndarray,
        latched: np.ndarray,
        r_comm: float,
        use_comms_degradation: bool,
        use_explicit_chi_gate: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        edge = swarm.build_graph(chi, r_comm)
        g_i = swarm.local_best(edge)
        if use_comms_degradation and use_explicit_chi_gate:
            social_gate = chi
        else:
            social_gate = np.ones(swarm.n_agents)
        return g_i, social_gate


@dataclass
class CentralizedStaticLeader(Architecture):
    """Single-leader broadcast; freezes permanently once the leader
    agent's own channel latches up (Sec. III-A)."""

    name: str = field(default="Centralized (static leader)", init=False)
    uses_comm_model: bool = field(default=True, init=False)
    leader_idx: int = 0
    g_best_global: np.ndarray = field(default=None)
    g_best_fitness: float = -np.inf

    @classmethod
    def initialize(cls, n_agents: int) -> "CentralizedStaticLeader":
        return cls(leader_idx=0, g_best_global=None, g_best_fitness=-np.inf)

    def step(
        self,
        swarm: AgentSwarm,
        chi: np.ndarray,
        latched: np.ndarray,
        r_comm: float,
        use_comms_degradation: bool,
        use_explicit_chi_gate: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.g_best_global is None:
            self.g_best_global = swarm.positions[0].copy()

        dist_to_leader = np.linalg.norm(swarm.positions - swarm.positions[self.leader_idx], axis=-1)
        can_report = (dist_to_leader <= r_comm) & (chi > 0)
        can_report[self.leader_idx] = True
        if chi[self.leader_idx] > 0:
            reporting = np.where(can_report)[0]
            best_j = reporting[np.argmax(swarm.personal_best_fitness[reporting])]
            if swarm.personal_best_fitness[best_j] > self.g_best_fitness:
                self.g_best_fitness = swarm.personal_best_fitness[best_j]
                self.g_best_global = swarm.personal_best[best_j].copy()

        g_i = np.tile(self.g_best_global, (swarm.n_agents, 1))
        social_gate = np.ones(swarm.n_agents)
        return g_i, social_gate


@dataclass
class CentralizedReElection(CentralizedStaticLeader):
    """Fault-tolerant variant of the static-leader baseline: an
    inoperative incumbent leader is replaced each step by the
    highest-fitness currently-operational, non-latched agent."""

    name: str = field(default="Centralized (re-election)", init=False)

    @classmethod
    def initialize(cls, n_agents: int) -> "CentralizedReElection":
        return cls(leader_idx=0, g_best_global=None, g_best_fitness=-np.inf)

    def step(
        self,
        swarm: AgentSwarm,
        chi: np.ndarray,
        latched: np.ndarray,
        r_comm: float,
        use_comms_degradation: bool,
        use_explicit_chi_gate: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if use_comms_degradation:
            operational = (chi > 0) & (~latched)
            if not operational[self.leader_idx] and np.any(operational):
                candidates = np.where(operational)[0]
                self.leader_idx = int(candidates[np.argmax(swarm.personal_best_fitness[candidates])])
            # If nobody is operational this step, leader_idx is left
            # unchanged and the broadcast simply fails to update below --
            # the election protocol itself requires reliable comms.
        return super().step(swarm, chi, latched, r_comm, use_comms_degradation, use_explicit_chi_gate)


@dataclass
class CentralizedHotStandby(CentralizedStaticLeader):
    """A deliberately bounded, dual-redundant centralized baseline.

    The active leader behaves like :class:`CentralizedStaticLeader`, but a
    single nearby standby is selected at the first timestep.  The active
    leader replicates its best-so-far state to that standby only when the two
    agents are in range and both physical radios are available.  A standby
    promotes after ``heartbeat_timeout_steps`` consecutive missing
    heartbeats, provided it is operational at the point of promotion.  It
    then becomes the sole active coordinator; this baseline intentionally
    does *not* add a chain of backups, a majority protocol, omniscient fault
    detection, or an error-free side channel.

    This definition gives centralized fault tolerance a fair, explicit
    recovery mechanism while retaining the paper's same stochastic
    range-and-channel model for every coordination message.  The initial
    standby is the closest non-leader agent at the first simulation step,
    rather than an arbitrary index, so the protocol starts from its most
    favorable available physical pairing without granting it a privileged
    location or a separate radio.
    """

    name: str = field(default="Centralized (hot standby)", init=False)
    heartbeat_timeout_steps: int = 3
    standby_idx: Optional[int] = None
    standby_g_best_global: np.ndarray = field(default=None)
    standby_g_best_fitness: float = -np.inf
    missed_heartbeats: int = 0
    failover_completed: bool = False

    @classmethod
    def initialize(cls, n_agents: int) -> "CentralizedHotStandby":
        return cls(heartbeat_timeout_steps=3)

    def _initialize_states(self, swarm: AgentSwarm) -> None:
        if self.g_best_global is None:
            self.g_best_global = swarm.positions[self.leader_idx].copy()
        if self.standby_idx is None and swarm.n_agents > 1:
            candidates = np.array([i for i in range(swarm.n_agents) if i != self.leader_idx])
            distances = np.linalg.norm(swarm.positions[candidates] - swarm.positions[self.leader_idx], axis=-1)
            self.standby_idx = int(candidates[np.argmin(distances)])
        if self.standby_idx is not None and self.standby_g_best_global is None:
            self.standby_g_best_global = swarm.positions[self.standby_idx].copy()

    def _update_active_state(
        self,
        swarm: AgentSwarm,
        chi: np.ndarray,
        r_comm: float,
    ) -> None:
        """Apply the same range-limited reporting rule as the static leader."""
        if chi[self.leader_idx] <= 0:
            return
        distances = np.linalg.norm(swarm.positions - swarm.positions[self.leader_idx], axis=-1)
        can_report = (distances <= r_comm) & (chi > 0)
        can_report[self.leader_idx] = True
        reporting = np.where(can_report)[0]
        best_j = reporting[np.argmax(swarm.personal_best_fitness[reporting])]
        if swarm.personal_best_fitness[best_j] > self.g_best_fitness:
            self.g_best_fitness = swarm.personal_best_fitness[best_j]
            self.g_best_global = swarm.personal_best[best_j].copy()

    def step(
        self,
        swarm: AgentSwarm,
        chi: np.ndarray,
        latched: np.ndarray,
        r_comm: float,
        use_comms_degradation: bool,
        use_explicit_chi_gate: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._initialize_states(swarm)
        self._update_active_state(swarm, chi, r_comm)

        if self.standby_idx is not None and not self.failover_completed:
            active_available = bool(chi[self.leader_idx] > 0 and not latched[self.leader_idx])
            standby_available = bool(chi[self.standby_idx] > 0 and not latched[self.standby_idx])
            in_range = bool(
                np.linalg.norm(swarm.positions[self.leader_idx] - swarm.positions[self.standby_idx]) <= r_comm
            )
            heartbeat_received = active_available and standby_available and in_range
            if heartbeat_received:
                self.standby_g_best_global = self.g_best_global.copy()
                self.standby_g_best_fitness = self.g_best_fitness
                self.missed_heartbeats = 0
            else:
                self.missed_heartbeats += 1

            if self.missed_heartbeats >= self.heartbeat_timeout_steps and standby_available:
                self.leader_idx = self.standby_idx
                self.g_best_global = self.standby_g_best_global.copy()
                self.g_best_fitness = self.standby_g_best_fitness
                self.standby_idx = None
                self.failover_completed = True

        g_i = np.tile(self.g_best_global, (swarm.n_agents, 1))
        return g_i, np.ones(swarm.n_agents)


@dataclass
class CentralizedHotStandbyH1(CentralizedHotStandby):
    """Hot standby with a one-timestep missed-heartbeat timeout."""

    name: str = field(default="Centralized (hot standby, h=1)", init=False)

    @classmethod
    def initialize(cls, n_agents: int) -> "CentralizedHotStandbyH1":
        return cls(heartbeat_timeout_steps=1)


@dataclass
class CentralizedHotStandbyH3(CentralizedHotStandby):
    """Hot standby with the pre-specified nominal three-step timeout."""

    name: str = field(default="Centralized (hot standby, h=3)", init=False)

    @classmethod
    def initialize(cls, n_agents: int) -> "CentralizedHotStandbyH3":
        return cls(heartbeat_timeout_steps=3)


@dataclass
class CentralizedHotStandbyH5(CentralizedHotStandby):
    """Hot standby with a five-timestep missed-heartbeat timeout."""

    name: str = field(default="Centralized (hot standby, h=5)", init=False)

    @classmethod
    def initialize(cls, n_agents: int) -> "CentralizedHotStandbyH5":
        return cls(heartbeat_timeout_steps=5)


@dataclass
class CanonicalPSO(Architecture):
    """Fully-connected, drag-unaware, communication-degradation-free
    baseline (Eq. 1, unmodified): global best visible to every agent every
    step, subject only to sensor-noise corruption of fitness."""

    name: str = field(default="Canonical PSO", init=False)
    uses_comm_model: bool = field(default=False, init=False)
    g_best_global: np.ndarray = field(default=None)
    g_best_fitness: float = -np.inf

    @classmethod
    def initialize(cls, n_agents: int) -> "CanonicalPSO":
        return cls(g_best_global=None, g_best_fitness=-np.inf)

    def step(
        self,
        swarm: AgentSwarm,
        chi: np.ndarray,
        latched: np.ndarray,
        r_comm: float,
        use_comms_degradation: bool,
        use_explicit_chi_gate: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.g_best_global is None:
            self.g_best_global = swarm.positions[0].copy()

        best_j = int(np.argmax(swarm.personal_best_fitness))
        if swarm.personal_best_fitness[best_j] > self.g_best_fitness:
            self.g_best_fitness = swarm.personal_best_fitness[best_j]
            self.g_best_global = swarm.personal_best[best_j].copy()

        g_i = np.tile(self.g_best_global, (swarm.n_agents, 1))
        social_gate = np.ones(swarm.n_agents)
        return g_i, social_gate


#: Lookup used by runner.py / scripts to construct an architecture by the
#: same name strings used in Table II/III of the paper.
ARCHITECTURE_REGISTRY = {
    "Decentralized M-PSO": DecentralizedMPSO,
    "Centralized Swarm": CentralizedStaticLeader,
    "Centralized Swarm (re-election)": CentralizedReElection,
    "Centralized Swarm (hot standby, h=1)": CentralizedHotStandbyH1,
    "Centralized Swarm (hot standby, h=3)": CentralizedHotStandbyH3,
    "Centralized Swarm (hot standby, h=5)": CentralizedHotStandbyH5,
    "Canonical PSO": CanonicalPSO,
}
