"""
pso_nuclear_sim.runner
=========================
Per-trial orchestration (the loop in Fig. 2 / Algorithm 1) and the
parallelized Monte Carlo campaign engine: SHA-256 condition-identity trial
seeding, Wilson-score 95% confidence intervals, trivial-spawn tracking, and
metric aggregation across trials.

.. important:: Known issue found during this refactor (drag / Canonical PSO)

    The original single-file simulator applied hydrodynamic drag under the
    condition ``architecture == "Canonical PSO" or not use_drag``. Because
    that is a boolean ``or``, drag was *unconditionally* skipped whenever
    ``architecture == "Canonical PSO"``, regardless of the ``use_drag``
    flag's value. This is correct and intentional for the paper's ordinary
    Canonical PSO baseline (Table II), which is defined as drag-free -- but
    it silently defeated ablation cell D ("Canonical PSO + drag added",
    ``use_drag=True``), which was *supposed* to test drag in combination
    with a non-decentralized architecture. Cell D's originally-reported
    26.6% therefore differs from the drag-free Canonical baseline (29.6%)
    by nothing but Monte Carlo sampling noise between two different
    condition-identity seeds -- it never actually applied drag at all.

    This module fixes that by removing the architecture-based override:
    drag is applied purely based on ``AblationFlags.use_drag``, full stop.
    The *caller* is responsible for constructing each named condition with
    the right flags -- see ``scripts/run_benchmarks.py``, which explicitly
    passes ``AblationFlags(use_drag=False)`` when building the ordinary
    Canonical PSO baseline rows, and ``AblationFlags(use_drag=True)``
    (the dataclass default) for ablation cell D. This keeps "which
    architecture computes coordination" and "does drag physics apply"
    as two independent, orthogonal settings, rather than one silently
    overriding the other. Re-running cell D under this fix is expected to
    change its reported success rate; see README "Known Issues Found
    During This Refactor" for the corrected value and what it implies for
    the M4 ablation discussion in the manuscript.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

from .agents import AgentSwarm, fraction_isolated
from .baselines import ARCHITECTURE_REGISTRY, Architecture
from .config import AblationFlags, PROFILES, RadiationProfile, SimulationConfig
from .environment import RadiationState, dose_rate, sensor_noise_std, true_signal

__all__ = [
    "trial_rng",
    "wilson_ci",
    "TrialResult",
    "ConditionResult",
    "run_single_trial",
    "run_condition",
]


def trial_rng(*key_parts: object, master_entropy: int) -> np.random.Generator:
    """Collision-proof per-trial RNG: SHA-256 hash of an explicit,
    human-readable identity string -> SeedSequence -> Generator.
    Deterministic and reproducible, but structurally incapable of
    colliding across distinct conditions regardless of call order, script,
    process, or session (Code and Data Availability, Sec. VI)."""
    key_str = "|".join(str(k) for k in key_parts) + f"|master={master_entropy}"
    digest = hashlib.sha256(key_str.encode()).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(np.random.SeedSequence(seed_int))


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion,
    expressed as a percentage. Used for every success-rate CI in the
    paper (z=1.96)."""
    if n_total == 0:
        return (float("nan"), float("nan"))
    p_hat = n_success / n_total
    denom = 1.0 + z**2 / n_total
    centre = p_hat + z**2 / (2 * n_total)
    adj = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total)
    return (centre - adj) / denom * 100.0, (centre + adj) / denom * 100.0


@dataclass
class TrialResult:
    """Outcome of a single Monte Carlo trial."""

    success_step: Optional[int]
    total_dose_gy: float
    trivial_spawn: bool
    min_dist_trace: List[float] = field(repr=False)
    position_trace: Optional[np.ndarray] = field(default=None, repr=False)
    #: Communication-fragility diagnostics (Sec. "communication-fragility
    #: robustness study"), populated only when ``record_connectivity=True``
    #: is passed to ``run_single_trial`` -- None otherwise, so ordinary
    #: campaigns (Tables II-V) are completely unaffected. See
    #: ``agents.fraction_isolated`` and the ``leader_idx`` check below.
    mean_frac_isolated: Optional[float] = field(default=None)
    leader_latched_final: Optional[bool] = field(default=None)
    hot_standby_failover_final: Optional[bool] = field(default=None)


def run_single_trial(
    config: SimulationConfig,
    architecture_name: str,
    profile: RadiationProfile,
    rng: np.random.Generator,
    ablation: AblationFlags = AblationFlags(),
    record_trajectory: bool = False,
    record_connectivity: bool = False,
) -> TrialResult:
    """Execute one full trial (Algorithm 1) and return its outcome.

    ``config.algorithm`` supplies ``n_agents``, ``comm_range_m``, and
    ``success_threshold_m`` for this trial -- callers sweeping N or
    r_comm should pass a config produced via
    ``config.with_overrides(algorithm=dataclasses.replace(config.algorithm, ...))``
    (see ``run_condition``'s ``n_agents``/``r_comm`` convenience kwargs).

    ``record_trajectory=True`` additionally records every agent's position
    at every timestep (shape ``(max_steps, n_agents, 2)``) for spatial
    dispersion plots (Fig. 2(b) / ``scripts/make_figures.py``). Off by
    default since it is not needed for aggregate Monte Carlo metrics and
    materially increases memory use across a 500-trial campaign.

    ``record_connectivity=True`` additionally computes, at every timestep,
    the *physical*-layer communication graph (range + chi_i(t), Eq. 3) via
    ``AgentSwarm.build_graph`` -- independent of which architecture is
    actually consuming it for coordination -- and tracks the fraction of
    agents with a fully collapsed neighborhood (``agents.fraction_isolated``),
    averaged over the trial into ``TrialResult.mean_frac_isolated``. It also
    checks, once at trial end, whether the architecture's *own* leader
    (``getattr(architecture, "leader_idx", None)``, present only for
    ``CentralizedStaticLeader``/``CentralizedReElection``) is permanently
    latched, into ``TrialResult.leader_latched_final``. Both are read-only
    diagnostics added for the communication-fragility robustness study
    (varying beta / Lambda_fail); they consume no RNG draws and do not
    alter the physics or coordination logic in any way, so every existing
    baseline/ablation/sweep result is bit-for-bit unaffected when this flag
    is left at its default of False.
    """
    pipe, agent_cfg, algo_cfg, rad_cfg = (
        config.pipe,
        config.agent,
        config.algorithm,
        config.radiation,
    )

    architecture: Architecture = ARCHITECTURE_REGISTRY[architecture_name].initialize(algo_cfg.n_agents)
    swarm = AgentSwarm.initialize(algo_cfg.n_agents, pipe, rng)
    radiation_state = RadiationState(algo_cfg.n_agents, rad_cfg.cumulative_dose_fail_threshold_gy)

    success_step: Optional[int] = None
    trivial_spawn = False
    min_dist_trace: List[float] = []
    position_trace: List[np.ndarray] = [] if record_trajectory else None
    frac_isolated_trace: List[float] = [] if record_connectivity else None

    for t in range(algo_cfg.max_steps):
        if record_trajectory:
            position_trace.append(swarm.positions.copy())
        d_now_gated = dose_rate(swarm.positions, profile, pipe) if ablation.use_comms_degradation \
            else np.zeros(algo_cfg.n_agents)
        d_now_real = dose_rate(swarm.positions, profile, pipe)  # true physical dose, always accumulated
        radiation_state.update(d_now_real, algo_cfg.timestep_s, gated=ablation.use_comms_degradation)

        s_true = true_signal(swarm.positions, pipe)
        if ablation.use_sensor_noise:
            noise = rng.normal(0.0, sensor_noise_std(d_now_real, rad_cfg))
        else:
            noise = np.zeros(algo_cfg.n_agents)
        swarm.update_personal_best(s_true + noise)

        if t == 0:
            dist0 = np.linalg.norm(swarm.personal_best - pipe.x_f, axis=-1)
            if np.min(dist0) <= algo_cfg.success_threshold_m:
                trivial_spawn = True

        if architecture.uses_comm_model:
            chi = swarm.channel_state(
                d_now_real, radiation_state.latched, rng, rad_cfg,
                ablation.use_comms_degradation, ablation.use_packet_loss,
            )
        else:
            chi = np.ones(algo_cfg.n_agents)

        if record_connectivity:
            physical_edge = swarm.build_graph(chi, algo_cfg.comm_range_m)
            frac_isolated_trace.append(fraction_isolated(physical_edge))

        g_i, social_gate = architecture.step(
            swarm, chi, radiation_state.latched, algo_cfg.comm_range_m,
            ablation.use_comms_degradation, ablation.use_explicit_chi_gate,
        )

        min_dist = float(np.min(np.linalg.norm(g_i - pipe.x_f, axis=-1)))
        min_dist_trace.append(min_dist)
        if success_step is None and min_dist <= algo_cfg.success_threshold_m:
            success_step = t

        # Drag applies purely based on ablation.use_drag -- see module
        # docstring re: the architecture-coupling bug this replaces. The
        # actual drag computation lives inside velocity_update itself so
        # that its RNG draw happens strictly after r1/r2 (see that
        # method's docstring for why this order is load-bearing).
        swarm.velocity_update(g_i, social_gate, rng, algo_cfg, agent_cfg, pipe, ablation.use_drag)

    leader_latched_final: Optional[bool] = None
    hot_standby_failover_final: Optional[bool] = None
    if record_connectivity:
        leader_idx = getattr(architecture, "leader_idx", None)
        if leader_idx is not None:
            leader_latched_final = bool(radiation_state.latched[leader_idx])
        failover_completed = getattr(architecture, "failover_completed", None)
        if failover_completed is not None:
            hot_standby_failover_final = bool(failover_completed)

    return TrialResult(
        success_step=success_step,
        total_dose_gy=radiation_state.total_dose,
        trivial_spawn=trivial_spawn,
        min_dist_trace=min_dist_trace,
        position_trace=np.array(position_trace) if record_trajectory else None,
        mean_frac_isolated=float(np.mean(frac_isolated_trace)) if record_connectivity else None,
        leader_latched_final=leader_latched_final,
        hot_standby_failover_final=hot_standby_failover_final,
    )


@dataclass
class ConditionResult:
    """Aggregated metrics for one Monte Carlo campaign (one row of Table
    II/III/IV/V), matching the fields written to every results CSV."""

    condition_id: str
    n_success: int
    n_total: int
    success_rate_pct: float
    ci95_low: float
    ci95_high: float
    mean_conv_steps: Optional[float]
    median_conv_steps: Optional[float]
    mean_cum_dose_gy: float
    std_cum_dose_gy: float
    n_trivial_spawns: int
    trivial_spawn_rate_pct: float
    nontrivial_success_rate_pct: float
    nontrivial_ci95_low: float
    nontrivial_ci95_high: float
    nontrivial_n_total: int
    #: Communication-fragility diagnostics (None unless
    #: ``record_connectivity=True`` was passed to ``run_condition``).
    #: ``mean_frac_isolated``: trial- and time-averaged fraction of agents
    #: with a fully collapsed physical-layer neighborhood (Eq. 3),
    #: independent of coordination architecture.
    #: ``leader_latchup_rate_pct``: fraction of trials (as a percentage) in
    #: which the architecture's own leader is permanently latched
    #: (Eq. 11) by mission end -- only defined for architectures exposing
    #: a ``leader_idx`` (CentralizedStaticLeader / CentralizedReElection);
    #: None for Decentralized M-PSO / Canonical PSO.
    #: ``hot_standby_failover_rate_pct``: percentage of hot-standby trials
    #: in which the pre-specified timeout triggered a completed promotion.
    mean_frac_isolated: Optional[float] = None
    leader_latchup_rate_pct: Optional[float] = None
    hot_standby_failover_rate_pct: Optional[float] = None


def _trial_worker(args) -> TrialResult:
    """Module-level (picklable) worker for ProcessPoolExecutor."""
    (condition_id, trial_idx, config, architecture_name, profile, ablation,
     master_entropy, record_connectivity) = args
    rng = trial_rng(condition_id, "trial", trial_idx, master_entropy=master_entropy)
    return run_single_trial(
        config, architecture_name, profile, rng, ablation,
        record_connectivity=record_connectivity,
    )


def run_condition(
    condition_id: str,
    config: SimulationConfig,
    architecture_name: str,
    profile: RadiationProfile,
    ablation: AblationFlags = AblationFlags(),
    n_trials: Optional[int] = None,
    n_agents: Optional[int] = None,
    r_comm: Optional[float] = None,
    success_threshold: Optional[float] = None,
    beta: Optional[float] = None,
    lambda_fail: Optional[float] = None,
    record_connectivity: bool = False,
    max_workers: Optional[int] = None,
) -> ConditionResult:
    """Run a full Monte Carlo campaign for one named condition and return
    aggregated metrics, including the non-trivial-spawn-controlled success
    rate reported throughout Table II.

    ``condition_id`` is the human-readable identity string baked into every
    trial's SHA-256 seed (e.g. ``"table2|Extreme|Decentralized M-PSO"``);
    two calls with the same ``condition_id`` and ``n_trials`` are
    bit-for-bit reproducible, and two calls with *different*
    ``condition_id`` values are seeded independently regardless of call
    order (this is the fix for the RNG-collision bug documented in the
    Code and Data Availability section of the paper).

    ``n_agents``, ``r_comm``, and ``success_threshold`` are convenience
    overrides for the swarm-size (Table IV) and communication-range
    (Table V) sensitivity sweeps and the threshold-sensitivity check,
    applied on top of ``config.algorithm`` without mutating the shared
    default config.

    ``beta`` and ``lambda_fail`` are the analogous convenience overrides
    for the communication-fragility robustness study (packet-loss
    sensitivity, Eq. 9, and cumulative-dose latch-up threshold, Eq. 11),
    applied on top of ``config.radiation`` without mutating the shared
    default config. Both are physically meaningful *system* parameters
    (hardware/channel properties), distinct from the radiological
    *environment* profiles (Control/Moderate/Extreme, which vary D_bg/K).

    ``record_connectivity`` forwards to ``run_single_trial``: when True,
    every trial additionally computes and returns the physical-layer
    isolation-fraction and leader-latch-up diagnostics described there.
    Left False by default so every pre-existing campaign (Tables II-V) is
    completely unaffected; the new communication-fragility sweep script
    is the only caller that sets it True.
    """
    n_trials = n_trials or config.n_monte_carlo
    algo_overrides = {}
    if n_agents is not None:
        algo_overrides["n_agents"] = n_agents
    if r_comm is not None:
        algo_overrides["comm_range_m"] = r_comm
    if success_threshold is not None:
        algo_overrides["success_threshold_m"] = success_threshold
    run_config = config if not algo_overrides else replace(
        config, algorithm=replace(config.algorithm, **algo_overrides)
    )

    rad_overrides = {}
    if beta is not None:
        rad_overrides["drop_sensitivity"] = beta
    if lambda_fail is not None:
        rad_overrides["cumulative_dose_fail_threshold_gy"] = lambda_fail
    if rad_overrides:
        run_config = replace(run_config, radiation=replace(run_config.radiation, **rad_overrides))

    work_items = [
        (condition_id, trial_idx, run_config, architecture_name, profile, ablation,
         config.master_entropy, record_connectivity)
        for trial_idx in range(n_trials)
    ]

    if max_workers == 1:
        results = [_trial_worker(item) for item in work_items]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_trial_worker, work_items))

    successes = [1 if r.success_step is not None else 0 for r in results]
    conv_times = [r.success_step for r in results if r.success_step is not None]
    doses = [r.total_dose_gy for r in results]
    trivial_flags = [r.trivial_spawn for r in results]

    n_success = int(sum(successes))
    success_rate = 100.0 * n_success / n_trials
    ci_low, ci_high = wilson_ci(n_success, n_trials)

    non_trivial_mask = [not tf for tf in trivial_flags]
    n_nontrivial_total = sum(non_trivial_mask)
    n_nontrivial_success = sum(s for s, keep in zip(successes, non_trivial_mask) if keep)
    if n_nontrivial_total > 0:
        nt_rate = 100.0 * n_nontrivial_success / n_nontrivial_total
        nt_ci = wilson_ci(n_nontrivial_success, n_nontrivial_total)
    else:
        nt_rate, nt_ci = float("nan"), (float("nan"), float("nan"))

    n_trivial = int(sum(trivial_flags))

    mean_frac_isolated: Optional[float] = None
    leader_latchup_rate_pct: Optional[float] = None
    hot_standby_failover_rate_pct: Optional[float] = None
    if record_connectivity:
        frac_isolated_vals = [r.mean_frac_isolated for r in results if r.mean_frac_isolated is not None]
        if frac_isolated_vals:
            mean_frac_isolated = float(np.mean(frac_isolated_vals))
        leader_flags = [r.leader_latched_final for r in results if r.leader_latched_final is not None]
        if leader_flags:
            leader_latchup_rate_pct = 100.0 * sum(leader_flags) / len(leader_flags)
        failover_flags = [
            r.hot_standby_failover_final
            for r in results
            if r.hot_standby_failover_final is not None
        ]
        if failover_flags:
            hot_standby_failover_rate_pct = 100.0 * sum(failover_flags) / len(failover_flags)

    return ConditionResult(
        condition_id=condition_id,
        n_success=n_success,
        n_total=n_trials,
        success_rate_pct=success_rate,
        ci95_low=ci_low,
        ci95_high=ci_high,
        mean_conv_steps=float(np.mean(conv_times)) if conv_times else None,
        median_conv_steps=float(np.median(conv_times)) if conv_times else None,
        mean_cum_dose_gy=float(np.mean(doses)),
        std_cum_dose_gy=float(np.std(doses)),
        n_trivial_spawns=n_trivial,
        trivial_spawn_rate_pct=100.0 * n_trivial / n_trials,
        nontrivial_success_rate_pct=nt_rate,
        nontrivial_ci95_low=nt_ci[0],
        nontrivial_ci95_high=nt_ci[1],
        nontrivial_n_total=n_nontrivial_total,
        mean_frac_isolated=mean_frac_isolated,
        leader_latchup_rate_pct=leader_latchup_rate_pct,
        hot_standby_failover_rate_pct=hot_standby_failover_rate_pct,
    )
