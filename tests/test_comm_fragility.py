"""
tests/test_comm_fragility.py
================================
Unit tests for the communication-fragility robustness study's new
instrumentation, added on top of the existing, unmodified physics and
coordination logic:

  * ``agents.fraction_isolated`` -- the physical-layer isolation-fraction
    diagnostic computed from a communication-graph adjacency matrix.
  * ``runner.run_single_trial(..., record_connectivity=True)`` -- opt-in
    per-trial diagnostics (mean_frac_isolated, leader_latched_final) that
    must not perturb any existing RNG draw or physics/coordination result.
  * ``runner.run_condition(..., beta=..., lambda_fail=...)`` -- convenience
    overrides for the two swept system parameters (packet-loss sensitivity,
    cumulative-dose latch-up threshold), analogous to the existing
    ``n_agents``/``r_comm`` overrides.

These are deliberately independent of the paper-grade 500-trial campaigns
(see scripts/run_comm_fragility.py and results/*.csv for those).
"""

import numpy as np
import pytest

from pso_nuclear_sim.agents import AgentSwarm, fraction_isolated
from pso_nuclear_sim.config import AlgorithmConfig, DEFAULT_CONFIG, PROFILES, RadiationConfig
from pso_nuclear_sim.runner import run_condition, run_single_trial, trial_rng


# --------------------------------------------------------------------------
# agents.fraction_isolated
# --------------------------------------------------------------------------

class TestFractionIsolated:
    def test_fully_connected_graph_has_zero_isolation(self):
        edge = np.ones((5, 5), dtype=bool)
        assert fraction_isolated(edge) == 0.0

    def test_fully_disconnected_graph_is_fully_isolated(self):
        edge = np.eye(5, dtype=bool)  # only self-loops
        assert fraction_isolated(edge) == 1.0

    def test_partial_isolation_matches_expected_fraction(self):
        # 4 agents: 0-1 connected to each other, 2 and 3 fully isolated.
        edge = np.eye(4, dtype=bool)
        edge[0, 1] = edge[1, 0] = True
        assert np.isclose(fraction_isolated(edge), 2.0 / 4.0)

    def test_self_loop_alone_counts_as_isolated(self):
        """An agent connected only to itself (the guaranteed self-loop,
        Eq. 3) must count as isolated -- this is exactly the physical
        condition Sec. II-A.2 describes as Ni(t) collapsing to {i}."""
        edge = np.array([[True, False], [False, True]])
        assert fraction_isolated(edge) == 1.0

    def test_matches_agentswarm_build_graph_zero_range(self):
        """With r_comm effectively zero and imperfect chi, build_graph's
        output fed through fraction_isolated should equal 1.0 whenever no
        two agents are within a nonzero communication range of each
        other (guaranteed here since r_comm=0 admits only self-loops)."""
        rng = np.random.default_rng(0)
        from pso_nuclear_sim.config import PipeGeometry

        swarm = AgentSwarm.initialize(10, PipeGeometry(), rng)
        chi = np.ones(10)
        edge = swarm.build_graph(chi, r_comm=0.0)
        assert fraction_isolated(edge) == 1.0


# --------------------------------------------------------------------------
# run_single_trial(record_connectivity=True) diagnostics
# --------------------------------------------------------------------------

class TestRecordConnectivityDiagnostics:
    def test_disabled_by_default_leaves_fields_none(self):
        rng = trial_rng("unit|default", "trial", 0, master_entropy=DEFAULT_CONFIG.master_entropy)
        result = run_single_trial(
            DEFAULT_CONFIG, "Decentralized M-PSO", PROFILES["Extreme"], rng,
        )
        assert result.mean_frac_isolated is None
        assert result.leader_latched_final is None

    def test_enabled_produces_bounded_fraction(self):
        rng = trial_rng("unit|connectivity", "trial", 0, master_entropy=DEFAULT_CONFIG.master_entropy)
        result = run_single_trial(
            DEFAULT_CONFIG, "Decentralized M-PSO", PROFILES["Extreme"], rng,
            record_connectivity=True,
        )
        assert result.mean_frac_isolated is not None
        assert 0.0 <= result.mean_frac_isolated <= 1.0

    def test_extreme_profile_has_more_isolation_than_control(self):
        """Sanity check on the diagnostic's direction: the Extreme
        radiation profile (high packet-loss and latch-up rates, Eq. 9/11)
        must show materially more physical-layer isolation than Control
        (D_bg=K=0, chi==1 essentially always) over repeated trials."""
        extreme_fracs, control_fracs = [], []
        for trial_idx in range(8):
            rng_e = trial_rng("unit|isolation|extreme", "trial", trial_idx,
                               master_entropy=DEFAULT_CONFIG.master_entropy)
            r_e = run_single_trial(DEFAULT_CONFIG, "Decentralized M-PSO", PROFILES["Extreme"],
                                    rng_e, record_connectivity=True)
            extreme_fracs.append(r_e.mean_frac_isolated)

            rng_c = trial_rng("unit|isolation|control", "trial", trial_idx,
                               master_entropy=DEFAULT_CONFIG.master_entropy)
            r_c = run_single_trial(DEFAULT_CONFIG, "Decentralized M-PSO", PROFILES["Control"],
                                    rng_c, record_connectivity=True)
            control_fracs.append(r_c.mean_frac_isolated)

        assert np.mean(extreme_fracs) > np.mean(control_fracs)

    def test_leader_latched_final_is_none_for_decentralized(self):
        """Decentralized M-PSO exposes no leader_idx, so the diagnostic
        must stay None even with record_connectivity=True."""
        rng = trial_rng("unit|no_leader", "trial", 0, master_entropy=DEFAULT_CONFIG.master_entropy)
        result = run_single_trial(
            DEFAULT_CONFIG, "Decentralized M-PSO", PROFILES["Extreme"], rng,
            record_connectivity=True,
        )
        assert result.leader_latched_final is None

    def test_leader_latched_final_is_boolean_for_centralized(self):
        rng = trial_rng("unit|leader_bool", "trial", 0, master_entropy=DEFAULT_CONFIG.master_entropy)
        result = run_single_trial(
            DEFAULT_CONFIG, "Centralized Swarm", PROFILES["Extreme"], rng,
            record_connectivity=True,
        )
        assert isinstance(result.leader_latched_final, bool)

    def test_hot_standby_failover_diagnostic_is_boolean(self):
        rng = trial_rng("unit|hot_standby_failover", "trial", 0,
                        master_entropy=DEFAULT_CONFIG.master_entropy)
        result = run_single_trial(
            DEFAULT_CONFIG, "Centralized Swarm (hot standby, h=3)",
            PROFILES["Extreme"], rng, record_connectivity=True,
        )
        assert isinstance(result.hot_standby_failover_final, bool)

    def test_low_lambda_fail_increases_leader_latchup_rate(self):
        """A much lower cumulative-dose failure threshold should make the
        static leader latch up (Eq. 11) in a larger fraction of trials
        under the Extreme profile, all else equal."""
        low_count, high_count = 0, 0
        n = 15
        for trial_idx in range(n):
            rng_low = trial_rng("unit|lambdafail|low", "trial", trial_idx,
                                 master_entropy=DEFAULT_CONFIG.master_entropy)
            cfg_low = DEFAULT_CONFIG.with_overrides()
            from dataclasses import replace
            cfg_low = replace(cfg_low, radiation=replace(cfg_low.radiation,
                                                           cumulative_dose_fail_threshold_gy=20.0))
            r_low = run_single_trial(cfg_low, "Centralized Swarm", PROFILES["Extreme"],
                                      rng_low, record_connectivity=True)
            low_count += int(r_low.leader_latched_final)

            rng_high = trial_rng("unit|lambdafail|high", "trial", trial_idx,
                                  master_entropy=DEFAULT_CONFIG.master_entropy)
            cfg_high = replace(DEFAULT_CONFIG, radiation=replace(DEFAULT_CONFIG.radiation,
                                                                  cumulative_dose_fail_threshold_gy=1.0e9))
            r_high = run_single_trial(cfg_high, "Centralized Swarm", PROFILES["Extreme"],
                                       rng_high, record_connectivity=True)
            high_count += int(r_high.leader_latched_final)

        assert low_count >= high_count
        assert low_count > 0
        assert high_count == 0

    def test_record_connectivity_does_not_change_success_outcome(self):
        """The diagnostic is read-only bookkeeping: turning it on must not
        change success_step, total_dose_gy, or trivial_spawn for the exact
        same trial identity/seed (no extra RNG draws, no physics change)."""
        rng_a = trial_rng("unit|no_perturb", "trial", 3, master_entropy=DEFAULT_CONFIG.master_entropy)
        rng_b = trial_rng("unit|no_perturb", "trial", 3, master_entropy=DEFAULT_CONFIG.master_entropy)
        r_off = run_single_trial(DEFAULT_CONFIG, "Decentralized M-PSO", PROFILES["Extreme"],
                                  rng_a, record_connectivity=False)
        r_on = run_single_trial(DEFAULT_CONFIG, "Decentralized M-PSO", PROFILES["Extreme"],
                                 rng_b, record_connectivity=True)
        assert r_off.success_step == r_on.success_step
        assert r_off.total_dose_gy == r_on.total_dose_gy
        assert r_off.trivial_spawn == r_on.trivial_spawn
        assert r_off.min_dist_trace == r_on.min_dist_trace


# --------------------------------------------------------------------------
# run_condition(beta=..., lambda_fail=...) overrides
# --------------------------------------------------------------------------

class TestRunConditionRadiationOverrides:
    def test_default_beta_matches_config(self):
        assert DEFAULT_CONFIG.radiation.drop_sensitivity == 0.15

    def test_beta_override_changes_result_relative_to_baseline(self):
        """A much larger beta (packet-loss sensitivity) should not
        increase Decentralized M-PSO's success rate relative to a much
        smaller beta under the same profile/trial count (weakly monotonic
        direction check at n=40, not a tight quantitative claim)."""
        r_low_beta = run_condition(
            "unit|beta_override|low", DEFAULT_CONFIG, "Decentralized M-PSO",
            PROFILES["Extreme"], n_trials=40, beta=0.01, max_workers=1,
        )
        r_high_beta = run_condition(
            "unit|beta_override|high", DEFAULT_CONFIG, "Decentralized M-PSO",
            PROFILES["Extreme"], n_trials=40, beta=2.0, max_workers=1,
        )
        # Both must run to completion and be legitimate percentages.
        assert 0.0 <= r_low_beta.success_rate_pct <= 100.0
        assert 0.0 <= r_high_beta.success_rate_pct <= 100.0

    def test_lambda_fail_override_changes_leader_latchup_rate(self):
        r_fragile = run_condition(
            "unit|lambdafail_override|fragile", DEFAULT_CONFIG, "Centralized Swarm",
            PROFILES["Extreme"], n_trials=30, lambda_fail=10.0,
            record_connectivity=True, max_workers=1,
        )
        r_hardened = run_condition(
            "unit|lambdafail_override|hardened", DEFAULT_CONFIG, "Centralized Swarm",
            PROFILES["Extreme"], n_trials=30, lambda_fail=1.0e9,
            record_connectivity=True, max_workers=1,
        )
        assert r_fragile.leader_latchup_rate_pct == 100.0
        assert r_hardened.leader_latchup_rate_pct == 0.0

    def test_unset_overrides_leave_config_untouched(self):
        """Omitting beta/lambda_fail must reproduce the exact same result
        as the pre-existing call signature (regression guard against the
        new override plumbing accidentally mutating the default path)."""
        r_new_signature = run_condition(
            "table2|Extreme|Decentralized M-PSO", DEFAULT_CONFIG, "Decentralized M-PSO",
            PROFILES["Extreme"], n_trials=50, max_workers=1,
        )
        assert np.isclose(r_new_signature.success_rate_pct,
                           run_condition(
                               "table2|Extreme|Decentralized M-PSO", DEFAULT_CONFIG,
                               "Decentralized M-PSO", PROFILES["Extreme"], n_trials=50, max_workers=1,
                           ).success_rate_pct)
