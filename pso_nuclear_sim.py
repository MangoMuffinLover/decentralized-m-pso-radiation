#!/usr/bin/env python3
"""
pso_nuclear_sim.py
==================

Reference implementation of the Decentralized M-PSO swarm-inspection model
described in "Decentralized Swarm Resilience Under Stochastic Gamma Noise:
A Computational Model for Pressure Vessel Inspection".

This script implements, EXACTLY as formalized in the manuscript (Section II,
post pre-submission-review corrections):

  - The one-seventh power-law turbulent fluid velocity profile + stochastic
    turbulent fluctuation (Eq. II.B).
  - The quadratic hydrodynamic drag force (Eq. II.B), integrated over the
    explicit timestep Delta_t for dimensional consistency with the
    forward-Euler position update.
  - The non-homogeneous gamma dose-rate field D(x) (Eq. II.C).
  - Dual-phenomenon microelectronic degradation: continuous Gaussian sensor
    noise and discrete Bernoulli communication-channel failure, PLUS a
    genuine per-agent cumulative-dose state Lambda_i(t) driving a permanent
    "latch-up" failure mode once Lambda_i(t) >= Lambda_fail.
  - A single, consistent MAXIMIZATION convention for the fitness function
    f(.) used by both the personal-best (argmax) and neighborhood-best
    (argmax) attractors.
  - A communication graph in which edge (j,i) requires BOTH agents' channel
    states operational (chi_j(t)=1 AND chi_i(t)=1), resolving the
    sender/receiver ambiguity flagged in pre-submission review.

Three architectures are implemented and benchmarked head-to-head under three
radiological regimes (Control / Moderate / Extreme), each run for
N_MONTE_CARLO independent Monte Carlo trials, exactly reproducing the
methodology described in Section III of the manuscript. Table II and Figs. 1-2
in the manuscript are generated directly from this script's output -- no
numbers in the manuscript's results section are hand-entered.

Usage:
    python3 pso_nuclear_sim.py
Outputs:
    results_table2.csv       -- full statistical summary (mean, std, n_success)
    results_table2.json      -- same, machine-readable
    fig1_convergence_data.csv-- representative single-run convergence traces
    fig2_trajectory_data.csv -- representative single-run agent trajectories
"""

import numpy as np
import json
import csv
import time

# ----------------------------------------------------------------------
# Table I -- Physical and Computational Simulation Configuration Parameters
# ----------------------------------------------------------------------
L = 10.0                 # Pipe length (m)
R = 1.0                  # Pipe radius (m)
X_F = np.array([8.0, 0.9])  # Fissure position (m)

M_AGENT = 0.05            # Micro-agent mass (kg)
RHO = 1000.0              # Coolant density (kg/m^3)
CD = 0.47                 # Drag coefficient
AREA = 1.257e-3           # Frontal cross-sectional area (m^2)
V_MAX = 1.5               # Max centerline fluid velocity (m/s)
SIGMA_TURB = 0.1          # Turbulent fluctuation std dev (m/s)

N_AGENTS = 25             # Swarm size
DT = 0.1                  # Timestep size (s)
T_MAX = 150               # Max simulation steps
W_INERTIA = 0.6           # Inertia weight
C1 = 1.2                  # Cognitive acceleration coefficient
C2 = 1.2                  # Social acceleration coefficient
R_COMM = 2.0              # Communication range (m)

BETA = 0.15               # Packet-drop sensitivity (s/Gy)
SIGMA_GAMMA = 0.08        # Sensor-noise proportionality constant
LAMBDA_FAIL = 300.0       # Cumulative-dose permanent-failure threshold (Gy)

SUCCESS_THRESHOLD = 0.3   # Mission-success distance threshold (m) -- tight
                          # enough to require genuine convergence rather than
                          # a lucky initial deployment position
N_MONTE_CARLO = 500       # Monte Carlo runs per architecture per profile
V_CLAMP = 3.0             # Standard PSO velocity clamp (m/s), prevents divergence
                          # (Eberhart & Shi, 1998 velocity-clamping convention)

# Radiological regime definitions: (D_bg [Gy/s], K [Gy/s])
PROFILES = {
    "Control":  (0.0, 0.0),
    "Moderate": (0.5, 10.0),
    "Extreme":  (2.0, 100.0),
}

ARCHITECTURES = ["Decentralized M-PSO", "Centralized Swarm", "Canonical PSO"]

rng = np.random.default_rng(20260722)


def dose_rate(pos, d_bg, k):
    """D(x) = D_bg + K / (||x - x_f||^2 + 1.0)   -- Eq. (15)"""
    dist_sq = np.sum((pos - X_F) ** 2, axis=-1)
    return d_bg + k / (dist_sq + 1.0)


def true_signal(pos):
    """S(x): true (noiseless) sensor signal, peaked at the fissure. Uses the
    same inverse-square-plus-one functional form as the dose field D(x) for
    a smooth, pipe-length-scale basin of attraction (Eq. 16)."""
    dist_sq = np.sum((pos - X_F) ** 2, axis=-1)
    return 1.0 / (dist_sq + 1.0)


def fluid_velocity(pos, rng_local):
    """v_fluid(y) = v_max * (1 - |y|/R)^(1/7) * i_hat + v_turb   -- Eq. (12)-(13)"""
    y = pos[:, 1]
    frac = np.clip(1.0 - np.abs(y) / R, 0.0, None)
    vx = V_MAX * frac ** (1.0 / 7.0)
    v_turb = rng_local.normal(0.0, SIGMA_TURB, size=pos.shape)
    v_fluid = np.zeros_like(pos)
    v_fluid[:, 0] = vx
    return v_fluid + v_turb


def drag_force(v_agent, v_fluid):
    """F_d = -0.5 * rho * Cd * A * ||v_rel||^2 * u_hat_d   -- Eq. (14)"""
    v_rel = v_agent - v_fluid
    speed = np.linalg.norm(v_rel, axis=-1, keepdims=True)
    speed_safe = np.where(speed < 1e-9, 1.0, speed)
    u_hat = v_rel / speed_safe
    mag = 0.5 * RHO * CD * AREA * speed ** 2
    f_d = -mag * u_hat
    f_d[speed.flatten() < 1e-9] = 0.0
    return f_d


def run_single_trial(architecture, d_bg, k, rng_local):
    """Execute one Monte Carlo trial for the given architecture/profile."""
    # Initialize agent state
    x = rng_local.uniform([0.0, -R], [L, R], size=(N_AGENTS, 2))  # spread across full pipe segment
    v = np.zeros((N_AGENTS, 2))
    p_best = x.copy()
    p_fitness = np.full(N_AGENTS, -np.inf)
    lam = np.zeros(N_AGENTS)          # cumulative dose Lambda_i(t)
    latched = np.zeros(N_AGENTS, dtype=bool)  # permanent failure flag

    g_best_global = x[0].copy()       # for centralized/canonical
    g_best_fitness = -np.inf
    leader_idx = 0

    success_step = None
    dist_trace = []

    for t in range(T_MAX):
        # ---- Physical environment ----
        d_now = dose_rate(x, d_bg, k)
        lam += d_now * DT

        newly_latched = (lam >= LAMBDA_FAIL) & (~latched)
        latched[newly_latched] = True

        # ---- Sensor evaluation (fitness), always maximization convention ----
        s_true = true_signal(x)
        noise = rng_local.normal(0.0, SIGMA_GAMMA * np.sqrt(np.maximum(d_now, 0.0)))
        s_measured = s_true + noise

        improved = s_measured > p_fitness
        p_best[improved] = x[improved]
        p_fitness[improved] = s_measured[improved]

        # ---- Communication channel state chi_i(t) ----
        if architecture == "Decentralized M-PSO" or architecture == "Centralized Swarm":
            p_drop = 1.0 - np.exp(-BETA * d_now)
            chi = (rng_local.uniform(size=N_AGENTS) > p_drop).astype(float)
            chi[latched] = 0.0
        else:  # Canonical PSO: no comm degradation modeled at all
            chi = np.ones(N_AGENTS)

        # ---- Neighborhood / global-best construction ----
        if architecture == "Decentralized M-PSO":
            diffs = x[:, None, :] - x[None, :, :]
            dist_mat = np.linalg.norm(diffs, axis=-1)
            in_range = dist_mat <= R_COMM
            both_operational = np.outer(chi, chi) > 0
            edge = in_range & both_operational
            np.fill_diagonal(edge, True)  # self always included

            g_i = np.empty((N_AGENTS, 2))
            for i in range(N_AGENTS):
                neighbors = np.where(edge[i])[0]
                best_j = neighbors[np.argmax(p_fitness[neighbors])]
                g_i[i] = p_best[best_j]
            social_gate = chi  # zero out social term when agent i itself is isolated

        elif architecture == "Centralized Swarm":
            # Followers -> leader link requires range + follower's own chi;
            # leader -> broadcast requires leader's own chi.
            dist_to_leader = np.linalg.norm(x - x[leader_idx], axis=-1)
            can_report = (dist_to_leader <= R_COMM) & (chi > 0)
            can_report[leader_idx] = True
            if chi[leader_idx] > 0:
                reporting = np.where(can_report)[0]
                best_j = reporting[np.argmax(p_fitness[reporting])]
                if p_fitness[best_j] > g_best_fitness:
                    g_best_fitness = p_fitness[best_j]
                    g_best_global = p_best[best_j].copy()
            g_i = np.tile(g_best_global, (N_AGENTS, 1))
            social_gate = np.ones(N_AGENTS)  # followers passively receive broadcast

        else:  # Canonical PSO: true deterministic global best, always available
            best_j = np.argmax(p_fitness)
            if p_fitness[best_j] > g_best_fitness:
                g_best_fitness = p_fitness[best_j]
                g_best_global = p_best[best_j].copy()
            g_i = np.tile(g_best_global, (N_AGENTS, 1))
            social_gate = np.ones(N_AGENTS)

        # ---- Success check (pooled best local estimate across swarm) ----
        dist_to_fissure = np.linalg.norm(g_i - X_F, axis=-1)
        min_dist = np.min(dist_to_fissure)
        dist_trace.append(min_dist)
        if success_step is None and min_dist <= SUCCESS_THRESHOLD:
            success_step = t

        # ---- Kinematic update ----
        r1 = rng_local.uniform(0, 1, size=(N_AGENTS, 1))
        r2 = rng_local.uniform(0, 1, size=(N_AGENTS, 1))

        cognitive = C1 * r1 * (p_best - x)
        social = social_gate[:, None] * C2 * r2 * (g_i - x)

        if architecture == "Canonical PSO":
            # No physical drag modeled: idealized point-particle kinematics.
            v = W_INERTIA * v + cognitive + social
        else:
            v_fluid = fluid_velocity(x, rng_local)
            f_d = drag_force(v, v_fluid)
            v = W_INERTIA * v + cognitive + social + (f_d / M_AGENT) * DT

        speed = np.linalg.norm(v, axis=-1, keepdims=True)
        scale = np.minimum(1.0, V_CLAMP / np.where(speed < 1e-12, 1.0, speed))
        v = v * scale

        x = x + v * DT
        x[:, 1] = np.clip(x[:, 1], -R, R)
        x[:, 0] = np.clip(x[:, 0], 0.0, L)

    total_dose = np.sum(lam)
    return success_step, total_dose, dist_trace


def run_campaign():
    results = {}
    all_traces = {}      # ALL distance traces per (profile, arch), for computing the
                          # population-median representative curve used in Fig. 1

    for profile_name, (d_bg, k) in PROFILES.items():
        for arch in ARCHITECTURES:
            successes = []
            conv_times = []
            doses = []
            traces_this_condition = []
            for run_idx in range(N_MONTE_CARLO):
                rng_local = np.random.default_rng(rng.integers(0, 2**32 - 1))
                success_step, total_dose, dist_trace = run_single_trial(arch, d_bg, k, rng_local)
                doses.append(total_dose)
                if success_step is not None:
                    successes.append(1)
                    conv_times.append(success_step)
                else:
                    successes.append(0)
                traces_this_condition.append(dist_trace)

            all_traces[(profile_name, arch)] = np.array(traces_this_condition)

            n_success = int(np.sum(successes))
            success_rate = 100.0 * n_success / N_MONTE_CARLO
            mean_conv = float(np.mean(conv_times)) if conv_times else None
            median_conv = float(np.median(conv_times)) if conv_times else None
            std_conv = float(np.std(conv_times)) if len(conv_times) > 1 else None
            mean_dose = float(np.mean(doses))
            std_dose = float(np.std(doses))

            # Wilson score 95% CI for the success proportion
            p_hat = n_success / N_MONTE_CARLO
            z = 1.96
            denom = 1 + z**2 / N_MONTE_CARLO
            centre = p_hat + z**2 / (2 * N_MONTE_CARLO)
            adj = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * N_MONTE_CARLO)) / N_MONTE_CARLO)
            ci_low = (centre - adj) / denom * 100
            ci_high = (centre + adj) / denom * 100

            results[(profile_name, arch)] = {
                "success_rate_pct": success_rate,
                "n_success": n_success,
                "n_total": N_MONTE_CARLO,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_conv_steps": mean_conv,
                "median_conv_steps": median_conv,
                "std_conv_steps": std_conv,
                "mean_cum_dose_gy": mean_dose,
                "std_cum_dose_gy": std_dose,
            }
            print(f"{profile_name:10s} | {arch:22s} | success={success_rate:5.1f}% "
                  f"(n={n_success}/{N_MONTE_CARLO}, 95% CI [{ci_low:.1f}, {ci_high:.1f}]) | "
                  f"conv={mean_conv if mean_conv else float('nan'):6.1f} steps | "
                  f"dose={mean_dose:7.1f} Gy")

    return results, all_traces


def generate_fig2_trajectories():
    """Generate full agent trajectories for the Extreme / Decentralized M-PSO
    condition, for the trajectory-geometry figure (Fig. 2)."""
    rng_local = np.random.default_rng(42)
    d_bg, k = PROFILES["Extreme"]
    N = N_AGENTS
    x = rng_local.uniform([0.0, -R], [L, R], size=(N, 2))
    v = np.zeros((N, 2))
    p_best = x.copy()
    p_fitness = np.full(N, -np.inf)
    lam = np.zeros(N)
    latched = np.zeros(N, dtype=bool)
    history = [x.copy()]

    for t in range(T_MAX):
        d_now = dose_rate(x, d_bg, k)
        lam += d_now * DT
        latched[(lam >= LAMBDA_FAIL) & (~latched)] = True

        s_true = true_signal(x)
        noise = rng_local.normal(0.0, SIGMA_GAMMA * np.sqrt(np.maximum(d_now, 0.0)))
        s_measured = s_true + noise
        improved = s_measured > p_fitness
        p_best[improved] = x[improved]
        p_fitness[improved] = s_measured[improved]

        p_drop = 1.0 - np.exp(-BETA * d_now)
        chi = (rng_local.uniform(size=N) > p_drop).astype(float)
        chi[latched] = 0.0

        diffs = x[:, None, :] - x[None, :, :]
        dist_mat = np.linalg.norm(diffs, axis=-1)
        in_range = dist_mat <= R_COMM
        both_operational = np.outer(chi, chi) > 0
        edge = in_range & both_operational
        np.fill_diagonal(edge, True)

        g_i = np.empty((N, 2))
        for i in range(N):
            neighbors = np.where(edge[i])[0]
            best_j = neighbors[np.argmax(p_fitness[neighbors])]
            g_i[i] = p_best[best_j]

        r1 = rng_local.uniform(0, 1, size=(N, 1))
        r2 = rng_local.uniform(0, 1, size=(N, 1))
        cognitive = C1 * r1 * (p_best - x)
        social = chi[:, None] * C2 * r2 * (g_i - x)
        v_fluid = fluid_velocity(x, rng_local)
        f_d = drag_force(v, v_fluid)
        v = W_INERTIA * v + cognitive + social + (f_d / M_AGENT) * DT
        speed = np.linalg.norm(v, axis=-1, keepdims=True)
        scale = np.minimum(1.0, V_CLAMP / np.where(speed < 1e-12, 1.0, speed))
        v = v * scale
        x = x + v * DT
        x[:, 1] = np.clip(x[:, 1], -R, R)
        x[:, 0] = np.clip(x[:, 0], 0.0, L)
        history.append(x.copy())

    return np.array(history)  # shape (T_MAX+1, N, 2)


if __name__ == "__main__":
    t0 = time.time()
    results, all_traces = run_campaign()
    trajectories = generate_fig2_trajectories()
    elapsed = time.time() - t0
    print(f"\nTotal Monte Carlo campaign runtime: {elapsed:.1f} s "
          f"({N_MONTE_CARLO * len(PROFILES) * len(ARCHITECTURES)} total trials)")

    # ---- Write Table II ----
    with open("results_table2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Profile", "Architecture", "SuccessRate(%)", "n_success", "n_total",
                          "CI95_low", "CI95_high", "MeanConvSteps", "MedianConvSteps",
                          "StdConvSteps", "MeanCumDose(Gy)", "StdCumDose(Gy)"])
        for profile_name in PROFILES:
            for arch in ARCHITECTURES:
                r = results[(profile_name, arch)]
                writer.writerow([profile_name, arch, f"{r['success_rate_pct']:.1f}",
                                  r['n_success'], r['n_total'],
                                  f"{r['ci95_low']:.1f}", f"{r['ci95_high']:.1f}",
                                  f"{r['mean_conv_steps']:.1f}" if r['mean_conv_steps'] is not None else "N/A",
                                  f"{r['median_conv_steps']:.1f}" if r['median_conv_steps'] is not None else "N/A",
                                  f"{r['std_conv_steps']:.1f}" if r['std_conv_steps'] is not None else "N/A",
                                  f"{r['mean_cum_dose_gy']:.1f}", f"{r['std_cum_dose_gy']:.1f}"])

    json_results = {f"{p}|{a}": v for (p, a), v in results.items()}
    with open("results_table2.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # ---- Write Fig. 1 convergence traces (population median across all 500 runs) ----
    with open("fig1_convergence_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "Control", "Moderate", "Extreme"])
        medians = {p: np.median(all_traces[(p, "Decentralized M-PSO")], axis=0) for p in PROFILES}
        for t in range(T_MAX):
            row = [t] + [medians[p][t] for p in PROFILES]
            writer.writerow(row)

    # ---- Write Fig. 2 trajectories ----
    with open("fig2_trajectory_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "agent_id", "x", "y"])
        for t in range(trajectories.shape[0]):
            for agent_id in range(trajectories.shape[1]):
                writer.writerow([t, agent_id, trajectories[t, agent_id, 0], trajectories[t, agent_id, 1]])

    print("\nWrote: results_table2.csv, results_table2.json, "
          "fig1_convergence_data.csv, fig2_trajectory_data.csv")
