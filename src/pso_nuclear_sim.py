#!/usr/bin/env python3
"""
pso_nuclear_sim.py -- reconstructed to exactly match the accepted IECON 2026
manuscript's formalized model (Eqs. 2-11, Algorithm 1, Table I).
"""
import numpy as np
import json, csv, time

# ---- Table I ----
L = 10.0
R = 1.0
X_F = np.array([8.0, 0.9])
M_AGENT = 0.05
RHO = 1000.0
CD = 0.47
AREA = 1.26e-3
V_MAX = 1.5
SIGMA_TURB = 0.1

N_AGENTS_DEFAULT = 25
DT = 0.1
T_MAX = 150
W_INERTIA = 0.6
C1 = 1.2
C2 = 1.2
R_COMM_DEFAULT = 2.0
V_CLAMP = 3.0

BETA = 0.15
SIGMA_GAMMA = 0.08
LAMBDA_FAIL = 300.0

SUCCESS_THRESHOLD = 0.3
N_MONTE_CARLO = 500

PROFILES = {
    "Control":  (0.0, 0.0),
    "Moderate": (0.5, 10.0),
    "Extreme":  (2.0, 100.0),
}
ARCHITECTURES = ["Decentralized M-PSO", "Centralized Swarm", "Canonical PSO"]

rng = np.random.default_rng(20260722)


def dose_rate(pos, d_bg, k):
    dist_sq = np.sum((pos - X_F) ** 2, axis=-1)
    return d_bg + k / (dist_sq + 1.0)


def true_signal(pos):
    dist_sq = np.sum((pos - X_F) ** 2, axis=-1)
    return 1.0 / (dist_sq + 1.0)


def fluid_velocity(pos, rng_local):
    y = pos[:, 1]
    frac = np.clip(1.0 - np.abs(y) / R, 0.0, None)
    vx = V_MAX * frac ** (1.0 / 7.0)
    v_turb = rng_local.normal(0.0, SIGMA_TURB, size=pos.shape)
    v_fluid = np.zeros_like(pos)
    v_fluid[:, 0] = vx
    return v_fluid + v_turb


def drag_force(v_agent, v_fluid):
    v_rel = v_agent - v_fluid
    speed = np.linalg.norm(v_rel, axis=-1, keepdims=True)
    speed_safe = np.where(speed < 1e-9, 1.0, speed)
    u_hat = v_rel / speed_safe
    mag = 0.5 * RHO * CD * AREA * speed ** 2
    f_d = -mag * u_hat
    f_d[speed.flatten() < 1e-9] = 0.0
    return f_d


def run_single_trial(architecture, d_bg, k, rng_local, n_agents=N_AGENTS_DEFAULT, r_comm=R_COMM_DEFAULT):
    N = n_agents
    x = rng_local.uniform([0.0, -R], [L, R], size=(N, 2))
    v = np.zeros((N, 2))
    p_best = x.copy()
    p_fitness = np.full(N, -np.inf)
    lam = np.zeros(N)
    latched = np.zeros(N, dtype=bool)

    g_best_global = x[0].copy()
    g_best_fitness = -np.inf
    leader_idx = 0

    success_step = None
    dist_trace = []

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

        if architecture in ("Decentralized M-PSO", "Centralized Swarm"):
            p_drop = 1.0 - np.exp(-BETA * d_now)
            chi = (rng_local.uniform(size=N) > p_drop).astype(float)
            chi[latched] = 0.0
        else:
            chi = np.ones(N)

        if architecture == "Decentralized M-PSO":
            diffs = x[:, None, :] - x[None, :, :]
            dist_mat = np.linalg.norm(diffs, axis=-1)
            in_range = dist_mat <= r_comm
            both_op = np.outer(chi, chi) > 0
            edge = in_range & both_op
            np.fill_diagonal(edge, True)
            g_i = np.empty((N, 2))
            for i in range(N):
                neighbors = np.where(edge[i])[0]
                best_j = neighbors[np.argmax(p_fitness[neighbors])]
                g_i[i] = p_best[best_j]
            social_gate = chi

        elif architecture == "Centralized Swarm":
            dist_to_leader = np.linalg.norm(x - x[leader_idx], axis=-1)
            can_report = (dist_to_leader <= r_comm) & (chi > 0)
            can_report[leader_idx] = True
            if chi[leader_idx] > 0:
                reporting = np.where(can_report)[0]
                best_j = reporting[np.argmax(p_fitness[reporting])]
                if p_fitness[best_j] > g_best_fitness:
                    g_best_fitness = p_fitness[best_j]
                    g_best_global = p_best[best_j].copy()
            g_i = np.tile(g_best_global, (N, 1))
            social_gate = np.ones(N)

        else:
            best_j = np.argmax(p_fitness)
            if p_fitness[best_j] > g_best_fitness:
                g_best_fitness = p_fitness[best_j]
                g_best_global = p_best[best_j].copy()
            g_i = np.tile(g_best_global, (N, 1))
            social_gate = np.ones(N)

        dist_to_fissure = np.linalg.norm(g_i - X_F, axis=-1)
        min_dist = np.min(dist_to_fissure)
        dist_trace.append(min_dist)
        if success_step is None and min_dist <= SUCCESS_THRESHOLD:
            success_step = t

        r1 = rng_local.uniform(0, 1, size=(N, 1))
        r2 = rng_local.uniform(0, 1, size=(N, 1))
        cognitive = C1 * r1 * (p_best - x)
        social = social_gate[:, None] * C2 * r2 * (g_i - x)

        if architecture == "Canonical PSO":
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


def wilson_ci(n_success, n_total):
    p_hat = n_success / n_total
    z = 1.96
    denom = 1 + z**2 / n_total
    centre = p_hat + z**2 / (2 * n_total)
    adj = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total)
    return (centre - adj) / denom * 100, (centre + adj) / denom * 100


def run_condition(architecture, d_bg, k, n_trials=N_MONTE_CARLO, n_agents=N_AGENTS_DEFAULT, r_comm=R_COMM_DEFAULT):
    successes, conv_times, doses = [], [], []
    for _ in range(n_trials):
        rng_local = np.random.default_rng(rng.integers(0, 2**32 - 1))
        s, d, _ = run_single_trial(architecture, d_bg, k, rng_local, n_agents=n_agents, r_comm=r_comm)
        doses.append(d)
        if s is not None:
            successes.append(1)
            conv_times.append(s)
        else:
            successes.append(0)
    n_success = int(np.sum(successes))
    success_rate = 100.0 * n_success / n_trials
    mean_conv = float(np.mean(conv_times)) if conv_times else None
    mean_dose = float(np.mean(doses))
    ci_low, ci_high = wilson_ci(n_success, n_trials)
    return {
        "success_rate_pct": success_rate, "n_success": n_success, "n_total": n_trials,
        "ci95_low": ci_low, "ci95_high": ci_high,
        "mean_conv_steps": mean_conv, "mean_cum_dose_gy": mean_dose,
    }


if __name__ == "__main__":
    t0 = time.time()

    print("=== VERIFICATION: reproducing published Table II ===")
    results = {}
    for profile_name, (d_bg, k) in PROFILES.items():
        for arch in ARCHITECTURES:
            r = run_condition(arch, d_bg, k)
            results[(profile_name, arch)] = r
            print(f"{profile_name:10s} | {arch:22s} | success={r['success_rate_pct']:5.1f}% "
                  f"(n={r['n_success']}/{r['n_total']}) | "
                  f"conv={r['mean_conv_steps'] if r['mean_conv_steps'] else float('nan'):6.1f} | "
                  f"dose={r['mean_cum_dose_gy']:8.1f} Gy")
    print(f"\nVerification runtime: {time.time()-t0:.1f}s")
