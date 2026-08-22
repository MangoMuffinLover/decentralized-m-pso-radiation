#!/usr/bin/env python3
"""sensitivity_sweep.py -- real Monte Carlo sensitivity sweeps over swarm
size N and communication range r_comm, using the verified pso_nuclear_sim.py
model. Addresses Reviewer 1 (Q1, Q2, Q3) and Reviewer 2 (Q5)."""

import numpy as np
import csv, time
from pso_nuclear_sim import run_condition, PROFILES

t0 = time.time()

# ---------------- Sweep 1: Swarm size N (Extreme profile) ----------------
N_VALUES = [5, 10, 15, 20, 25, 30, 40, 50]
d_bg, k = PROFILES["Extreme"]

print("=== SWEEP 1: Swarm size N (Decentralized M-PSO, Extreme profile) ===")
n_sweep_results = []
for N in N_VALUES:
    r = run_condition("Decentralized M-PSO", d_bg, k, n_trials=500, n_agents=N)
    n_sweep_results.append((N, r))
    print(f"N={N:3d} | success={r['success_rate_pct']:5.1f}% (n={r['n_success']}/{r['n_total']}, "
          f"95% CI [{r['ci95_low']:.1f},{r['ci95_high']:.1f}]) | "
          f"conv={r['mean_conv_steps'] if r['mean_conv_steps'] else float('nan'):6.1f} | "
          f"dose={r['mean_cum_dose_gy']:9.1f} Gy")

with open("sweep_N.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "SuccessRate(%)", "n_success", "n_total", "CI95_low", "CI95_high", "MeanConvSteps", "MeanCumDose(Gy)"])
    for N, r in n_sweep_results:
        writer.writerow([N, f"{r['success_rate_pct']:.1f}", r['n_success'], r['n_total'],
                          f"{r['ci95_low']:.1f}", f"{r['ci95_high']:.1f}",
                          f"{r['mean_conv_steps']:.1f}" if r['mean_conv_steps'] is not None else "N/A",
                          f"{r['mean_cum_dose_gy']:.1f}"])

# ---------------- Sweep 2: Communication range r_comm (Extreme profile) ----------------
RCOMM_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]
print("\n=== SWEEP 2: Communication range r_comm (Extreme profile) ===")
rcomm_sweep_results = []
for r_comm in RCOMM_VALUES:
    for arch in ["Decentralized M-PSO", "Centralized Swarm"]:
        r = run_condition(arch, d_bg, k, n_trials=500, r_comm=r_comm)
        rcomm_sweep_results.append((r_comm, arch, r))
        print(f"r_comm={r_comm:4.1f}m | {arch:22s} | success={r['success_rate_pct']:5.1f}% "
              f"(n={r['n_success']}/{r['n_total']}, 95% CI [{r['ci95_low']:.1f},{r['ci95_high']:.1f}]) | "
              f"conv={r['mean_conv_steps'] if r['mean_conv_steps'] else float('nan'):6.1f} | "
              f"dose={r['mean_cum_dose_gy']:9.1f} Gy")

with open("sweep_rcomm.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["r_comm(m)", "Architecture", "SuccessRate(%)", "n_success", "n_total", "CI95_low", "CI95_high", "MeanConvSteps", "MeanCumDose(Gy)"])
    for r_comm, arch, r in rcomm_sweep_results:
        writer.writerow([r_comm, arch, f"{r['success_rate_pct']:.1f}", r['n_success'], r['n_total'],
                          f"{r['ci95_low']:.1f}", f"{r['ci95_high']:.1f}",
                          f"{r['mean_conv_steps']:.1f}" if r['mean_conv_steps'] is not None else "N/A",
                          f"{r['mean_cum_dose_gy']:.1f}"])

print(f"\nTotal sensitivity sweep runtime: {time.time()-t0:.1f}s")
print("Wrote: sweep_N.csv, sweep_rcomm.csv")
