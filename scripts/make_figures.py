#!/usr/bin/env python3
"""
scripts/make_figures.py
==========================
Publication-ready figure generation, self-contained: every plot is driven
directly by the simulator (no missing intermediate CSV dependency, unlike
the original single-file version of this script, which read from a
``fig2a_convergence_corrected.csv`` that was never checked into the
archive).

Produces:
    figures/pso_convergence_rates.pdf   -- Fig. 2(a): population-median
                                            convergence distance across the
                                            three regimes.
    figures/pso_swarm_trajectories.pdf  -- Fig. 2(b): agent pathways under
                                            Extreme, one representative run.
    figures/dose_accumulation.pdf       -- supplementary: cumulative dose
                                            growth vs. swarm size (Table IV).

Terminology note (m5): the Moderate-profile curve is labeled
"$D(\\vec{x})$-driven degradation", matching the manuscript's own dose-rate
notation (Eq. 7) -- *not* "TID" (Total Ionizing Dose), which refers to
cumulative dose Lambda_i(t) (Eq. 10), a different quantity. See
``pso_nuclear_sim.runner`` module docstring / README for the corrected-vs-
original comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pso_nuclear_sim import DEFAULT_CONFIG, PROFILES, run_condition
from pso_nuclear_sim.runner import run_single_trial, trial_rng

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def make_convergence_figure(n_trials: int = 500) -> None:
    """Fig. 2(a): population-median distance-to-fissure across timesteps,
    for each of the three radiation profiles, computed directly from
    ``n_trials`` fresh Monte Carlo trials rather than a pre-baked CSV."""
    print(f"Generating convergence curves from {n_trials} trials/profile (Decentralized M-PSO)...")
    curves = {}
    for profile_name, profile in PROFILES.items():
        traces = []
        for trial_idx in range(n_trials):
            rng = trial_rng(f"figure2a|{profile_name}", "trial", trial_idx,
                             master_entropy=DEFAULT_CONFIG.master_entropy)
            result = run_single_trial(DEFAULT_CONFIG, "Decentralized M-PSO", profile, rng)
            traces.append(result.min_dist_trace)
        curves[profile_name] = np.median(np.array(traces), axis=0)
        print(f"  {profile_name}: final median distance = {curves[profile_name][-1]:.3f} m")

    steps = np.arange(DEFAULT_CONFIG.algorithm.max_steps)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(steps, curves["Control"], color="#1f77b4", linewidth=2.0,
            label="Control Profile (0% Radiation background)")
    ax.plot(steps, curves["Moderate"], color="#ff7f0e", linewidth=2.0,
            label=r"Moderate Radiation Field ($D(\vec{x})$-driven degradation)")
    ax.plot(steps, curves["Extreme"], color="#d62728", linewidth=2.0,
            label="Extreme Stochastic Gamma Field")
    ax.set_xlabel("Simulation Steps ($t$)")
    ax.set_ylabel(r"Median Distance to Wall Fissure ($\|\vec{g}-\vec{x}_f\|$ meters)")
    ax.set_title("Swarm Convergence Rate in PWR Pipe Environment\n"
                 f"(Decentralized M-PSO, population median across {n_trials} Monte Carlo runs)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "pso_convergence_rates.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_trajectory_figure(seed_trial_idx: int = 0) -> None:
    """Fig. 2(b): agent pathways under Extreme with full turbulent shear,
    one representative run. The caption (m2) is explicit that this single
    run illustrates dispersion *typical of* the full 500-trial campaign
    (Table II), not an aggregate statistic in its own right."""
    print("Generating representative Extreme-profile trajectory plot...")
    rng = trial_rng("figure2b|representative", "trial", seed_trial_idx,
                     master_entropy=DEFAULT_CONFIG.master_entropy)
    result = run_single_trial(DEFAULT_CONFIG, "Decentralized M-PSO", PROFILES["Extreme"], rng,
                               record_trajectory=True)
    trajectory = result.position_trace  # (max_steps, n_agents, 2)

    pipe = DEFAULT_CONFIG.pipe
    fig, ax = plt.subplots(figsize=(7, 4.2))
    n_agents = trajectory.shape[1]
    for i in range(n_agents):
        ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], alpha=0.35, linewidth=0.8)
    ax.axhline(pipe.radius, color="black", linewidth=2.5, label="Pipe Boundary Wall")
    ax.axhline(-pipe.radius, color="black", linewidth=2.5)
    ax.scatter([pipe.x_f[0]], [pipe.x_f[1]], marker="*", color="crimson", s=250, zorder=5,
               label="Target Micro-Fissure")
    ax.set_xlabel("Axial Dimension ($x$ meters)")
    ax.set_ylabel("Radial Dimension ($y$ meters)")
    ax.set_title("Extreme Stochastic Gamma Field: Path Deviation Under Turbulent Drag\n"
                 f"(Decentralized M-PSO, all {n_agents} agents, single representative run)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, pipe.length)
    ax.set_ylim(-pipe.radius * 1.1, pipe.radius * 1.1)
    fig.tight_layout()

    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "pso_swarm_trajectories.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print("NOTE (m2): caption in main.tex already states this run illustrates dispersion")
    print("typical of the full 500-trial Extreme-profile campaign, not an aggregate statistic.")


def make_dose_accumulation_figure(n_trials: int = 500) -> None:
    """Supplementary figure: mean cumulative swarm dose vs. swarm size N,
    directly reproducing the trend underlying Table IV's Cum. Dose column."""
    print(f"Generating dose-accumulation-vs-N figure from {n_trials} trials/N...")
    n_values = [5, 10, 15, 20, 25, 30, 40, 50]
    doses, ci_lo, ci_hi = [], [], []
    for n_agents in n_values:
        r = run_condition(f"figS|dose_vs_N|N={n_agents}", DEFAULT_CONFIG, "Decentralized M-PSO",
                           PROFILES["Extreme"], n_agents=n_agents, n_trials=n_trials)
        doses.append(r.mean_cum_dose_gy)
        ci_lo.append(r.mean_cum_dose_gy - r.std_cum_dose_gy)
        ci_hi.append(r.mean_cum_dose_gy + r.std_cum_dose_gy)
        print(f"  N={n_agents:3d}: mean dose = {r.mean_cum_dose_gy:9.1f} Gy")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_values, doses, marker="o", color="#2ca02c", linewidth=2.0)
    ax.fill_between(n_values, ci_lo, ci_hi, color="#2ca02c", alpha=0.15, label="±1 std. dev.")
    ax.set_xlabel("Swarm Size $N$")
    ax.set_ylabel("Mean Cumulative Swarm Dose (Gy)")
    ax.set_title("Cumulative Radiological Exposure vs. Swarm Size\n(Extreme Profile)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "dose_accumulation.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate publication-ready figures.")
    parser.add_argument("--trials", type=int, default=500,
                         help="Trials per profile/condition (default: 500; use a small "
                              "value like 20 for a fast smoke test of the plotting code).")
    parser.add_argument("--figure", choices=["convergence", "trajectory", "dose", "all"],
                         default="all")
    args = parser.parse_args()

    if args.figure in ("convergence", "all"):
        make_convergence_figure(args.trials)
    if args.figure in ("trajectory", "all"):
        make_trajectory_figure()
    if args.figure in ("dose", "all"):
        make_dose_accumulation_figure(args.trials)


if __name__ == "__main__":
    main()
