#!/usr/bin/env python3
"""
scripts/run_benchmarks.py
============================
CLI entry point that reproduces every Monte Carlo campaign in the paper
using the modular ``pso_nuclear_sim`` package, and writes structured CSV
results identical in format to the tables in the manuscript.

Usage
-----
    python scripts/run_benchmarks.py --campaign all
    python scripts/run_benchmarks.py --campaign baseline --trials 500
    python scripts/run_benchmarks.py --campaign ablation --workers 4
    python scripts/run_benchmarks.py --campaign all --trials 20   # fast smoke test

Campaigns
---------
baseline      Table II: Decentralized M-PSO vs. both centralized variants
              vs. Canonical PSO, across Control/Moderate/Extreme.
              -> results/table_baseline.csv
reelection    Re-election-specific view across all three profiles
              (a convenience subset of `baseline`, kept separate for
              direct comparison to the paper's original table_reelection.csv).
              -> results/table_reelection.csv
ablation      Table III: the full seven-cell ablation (Extreme profile).
              -> results/table_ablation.csv
sweep-n       Table IV: swarm-size sensitivity (Decentralized M-PSO,
              Extreme profile).
              -> results/table_sweep_n.csv
sweep-rcomm   Table V: communication-range sensitivity (Extreme profile).
              -> results/table_sweep_rcomm.csv
all           All of the above, in sequence.

.. note:: Known corrected result -- ablation cell D

    The original single-file simulator had a bug (see
    ``pso_nuclear_sim.runner`` module docstring) that silently skipped
    drag physics for ablation cell D ("Canonical PSO + drag"), regardless
    of the ``use_drag`` flag. This script constructs cell D with drag
    genuinely enabled and every ordinary Canonical PSO baseline row with
    drag explicitly disabled (matching the paper's own definition of
    Canonical PSO). Cell D's success rate here is therefore expected to
    read higher than the 26.6% originally printed in the manuscript --
    re-run ``--campaign ablation`` and see README "Known Issues Found
    During This Refactor" before citing cell D from this repository.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pso_nuclear_sim import (  # noqa: E402
    AblationFlags,
    DEFAULT_CONFIG,
    PROFILES,
    run_condition,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

ARCHS_BASELINE = ["Decentralized M-PSO", "Centralized Swarm", "Canonical PSO"]


def _ablation_for(architecture: str) -> AblationFlags:
    """Canonical PSO is defined (Sec. III) as drag-free; every other
    architecture uses the full-model default. This is the explicit,
    caller-side fix for the drag/architecture coupling bug -- see the
    runner module docstring."""
    if architecture == "Canonical PSO":
        return AblationFlags(use_drag=False)
    return AblationFlags()


def run_baseline(n_trials: int, workers: int | None) -> Dict[Tuple[str, str], object]:
    print("=== Table II: baseline comparison ===")
    results = {}
    for profile_name, profile in PROFILES.items():
        for arch in ARCHS_BASELINE:
            cid = f"table2|{profile_name}|{arch}"
            r = run_condition(
                cid, DEFAULT_CONFIG, arch, profile,
                ablation=_ablation_for(arch), n_trials=n_trials, max_workers=workers,
            )
            results[(profile_name, arch)] = r
            print(f"{profile_name:10s} | {arch:22s} | success={r.success_rate_pct:5.1f}% "
                  f"(n={r.n_success}/{r.n_total}) | 95%CI=[{r.ci95_low:.1f},{r.ci95_high:.1f}] | "
                  f"non-trivial={r.nontrivial_success_rate_pct:.1f}%")

        # Re-election variant, Table II's 4th row per profile.
        cid = f"reelect|{profile_name}"
        r = run_condition(
            cid, DEFAULT_CONFIG, "Centralized Swarm (re-election)", profile,
            n_trials=n_trials, max_workers=workers,
        )
        results[(profile_name, "Centralized Swarm (re-election)")] = r
        print(f"{profile_name:10s} | {'Centralized (re-election)':22s} | success={r.success_rate_pct:5.1f}% "
              f"(n={r.n_success}/{r.n_total}) | 95%CI=[{r.ci95_low:.1f},{r.ci95_high:.1f}]")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "table_baseline.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Profile", "Architecture", "Success(%)", "n_success", "n_total",
                          "CI95_low", "CI95_high", "MeanConv", "MedianConv", "CumDose(Gy)",
                          "TrivialSpawn(%)", "NonTrivialSuccess(%)"])
        for profile_name in PROFILES:
            for arch in ARCHS_BASELINE + ["Centralized Swarm (re-election)"]:
                r = results[(profile_name, arch)]
                writer.writerow([
                    profile_name, arch, f"{r.success_rate_pct:.1f}", r.n_success, r.n_total,
                    f"{r.ci95_low:.1f}", f"{r.ci95_high:.1f}",
                    f"{r.mean_conv_steps:.1f}" if r.mean_conv_steps is not None else "N/A",
                    f"{r.median_conv_steps:.1f}" if r.median_conv_steps is not None else "N/A",
                    f"{r.mean_cum_dose_gy:.1f}", f"{r.trivial_spawn_rate_pct:.1f}",
                    f"{r.nontrivial_success_rate_pct:.1f}",
                ])
    print(f"Wrote {RESULTS_DIR / 'table_baseline.csv'}")
    return results


def run_reelection(n_trials: int, workers: int | None) -> None:
    print("=== Re-election, all three profiles ===")
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "table_reelection.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Profile", "Success(%)", "n_success", "n_total", "CI95_low", "CI95_high",
                          "CumDose(Gy)", "NonTrivialSuccess(%)"])
        for profile_name, profile in PROFILES.items():
            cid = f"reelect|{profile_name}"
            r = run_condition(
                cid, DEFAULT_CONFIG, "Centralized Swarm (re-election)", profile,
                n_trials=n_trials, max_workers=workers,
            )
            print(f"{profile_name:10s} | success={r.success_rate_pct:5.1f}% (n={r.n_success}/{r.n_total}) | "
                  f"95%CI=[{r.ci95_low:.1f},{r.ci95_high:.1f}] | dose={r.mean_cum_dose_gy:.1f} Gy")
            writer.writerow([profile_name, f"{r.success_rate_pct:.1f}", r.n_success, r.n_total,
                              f"{r.ci95_low:.1f}", f"{r.ci95_high:.1f}", f"{r.mean_cum_dose_gy:.1f}",
                              f"{r.nontrivial_success_rate_pct:.1f}"])
    print(f"Wrote {RESULTS_DIR / 'table_reelection.csv'}")


def run_ablation(n_trials: int, workers: int | None) -> None:
    print("=== Table III: seven-cell ablation (Extreme profile) ===")
    profile = PROFILES["Extreme"]
    cells: List[Tuple[str, str, AblationFlags]] = [
        ("A_full_model", "Decentralized M-PSO", AblationFlags()),
        ("B_no_drag", "Decentralized M-PSO", AblationFlags(use_drag=False)),
        ("C_no_comms_degrad", "Decentralized M-PSO", AblationFlags(use_comms_degradation=False)),
        ("D_canonical_plus_drag", "Canonical PSO", AblationFlags(use_drag=True)),
        ("E_no_explicit_gate", "Decentralized M-PSO", AblationFlags(use_explicit_chi_gate=False)),
        ("F_sensor_noise_only", "Decentralized M-PSO", AblationFlags(use_packet_loss=False)),
        ("G_packet_loss_only", "Decentralized M-PSO", AblationFlags(use_sensor_noise=False)),
    ]
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "table_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cell", "Success(%)", "n_success", "n_total", "CI95_low", "CI95_high",
                          "CumDose(Gy)", "NonTrivialSuccess(%)"])
        for label, arch, ablation in cells:
            cid = f"ablation|{label}"
            r = run_condition(cid, DEFAULT_CONFIG, arch, profile, ablation=ablation,
                               n_trials=n_trials, max_workers=workers)
            flag = "  <-- was a silent no-op before this refactor's bugfix" if label == "D_canonical_plus_drag" else ""
            print(f"{label:24s} | success={r.success_rate_pct:5.1f}% (n={r.n_success}/{r.n_total}) | "
                  f"95%CI=[{r.ci95_low:.1f},{r.ci95_high:.1f}] | dose={r.mean_cum_dose_gy:.1f} Gy{flag}")
            writer.writerow([label, f"{r.success_rate_pct:.1f}", r.n_success, r.n_total,
                              f"{r.ci95_low:.1f}", f"{r.ci95_high:.1f}", f"{r.mean_cum_dose_gy:.1f}",
                              f"{r.nontrivial_success_rate_pct:.1f}"])
    print(f"Wrote {RESULTS_DIR / 'table_ablation.csv'}")


def run_sweep_n(n_trials: int, workers: int | None) -> None:
    print("=== Table IV: swarm-size sensitivity (Extreme profile) ===")
    profile = PROFILES["Extreme"]
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "table_sweep_n.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "Success(%)", "n_success", "n_total", "CI95_low", "CI95_high",
                          "MeanConv", "CumDose(Gy)"])
        for n_agents in [5, 10, 15, 20, 25, 30, 40, 50]:
            cid = f"table3|N={n_agents}"
            r = run_condition(cid, DEFAULT_CONFIG, "Decentralized M-PSO", profile,
                               n_agents=n_agents, n_trials=n_trials, max_workers=workers)
            print(f"N={n_agents:3d} | success={r.success_rate_pct:5.1f}% (n={r.n_success}/{r.n_total}) | "
                  f"95%CI=[{r.ci95_low:.1f},{r.ci95_high:.1f}] | dose={r.mean_cum_dose_gy:9.1f} Gy")
            writer.writerow([n_agents, f"{r.success_rate_pct:.1f}", r.n_success, r.n_total,
                              f"{r.ci95_low:.1f}", f"{r.ci95_high:.1f}",
                              f"{r.mean_conv_steps:.1f}" if r.mean_conv_steps is not None else "N/A",
                              f"{r.mean_cum_dose_gy:.1f}"])
    print(f"Wrote {RESULTS_DIR / 'table_sweep_n.csv'}")


def run_sweep_rcomm(n_trials: int, workers: int | None) -> None:
    print("=== Table V: communication-range sensitivity (Extreme profile) ===")
    profile = PROFILES["Extreme"]
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "table_sweep_rcomm.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_comm(m)", "Architecture", "Success(%)", "n_success", "n_total",
                          "CI95_low", "CI95_high"])
        for r_comm in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for arch in ["Decentralized M-PSO", "Centralized Swarm"]:
                cid = f"table4|r_comm={r_comm}|{arch}"
                r = run_condition(cid, DEFAULT_CONFIG, arch, profile,
                                   r_comm=r_comm, n_trials=n_trials, max_workers=workers)
                print(f"r_comm={r_comm:4.1f}m | {arch:22s} | success={r.success_rate_pct:5.1f}% "
                      f"(n={r.n_success}/{r.n_total}) | 95%CI=[{r.ci95_low:.1f},{r.ci95_high:.1f}]")
                writer.writerow([r_comm, arch, f"{r.success_rate_pct:.1f}", r.n_success, r.n_total,
                                  f"{r.ci95_low:.1f}", f"{r.ci95_high:.1f}"])
    print(f"Wrote {RESULTS_DIR / 'table_sweep_rcomm.csv'}")


CAMPAIGNS = {
    "baseline": run_baseline,
    "reelection": run_reelection,
    "ablation": run_ablation,
    "sweep-n": run_sweep_n,
    "sweep-rcomm": run_sweep_rcomm,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the paper's Monte Carlo campaigns.")
    parser.add_argument("--campaign", choices=list(CAMPAIGNS) + ["all"], default="all",
                         help="Which campaign to run (default: all).")
    parser.add_argument("--trials", type=int, default=500,
                         help="Trials per condition (default: 500, matching the paper; "
                              "use a small value like 20 for a fast smoke test).")
    parser.add_argument("--workers", type=int, default=None,
                         help="Process-pool workers (default: all available cores). "
                              "Pass 1 to force single-threaded, fully-deterministic execution order.")
    args = parser.parse_args()

    t0 = time.time()
    campaigns = list(CAMPAIGNS) if args.campaign == "all" else [args.campaign]
    for name in campaigns:
        CAMPAIGNS[name](args.trials, args.workers)
        print()
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
