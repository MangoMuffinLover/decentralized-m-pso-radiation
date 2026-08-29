#!/usr/bin/env python3
"""
scripts/run_comm_fragility.py
=================================
Communication-fragility robustness study (candidate enhancement, not part
of the accepted manuscript's original campaigns). Varies two physically
meaningful *system* parameters governing the communication-degradation
model (Sec. II-C of the paper) -- independent of the radiological
*environment* profile (Control/Moderate/Extreme, which vary D_bg/K):

  beta (packet-loss sensitivity, Eq. 9): hardware's dose-to-failure
      sensitivity for transient Bernoulli channel drop.
  Lambda_fail (cumulative-dose latch-up threshold, Eq. 11): hardware
      radiation-hardening level governing permanent transceiver failure.

Both sweeps compare Decentralized M-PSO against the static-leader
Centralized Swarm baseline (per MISSION.md's "Preferred direction"), under
the Extreme radiological profile -- the regime in which the existing
manuscript already shows the largest architectural gap (Table II) and
where communication degradation is most severe, matching the precedent set
by the existing swarm-size (Table IV) and communication-range (Table V)
sensitivity sweeps, which also fix the profile at Extreme.

Every condition additionally records the physical-layer connectivity
diagnostics added in this patch (``record_connectivity=True``):
  - mean_frac_isolated: trial- and time-averaged fraction of agents with a
    fully collapsed neighborhood (architecture-independent).
  - leader_latchup_rate_pct: fraction of trials in which the (Centralized
    Swarm's) leader is permanently latched by mission end -- undefined
    (blank) for Decentralized M-PSO, which has no leader.

Usage
-----
    python scripts/run_comm_fragility.py --mode smoke   # 20 trials/condition
    python scripts/run_comm_fragility.py --mode full     # 500 trials/condition
    python scripts/run_comm_fragility.py --mode full --workers 1

Outputs (--mode full only)
---------------------------
    results/table_beta_sweep.csv
    results/table_lambdafail_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pso_nuclear_sim import DEFAULT_CONFIG, PROFILES, run_condition  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

PROFILE = PROFILES["Extreme"]
ARCHS = ["Decentralized M-PSO", "Centralized Swarm"]

# Baseline (Table I) values: beta=0.15 s/Gy, Lambda_fail=300.0 Gy.
BETA_GRID = [0.05, 0.10, 0.15, 0.30, 0.50]           # s/Gy
LAMBDA_FAIL_GRID = [100.0, 200.0, 300.0, 500.0, 800.0]  # Gy


def _fmt(x, nd=1):
    return "N/A" if x is None else f"{x:.{nd}f}"


def run_beta_sweep(n_trials: int, workers: int | None, tag: str) -> List[dict]:
    print(f"=== Communication-fragility: beta (packet-loss sensitivity) sweep [{tag}] ===")
    rows = []
    for beta in BETA_GRID:
        for arch in ARCHS:
            cid = f"commfrag|beta={beta}|{arch}|{tag}"
            r = run_condition(
                cid, DEFAULT_CONFIG, arch, PROFILE,
                n_trials=n_trials, beta=beta, record_connectivity=True, max_workers=workers,
            )
            print(f"beta={beta:5.2f} s/Gy | {arch:22s} | success={r.success_rate_pct:5.1f}% "
                  f"(n={r.n_success}/{r.n_total}) | 95%CI=[{r.ci95_low:.1f},{r.ci95_high:.1f}] | "
                  f"frac_isolated={_fmt(r.mean_frac_isolated, 3)} | "
                  f"leader_latchup={_fmt(r.leader_latchup_rate_pct)}%")
            rows.append({
                "beta_s_per_Gy": beta, "Architecture": arch,
                "Success(%)": r.success_rate_pct, "n_success": r.n_success, "n_total": r.n_total,
                "CI95_low": r.ci95_low, "CI95_high": r.ci95_high,
                "NonTrivialSuccess(%)": r.nontrivial_success_rate_pct,
                "MeanFracIsolated": r.mean_frac_isolated,
                "LeaderLatchupRate(%)": r.leader_latchup_rate_pct,
                "CumDose(Gy)": r.mean_cum_dose_gy,
            })
    return rows


def run_lambda_fail_sweep(n_trials: int, workers: int | None, tag: str) -> List[dict]:
    print(f"=== Communication-fragility: Lambda_fail (latch-up threshold) sweep [{tag}] ===")
    rows = []
    for lam in LAMBDA_FAIL_GRID:
        for arch in ARCHS:
            cid = f"commfrag|lambdafail={lam}|{arch}|{tag}"
            r = run_condition(
                cid, DEFAULT_CONFIG, arch, PROFILE,
                n_trials=n_trials, lambda_fail=lam, record_connectivity=True, max_workers=workers,
            )
            print(f"Lambda_fail={lam:6.1f} Gy | {arch:22s} | success={r.success_rate_pct:5.1f}% "
                  f"(n={r.n_success}/{r.n_total}) | 95%CI=[{r.ci95_low:.1f},{r.ci95_high:.1f}] | "
                  f"frac_isolated={_fmt(r.mean_frac_isolated, 3)} | "
                  f"leader_latchup={_fmt(r.leader_latchup_rate_pct)}%")
            rows.append({
                "Lambda_fail_Gy": lam, "Architecture": arch,
                "Success(%)": r.success_rate_pct, "n_success": r.n_success, "n_total": r.n_total,
                "CI95_low": r.ci95_low, "CI95_high": r.ci95_high,
                "NonTrivialSuccess(%)": r.nontrivial_success_rate_pct,
                "MeanFracIsolated": r.mean_frac_isolated,
                "LeaderLatchupRate(%)": r.leader_latchup_rate_pct,
                "CumDose(Gy)": r.mean_cum_dose_gy,
            })
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Communication-fragility robustness study (beta, Lambda_fail).")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                         help="'smoke': 20 trials/condition, diagnostic only, never written to results/ "
                              "as manuscript evidence. 'full': 500 trials/condition, written to results/.")
    parser.add_argument("--workers", type=int, default=1,
                         help="Process-pool workers (default: 1 -- this sandbox has a single core; "
                              "see logs/runtime_estimate.md).")
    parser.add_argument("--sweep", choices=["beta", "lambdafail", "both"], default="both")
    args = parser.parse_args()

    n_trials = 20 if args.mode == "smoke" else 500
    tag = "SMOKE_DIAGNOSTIC_ONLY" if args.mode == "smoke" else "n500"

    t0 = time.time()
    beta_rows: List[dict] = []
    lambda_rows: List[dict] = []
    if args.sweep in ("beta", "both"):
        beta_rows = run_beta_sweep(n_trials, args.workers, tag)
        print()
    if args.sweep in ("lambdafail", "both"):
        lambda_rows = run_lambda_fail_sweep(n_trials, args.workers, tag)
        print()
    elapsed = time.time() - t0

    n_conditions = len(beta_rows) // 1 + len(lambda_rows) // 1  # already per-condition rows
    print(f"Total runtime: {elapsed:.1f}s for {len(beta_rows) + len(lambda_rows)} conditions "
          f"({n_trials} trials each).")

    if args.mode == "full":
        write_csv(beta_rows, RESULTS_DIR / "table_beta_sweep.csv")
        write_csv(lambda_rows, RESULTS_DIR / "table_lambdafail_sweep.csv")
    else:
        print("\n[SMOKE MODE] Results NOT written to results/ -- diagnostic only, per "
              "the execution-budget rule (do not use reduced-trial runs as manuscript evidence).")
        LOGS_DIR.mkdir(exist_ok=True)
        with open(LOGS_DIR / "smoke_test_output.txt", "a") as f:
            f.write(f"\n--- smoke run, sweep={args.sweep}, workers={args.workers}, "
                    f"n_trials={n_trials}, elapsed={elapsed:.2f}s ---\n")
            for row in beta_rows + lambda_rows:
                f.write(str(row) + "\n")


if __name__ == "__main__":
    main()
