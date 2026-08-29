#!/usr/bin/env python3
"""
scripts/_run_one_condition.py
=================================
Internal helper (not part of the public reproduction workflow -- see
README/run_comm_fragility.py for that). Runs exactly ONE 500-trial
condition of the communication-fragility study and appends its row to
the appropriate results CSV. Exists purely so the full study can be
executed as a sequence of short-lived, individually-timed processes
instead of one long-running one, with identical `condition_id`s (hence
identical SHA-256 seeds) and identical aggregation logic to
`run_comm_fragility.py --mode full`. Every condition here is
independently reproducible by re-running this same command.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pso_nuclear_sim import DEFAULT_CONFIG, PROFILES, run_condition  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
PROFILE = PROFILES["Extreme"]
N_TRIALS = 500


def append_row(path: Path, row: dict, fieldnames: list) -> None:
    path.parent.mkdir(exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_beta(beta: float, arch: str) -> None:
    cid = f"commfrag|beta={beta}|{arch}|n500"
    t0 = time.time()
    r = run_condition(cid, DEFAULT_CONFIG, arch, PROFILE, n_trials=N_TRIALS,
                       beta=beta, record_connectivity=True, max_workers=1)
    dt = time.time() - t0
    row = {
        "beta_s_per_Gy": beta, "Architecture": arch,
        "Success(%)": f"{r.success_rate_pct:.1f}", "n_success": r.n_success, "n_total": r.n_total,
        "CI95_low": f"{r.ci95_low:.1f}", "CI95_high": f"{r.ci95_high:.1f}",
        "NonTrivialSuccess(%)": f"{r.nontrivial_success_rate_pct:.1f}",
        "MeanFracIsolated": "" if r.mean_frac_isolated is None else f"{r.mean_frac_isolated:.4f}",
        "LeaderLatchupRate(%)": "" if r.leader_latchup_rate_pct is None else f"{r.leader_latchup_rate_pct:.1f}",
        "CumDose(Gy)": f"{r.mean_cum_dose_gy:.1f}",
    }
    append_row(RESULTS_DIR / "table_beta_sweep.csv", row,
               list(row.keys()))
    with open(LOGS_DIR / "commands.md", "a") as f:
        f.write(f"beta={beta} {arch}: success={r.success_rate_pct:.1f}% "
                f"(n={r.n_success}/{r.n_total}) elapsed={dt:.2f}s cid={cid}\n")
    print(f"beta={beta} {arch}: success={r.success_rate_pct:.1f}% "
          f"(n={r.n_success}/{r.n_total}) 95%CI=[{r.ci95_low:.1f},{r.ci95_high:.1f}] "
          f"frac_isolated={r.mean_frac_isolated} leader_latchup={r.leader_latchup_rate_pct} "
          f"elapsed={dt:.2f}s")


def run_lambda(lam: float, arch: str) -> None:
    cid = f"commfrag|lambdafail={lam}|{arch}|n500"
    t0 = time.time()
    r = run_condition(cid, DEFAULT_CONFIG, arch, PROFILE, n_trials=N_TRIALS,
                       lambda_fail=lam, record_connectivity=True, max_workers=1)
    dt = time.time() - t0
    row = {
        "Lambda_fail_Gy": lam, "Architecture": arch,
        "Success(%)": f"{r.success_rate_pct:.1f}", "n_success": r.n_success, "n_total": r.n_total,
        "CI95_low": f"{r.ci95_low:.1f}", "CI95_high": f"{r.ci95_high:.1f}",
        "NonTrivialSuccess(%)": f"{r.nontrivial_success_rate_pct:.1f}",
        "MeanFracIsolated": "" if r.mean_frac_isolated is None else f"{r.mean_frac_isolated:.4f}",
        "LeaderLatchupRate(%)": "" if r.leader_latchup_rate_pct is None else f"{r.leader_latchup_rate_pct:.1f}",
        "CumDose(Gy)": f"{r.mean_cum_dose_gy:.1f}",
    }
    append_row(RESULTS_DIR / "table_lambdafail_sweep.csv", row,
               list(row.keys()))
    with open(LOGS_DIR / "commands.md", "a") as f:
        f.write(f"Lambda_fail={lam} {arch}: success={r.success_rate_pct:.1f}% "
                f"(n={r.n_success}/{r.n_total}) elapsed={dt:.2f}s cid={cid}\n")
    print(f"Lambda_fail={lam} {arch}: success={r.success_rate_pct:.1f}% "
          f"(n={r.n_success}/{r.n_total}) 95%CI=[{r.ci95_low:.1f},{r.ci95_high:.1f}] "
          f"frac_isolated={r.mean_frac_isolated} leader_latchup={r.leader_latchup_rate_pct} "
          f"elapsed={dt:.2f}s")


if __name__ == "__main__":
    kind = sys.argv[1]
    value = float(sys.argv[2])
    arch = sys.argv[3]
    if kind == "beta":
        run_beta(value, arch)
    elif kind == "lambdafail":
        run_lambda(value, arch)
    else:
        raise SystemExit(f"unknown kind {kind}")
