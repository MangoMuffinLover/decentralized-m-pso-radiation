#!/usr/bin/env python3
"""Run the pre-specified hot-standby centralized-baseline study.

The study answers one bounded question raised during final peer review:
does a state-replicating leader with a single nearby hot standby and a
missed-heartbeat failover rule materially close the Extreme-profile gap to
decentralized M-PSO?  All coordination messages use the same range, packet-
loss, and cumulative-dose latch-up model as the existing centralized
baselines.  The three timeout values are fixed before inspection (1, 3, and
5 simulation steps) to avoid selecting a favorable recovery latency after
seeing outcomes.

Usage
-----
    python scripts/run_hot_standby.py --mode smoke
    python scripts/run_hot_standby.py --mode full --workers 1

Only the 500-trial ``full`` campaign writes manuscript-grade evidence to
``results/hot_standby_extreme.csv``.  Smoke output is diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pso_nuclear_sim import DEFAULT_CONFIG, PROFILES, run_condition  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
PROFILE = PROFILES["Extreme"]
ARCHITECTURES = [
    "Decentralized M-PSO",
    "Centralized Swarm",
    "Centralized Swarm (re-election)",
    "Centralized Swarm (hot standby, h=1)",
    "Centralized Swarm (hot standby, h=3)",
    "Centralized Swarm (hot standby, h=5)",
]


def as_row(architecture: str, result) -> dict[str, object]:
    return {
        "Architecture": architecture,
        "Success(%)": result.success_rate_pct,
        "n_success": result.n_success,
        "n_total": result.n_total,
        "CI95_low": result.ci95_low,
        "CI95_high": result.ci95_high,
        "NonTrivialSuccess(%)": result.nontrivial_success_rate_pct,
        "MeanConvSteps": result.mean_conv_steps,
        "MeanFracIsolated": result.mean_frac_isolated,
        "LeaderLatchupRate(%)": result.leader_latchup_rate_pct,
        "HotStandbyFailoverRate(%)": result.hot_standby_failover_rate_pct,
        "CumDose(Gy)": result.mean_cum_dose_gy,
    }


def run_campaign(n_trials: int, workers: int | None, tag: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        condition_id = f"hotstandby|Extreme|{architecture}|{tag}"
        result = run_condition(
            condition_id,
            DEFAULT_CONFIG,
            architecture,
            PROFILE,
            n_trials=n_trials,
            record_connectivity=True,
            max_workers=workers,
        )
        print(
            f"{architecture:40s} | success={result.success_rate_pct:5.1f}% "
            f"(n={result.n_success}/{result.n_total}) | "
            f"95%CI=[{result.ci95_low:.1f},{result.ci95_high:.1f}] | "
            f"nontrivial={result.nontrivial_success_rate_pct:5.1f}% | "
            f"failover={result.hot_standby_failover_rate_pct if result.hot_standby_failover_rate_pct is not None else 'N/A'} | "
            f"dose={result.mean_cum_dose_gy:.1f} Gy"
        )
        rows.append(as_row(architecture, result))
    return rows


def write_csv(rows: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    n_trials = 20 if args.mode == "smoke" else 500
    tag = "SMOKE_DIAGNOSTIC_ONLY" if args.mode == "smoke" else "n500"
    started = time.time()
    rows = run_campaign(n_trials, args.workers, tag)
    elapsed = time.time() - started
    print(f"Completed {len(rows)} conditions x {n_trials} trials in {elapsed:.1f}s.")

    if args.mode == "full":
        destination = RESULTS_DIR / "hot_standby_extreme.csv"
        write_csv(rows, destination)
        print(f"Wrote manuscript-grade results to {destination}")
    else:
        LOGS_DIR.mkdir(exist_ok=True)
        with (LOGS_DIR / "hot_standby_smoke.txt").open("w") as stream:
            stream.write(f"n_trials={n_trials}; elapsed_s={elapsed:.3f}\n")
            for row in rows:
                stream.write(f"{row}\n")
        print("[SMOKE MODE] Diagnostic results were not written to results/.")


if __name__ == "__main__":
    main()
