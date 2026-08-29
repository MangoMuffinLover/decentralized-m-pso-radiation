#!/usr/bin/env python3
"""
scripts/make_comm_fragility_figure.py
=========================================
Compact, publication-ready two-panel figure for the communication-
fragility robustness study (candidate enhancement). Reads directly from
results/table_beta_sweep.csv and results/table_lambdafail_sweep.csv (no
hand-entered numbers) and produces a single PDF sized to sit as one
IEEE double-column figure (fig* two-column width), matching the
compactness constraint of a strict six-page camera-ready limit.

Output: figures/comm_fragility_sweep.pdf
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    beta_rows = read_csv(RESULTS_DIR / "table_beta_sweep.csv")
    lam_rows = read_csv(RESULTS_DIR / "table_lambdafail_sweep.csv")

    archs = ["Decentralized M-PSO", "Centralized Swarm"]
    colors = {"Decentralized M-PSO": "#1f77b4", "Centralized Swarm": "#d62728"}
    markers = {"Decentralized M-PSO": "o", "Centralized Swarm": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85))  # IEEE two-column width

    ax = axes[0]
    for arch in archs:
        rows = [r for r in beta_rows if r["Architecture"] == arch]
        rows.sort(key=lambda r: float(r["beta_s_per_Gy"]))
        x = [float(r["beta_s_per_Gy"]) for r in rows]
        y = [float(r["Success(%)"]) for r in rows]
        lo = [float(r["Success(%)"]) - float(r["CI95_low"]) for r in rows]
        hi = [float(r["CI95_high"]) - float(r["Success(%)"]) for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], marker=markers[arch], color=colors[arch],
                    label=arch, capsize=2, linewidth=1.2, markersize=4)
    ax.axvline(0.15, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$\beta$ (packet-loss sensitivity, s/Gy)", fontsize=8)
    ax.set_ylabel("Success rate (%)", fontsize=8)
    ax.set_title("(a) Packet-loss sensitivity sweep", fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.set_ylim(-3, 100)
    ax.grid(alpha=0.3, linewidth=0.5)

    ax2 = axes[1]
    for arch in archs:
        rows = [r for r in lam_rows if r["Architecture"] == arch]
        rows.sort(key=lambda r: float(r["Lambda_fail_Gy"]))
        x = [float(r["Lambda_fail_Gy"]) for r in rows]
        y = [float(r["Success(%)"]) for r in rows]
        lo = [float(r["Success(%)"]) - float(r["CI95_low"]) for r in rows]
        hi = [float(r["CI95_high"]) - float(r["Success(%)"]) for r in rows]
        ax2.errorbar(x, y, yerr=[lo, hi], marker=markers[arch], color=colors[arch],
                     label=arch, capsize=2, linewidth=1.2, markersize=4)
    ax2.axvline(300.0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_xlabel(r"$\Lambda_{\mathrm{fail}}$ (latch-up threshold, Gy)", fontsize=8)
    ax2.set_title("(b) Latch-up threshold sweep", fontsize=8.5)
    ax2.tick_params(labelsize=7)
    ax2.set_ylim(-3, 100)
    ax2.grid(alpha=0.3, linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / "comm_fragility_sweep.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
