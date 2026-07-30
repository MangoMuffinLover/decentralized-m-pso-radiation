#!/usr/bin/env python3
"""Generate Fig. 1 (convergence rate) and Fig. 2 (trajectory geometry) directly
from pso_nuclear_sim.py's real output CSVs. No hand-entered data."""

import csv
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Fig. 1: Convergence Rate ----------------
steps, control, moderate, extreme = [], [], [], []
with open("fig1_convergence_data.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        control.append(float(row["Control"]))
        moderate.append(float(row["Moderate"]))
        extreme.append(float(row["Extreme"]))

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(steps, control, color="#1f77b4", linewidth=2.0, label="Control Profile (0% Radiation background)")
ax.plot(steps, moderate, color="#ff7f0e", linewidth=2.0, label="Moderate Radiation Field (TID degradation)")
ax.plot(steps, extreme, color="#d62728", linewidth=2.0, label="Extreme Stochastic Gamma Field")
ax.axhline(0.3, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
ax.text(2, 0.32, "0.3 m success threshold", fontsize=8, color="gray")
ax.set_xlabel("Simulation Steps ($t$)")
ax.set_ylabel(r"Median Distance to Wall Fissure ($\|\vec{g}-\vec{x}_f\|$ meters)")
ax.set_title("Swarm Convergence Rate in PWR Pipe Environment\n(Decentralized M-PSO, population median across 500 Monte Carlo runs)", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("pso_convergence_rates.pdf")
print("Wrote pso_convergence_rates.pdf")

# ---------------- Fig. 2: Trajectory Geometry ----------------
traj = {}  # agent_id -> list of (x,y)
with open("fig2_trajectory_data.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        aid = int(row["agent_id"])
        traj.setdefault(aid, []).append((float(row["x"]), float(row["y"])))

fig2, ax2 = plt.subplots(figsize=(9, 4.0))
ax2.axhline(1.0, color="black", linewidth=2.5, label="Pipe Boundary Wall")
ax2.axhline(-1.0, color="black", linewidth=2.5)
for aid, pts in traj.items():
    pts = np.array(pts)
    ax2.plot(pts[:, 0], pts[:, 1], color="#7f8fdb", alpha=0.35, linewidth=0.8)
ax2.scatter([8.0], [0.9], marker="*", color="crimson", s=250, zorder=5, label="Target Micro-Fissure")
ax2.set_xlim(0, 10)
ax2.set_ylim(-1.15, 1.15)
ax2.set_xlabel("Axial Dimension ($x$ meters)")
ax2.set_ylabel("Radial Dimension ($y$ meters)")
ax2.set_title("Extreme Stochastic Gamma Field: Path Deviation Under Turbulent Drag\n(Decentralized M-PSO, all 25 agents, single representative run)", fontsize=10)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, linestyle=":", alpha=0.3)
fig2.tight_layout()
fig2.savefig("pso_swarm_trajectories.pdf")
print("Wrote pso_swarm_trajectories.pdf")
