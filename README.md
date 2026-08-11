\# Decentralized M-PSO in High-Radiation Swarm Environments



This repository contains the simulation framework, experimental dataset, and figure generation pipeline for the manuscript:

\*\*"Decentralized M-PSO in High-Radiation Swarm Environments"\*\*



\---



\## Repository Structure



```text

decentralized-m-pso-radiation/

│

├── data/

│   ├── fig1\\\_convergence\\\_data.csv    # Step-by-step convergence trace data

│   ├── fig2\\\_trajectory\\\_data.csv     # Agent spatial coordinates (x, y, z) over time

│   ├── results\\\_table2.csv           # Benchmark statistics (success rates, mean convergence)

│   └── results\\\_table2.json          # Machine-readable output for Table II summary

│

├── make\\\_figures.py                  # Script to render publication-ready vector PDF figures

├── pso\\\_nuclear\\\_sim.py              # Core Monte Carlo simulation engine

├── README.md                        # Repository documentation

└── LICENSE                          # MIT License


