# Reviewer-response map

This is a drafting aid. Adapt its tone to the conference's required response
format, if one is requested; it is not normally uploaded with a camera-ready
paper.

| Reviewer point | Final manuscript response |
|---|---|
| Explain swarm size and horizon selection | Section III reports the $N=10,25,50$ sweep, identifies $N=25$ as the empirical coverage--dose knee, and explains that 150 steps exceeds the observed convergence tail. |
| Test swarm-size sensitivity | Table IV reports the 500-trial Extreme-profile swarm-size sensitivity results. |
| Neighbors and communication links can fail too | Section II defines failure for every agent's transceiver; Table V and Fig. 3 add packet-loss sensitivity and $\Lambda_{\rm fail}$ sweeps for the full communication model. Table VI reports a separate 500-trial-per-timeout hot-standby campaign: failover activates in 78.6--81.2% of trials, but the same range/$\chi$-gated information path remains and success stays at 0.6--1.6%. |
| Add recent, relevant literature | The introduction adds Mitchell et al. (2023) on multi-robot nuclear environments and Refis et al. (2025) on swarm robotics. |
| Define notation consistently | Section II introduces each M-PSO state and parameter before it is used in the corresponding equations. |
| Compare with AI/MARL approaches | Section III adds a concise, explicitly qualitative QMIX/MAPPO positioning statement. It does not invent an unmatched numerical baseline. |
| Add a methodology diagram | Fig. 1 gives the per-timestep physical, radiological, graph, and M-PSO workflow. |
| Vary system parameters beyond radiation/noise | Table IV and Table V/Fig. 3 vary swarm size plus the hardware/channel parameters $\beta$ and $\Lambda_{\rm fail}$, each with 500 independent trials per setting. |

## Claim boundary

The conclusions are restricted to the specified reduced-order simulation.
They are not claims of reactor deployment, validated electronics lifetime,
or superiority over MARL methods without a matched experimental comparison.
