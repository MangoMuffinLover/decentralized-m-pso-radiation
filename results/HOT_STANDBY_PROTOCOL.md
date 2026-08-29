# Pre-specified hot-standby robustness study

## Question

Does a single state-replicating hot standby with missed-heartbeat failover
materially close the Extreme-profile gap between a centralized coordinator
and decentralized M-PSO?

## Protocol fixed before inspecting results

- The active leader follows the same range-limited reporting rule as the
  existing static-leader baseline.
- At the first timestep, the closest non-leader agent becomes the single hot
  standby; it receives no privileged position, radio, or side channel.
- Leader state is replicated only when leader and standby are within
  `r_comm` and both radiation-degraded channel states are operational.
- The standby promotes after `h` consecutive missed heartbeats. The timeout
  grid `h = {1, 3, 5}` was fixed before the full campaign.
- After promotion, the standby becomes the sole coordinator; no second backup,
  quorum, or omniscient failure detector is added.
- All conditions use the Extreme profile, 25 agents, 150 timesteps, the same
  physical/radiological model, 500 independent SHA-256-seeded trials, and
  Wilson 95% confidence intervals.

## Results

| Architecture | Success [Wilson 95% CI] | Non-trivial success | Failover completed |
|---|---:|---:|---:|
| Decentralized M-PSO | 87.0% [83.8, 89.7] | 82.5% | n/a |
| Static leader | 1.8% [0.9, 3.4] | 0.26% | n/a |
| Re-election | 4.6% [3.1, 6.8] | 2.86% | n/a |
| Hot standby, h=1 | 1.4% [0.7, 2.9] | 0.26% | 81.2% |
| Hot standby, h=3 | 1.6% [0.8, 3.1] | 0.00% | 80.4% |
| Hot standby, h=5 | 0.6% [0.2, 1.7] | 0.00% | 78.6% |

The hot-standby non-trivial rates use the trials that did not begin inside the
success radius as denominators: 1/391 for `h=1`, 0/392 for `h=3`, and 0/393
for `h=5`. Thus, the reported maximum of 0.26% is a condition-level rate,
not a pooled estimate across the timeout grid.

For the pre-specified nominal `h=3` comparator, decentralized M-PSO exceeds
hot-standby success by 85.4 percentage points (Wald z=27.18; approximate 95%
difference interval [82.25, 88.55] points). The failover diagnostic shows that
the recovery mechanism activated in most trials; its low success is not
explained by an inert implementation. Failover changes coordinator identity
but not the information pathway: leader reporting and state replication use
the same range/channel gates, while both coordinator agents accumulate dose.
The campaign did not instrument state age, so no empirical claim about the
distribution of replication staleness is made.

## Bounded interpretation

Within this reduced-order model, one state-replicating backup does not remove
the centralized aggregation bottleneck under severe range-and-channel
fragmentation. This experiment does not establish that every centralized
fault-tolerance design fails: multi-standby, quorum, relay-assisted, or
physically hardened side-channel protocols remain outside the evaluated set.
