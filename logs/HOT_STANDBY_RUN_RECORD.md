# Hot-standby campaign run record

- Protocol: one nearest-agent, state-replicating hot standby; missed-heartbeat
  timeout grid fixed at `h={1,3,5}` before the full run.
- Environment: Extreme radiation profile; `N=25`; `t_max=150`; 500 independent
  SHA-256-seeded trials per architecture; one worker.
- Smoke gate: 20 trials per architecture; diagnostic only, excluded from paper.
- Full command: `python scripts/run_hot_standby.py --mode full --workers 1`.
- Full campaign runtime: 160.1 seconds for six architectures x 500 trials.
- Regression suite after instrumentation: 62 passed, 0 failed.
- Additive-diagnostic check: the nominal `h=3` condition reproduced exactly at
  8/500 successes before and after adding failover-rate bookkeeping.

The authoritative output is `results/hot_standby_extreme.csv`; the protocol
and bounded interpretation are in `results/HOT_STANDBY_PROTOCOL.md`.
