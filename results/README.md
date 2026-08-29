# Communication-fragility evidence

The two CSV files contain the 500-trial aggregate outputs for the added
packet-loss-sensitivity and latch-up-threshold sweeps. Each row is an
independent SHA-256-seeded campaign condition. Recreate the CSVs with:

```text
python scripts/run_comm_fragility.py --mode full --workers 1
```

The manuscript reports a compact subset selected to fit the six-page IEEE
limit. The full grids remain here for transparent reproduction.

`hot_standby_extreme.csv` contains the post-review 500-trial comparison of
decentralized M-PSO, the original centralized baselines, and a
state-replicating single-hot-standby baseline at the pre-specified
missed-heartbeat timeouts `h={1,3,5}`. Recreate it with:

```text
python scripts/run_hot_standby.py --mode full --workers 1
```

`HOT_STANDBY_PROTOCOL.md` records the frozen protocol, results, and bounded
interpretation.
