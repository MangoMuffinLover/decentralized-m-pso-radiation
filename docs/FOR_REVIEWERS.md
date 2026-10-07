# A five-minute route through the project

This page is for faculty members, reviewers, and prospective collaborators who want to evaluate the work efficiently.

## 1. Start with the claim

The study does not claim a new PSO family or universal superiority of decentralized control. Its contribution is a physically grounded process that generates a time-varying communication topology from range, transient dose-dependent packet loss, and permanent cumulative-dose latch-up, then evaluates how that topology affects swarm coordination.

## 2. Inspect the strongest evidence

- Main paper, Table II: architecture comparison across Control, Moderate, and Extreme profiles.
- Main paper, Table III: seven-cell ablation isolating drag and degradation mechanisms.
- Main paper, Fig. 3 and Table V: communication sensitivity to packet-loss coefficient and latch-up threshold.
- Main paper, Table VI: state-replicating hot standby with pre-specified heartbeat timeouts.
- `docs/RESULTS.md`: concise guide to campaign identities and why nominal replications differ slightly.

## 3. Inspect the implementation boundary

- `docs/PAPER_CODE_MAP.md` maps equations and tables to code.
- `docs/MODEL_SCOPE.md` separates supported conclusions from untested extrapolations.
- `results/HOT_STANDBY_PROTOCOL.md` records the hardened-baseline protocol and its bounded interpretation.

## 4. Verify the software

```bash
python -m pip install -r requirements.txt
python tests/run_tests.py
```

Expected result: 62 passed, 0 failed.

## 5. Reproduce before extending

Use the smoke campaigns in `docs/REPRODUCIBILITY.md` for a quick end-to-end check. The same guide gives the full 500-trial commands and explains the SHA-256 seed policy.

## Productive extension directions

The highest-value next steps are hardware-in-the-loop validation, 3D/CFD coupling, matched-observation MARL comparisons, and centralized comparators with multiple standbys, quorum logic, relay support, or a physically independent side channel.

