# Contributing

Reproduction reports, bug reports, and carefully scoped extensions are welcome.

Before opening a pull request:

1. Describe the scientific or software question being addressed.
2. Keep changes to random-number generation explicit; do not silently change condition identities or call order.
3. Add or update tests for changes to physics, communication, or controller behavior.
4. Run `python tests/run_tests.py` and report the result.
5. Distinguish smoke-test output from 500-trial evidence.
6. Do not replace archived release CSVs without documenting the code version, protocol, and seed-policy implications.

For a proposed new baseline, document its information privileges, failure detector, side channels, training budget if applicable, and whether it shares the same physical degradation model.

