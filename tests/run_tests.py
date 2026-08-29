"""Small dependency-free runner for this repository's test subset.

Use ``python -m pytest -q`` when pytest is installed.  This runner exists
for restricted environments and intentionally supports only the module-level
fixtures and plain assertions used by the three local test files.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
import types
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPOSITORY_ROOT / "tests"
sys.path.insert(0, str(REPOSITORY_ROOT))


def fixture(function):
    function.__mini_fixture__ = True
    return function


sys.modules.setdefault("pytest", types.SimpleNamespace(fixture=fixture))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(function, instance, fixtures: dict[str, object]) -> None:
    kwargs = {}
    for parameter in inspect.signature(function).parameters:
        if parameter == "self":
            continue
        if parameter not in fixtures:
            raise LookupError(f"Missing fixture '{parameter}' for {function.__qualname__}")
        kwargs[parameter] = fixtures[parameter]()
    if instance is None:
        function(**kwargs)
    else:
        function(instance, **kwargs)


def main() -> int:
    passed = failed = 0
    for path in sorted(TEST_ROOT.glob("test_*.py")):
        module = load_module(path)
        fixtures = {
            name: value
            for name, value in vars(module).items()
            if inspect.isfunction(value) and getattr(value, "__mini_fixture__", False)
        }
        tests = []
        for name, value in vars(module).items():
            if name.startswith("test_") and inspect.isfunction(value):
                tests.append((f"{path.name}::{name}", value, None))
            elif name.startswith("Test") and inspect.isclass(value):
                instance = value()
                tests.extend(
                    (f"{path.name}::{name}::{method_name}", method, instance)
                    for method_name, method in vars(value).items()
                    if method_name.startswith("test_") and inspect.isfunction(method)
                )
        for label, function, instance in tests:
            try:
                invoke(function, instance, fixtures)
                passed += 1
                print(f"PASS  {label}")
            except Exception:
                failed += 1
                print(f"FAIL  {label}\n{traceback.format_exc()}")
    print(f"TOTAL: {passed} passed, {failed} failed")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
