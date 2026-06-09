# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Public API for tests that can also be used as benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import json
from typing import Any


SUPPORTED_FRAMEWORKS = {"pytorch", "jax"}


class BenchmarkSkip(Exception):
    """Raised by benchmarkable cases when a backend or hardware feature is unavailable."""


@dataclass(frozen=True)
class BenchmarkVariant:
    """A callable benchmark path for a case."""

    name: str
    function: Callable[[Any], Any]
    is_reference: bool = False


MetricFn = Callable[[Any, float], float]


@dataclass
class BenchmarkCase:
    """A correctness test case that can also be timed by the benchmark runner.

    ``setup`` creates deterministic inputs and returns an arbitrary state object.
    ``check`` validates correctness using that state and is intended to be called
    directly from a pytest test. The benchmark runner calls the same ``setup``,
    ``reference`` and ``evaluate`` functions when collecting timings.
    """

    case_id: str
    framework: str
    component: str
    operation: str
    params: Mapping[str, Any]
    setup: Callable[[], Any]
    evaluate: Callable[[Any], Any]
    reference: Callable[[Any], Any] | None = None
    check: Callable[[Any], None] | None = None
    tags: Iterable[str] = field(default_factory=tuple)
    metrics: Mapping[str, MetricFn] = field(default_factory=dict)
    unit_test: str | None = None
    source: str | None = None
    regression_threshold: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        framework = self.framework.lower()
        if framework not in SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unsupported framework '{self.framework}'. "
                f"Expected one of {sorted(SUPPORTED_FRAMEWORKS)}."
            )
        self.framework = framework
        self.tags = tuple(sorted(set(self.tags)))
        self.params = _json_mapping("params", self.params)
        self.metrics = dict(self.metrics)
        if self.regression_threshold is not None:
            self.regression_threshold = _json_mapping(
                "regression_threshold", self.regression_threshold
            )

    def iter_variants(self, include_reference: bool = True) -> list[BenchmarkVariant]:
        """Return the timed variants for this case."""
        variants: list[BenchmarkVariant] = []
        if include_reference and self.reference is not None:
            variants.append(
                BenchmarkVariant(name="reference", function=self.reference, is_reference=True)
            )
        variants.append(BenchmarkVariant(name="evaluation", function=self.evaluate))
        return variants

    def run_check(self) -> None:
        """Run the case's correctness check with freshly created state."""
        if self.check is None:
            raise ValueError(f"Benchmark case '{self.case_id}' does not define check().")
        self.check(self.setup())

    def metadata(self) -> dict[str, Any]:
        """Return stable metadata shared by all records for this case."""
        return {
            "case_id": self.case_id,
            "framework": self.framework,
            "component": self.component,
            "operation": self.operation,
            "params": dict(self.params),
            "tags": list(self.tags),
            "unit_test": self.unit_test,
            "source": self.source,
            "regression_threshold": (
                dict(self.regression_threshold) if self.regression_threshold is not None else None
            ),
        }


def benchmarkable_test(case_id: str | None = None, **metadata: Any) -> Callable[[Any], Any]:
    """Decorate a pytest test as benchmarkable without making pytest a hard dependency."""

    def decorator(test_fn: Any) -> Any:
        setattr(test_fn, "benchmarkable_case_id", case_id or getattr(test_fn, "__name__", None))
        setattr(test_fn, "benchmarkable_metadata", metadata)
        try:
            import pytest

            return pytest.mark.benchmarkable(test_fn)
        except ImportError:
            return test_fn

    return decorator


def _json_mapping(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that a mapping is JSON serializable and return a normal dict."""
    data = dict(value)
    try:
        json.dumps(data, sort_keys=True)
    except TypeError as exc:
        raise TypeError(f"Benchmark case {name} must be JSON serializable.") from exc
    return data
