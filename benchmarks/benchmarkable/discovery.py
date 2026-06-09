# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Benchmarkable case discovery."""

from __future__ import annotations

from collections.abc import Iterable
import importlib
from types import ModuleType
from typing import Any

from .api import BenchmarkCase


def import_module(module_name: str) -> ModuleType:
    """Import a module containing benchmarkable cases."""
    return importlib.import_module(module_name)


def cases_from_module(module: ModuleType) -> list[BenchmarkCase]:
    """Return benchmark cases registered by a module.

    Modules should expose either ``iter_benchmark_cases()`` or
    ``BENCHMARKABLE_CASES``. Discovery does not execute pytest test functions.
    """
    if hasattr(module, "iter_benchmark_cases"):
        raw_cases = module.iter_benchmark_cases()
    else:
        raw_cases = getattr(module, "BENCHMARKABLE_CASES", [])

    cases: list[BenchmarkCase] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, BenchmarkCase):
            raise TypeError(
                f"{module.__name__} produced {type(raw_case).__name__}; "
                "expected benchmarks.benchmarkable.BenchmarkCase."
            )
        if raw_case.source is None:
            raw_case.source = module.__name__
        cases.append(raw_case)
    return cases


def discover_cases(
    modules: Iterable[str],
    framework: str = "all",
    case_ids: Iterable[str] | None = None,
    tags: Iterable[str] | None = None,
    components: Iterable[str] | None = None,
) -> list[BenchmarkCase]:
    """Import modules and return benchmark cases matching the requested filters."""
    wanted_case_ids = set(case_ids or [])
    wanted_tags = set(tags or [])
    wanted_components = set(components or [])
    wanted_framework = framework.lower()

    discovered: list[BenchmarkCase] = []
    for module_name in modules:
        discovered.extend(cases_from_module(import_module(module_name)))

    return [
        case
        for case in discovered
        if _matches_case(case, wanted_framework, wanted_case_ids, wanted_tags, wanted_components)
    ]


def filter_case_index_mod(
    cases: list[BenchmarkCase],
    index_mod: tuple[int, int] | None,
) -> list[BenchmarkCase]:
    """Return the subset assigned to ``index % count == worker_index``."""
    if index_mod is None:
        return cases
    worker_index, worker_count = index_mod
    return [case for index, case in enumerate(cases) if index % worker_count == worker_index]


def case_summary(case: BenchmarkCase) -> dict[str, Any]:
    """Return a stable, JSON-friendly case summary."""
    data = case.metadata()
    data["has_reference"] = case.reference is not None
    data["has_check"] = case.check is not None
    return data


def _matches_case(
    case: BenchmarkCase,
    framework: str,
    case_ids: set[str],
    tags: set[str],
    components: set[str],
) -> bool:
    if framework != "all" and case.framework != framework:
        return False
    if case_ids and case.case_id not in case_ids:
        return False
    if tags and not tags.issubset(set(case.tags)):
        return False
    if components and case.component not in components:
        return False
    return True
