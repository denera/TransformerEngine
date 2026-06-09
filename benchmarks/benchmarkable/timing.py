# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Timing backends for benchmarkable cases."""

from __future__ import annotations

import ctypes
import math
import statistics
import time
from typing import Any

from .api import BenchmarkCase, BenchmarkSkip, BenchmarkVariant


def run_variant(
    case: BenchmarkCase,
    variant: BenchmarkVariant,
    warmup: int,
    iterations: int,
    min_run_time: float,
    profile: bool = False,
) -> dict[str, Any]:
    """Run and time one benchmark variant."""
    record = _base_record(case, variant.name)
    try:
        state = case.setup()
        for _ in range(warmup):
            output = variant.function(state)
            synchronize(case.framework, output)
        synchronize(case.framework)

        samples_ms: list[float] = []
        profiler = _CudaProfiler(case.framework) if profile else _NullProfiler()
        profiler.start()
        try:
            timed_start = time.perf_counter()
            while len(samples_ms) < iterations or time.perf_counter() - timed_start < min_run_time:
                start = time.perf_counter()
                output = variant.function(state)
                synchronize(case.framework, output)
                samples_ms.append((time.perf_counter() - start) * 1000.0)
        finally:
            profiler.stop()
            synchronize(case.framework)

        stats = timing_stats(samples_ms)
        record.update(
            {
                "status": "completed",
                "warmup_iterations": warmup,
                "iterations": len(samples_ms),
                "min_run_time_sec": min_run_time,
                "profile_enabled": profile,
                "profile_after_warmup": profile,
                "samples_ms": samples_ms,
                "timing": stats,
                "metrics": _case_metrics(case, state, stats["median_ms"]),
            }
        )
    except BenchmarkSkip as exc:
        record.update({"status": "skipped", "reason": str(exc)})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        record.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        )
    return record


def timing_stats(samples_ms: list[float]) -> dict[str, float]:
    """Compute timing statistics in milliseconds."""
    if not samples_ms:
        return {
            "median_ms": 0.0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "stddev_ms": 0.0,
            "p95_ms": 0.0,
        }

    ordered = sorted(samples_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stddev_ms": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "p95_ms": ordered[p95_index],
    }


def synchronize(framework: str, output: Any | None = None) -> None:
    """Synchronize framework async work and block returned arrays when needed."""
    if framework == "pytorch":
        _synchronize_torch()
    elif framework == "jax":
        _block_until_ready(output)


def _base_record(case: BenchmarkCase, variant: str) -> dict[str, Any]:
    record = {
        "schema_version": "benchmark_record/v1",
        "variant": variant,
        "status": "pending",
    }
    record.update(case.metadata())
    return record


def _case_metrics(case: BenchmarkCase, state: Any, median_ms: float) -> dict[str, float]:
    metrics = {}
    for name, metric_fn in case.metrics.items():
        metrics[name] = float(metric_fn(state, median_ms))
    return metrics


def _synchronize_torch() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def _block_until_ready(value: Any) -> None:
    if value is None:
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _block_until_ready(item)


class _NullProfiler:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return


class _CudaProfiler:
    """Start CUDA profiler capture after warmup and stop before teardown."""

    def __init__(self, framework: str) -> None:
        self.framework = framework
        self._runtime = None

    def start(self) -> None:
        if self.framework == "pytorch" and self._start_torch():
            return
        self._runtime = _load_cudart()
        if self._runtime is not None:
            self._runtime.cudaProfilerStart()

    def stop(self) -> None:
        if self.framework == "pytorch" and self._stop_torch():
            return
        if self._runtime is not None:
            self._runtime.cudaProfilerStop()

    @staticmethod
    def _start_torch() -> bool:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.cudart().cudaProfilerStart()
                return True
        except Exception:  # pylint: disable=broad-exception-caught
            return False
        return False

    @staticmethod
    def _stop_torch() -> bool:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.cudart().cudaProfilerStop()
                return True
        except Exception:  # pylint: disable=broad-exception-caught
            return False
        return False


def _load_cudart() -> Any | None:
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.dylib"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None
