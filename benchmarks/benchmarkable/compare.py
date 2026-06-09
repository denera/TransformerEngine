# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Compare benchmarkable reports against historical artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import record_key


DEFAULT_RELATIVE_THRESHOLD = 0.05
DEFAULT_ABSOLUTE_THRESHOLD_MS = 0.01


def compare_reports(
    baseline_report: dict[str, Any],
    current_report: dict[str, Any],
    relative_threshold: float = DEFAULT_RELATIVE_THRESHOLD,
    absolute_threshold_ms: float = DEFAULT_ABSOLUTE_THRESHOLD_MS,
) -> dict[str, Any]:
    """Compare completed records in two reports."""
    baseline_records = _completed_records_by_key(baseline_report)
    current_records = _completed_records_by_key(current_report)

    regressions = []
    improvements = []
    unchanged = []
    incompatible = []

    for key, current in sorted(current_records.items()):
        baseline = baseline_records.get(key)
        if baseline is None:
            continue
        if not _hardware_compatible(baseline_report, current_report):
            incompatible.append({"key": key, "reason": "hardware metadata differs"})
            continue

        baseline_ms = _median_ms(baseline)
        current_ms = _median_ms(current)
        delta_ms = current_ms - baseline_ms
        relative_delta = delta_ms / baseline_ms if baseline_ms > 0 else 0.0
        threshold = _threshold_for_record(current, relative_threshold, absolute_threshold_ms)
        entry = {
            "key": key,
            "case_id": current.get("case_id"),
            "variant": current.get("variant"),
            "params": current.get("params", {}),
            "baseline_median_ms": baseline_ms,
            "current_median_ms": current_ms,
            "delta_ms": delta_ms,
            "relative_delta": relative_delta,
            "relative_threshold": threshold["relative"],
            "absolute_threshold_ms": threshold["absolute_ms"],
        }
        if delta_ms > threshold["absolute_ms"] and relative_delta > threshold["relative"]:
            regressions.append(entry)
        elif delta_ms < -threshold["absolute_ms"] and -relative_delta > threshold["relative"]:
            improvements.append(entry)
        else:
            unchanged.append(entry)

    missing = sorted(key for key in baseline_records if key not in current_records)
    new = sorted(key for key in current_records if key not in baseline_records)
    return {
        "schema_version": "benchmark_comparison/v1",
        "summary": {
            "baseline_records": len(baseline_records),
            "current_records": len(current_records),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "unchanged": len(unchanged),
            "missing": len(missing),
            "new": len(new),
            "incompatible": len(incompatible),
        },
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "missing": missing,
        "new": new,
        "incompatible": incompatible,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare benchmarkable JSON reports.")
    parser.add_argument("--baseline", required=True, help="Historical benchmark_report.json")
    parser.add_argument("--current", required=True, help="Current benchmark_report.json")
    parser.add_argument("--output", required=True, help="Comparison JSON output path")
    parser.add_argument(
        "--relative-threshold",
        type=float,
        default=DEFAULT_RELATIVE_THRESHOLD,
        help="Default relative regression threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--absolute-threshold-ms",
        type=float,
        default=DEFAULT_ABSOLUTE_THRESHOLD_MS,
        help="Default absolute regression threshold in ms (default: %(default)s)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit nonzero when any regression is detected.",
    )
    args = parser.parse_args()

    baseline = _read_json(Path(args.baseline))
    current = _read_json(Path(args.current))
    comparison = compare_reports(
        baseline,
        current,
        relative_threshold=args.relative_threshold,
        absolute_threshold_ms=args.absolute_threshold_ms,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.fail_on_regression and comparison["regressions"]:
        return 2
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _completed_records_by_key(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {}
    for record in report.get("records", []):
        if record.get("status") == "completed":
            records[record_key(record)] = record
    return records


def _median_ms(record: dict[str, Any]) -> float:
    return float(record.get("timing", {}).get("median_ms", 0.0))


def _threshold_for_record(
    record: dict[str, Any],
    default_relative: float,
    default_absolute_ms: float,
) -> dict[str, float]:
    override = record.get("regression_threshold") or {}
    return {
        "relative": float(override.get("relative", default_relative)),
        "absolute_ms": float(override.get("absolute_ms", default_absolute_ms)),
    }


def _hardware_compatible(
    baseline_report: dict[str, Any],
    current_report: dict[str, Any],
) -> bool:
    baseline_device = _primary_device_identity(baseline_report)
    current_device = _primary_device_identity(current_report)
    return not baseline_device or not current_device or baseline_device == current_device


def _primary_device_identity(report: dict[str, Any]) -> tuple[Any, ...] | None:
    env = report.get("environment", {})
    torch_cuda = env.get("devices", {}).get("pytorch_cuda", {})
    devices = torch_cuda.get("devices", [])
    if devices:
        device = devices[0]
        return (device.get("name"), tuple(device.get("compute_capability", [])))

    jax_devices = env.get("devices", {}).get("jax", {}).get("devices", [])
    if jax_devices:
        device = jax_devices[0]
        return (device.get("platform"), device.get("device_kind"))
    return None


if __name__ == "__main__":
    raise SystemExit(main())
