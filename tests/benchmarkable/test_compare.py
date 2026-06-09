# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for benchmarkable report comparison."""

from benchmarks.benchmarkable.compare import compare_reports


def _report(median_ms):
    return {
        "schema_version": "benchmark_report/v1",
        "environment": {
            "devices": {
                "pytorch_cuda": {
                    "devices": [
                        {
                            "name": "Fake GPU",
                            "compute_capability": [9, 0],
                        }
                    ]
                }
            }
        },
        "records": [
            {
                "schema_version": "benchmark_record/v1",
                "status": "completed",
                "framework": "pytorch",
                "case_id": "fake.case",
                "component": "fake",
                "operation": "add",
                "variant": "evaluation",
                "params": {"size": 1},
                "timing": {"median_ms": median_ms},
            }
        ],
    }


def test_compare_reports_flags_regression():
    comparison = compare_reports(_report(1.0), _report(1.2))

    assert comparison["summary"]["regressions"] == 1
    assert comparison["regressions"][0]["relative_delta"] > 0.05


def test_compare_reports_accepts_unchanged_with_absolute_threshold():
    comparison = compare_reports(_report(1.0), _report(1.005))

    assert comparison["summary"]["regressions"] == 0
    assert comparison["summary"]["unchanged"] == 1
