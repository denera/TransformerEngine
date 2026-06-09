# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for benchmarkable runner artifact generation."""

from __future__ import annotations

import json

from benchmarks.benchmarkable.artifacts import merge_worker_reports
from benchmarks.benchmarkable.runner import run_cases


def test_runner_writes_report_jsonl_and_summary(tmp_path, monkeypatch):
    module_path = tmp_path / "fake_bench_cases.py"
    module_path.write_text(
        """
from benchmarks.benchmarkable import BenchmarkCase

class State:
    def __init__(self):
        self.value = 1

def setup():
    return State()

def reference(state):
    return state.value + 1

def evaluate(state):
    return state.value + 2

def check(state):
    assert reference(state) == 2
    assert evaluate(state) == 3

def throughput(state, median_ms):
    return 10.0 / max(median_ms, 1.0e-9)

BENCHMARKABLE_CASES = [
    BenchmarkCase(
        case_id="fake.case",
        framework="pytorch",
        component="fake",
        operation="add",
        params={"size": 1},
        setup=setup,
        reference=reference,
        evaluate=evaluate,
        check=check,
        tags=("smoke",),
        metrics={"items_per_ms": throughput},
    )
]
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    output_dir = tmp_path / "out"
    report_file = tmp_path / "raw_report.json"

    paths = run_cases(
        modules=["fake_bench_cases"],
        framework="pytorch",
        case_ids=[],
        tags=[],
        components=[],
        output_dir=output_dir,
        warmup=1,
        iterations=2,
        min_run_time=0.0,
        include_reference=True,
        profile=False,
        report_file=report_file,
    )

    assert paths["report"].is_file()
    assert paths["records"].is_file()
    assert paths["summary"].is_file()
    assert paths["requested_report"] == report_file
    assert report_file.is_file()
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    copied_report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["schema_version"] == "benchmark_report/v1"
    assert copied_report["schema_version"] == "benchmark_report/v1"
    assert report["summary"]["record_count"] == 2
    assert report["summary"]["status_counts"]["completed"] == 2
    assert {record["variant"] for record in report["records"]} == {"reference", "evaluation"}
    assert all(record["timing"]["median_ms"] >= 0 for record in report["records"])
    assert all("items_per_ms" in record["metrics"] for record in report["records"])


def test_merge_worker_reports_validates_expected_case_variants(tmp_path):
    case_summary = {
        "case_id": "fake.case",
        "framework": "pytorch",
        "component": "fake",
        "operation": "add",
        "params": {"size": 1},
        "has_reference": True,
    }
    records = [
        {
            "schema_version": "benchmark_record/v1",
            "status": "completed",
            "framework": "pytorch",
            "case_id": "fake.case",
            "component": "fake",
            "operation": "add",
            "variant": variant,
            "params": {"size": 1},
            "timing": {"median_ms": 1.0},
        }
        for variant in ("reference", "evaluation")
    ]
    worker_report = {
        "schema_version": "benchmark_report/v1",
        "summary": {"record_count": 2},
        "selection": {
            "include_reference": True,
            "selected_cases": [case_summary],
        },
        "records": records,
    }
    worker_path = tmp_path / "worker_000" / "benchmark_report.json"
    worker_path.parent.mkdir()
    worker_path.write_text(json.dumps(worker_report), encoding="utf-8")

    paths = merge_worker_reports(
        tmp_path / "merged",
        [worker_path],
        ["python3", "-m", "benchmarks.benchmarkable.runner"],
        {"include_reference": True},
        {"enabled": True},
    )

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    validation = report["sharding"]["merge_validation"]
    assert validation["valid"]
    assert validation["duplicate_keys"] == []
    assert validation["missing_keys"] == []
