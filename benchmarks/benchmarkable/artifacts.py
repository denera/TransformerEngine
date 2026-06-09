# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Artifact helpers for benchmarkable runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any


REPORT_SCHEMA_VERSION = "benchmark_report/v1"


def write_run_artifacts(
    output_dir: Path,
    records: list[dict[str, Any]],
    command: list[str],
    selection: dict[str, Any],
    sharding: dict[str, Any] | None = None,
    report_name: str = "benchmark_report.json",
    records_name: str = "benchmark_records.jsonl",
    summary_name: str = "benchmark_summary.csv",
) -> dict[str, Path]:
    """Write JSON, JSONL and CSV artifacts for one benchmark run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(records, command, selection, sharding=sharding)

    report_path = output_dir / report_name
    records_path = output_dir / records_name
    summary_path = output_dir / summary_name

    _write_json(report_path, report)
    _write_jsonl(records_path, records)
    _write_summary_csv(summary_path, records)
    return {"report": report_path, "records": records_path, "summary": summary_path}


def build_report(
    records: list[dict[str, Any]],
    command: list[str],
    selection: dict[str, Any],
    sharding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-contained machine-readable benchmark report."""
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "selection": selection,
        "environment": collect_environment(command),
        "sharding": sharding or {"enabled": False},
        "summary": {
            "record_count": len(records),
            "status_counts": status_counts,
        },
        "records": records,
    }


def collect_environment(command: list[str] | None = None) -> dict[str, Any]:
    """Collect stable environment metadata without copying arbitrary environment variables."""
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        },
        "host": {
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
        },
        "git": _git_metadata(),
        "frameworks": _framework_versions(),
        "devices": _device_metadata(),
        "scheduler": _scheduler_metadata(),
        "command": command or [],
    }


def merge_worker_reports(
    output_dir: Path,
    worker_report_paths: list[Path],
    command: list[str],
    selection: dict[str, Any],
    sharding: dict[str, Any],
) -> dict[str, Path]:
    """Merge worker reports into the standard top-level artifact set."""
    merged_records: list[dict[str, Any]] = []
    worker_summaries = []
    for report_path in sorted(worker_report_paths):
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        worker_summaries.append(
            {
                "path": str(report_path),
                "summary": report.get("summary", {}),
                "selection": report.get("selection", {}),
            }
        )
        merged_records.extend(report.get("records", []))

    expected_records = _expected_records_from_worker_summaries(worker_summaries)
    sharding = dict(sharding)
    sharding["worker_reports"] = worker_summaries
    sharding["merge_validation"] = _validate_merged_records(
        merged_records,
        expected_records=expected_records,
    )
    return write_run_artifacts(
        output_dir,
        merged_records,
        command,
        selection,
        sharding=sharding,
    )


def record_key(record: dict[str, Any]) -> str:
    """Return a deterministic key for comparing benchmark records."""
    key = {
        "case_id": record.get("case_id"),
        "framework": record.get("framework"),
        "operation": record.get("operation"),
        "variant": record.get("variant"),
        "params": record.get("params", {}),
    }
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def _validate_merged_records(
    records: list[dict[str, Any]],
    expected_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in records:
        key = record_key(record)
        counts[key] = counts.get(key, 0) + 1
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    missing = []
    if expected_records is not None:
        expected_keys = sorted({record_key(record) for record in expected_records})
        missing = sorted(key for key in expected_keys if key not in counts)
    return {
        "record_count": len(records),
        "unique_record_count": len(counts),
        "duplicate_keys": duplicates,
        "missing_keys": missing,
        "valid": not duplicates and not missing,
    }


def _expected_records_from_worker_summaries(
    worker_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_records = []
    for worker in worker_summaries:
        selection = worker.get("selection", {})
        include_reference = selection.get("include_reference", True)
        for case in selection.get("selected_cases", []):
            variants = []
            if include_reference and case.get("has_reference"):
                variants.append("reference")
            variants.append("evaluation")
            for variant in variants:
                expected_records.append(
                    {
                        "case_id": case.get("case_id"),
                        "framework": case.get("framework"),
                        "operation": case.get("operation"),
                        "variant": variant,
                        "params": case.get("params", {}),
                    }
                )
    return expected_records


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "status",
        "framework",
        "case_id",
        "variant",
        "component",
        "operation",
        "median_ms",
        "mean_ms",
        "p95_ms",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            timing = record.get("timing", {})
            writer.writerow(
                {
                    "status": record.get("status"),
                    "framework": record.get("framework"),
                    "case_id": record.get("case_id"),
                    "variant": record.get("variant"),
                    "component": record.get("component"),
                    "operation": record.get("operation"),
                    "median_ms": timing.get("median_ms"),
                    "mean_ms": timing.get("mean_ms"),
                    "p95_ms": timing.get("p95_ms"),
                    "reason": record.get("reason"),
                }
            )


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run_git(["status", "--porcelain"])),
    }


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            encoding="utf-8",
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _framework_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "transformer_engine": _module_version("transformer_engine"),
        "torch": _module_version("torch"),
        "jax": _module_version("jax"),
        "jaxlib": _module_version("jaxlib"),
    }
    return versions


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    return getattr(module, "__version__", None)


def _device_metadata() -> dict[str, Any]:
    return {
        "pytorch_cuda": _torch_device_metadata(),
        "jax": _jax_device_metadata(),
    }


def _torch_device_metadata() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}

    if not torch.cuda.is_available():
        return {"available": False, "device_count": 0}

    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        uuid = getattr(props, "uuid", None)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory": props.total_memory,
                "compute_capability": [props.major, props.minor],
                "multi_processor_count": props.multi_processor_count,
                "uuid": str(uuid) if uuid is not None else None,
            }
        )
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "current_device": torch.cuda.current_device(),
        "devices": devices,
    }


def _jax_device_metadata() -> dict[str, Any]:
    try:
        import jax
    except ImportError:
        return {"available": False, "reason": "jax not installed"}

    try:
        devices = [
            {
                "id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
                "process_index": device.process_index,
            }
            for device in jax.devices()
        ]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"available": False, "reason": str(exc)}

    return {"available": bool(devices), "device_count": len(devices), "devices": devices}


def _scheduler_metadata() -> dict[str, Any]:
    names = [
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_GPUS",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_STEP_GPUS",
    ]
    metadata = {name: os.environ.get(name) for name in names if os.environ.get(name) is not None}
    visible_devices = _parse_device_list(os.environ.get("CUDA_VISIBLE_DEVICES"))
    allocated_devices = _scheduler_allocated_devices()
    metadata.update(
        {
            "visible_cuda_devices": visible_devices,
            "visible_gpu_count": len(visible_devices),
            "scheduler_allocated_devices": allocated_devices,
            "scheduler_allocated_gpu_count": len(allocated_devices),
        }
    )
    return metadata


def _scheduler_allocated_devices() -> list[str]:
    for name in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"):
        devices = _parse_device_list(os.environ.get(name))
        if devices:
            return devices
    return []


def _parse_device_list(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]
