# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""CLI runner for benchmarkable tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from .artifacts import merge_worker_reports, write_run_artifacts
from .discovery import case_summary, discover_cases, filter_case_index_mod
from .timing import run_variant


DEFAULT_MODULES = [
    "tests.pytorch.test_benchmarkable_permutation",
    "tests.jax.test_benchmarkable_softmax",
]


def run_cases(
    modules: list[str],
    framework: str,
    case_ids: list[str],
    tags: list[str],
    components: list[str],
    output_dir: Path,
    warmup: int,
    iterations: int,
    min_run_time: float,
    include_reference: bool,
    profile: bool,
    index_mod: tuple[int, int] | None = None,
    command: list[str] | None = None,
    sharding: dict[str, Any] | None = None,
    report_file: Path | None = None,
) -> dict[str, Path]:
    """Discover, run and write artifacts for selected benchmark cases."""
    cases = discover_cases(
        modules=modules,
        framework=framework,
        case_ids=case_ids,
        tags=tags,
        components=components,
    )
    cases = filter_case_index_mod(cases, index_mod)
    records = []
    for case in cases:
        for variant in case.iter_variants(include_reference=include_reference):
            records.append(
                run_variant(
                    case,
                    variant,
                    warmup=warmup,
                    iterations=iterations,
                    min_run_time=min_run_time,
                    profile=profile,
                )
            )

    selection = {
        "modules": modules,
        "framework": framework,
        "case_ids": case_ids,
        "tags": tags,
        "components": components,
        "include_reference": include_reference,
        "index_mod": index_mod,
        "selected_cases": [case_summary(case) for case in cases],
    }
    paths = write_run_artifacts(
        output_dir,
        records,
        command or sys.argv,
        selection,
        sharding=sharding,
    )
    if report_file is not None:
        _copy_report(paths["report"], report_file)
        paths["requested_report"] = report_file
    return paths


def main() -> int:
    parser = _make_parser()
    args = parser.parse_args()

    modules = args.module or DEFAULT_MODULES
    index_mod = _parse_index_mod(args.case_index_mod)

    if args.list_cases:
        cases = discover_cases(
            modules=modules,
            framework=args.framework,
            case_ids=args.case_id,
            tags=args.tag,
            components=args.component,
        )
        print(json.dumps([case_summary(case) for case in cases], indent=2, sort_keys=True))
        return 0

    if args.output_dir is None:
        parser.error("--output-dir is required unless --list-cases is used.")

    if (
        args.shard_devices == "auto"
        and index_mod is None
        and not args.profile
        and not args.no_spawn_workers
    ):
        devices = _scheduler_allocated_devices()
        if len(devices) > 1:
            return _run_sharded(args, modules, devices)

    paths = run_cases(
        modules=modules,
        framework=args.framework,
        case_ids=args.case_id,
        tags=args.tag,
        components=args.component,
        output_dir=Path(args.output_dir),
        warmup=args.warmup,
        iterations=args.iterations,
        min_run_time=args.min_run_time,
        include_reference=not args.no_reference,
        profile=args.profile,
        index_mod=index_mod,
        command=sys.argv,
        sharding=_single_process_sharding(args),
        report_file=Path(args.report_file) if args.report_file else None,
    )
    print(f"Wrote benchmark report: {paths['report']}")
    if "requested_report" in paths:
        print(f"Copied benchmark report: {paths['requested_report']}")
    return 0


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmarkable Transformer Engine tests.")
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="Module containing BENCHMARKABLE_CASES or iter_benchmark_cases().",
    )
    parser.add_argument(
        "--framework",
        choices=["all", "pytorch", "jax"],
        default="all",
        help="Framework filter (default: %(default)s)",
    )
    parser.add_argument("--case-id", action="append", default=[], help="Case id filter.")
    parser.add_argument("--tag", action="append", default=[], help="Require tag filter.")
    parser.add_argument("--component", action="append", default=[], help="Component filter.")
    parser.add_argument("--output-dir", default=None, help="Directory for benchmark artifacts.")
    parser.add_argument(
        "--report-file",
        default=None,
        help=(
            "Optional additional path for the top-level benchmark_report/v1 JSON report. "
            "This is useful when a harness requires a fixed raw-report filename."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per variant.")
    parser.add_argument("--iterations", type=int, default=20, help="Minimum timed iterations.")
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=0.0,
        help="Minimum timed seconds per variant in addition to --iterations.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Time only the evaluation path, even when reference is available.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable post-warmup CUDA profiler start/stop around timed iterations.",
    )
    parser.add_argument(
        "--shard-devices",
        choices=["auto", "off"],
        default="auto",
        help="Shard independent cases across scheduler-allocated GPUs (default: %(default)s).",
    )
    parser.add_argument(
        "--case-index-mod",
        default=None,
        help="Internal worker filter in the form WORKER_INDEX/WORKER_COUNT.",
    )
    parser.add_argument(
        "--no-spawn-workers",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--list-cases", action="store_true", help="Print selected cases and exit.")
    return parser


def _run_sharded(args: argparse.Namespace, modules: list[str], devices: list[str]) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    worker_commands = []
    for worker_index, device in enumerate(devices):
        worker_dir = output_dir / f"worker_{worker_index:03d}"
        worker_cmd = _worker_command(args, modules, worker_index, len(devices), worker_dir)
        worker_env = dict(os.environ)
        worker_env.update(
            {
                "CUDA_VISIBLE_DEVICES": device,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        worker_commands.append({"device": device, "output_dir": str(worker_dir), "cmd": worker_cmd})
        processes.append(subprocess.Popen(worker_cmd, env=worker_env))

    failed = []
    for index, process in enumerate(processes):
        returncode = process.wait()
        if returncode != 0:
            failed.append({"worker_index": index, "returncode": returncode})

    sharding = {
        "enabled": True,
        "reason": "independent benchmark cases sharded across scheduler-allocated devices",
        "scheduler_allocated_gpu_count": len(devices),
        "visible_gpu_count": len(_visible_cuda_devices()),
        "selected_devices": devices,
        "worker_commands": worker_commands,
        "sharding_strategy": "case_index_mod",
        "failed_workers": failed,
    }
    worker_reports = [
        Path(item["output_dir"]) / "benchmark_report.json" for item in worker_commands
    ]
    if failed:
        print(json.dumps({"failed_workers": failed}, indent=2, sort_keys=True))
        return 1
    paths = merge_worker_reports(
        output_dir,
        worker_reports,
        sys.argv,
        {
            "modules": modules,
            "framework": args.framework,
            "case_ids": args.case_id,
            "tags": args.tag,
            "components": args.component,
            "include_reference": not args.no_reference,
        },
        sharding,
    )
    if args.report_file:
        _copy_report(paths["report"], Path(args.report_file))
    print(f"Wrote merged benchmark report: {output_dir / 'benchmark_report.json'}")
    if args.report_file:
        print(f"Copied benchmark report: {args.report_file}")
    return 0


def _worker_command(
    args: argparse.Namespace,
    modules: list[str],
    worker_index: int,
    worker_count: int,
    worker_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.benchmarkable.runner",
        "--framework",
        args.framework,
        "--output-dir",
        str(worker_dir),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--min-run-time",
        str(args.min_run_time),
        "--shard-devices",
        "off",
        "--case-index-mod",
        f"{worker_index}/{worker_count}",
        "--no-spawn-workers",
    ]
    for module in modules:
        cmd.extend(["--module", module])
    for case_id in args.case_id:
        cmd.extend(["--case-id", case_id])
    for tag in args.tag:
        cmd.extend(["--tag", tag])
    for component in args.component:
        cmd.extend(["--component", component])
    if args.no_reference:
        cmd.append("--no-reference")
    return cmd


def _copy_report(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _single_process_sharding(args: argparse.Namespace) -> dict[str, Any]:
    if args.shard_devices == "off":
        return {"enabled": False, "reason": "disabled by --shard-devices=off"}
    if args.profile:
        return {
            "enabled": False,
            "reason": (
                "disabled for profiler attribution; use unprofiled run for full sharded sweep"
            ),
        }
    devices = _scheduler_allocated_devices()
    return {
        "enabled": False,
        "reason": "fewer than two scheduler-allocated devices detected",
        "scheduler_allocated_gpu_count": len(devices),
        "selected_devices": devices,
    }


def _parse_index_mod(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    left, right = raw.split("/", maxsplit=1)
    worker_index = int(left)
    worker_count = int(right)
    if worker_count < 1 or worker_index < 0 or worker_index >= worker_count:
        raise ValueError("--case-index-mod must be WORKER_INDEX/WORKER_COUNT.")
    return worker_index, worker_count


def _scheduler_allocated_devices() -> list[str]:
    for name in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"):
        devices = _parse_device_list(os.environ.get(name))
        if devices:
            return devices
    return []


def _visible_cuda_devices() -> list[str]:
    return _parse_device_list(os.environ.get("CUDA_VISIBLE_DEVICES"))


def _parse_device_list(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return []
    devices = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("GPU-"):
            devices.append(item)
        else:
            devices.append(item)
    return devices


if __name__ == "__main__":
    raise SystemExit(main())
