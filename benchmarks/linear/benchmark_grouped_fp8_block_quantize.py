#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 block-scaling quantization."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from typing import Callable, Dict, Iterable, List

import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import Float8BlockQuantizer


def _csv_ints(value: str) -> List[int]:
    return [int(v) for v in value.split(",") if v]


def _csv_strings(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _run_git(args: List[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _fp8_dtype_from_name(name: str) -> te.DType:
    name = name.lower()
    if name in ("e4m3", "float8e4m3"):
        return te.DType.kFloat8E4M3
    if name in ("e5m2", "float8e5m2"):
        return te.DType.kFloat8E5M2
    raise ValueError(f"Unsupported FP8 dtype: {name}")


def _mode_to_usage(mode: str) -> tuple[bool, bool]:
    if mode == "rowwise":
        return True, False
    if mode == "columnwise":
        return False, True
    if mode == "both":
        return True, True
    raise ValueError(f"Unsupported output mode: {mode}")


def _roundup(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _scale_shape(rows: int, cols: int, dim: int, columnwise: bool) -> tuple[int, int]:
    if dim == 2:
        if columnwise:
            return math.ceil(cols / 128), _roundup(math.ceil(rows / 128), 4)
        return math.ceil(rows / 128), _roundup(math.ceil(cols / 128), 4)
    if columnwise:
        return math.ceil(rows / 128), _roundup(cols, 4)
    return math.ceil(cols / 128), _roundup(rows, 4)


def _scale_elements(rows: int, cols: int, dim: int, columnwise: bool) -> int:
    shape = _scale_shape(rows, cols, dim, columnwise)
    return shape[0] * shape[1]


def _make_rows(layout: str, num_groups: int, base_rows: int) -> List[int]:
    if layout == "uniform":
        return [base_rows] * num_groups
    if layout != "jagged":
        raise ValueError(f"Unsupported layout: {layout}")
    deltas = [0, -1, 1, -64, 64, -127, 127, -128, 128, -255, 255, -3, 3, -17, 17, 33]
    rows = [max(1, base_rows + deltas[i % len(deltas)]) for i in range(num_groups)]
    if num_groups >= 4:
        rows[0] = max(1, min(rows[0], 127))
        rows[1] = max(rows[1], 128)
        rows[2] = max(rows[2], 129)
    return rows


def _make_quantizer(dim: int, mode: str, fp8_dtype: te.DType) -> Float8BlockQuantizer:
    rowwise, columnwise = _mode_to_usage(mode)
    return Float8BlockQuantizer(
        fp8_dtype=fp8_dtype,
        rowwise=rowwise,
        columnwise=columnwise,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=dim,
    )


def _make_inputs(rows: List[int], cols: int, dtype: torch.dtype) -> List[torch.Tensor]:
    tensors = []
    for i, row_count in enumerate(rows):
        tensor = torch.randn((row_count, cols), dtype=dtype, device="cuda")
        tensors.append(tensor * float(i + 1))
    return tensors


def _calibrate_copy_bandwidth(num_bytes: int, warmup: int, iterations: int) -> float:
    numel = max(1, num_bytes // 2)
    src = torch.empty(numel, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        dst.copy_(src)
    end.record()
    end.synchronize()
    elapsed_s = start.elapsed_time(end) / 1000.0
    return (num_bytes * iterations) / elapsed_s / 1.0e9 if elapsed_s > 0 else 0.0


def _time_callable(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    min_sample_ms: float,
    profile: bool,
    nvtx_name: str,
) -> Dict[str, object]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    inner = 1
    while True:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
        if elapsed_ms >= min_sample_ms or inner >= 4096:
            break
        inner *= 2

    if profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(nvtx_name)

    samples_ms = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / inner)

    if profile:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()

    mean_ms = statistics.mean(samples_ms)
    median_ms = statistics.median(samples_ms)
    stdev_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return {
        "inner_iterations": inner,
        "samples_ms": samples_ms,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p90_ms": sorted(samples_ms)[int(0.9 * (len(samples_ms) - 1))],
        "stdev_ms": stdev_ms,
        "cv": stdev_ms / mean_ms if mean_ms else 0.0,
    }


def _bytes_model(
    rows: List[int],
    cols: int,
    dim: int,
    mode: str,
    input_dtype: torch.dtype,
) -> Dict[str, int]:
    input_bytes_per_elem = torch.empty((), dtype=input_dtype).element_size()
    total_elements = sum(rows) * cols
    rowwise, columnwise = _mode_to_usage(mode)
    input_bytes = total_elements * input_bytes_per_elem
    rowwise_output_bytes = total_elements if rowwise else 0
    columnwise_output_bytes = total_elements if columnwise else 0
    rowwise_scale_bytes = (
        4 * sum(_scale_elements(r, cols, dim, False) for r in rows) if rowwise else 0
    )
    columnwise_scale_bytes = (
        4 * sum(_scale_elements(r, cols, dim, True) for r in rows) if columnwise else 0
    )
    useful = (
        input_bytes
        + rowwise_output_bytes
        + columnwise_output_bytes
        + rowwise_scale_bytes
        + columnwise_scale_bytes
    )
    input_passes = 2 if dim == 1 and rowwise and columnwise else 1
    physical = (
        input_bytes * input_passes
        + rowwise_output_bytes
        + columnwise_output_bytes
        + rowwise_scale_bytes
        + columnwise_scale_bytes
    )
    return {
        "input_bytes": input_bytes,
        "rowwise_output_bytes": rowwise_output_bytes,
        "columnwise_output_bytes": columnwise_output_bytes,
        "rowwise_scale_bytes": rowwise_scale_bytes,
        "columnwise_scale_bytes": columnwise_scale_bytes,
        "useful_bytes": useful,
        "estimated_physical_bytes": physical,
    }


def _run_case(args: argparse.Namespace, case: Dict[str, object], env: Dict[str, object]) -> Dict[str, object]:
    rows = case["rows"]
    cols = case["cols"]
    dim = case["block_scaling_dim"]
    mode = case["output_mode"]
    dtype = _dtype_from_name(args.dtype)
    fp8_dtype = _fp8_dtype_from_name(args.fp8_dtype)

    tensors = _make_inputs(rows, cols, dtype)
    grouped_input = torch.cat(tensors, dim=0)
    first_dims = torch.tensor(rows, dtype=torch.int64, device="cuda")
    quantizer = _make_quantizer(dim, mode, fp8_dtype)
    quantizers = [_make_quantizer(dim, mode, fp8_dtype) for _ in rows]

    def candidate_group_quantize() -> object:
        return tex.group_quantize(grouped_input, quantizer, len(rows), first_dims)

    def candidate_split_quantize() -> object:
        return tex.split_quantize(grouped_input, rows, quantizers)

    def baseline_manual_loop() -> object:
        return [q(t) for q, t in zip(quantizers, tensors)]

    candidate_fn = (
        candidate_group_quantize if case["api"] == "group_quantize" else candidate_split_quantize
    )
    bytes_model = _bytes_model(rows, cols, dim, mode, dtype)
    copy_roofline_gbps = _calibrate_copy_bandwidth(
        max(bytes_model["estimated_physical_bytes"], 16 * 1024 * 1024),
        max(1, args.warmup // 4),
        max(5, args.iterations // 4),
    )

    profile_this_case = args.profile and (
        args.profile_case == "all" or args.profile_case == case["case_label"]
    )
    candidate_timing = _time_callable(
        candidate_fn,
        warmup=args.warmup,
        iterations=args.iterations,
        min_sample_ms=args.min_sample_ms,
        profile=profile_this_case,
        nvtx_name=f"candidate_{case['api']}_{case['case_label']}",
    )
    baseline_timing = _time_callable(
        baseline_manual_loop,
        warmup=args.warmup,
        iterations=args.iterations,
        min_sample_ms=args.min_sample_ms,
        profile=profile_this_case,
        nvtx_name=f"baseline_manual_loop_{case['case_label']}",
    )

    candidate_seconds = candidate_timing["mean_ms"] / 1000.0
    baseline_seconds = baseline_timing["mean_ms"] / 1000.0
    candidate_gbps = bytes_model["useful_bytes"] / candidate_seconds / 1.0e9
    baseline_gbps = bytes_model["useful_bytes"] / baseline_seconds / 1.0e9
    candidate_physical_gbps = bytes_model["estimated_physical_bytes"] / candidate_seconds / 1.0e9
    baseline_physical_gbps = bytes_model["estimated_physical_bytes"] / baseline_seconds / 1.0e9
    speedup = baseline_seconds / candidate_seconds

    rowwise, columnwise = _mode_to_usage(mode)
    record = {
        **env,
        **case,
        "input_dtype": str(dtype).replace("torch.", ""),
        "fp8_dtype": args.fp8_dtype,
        "force_pow_2_scales": True,
        "amax_epsilon": 0.0,
        "num_groups": len(rows),
        "rows": rows,
        "cols": cols,
        "first_dims": rows,
        "tensor_offsets": [0] + [sum(rows[: i + 1]) * cols for i in range(len(rows))],
        "rowwise_scale_shapes": [_scale_shape(r, cols, dim, False) for r in rows]
        if rowwise
        else [],
        "columnwise_scale_shapes": [_scale_shape(r, cols, dim, True) for r in rows]
        if columnwise
        else [],
        **bytes_model,
        "calibrated_copy_roofline_GBps": copy_roofline_gbps,
        "candidate_timing": candidate_timing,
        "baseline_timing": baseline_timing,
        "bandwidth_GBps_actual_bytes": candidate_gbps,
        "baseline_bandwidth_GBps_actual_bytes": baseline_gbps,
        "candidate_physical_bandwidth_GBps": candidate_physical_gbps,
        "baseline_physical_bandwidth_GBps": baseline_physical_gbps,
        "candidate_speedup_over_manual_loop": speedup,
        "candidate_fraction_of_calibrated_roofline": candidate_physical_gbps / copy_roofline_gbps
        if copy_roofline_gbps
        else None,
        "baseline_fraction_of_calibrated_roofline": baseline_physical_gbps / copy_roofline_gbps
        if copy_roofline_gbps
        else None,
        "launch_evidence": {
            "candidate_path": case["api"],
            "candidate_grouped_quantize_requests_per_call": 1,
            "baseline_path": "manual_loop_single_tensor_quantizer",
            "baseline_quantize_requests_per_call": len(rows),
            "profiler_requested": profile_this_case,
        },
        "baseline_stability": {
            "sample_count": len(baseline_timing["samples_ms"]),
            "cv": baseline_timing["cv"],
        },
    }
    return record


def _environment(command: str) -> Dict[str, object]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "gpu_model": props.name,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "container_image": os.getenv("NVIDIA_PYTORCH_VERSION", os.getenv("PYXIS_IMAGE", "unknown")),
        "te_commit": _run_git(["rev-parse", "HEAD"]),
        "te_branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "build_mode": os.getenv("NVTE_BUILD_DEBUG", "release"),
        "command": command,
    }


def _build_cases(args: argparse.Namespace) -> List[Dict[str, object]]:
    cases = []
    for api in _csv_strings(args.api):
        for dim in _csv_ints(args.dims):
            for mode in _csv_strings(args.output_modes):
                for layout in _csv_strings(args.layouts):
                    for num_groups in _csv_ints(args.num_groups):
                        for rows_base in _csv_ints(args.rows_sweep):
                            for cols in _csv_ints(args.cols):
                                rows = _make_rows(layout, num_groups, rows_base)
                                label = (
                                    f"{api}_d{dim}_{mode}_{layout}_g{num_groups}"
                                    f"_r{rows_base}_c{cols}"
                                )
                                cases.append(
                                    {
                                        "api": api,
                                        "block_scaling_dim": dim,
                                        "output_mode": mode,
                                        "layout": layout,
                                        "rows_base": rows_base,
                                        "rows": rows,
                                        "cols": cols,
                                        "case_label": label,
                                    }
                                )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="group_quantize", help="Comma-separated API list.")
    parser.add_argument("--dims", default="1,2", help="Comma-separated block scaling dims.")
    parser.add_argument("--output-modes", default="rowwise,columnwise,both")
    parser.add_argument("--layouts", default="uniform,jagged")
    parser.add_argument("--num-groups", default="1,2,4,8,16")
    parser.add_argument("--rows-sweep", default="128,512,2048,8192")
    parser.add_argument("--cols", default="256,1024,4096,8192")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--fp8-dtype", default="e4m3")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--min-sample-ms", type=float, default=50.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-case", default="all")
    parser.add_argument("--require-speed-of-light", action="store_true")
    parser.add_argument("--launch-evidence", default="internal")
    parser.add_argument("--output", default=os.getenv("ORCHESTRA_BENCHMARK_RAW_REPORT", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not available:
        raise RuntimeError(f"FP8 block scaling is not available: {reason}")

    env = _environment(" ".join(sys.argv))
    cases = _build_cases(args)
    records = []
    for case in cases:
        records.append(_run_case(args, case, env))

    report = {
        "schema_version": "grouped_fp8_block_quantize_benchmark/v1",
        "summary": {
            "num_cases": len(records),
            "apis": _csv_strings(args.api),
            "dims": _csv_ints(args.dims),
            "output_modes": _csv_strings(args.output_modes),
            "layouts": _csv_strings(args.layouts),
        },
        "records": records,
    }

    output = args.output
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
