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
import tempfile
from typing import Callable, Dict, List, Optional

import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch import Float8BlockQuantizer


_DIM2_COLUMNWISE_ONLY_UNSUPPORTED_BASELINE = (
    "non_grouped_float8_block_dim2_columnwise_only_manual_loop_baseline_unsupported"
)

_DIM2_COLUMNWISE_ONLY_UNSUPPORTED_RATIONALE = (
    "The established non-grouped 2D FP8 block-scaling quantizer requires a "
    "rowwise-shaped primary output buffer when columnwise transpose output is "
    "requested. The benchmark therefore excludes dim=2 columnwise-only primary "
    "performance cases instead of timing an unsupported rowwise=false, "
    "columnwise=true manual-loop baseline or comparing against a both-output "
    "baseline with extra rowwise work."
)

_GROUPED_LINEAR_FP8_BLOCK_SCALING_M_SPLIT_MULTIPLE = 4

_GROUPED_LINEAR_FP8_BLOCK_SCALING_UNSUPPORTED_SPLITS = (
    "grouped_linear_float8_block_scaling_m_split_not_divisible_by_4"
)

_GROUPED_LINEAR_FP8_BLOCK_SCALING_UNSUPPORTED_RATIONALE = (
    "GroupedLinear with Float8BlockScaling routes through grouped GEMM scale-factor "
    "swizzling, which requires each non-empty m_split to be divisible by 4. "
    "Requested secondary GroupedLinear cases with jagged rows that violate this "
    "constraint are reported as skipped secondary records instead of being timed."
)


def _csv_ints(value: str) -> List[int]:
    return [int(v) for v in value.split(",") if v]


def _csv_strings(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _env_list(value: str) -> List[str]:
    return [v.strip() for v in value.replace(" ", ",").split(",") if v.strip()]


def _slurm_gpu_count(value: str) -> int:
    count = 0
    for entry in _env_list(value):
        if "-" in entry:
            start, end = entry.split("-", 1)
            if start.isdigit() and end.isdigit():
                count += int(end) - int(start) + 1
                continue
        count += 1
    return count


def _scheduler_selected_devices() -> List[str]:
    visible = _env_list(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    if not visible:
        visible = [str(i) for i in range(torch.cuda.device_count())]

    slurm_job_gpus = os.getenv("SLURM_JOB_GPUS", "")
    slurm_gpus_on_node = os.getenv("SLURM_GPUS_ON_NODE", "")
    allocated_count = _slurm_gpu_count(slurm_job_gpus)
    if allocated_count == 0 and slurm_gpus_on_node.isdigit():
        allocated_count = int(slurm_gpus_on_node)

    if allocated_count <= 0:
        return visible[:1]
    return visible[: min(allocated_count, len(visible))]


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


def _has_same_output_manual_loop_baseline(dim: int, mode: str) -> bool:
    return not (dim == 2 and mode == "columnwise")


def _grouped_linear_invalid_m_split_indices(rows: List[int]) -> List[int]:
    return [
        i
        for i, row_count in enumerate(rows)
        if row_count > 0
        and row_count % _GROUPED_LINEAR_FP8_BLOCK_SCALING_M_SPLIT_MULTIPLE != 0
    ]


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


def _row_tile_launch_evidence(rows: List[int], cols: int) -> Dict[str, object]:
    block_len = 128
    tiles_n = math.ceil(cols / block_len)
    useful_row_tiles = [math.ceil(row_count / block_len) for row_count in rows]
    launch_row_tiles_per_group = math.ceil(max(rows) / block_len) if rows else 0
    planned_row_tile_ctas = len(rows) * launch_row_tiles_per_group
    useful_row_tile_ctas = sum(useful_row_tiles)
    planned_total_ctas = planned_row_tile_ctas * tiles_n
    useful_total_ctas = useful_row_tile_ctas * tiles_n
    overlaunch_factor = planned_total_ctas / useful_total_ctas if useful_total_ctas else None
    return {
        "block_len": block_len,
        "tiles_n": tiles_n,
        "useful_row_tiles_per_group": useful_row_tiles,
        "candidate_launch_row_tiles_per_group": launch_row_tiles_per_group,
        "candidate_planned_row_tile_ctas": planned_row_tile_ctas,
        "candidate_useful_row_tile_ctas": useful_row_tile_ctas,
        "candidate_planned_total_ctas": planned_total_ctas,
        "candidate_useful_total_ctas": useful_total_ctas,
        "candidate_total_cta_overlaunch_factor": overlaunch_factor,
        "candidate_launch_geometry_source": "max_member_rows_from_grouped_output_shapes",
    }


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


def _make_fp8_block_recipe(dim: int) -> recipe.Float8BlockScaling:
    weight_dim = 1 if dim == 2 else 2
    return recipe.Float8BlockScaling(
        x_block_scaling_dim=dim,
        w_block_scaling_dim=weight_dim,
        grad_block_scaling_dim=1,
    )


def _make_inputs(rows: List[int], cols: int, dtype: torch.dtype) -> List[torch.Tensor]:
    tensors = []
    for i, row_count in enumerate(rows):
        tensor = torch.randn((row_count, cols), dtype=dtype, device="cuda")
        tensors.append(tensor * float(i + 1))
    return tensors


def _calibrate_copy_bandwidth(
    target_read_write_bytes: int, warmup: int, iterations: int
) -> Dict[str, object]:
    element_size_bytes = torch.empty((), dtype=torch.float16).element_size()
    one_way_bytes = max(1, (target_read_write_bytes + 1) // 2)
    numel = max(1, (one_way_bytes + element_size_bytes - 1) // element_size_bytes)
    src = torch.empty(numel, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    calibrated_one_way_bytes = numel * element_size_bytes
    calibrated_read_write_bytes = 2 * calibrated_one_way_bytes
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
    one_way_gbps = (
        calibrated_one_way_bytes * iterations / elapsed_s / 1.0e9 if elapsed_s > 0 else 0.0
    )
    read_write_gbps = (
        calibrated_read_write_bytes * iterations / elapsed_s / 1.0e9 if elapsed_s > 0 else 0.0
    )
    return {
        "copy_calibration_kernel": "torch.Tensor.copy_ device_to_device",
        "copy_calibration_dtype": "float16",
        "copy_calibration_accounting": "read_plus_write_global_memory_traffic",
        "copy_calibration_requested_read_write_bytes": target_read_write_bytes,
        "copy_calibration_one_way_bytes": calibrated_one_way_bytes,
        "copy_calibration_read_write_bytes": calibrated_read_write_bytes,
        "copy_calibration_warmup": warmup,
        "copy_calibration_iterations": iterations,
        "copy_calibration_elapsed_s": elapsed_s,
        "calibrated_copy_one_way_bandwidth_GBps": one_way_gbps,
        "calibrated_copy_read_write_roofline_GBps": read_write_gbps,
    }


def _roofline_fraction(physical_gbps: float, roofline_gbps: float) -> Dict[str, object]:
    if roofline_gbps <= 0:
        return {
            "fraction": None,
            "raw_fraction": None,
            "valid": False,
            "invalid_reason": "calibrated_read_write_roofline_not_positive",
        }
    raw_fraction = physical_gbps / roofline_gbps
    if raw_fraction <= 1.0:
        return {
            "fraction": raw_fraction,
            "raw_fraction": raw_fraction,
            "valid": True,
            "invalid_reason": None,
        }
    return {
        "fraction": None,
        "raw_fraction": raw_fraction,
        "valid": False,
        "invalid_reason": "measured_physical_bandwidth_exceeds_calibrated_read_write_roofline",
    }


def _time_callable(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    min_sample_ms: float,
    profile: bool,
    nvtx_name: str,
    use_cuda_graph: bool,
    cuda_graph_repetitions: int,
) -> Dict[str, object]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timed_fn = fn
    calls_per_timed_invocation = 1
    graph_repetitions = 0
    if use_cuda_graph:
        graph_repetitions = max(1, cuda_graph_repetitions)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for _ in range(graph_repetitions):
                fn()

        def replay_graph() -> None:
            graph.replay()

        timed_fn = replay_graph
        calls_per_timed_invocation = graph_repetitions
        torch.cuda.synchronize()

    inner = 1
    while True:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            timed_fn()
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
            timed_fn()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / (inner * calls_per_timed_invocation))

    if profile:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()

    mean_ms = statistics.mean(samples_ms)
    median_ms = statistics.median(samples_ms)
    stdev_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    measured_timed_invocation_count = iterations * inner
    measured_logical_iteration_count = (
        measured_timed_invocation_count * calls_per_timed_invocation
    )
    return {
        "warmup_count": warmup,
        "requested_sample_count": iterations,
        "sample_count": len(samples_ms),
        "min_sample_ms": min_sample_ms,
        "inner_iterations": inner,
        "logical_iterations_per_inner": calls_per_timed_invocation,
        "measured_timed_invocation_count": measured_timed_invocation_count,
        "measured_logical_iteration_count": measured_logical_iteration_count,
        "cuda_graph_repetitions": graph_repetitions,
        "uses_cuda_graph": use_cuda_graph,
        "samples_ms": samples_ms,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p90_ms": sorted(samples_ms)[int(0.9 * (len(samples_ms) - 1))],
        "stdev_ms": stdev_ms,
        "cv": stdev_ms / mean_ms if mean_ms else 0.0,
    }


def _timing_summary(timing: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if timing is None:
        return None
    return {
        "warmup_count": timing["warmup_count"],
        "sample_count": timing["sample_count"],
        "requested_sample_count": timing["requested_sample_count"],
        "mean_ms": timing["mean_ms"],
        "median_ms": timing["median_ms"],
        "stdev_ms": timing["stdev_ms"],
        "cv": timing["cv"],
        "inner_iterations": timing["inner_iterations"],
        "logical_iterations_per_inner": timing["logical_iterations_per_inner"],
        "measured_logical_iteration_count": timing["measured_logical_iteration_count"],
    }


def _bytes_model(
    rows: List[int],
    cols: int,
    dim: int,
    mode: str,
    input_dtype: torch.dtype,
) -> Dict[str, object]:
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
    input_read_bytes = input_bytes
    output_write_bytes = (
        rowwise_output_bytes
        + columnwise_output_bytes
        + rowwise_scale_bytes
        + columnwise_scale_bytes
    )
    physical = (
        input_read_bytes
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
        "estimated_physical_input_read_bytes": input_read_bytes,
        "estimated_physical_input_read_passes": 1,
        "estimated_physical_output_write_bytes": output_write_bytes,
        "estimated_physical_bytes": physical,
        "physical_byte_accounting": (
            "single input read plus requested FP8 output writes and FP32 scale-inverse writes"
        ),
        "physical_byte_accounting_counts": "read_plus_write_global_memory_traffic",
        "physical_byte_model_version": "single_read_requested_outputs_v2",
    }


def _monolithic_comparability(rows: List[int], cols: int, dim: int, mode: str) -> Dict[str, object]:
    exact_block_aligned = all(row % 128 == 0 for row in rows) and cols % 128 == 0
    uniform = all(row == rows[0] for row in rows)
    if uniform and exact_block_aligned:
        return {
            "classification": "comparable",
            "timed": True,
            "reason": (
                "Uniform exact-block case: collapsed tensor block boundaries align with "
                "group boundaries, so this is a practical upper reference for grouped "
                "throughput."
            ),
        }
    if dim == 1 and mode == "rowwise":
        return {
            "classification": "partially_comparable",
            "timed": True,
            "reason": (
                "1D rowwise scales are per row and do not cross group boundaries, but "
                "the grouped scale buffer still uses per-member padded slices."
            ),
        }
    return {
        "classification": "not_comparable",
        "timed": False,
        "reason": (
            "Collapsed monolithic quantize is not timed because jagged or partial scale "
            "blocks, columnwise split layout, or per-group boundary isolation would make "
            "the reference semantically misleading for this case."
        ),
    }


def _run_case(
    args: argparse.Namespace, case: Dict[str, object], env: Dict[str, object]
) -> Dict[str, object]:
    rows = case["rows"]
    cols = case["cols"]
    dim = case["block_scaling_dim"]
    mode = case["output_mode"]
    if not _has_same_output_manual_loop_baseline(dim, mode):
        raise ValueError(
            f"Case {case['case_label']} has no supported same-output-mode manual-loop baseline: "
            f"{_DIM2_COLUMNWISE_ONLY_UNSUPPORTED_RATIONALE}"
        )
    dtype = _dtype_from_name(args.dtype)
    fp8_dtype = _fp8_dtype_from_name(args.fp8_dtype)

    tensors = _make_inputs(rows, cols, dtype)
    grouped_input = torch.cat(tensors, dim=0)
    first_dims = torch.tensor(rows, dtype=torch.int64, device="cuda")
    quantizer = _make_quantizer(dim, mode, fp8_dtype)
    quantizers = [_make_quantizer(dim, mode, fp8_dtype) for _ in rows]

    candidate_output = tex.group_quantize(grouped_input, quantizer, len(rows), first_dims)
    baseline_outputs = [q(t) for q, t in zip(quantizers, tensors)]
    monolithic = _monolithic_comparability(rows, cols, dim, mode)
    monolithic_output = quantizer(grouped_input) if monolithic["timed"] else None

    def candidate_group_quantize_preallocated() -> object:
        return tex.group_quantize_out(grouped_input, candidate_output)

    def baseline_manual_loop_preallocated() -> None:
        for q, tensor, output in zip(quantizers, tensors, baseline_outputs):
            q.quantize(tensor, out=output)

    def monolithic_preallocated() -> None:
        quantizer.quantize(grouped_input, out=monolithic_output)

    bytes_model = _bytes_model(rows, cols, dim, mode, dtype)
    copy_calibration = _calibrate_copy_bandwidth(
        max(bytes_model["estimated_physical_bytes"], 16 * 1024 * 1024),
        max(1, args.warmup // 4),
        max(5, args.iterations // 4),
    )

    profile_this_case = args.profile and (
        args.profile_case == "all" or args.profile_case == case["case_label"]
    )
    candidate_timing = _time_callable(
        candidate_group_quantize_preallocated,
        warmup=args.warmup,
        iterations=args.iterations,
        min_sample_ms=args.min_sample_ms,
        profile=profile_this_case,
        nvtx_name=f"candidate_{case['api']}_{case['case_label']}",
        use_cuda_graph=args.use_cuda_graph,
        cuda_graph_repetitions=args.cuda_graph_repetitions,
    )
    baseline_timing = _time_callable(
        baseline_manual_loop_preallocated,
        warmup=args.warmup,
        iterations=args.iterations,
        min_sample_ms=args.min_sample_ms,
        profile=profile_this_case,
        nvtx_name=f"baseline_manual_loop_{case['case_label']}",
        use_cuda_graph=args.use_cuda_graph,
        cuda_graph_repetitions=args.cuda_graph_repetitions,
    )
    monolithic_timing = None
    if monolithic_output is not None:
        monolithic_timing = _time_callable(
            monolithic_preallocated,
            warmup=args.warmup,
            iterations=args.iterations,
            min_sample_ms=args.min_sample_ms,
            profile=profile_this_case,
            nvtx_name=f"monolithic_collapsed_{case['case_label']}",
            use_cuda_graph=args.use_cuda_graph,
            cuda_graph_repetitions=args.cuda_graph_repetitions,
        )

    candidate_seconds = candidate_timing["mean_ms"] / 1000.0
    baseline_seconds = baseline_timing["mean_ms"] / 1000.0
    monolithic_seconds = (
        monolithic_timing["mean_ms"] / 1000.0 if monolithic_timing is not None else None
    )
    candidate_gbps = bytes_model["useful_bytes"] / candidate_seconds / 1.0e9
    baseline_gbps = bytes_model["useful_bytes"] / baseline_seconds / 1.0e9
    monolithic_gbps = (
        bytes_model["useful_bytes"] / monolithic_seconds / 1.0e9
        if monolithic_seconds is not None
        else None
    )
    candidate_physical_gbps = bytes_model["estimated_physical_bytes"] / candidate_seconds / 1.0e9
    baseline_physical_gbps = bytes_model["estimated_physical_bytes"] / baseline_seconds / 1.0e9
    monolithic_physical_gbps = (
        bytes_model["estimated_physical_bytes"] / monolithic_seconds / 1.0e9
        if monolithic_seconds is not None
        else None
    )
    copy_roofline_gbps = float(copy_calibration["calibrated_copy_read_write_roofline_GBps"])
    candidate_roofline = _roofline_fraction(candidate_physical_gbps, copy_roofline_gbps)
    baseline_roofline = _roofline_fraction(baseline_physical_gbps, copy_roofline_gbps)
    monolithic_roofline = (
        _roofline_fraction(monolithic_physical_gbps, copy_roofline_gbps)
        if monolithic_physical_gbps is not None
        else {"fraction": None, "raw_fraction": None, "valid": None, "invalid_reason": None}
    )
    speedup = baseline_seconds / candidate_seconds
    monolithic_ratio = (
        candidate_gbps / monolithic_gbps
        if monolithic_gbps is not None and monolithic_gbps > 0
        else None
    )
    row_tile_evidence = _row_tile_launch_evidence(rows, cols)
    candidate_measured_quantize_requests = candidate_timing["measured_logical_iteration_count"]
    baseline_measured_quantize_requests = (
        baseline_timing["measured_logical_iteration_count"] * len(rows)
    )
    monolithic_measured_quantize_requests = (
        monolithic_timing["measured_logical_iteration_count"]
        if monolithic_timing is not None
        else 0
    )

    rowwise, columnwise = _mode_to_usage(mode)
    record = {
        **env,
        **case,
        "record_type": "steady_state_preallocated_grouped_quantize",
        "primary_performance_case": True,
        "primary_evidence_layer": "direct_extension_binding_group_quantize_out",
        "primary_evidence_excludes": [
            "pytorch_module",
            "grouped_linear",
            "autograd",
            "training_loop",
        ],
        "input_dtype": str(dtype).replace("torch.", ""),
        "fp8_dtype": args.fp8_dtype,
        "force_pow_2_scales": True,
        "amax_epsilon": 0.0,
        "candidate_output_mode": mode,
        "candidate_quantizer_rowwise": rowwise,
        "candidate_quantizer_columnwise": columnwise,
        "baseline_output_mode": mode,
        "baseline_quantizer_rowwise": rowwise,
        "baseline_quantizer_columnwise": columnwise,
        "same_output_mode_manual_loop_baseline": True,
        "num_groups": len(rows),
        "rows": rows,
        "cols": cols,
        "tensor_rows": rows,
        "tensor_cols": cols,
        "tensor_elements": [row_count * cols for row_count in rows],
        "total_tensor_elements": sum(rows) * cols,
        "min_tensor_rows": min(rows),
        "max_tensor_rows": max(rows),
        "first_dims": rows,
        "tensor_offsets": [0] + [sum(rows[: i + 1]) * cols for i in range(len(rows))],
        "rowwise_scale_shapes": [_scale_shape(r, cols, dim, False) for r in rows]
        if rowwise
        else [],
        "columnwise_scale_shapes": [_scale_shape(r, cols, dim, True) for r in rows]
        if columnwise
        else [],
        **bytes_model,
        **copy_calibration,
        "calibrated_copy_roofline_GBps": copy_roofline_gbps,
        "calibrated_copy_roofline_accounting": "read_plus_write_global_memory_traffic",
        "calibrated_copy_roofline_matches_physical_byte_accounting": True,
        "roofline_fraction_required": args.require_speed_of_light,
        "candidate_timing": candidate_timing,
        "baseline_timing": baseline_timing,
        "monolithic_timing": monolithic_timing,
        "timing_stability": {
            "candidate": _timing_summary(candidate_timing),
            "manual_loop_baseline": _timing_summary(baseline_timing),
            "monolithic_collapsed_reference": _timing_summary(monolithic_timing),
            "validity_policy": (
                "Unexplained high variance, large baseline drift, impossible roofline "
                "fractions, or erratic adjacent-size performance are benchmark validity "
                "alarms and should not be interpreted as optimization wins."
            ),
        },
        "measurement_counts": {
            "warmup_count": args.warmup,
            "requested_sample_count": args.iterations,
            "candidate_measured_grouped_quantize_requests": (
                candidate_measured_quantize_requests
            ),
            "baseline_measured_single_tensor_quantize_requests": (
                baseline_measured_quantize_requests
            ),
            "monolithic_measured_quantize_requests": monolithic_measured_quantize_requests,
            "candidate_quantize_requests_per_logical_iteration": 1,
            "baseline_quantize_requests_per_logical_iteration": len(rows),
            "monolithic_quantize_requests_per_logical_iteration": (
                1 if monolithic_timing is not None else 0
            ),
        },
        "bandwidth_GBps_actual_bytes": candidate_gbps,
        "baseline_bandwidth_GBps_actual_bytes": baseline_gbps,
        "monolithic_bandwidth_GBps_actual_bytes": monolithic_gbps,
        "candidate_physical_bandwidth_GBps": candidate_physical_gbps,
        "baseline_physical_bandwidth_GBps": baseline_physical_gbps,
        "monolithic_physical_bandwidth_GBps": monolithic_physical_gbps,
        "candidate_speedup_over_manual_loop": speedup,
        "candidate_ratio_to_monolithic": monolithic_ratio,
        "monolithic_comparability": monolithic["classification"],
        "monolithic_comparability_reason": monolithic["reason"],
        "monolithic_reference_timed": monolithic["timed"],
        "primary_measurement_region": "steady_state_preallocated_cuda_graph"
        if args.use_cuda_graph
        else "steady_state_preallocated",
        "excludes_setup_allocation_and_output_construction": True,
        "preallocated_candidate_output": True,
        "preallocated_manual_loop_outputs": True,
        "candidate_fraction_of_calibrated_roofline": candidate_roofline["fraction"],
        "baseline_fraction_of_calibrated_roofline": baseline_roofline["fraction"],
        "monolithic_fraction_of_calibrated_roofline": monolithic_roofline["fraction"],
        "candidate_raw_fraction_of_calibrated_roofline": candidate_roofline["raw_fraction"],
        "baseline_raw_fraction_of_calibrated_roofline": baseline_roofline["raw_fraction"],
        "monolithic_raw_fraction_of_calibrated_roofline": monolithic_roofline["raw_fraction"],
        "candidate_roofline_fraction_valid": candidate_roofline["valid"],
        "baseline_roofline_fraction_valid": baseline_roofline["valid"],
        "monolithic_roofline_fraction_valid": monolithic_roofline["valid"],
        "roofline_fraction_valid": candidate_roofline["valid"]
        and baseline_roofline["valid"]
        and (monolithic_roofline["valid"] is not False),
        "candidate_roofline_invalid_reason": candidate_roofline["invalid_reason"],
        "baseline_roofline_invalid_reason": baseline_roofline["invalid_reason"],
        "monolithic_roofline_invalid_reason": monolithic_roofline["invalid_reason"],
        "invalid_roofline_fraction_policy": (
            "Records with raw candidate or baseline fraction above 1.0 are not valid roofline "
            "evidence; bounded fraction fields are set to null instead of reporting an "
            "impossible speed-of-light value."
        ),
        "launch_evidence": {
            "candidate_path": "group_quantize_out_preallocated_fp8_block",
            "candidate_frontdoor_api": case["api"],
            "candidate_grouped_quantize_requests_per_logical_iteration": 1,
            "baseline_path": "manual_loop_single_tensor_quantizer_preallocated_outputs",
            "baseline_quantize_requests_per_logical_iteration": len(rows),
            "monolithic_path": "single_collapsed_tensor_quantizer_preallocated_output",
            "monolithic_quantize_requests_per_logical_iteration": 1
            if monolithic_timing is not None
            else 0,
            "candidate_measured_grouped_quantize_requests": (
                candidate_measured_quantize_requests
            ),
            "baseline_measured_single_tensor_quantize_requests": (
                baseline_measured_quantize_requests
            ),
            "monolithic_measured_quantize_requests": monolithic_measured_quantize_requests,
            "cuda_graph_repetitions": candidate_timing["cuda_graph_repetitions"],
            "profiler_requested": profile_this_case,
            **row_tile_evidence,
        },
        "baseline_stability": {
            "sample_count": len(baseline_timing["samples_ms"]),
            "mean_ms": baseline_timing["mean_ms"],
            "median_ms": baseline_timing["median_ms"],
            "stdev_ms": baseline_timing["stdev_ms"],
            "cv": baseline_timing["cv"],
            "warmup_count": baseline_timing["warmup_count"],
            "measured_single_tensor_quantize_requests": baseline_measured_quantize_requests,
        },
    }
    return record


def _skipped_primary_case_record(
    args: argparse.Namespace, case: Dict[str, object], env: Dict[str, object]
) -> Dict[str, object]:
    rows = case["rows"]
    cols = case["cols"]
    dim = case["block_scaling_dim"]
    mode = case["output_mode"]
    rowwise, columnwise = _mode_to_usage(mode)
    return {
        **env,
        **case,
        "record_type": "skipped_primary_case",
        "primary_performance_case": False,
        "input_dtype": str(_dtype_from_name(args.dtype)).replace("torch.", ""),
        "fp8_dtype": args.fp8_dtype,
        "num_groups": len(rows),
        "rows": rows,
        "cols": cols,
        "tensor_rows": rows,
        "tensor_cols": cols,
        "tensor_elements": [row_count * cols for row_count in rows],
        "total_tensor_elements": sum(rows) * cols,
        "min_tensor_rows": min(rows),
        "max_tensor_rows": max(rows),
        "first_dims": rows,
        "candidate_output_mode": mode,
        "candidate_quantizer_rowwise": rowwise,
        "candidate_quantizer_columnwise": columnwise,
        "baseline_output_mode": None,
        "baseline_quantizer_rowwise": None,
        "baseline_quantizer_columnwise": None,
        "would_require_unsupported_baseline_output_mode": mode,
        "same_output_mode_manual_loop_baseline": False,
        "manual_loop_baseline_available": False,
        "monolithic_comparability": "not_comparable",
        "monolithic_reference_timed": False,
        "monolithic_bandwidth_GBps_actual_bytes": None,
        "candidate_ratio_to_monolithic": None,
        "candidate_not_timed": True,
        "baseline_not_timed": True,
        "excluded_from_primary_speedup": True,
        "skip_reason": _DIM2_COLUMNWISE_ONLY_UNSUPPORTED_BASELINE,
        "unsupported_mode_rationale": _DIM2_COLUMNWISE_ONLY_UNSUPPORTED_RATIONALE,
        "unsupported_baseline_quantizer": {
            "block_scaling_dim": dim,
            "rowwise": rowwise,
            "columnwise": columnwise,
        },
        "invalid_alternative_not_used": (
            "The benchmark does not use a both-output manual-loop baseline for this "
            "columnwise-only candidate because that would add rowwise output work to "
            "the baseline and invalidate the primary speedup comparison."
        ),
    }


def _run_grouped_linear_case(
    args: argparse.Namespace, case: Dict[str, object], env: Dict[str, object]
) -> Dict[str, object]:
    rows = case["rows"]
    cols = case["cols"]
    dim = case["block_scaling_dim"]
    dtype = _dtype_from_name(args.dtype)
    out_features = args.grouped_linear_out_features or cols
    fp8_recipe = _make_fp8_block_recipe(dim)

    grouped_linear = te.GroupedLinear(
        len(rows),
        cols,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    ).train()
    inp = torch.randn(
        (sum(rows), cols),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )

    def grouped_linear_forward_backward() -> None:
        grouped_linear.zero_grad(set_to_none=True)
        if inp.grad is not None:
            inp.grad.zero_()
        with te.autocast(enabled=True, recipe=fp8_recipe):
            out = grouped_linear(inp, rows)
        out.float().sum().backward()

    profile_this_case = args.profile and (
        args.profile_case == "all" or args.profile_case == case["case_label"]
    )
    timing = _time_callable(
        grouped_linear_forward_backward,
        warmup=max(1, args.warmup // 2),
        iterations=max(1, args.iterations // 4),
        min_sample_ms=args.min_sample_ms,
        profile=profile_this_case,
        nvtx_name=f"grouped_linear_e2e_{case['case_label']}",
        use_cuda_graph=False,
        cuda_graph_repetitions=1,
    )
    seconds = timing["mean_ms"] / 1000.0
    tokens_per_second = sum(rows) / seconds if seconds > 0 else 0.0
    return {
        **env,
        **case,
        "record_type": "grouped_linear_end_to_end",
        "input_dtype": str(dtype).replace("torch.", ""),
        "num_groups": len(rows),
        "rows": rows,
        "cols": cols,
        "tensor_rows": rows,
        "tensor_cols": cols,
        "tensor_elements": [row_count * cols for row_count in rows],
        "total_tensor_elements": sum(rows) * cols,
        "out_features": out_features,
        "x_block_scaling_dim": fp8_recipe.x_block_scaling_dim,
        "w_block_scaling_dim": fp8_recipe.w_block_scaling_dim,
        "grad_block_scaling_dim": fp8_recipe.grad_block_scaling_dim,
        "measurement_region": "end_to_end_grouped_linear_forward_backward_api",
        "timing": timing,
        "tokens_per_second": tokens_per_second,
        "notes": (
            "Secondary end-to-end GroupedLinear coverage for the Float8BlockScaling "
            "split_quantize callsite. This includes module/autograd/GEMM overhead and is "
            "not the primary grouped quantize kernel-throughput metric."
        ),
    }


def _skipped_grouped_linear_case_record(
    args: argparse.Namespace, case: Dict[str, object], env: Dict[str, object]
) -> Dict[str, object]:
    rows = case["rows"]
    cols = case["cols"]
    dim = case["block_scaling_dim"]
    dtype = _dtype_from_name(args.dtype)
    out_features = args.grouped_linear_out_features or cols
    fp8_recipe = _make_fp8_block_recipe(dim)
    invalid_indices = _grouped_linear_invalid_m_split_indices(rows)
    valid_indices = [
        i
        for i, row_count in enumerate(rows)
        if row_count > 0
        and row_count % _GROUPED_LINEAR_FP8_BLOCK_SCALING_M_SPLIT_MULTIPLE == 0
    ]
    return {
        **env,
        **case,
        "record_type": "skipped_grouped_linear_case",
        "primary_performance_case": False,
        "secondary_grouped_linear_case": True,
        "input_dtype": str(dtype).replace("torch.", ""),
        "num_groups": len(rows),
        "rows": rows,
        "cols": cols,
        "tensor_rows": rows,
        "tensor_cols": cols,
        "tensor_elements": [row_count * cols for row_count in rows],
        "total_tensor_elements": sum(rows) * cols,
        "out_features": out_features,
        "x_block_scaling_dim": fp8_recipe.x_block_scaling_dim,
        "w_block_scaling_dim": fp8_recipe.w_block_scaling_dim,
        "grad_block_scaling_dim": fp8_recipe.grad_block_scaling_dim,
        "measurement_region": "end_to_end_grouped_linear_forward_backward_api",
        "grouped_linear_not_timed": True,
        "excluded_from_primary_speedup": True,
        "skip_reason": _GROUPED_LINEAR_FP8_BLOCK_SCALING_UNSUPPORTED_SPLITS,
        "unsupported_mode_rationale": _GROUPED_LINEAR_FP8_BLOCK_SCALING_UNSUPPORTED_RATIONALE,
        "required_m_split_multiple": _GROUPED_LINEAR_FP8_BLOCK_SCALING_M_SPLIT_MULTIPLE,
        "invalid_m_split_indices": invalid_indices,
        "invalid_m_splits": [rows[i] for i in invalid_indices],
        "valid_m_split_indices": valid_indices,
        "valid_m_splits": [rows[i] for i in valid_indices],
        "non_empty_m_splits": [row_count for row_count in rows if row_count > 0],
        "skipped_before_cuda_execution": True,
        "notes": (
            "Secondary GroupedLinear coverage is skipped for this requested split layout "
            "because the current Float8BlockScaling grouped GEMM path rejects padded "
            "scale-factor rows when any non-empty m_split is not divisible by 4."
        ),
    }


def _environment(
    command: str,
    *,
    worker_count: int = 1,
    worker_index: Optional[int] = None,
    selected_devices: Optional[List[str]] = None,
    per_worker_commands: Optional[List[str]] = None,
    per_worker_artifact_paths: Optional[List[str]] = None,
    merge_validation: str = "single worker writes the required raw report directly",
    sharding_strategy: str = "single_worker",
    sharding_rationale: str = "single worker execution",
) -> Dict[str, object]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    slurm_job_gpus = os.getenv("SLURM_JOB_GPUS", "")
    slurm_gpus_on_node = os.getenv("SLURM_GPUS_ON_NODE", "")
    visible_gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    selected_device = (
        cuda_visible_devices.split(",")[current_device]
        if cuda_visible_devices
        else str(current_device)
    )
    if selected_devices is None:
        selected_devices = [selected_device]
    if per_worker_commands is None:
        per_worker_commands = [command]
    if per_worker_artifact_paths is None:
        per_worker_artifact_paths = [os.getenv("ORCHESTRA_BENCHMARK_RAW_REPORT", "")]
    allocated_gpu_count = _slurm_gpu_count(slurm_job_gpus)
    if allocated_gpu_count == 0 and slurm_gpus_on_node.isdigit():
        allocated_gpu_count = int(slurm_gpus_on_node)
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
        "scheduler_allocated_gpus": slurm_job_gpus,
        "scheduler_allocated_gpu_count": allocated_gpu_count,
        "slurm_gpus_on_node": slurm_gpus_on_node,
        "cuda_visible_devices": cuda_visible_devices,
        "visible_gpu_count": visible_gpu_count,
        "selected_devices": selected_devices,
        "worker_selected_device": selected_device,
        "benchmark_worker_count": worker_count,
        "benchmark_worker_index": worker_index,
        "benchmark_sharding_strategy": sharding_strategy,
        "benchmark_sharding_rationale": sharding_rationale,
        "per_worker_commands": per_worker_commands,
        "per_worker_artifact_paths": per_worker_artifact_paths,
        "merge_validation": merge_validation,
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


def _build_grouped_linear_cases(args: argparse.Namespace) -> List[Dict[str, object]]:
    cases = []
    for dim in _csv_ints(args.dims):
        for layout in _csv_strings(args.grouped_linear_layouts):
            for num_groups in _csv_ints(args.grouped_linear_num_groups):
                for rows_base in _csv_ints(args.grouped_linear_rows_sweep):
                    for cols in _csv_ints(args.grouped_linear_cols):
                        rows = _make_rows(layout, num_groups, rows_base)
                        label = f"grouped_linear_d{dim}_{layout}_g{num_groups}_r{rows_base}_c{cols}"
                        cases.append(
                            {
                                "api": "grouped_linear",
                                "block_scaling_dim": dim,
                                "layout": layout,
                                "rows_base": rows_base,
                                "rows": rows,
                                "cols": cols,
                                "case_label": label,
                            }
                        )
    return cases


def _max_or_none(values: List[float]) -> Optional[float]:
    return max(values) if values else None


def _build_report_diagnostics(records: List[Dict[str, object]]) -> Dict[str, object]:
    high_cv_threshold = 0.10
    adjacent_drop_ratio_threshold = 0.50
    adjacent_jump_ratio_threshold = 2.00
    max_reported_adjacent_anomalies = 256

    candidate_cvs = [
        float(record["candidate_timing"]["cv"])
        for record in records
        if record.get("candidate_timing") is not None
    ]
    baseline_cvs = [
        float(record["baseline_timing"]["cv"])
        for record in records
        if record.get("baseline_timing") is not None
    ]
    monolithic_cvs = [
        float(record["monolithic_timing"]["cv"])
        for record in records
        if record.get("monolithic_timing") is not None
    ]
    high_cv_records = []
    for record in records:
        for role, timing_key in (
            ("candidate", "candidate_timing"),
            ("manual_loop_baseline", "baseline_timing"),
            ("monolithic_collapsed_reference", "monolithic_timing"),
        ):
            timing = record.get(timing_key)
            if timing is not None and float(timing["cv"]) > high_cv_threshold:
                high_cv_records.append(
                    {
                        "case_label": record["case_label"],
                        "role": role,
                        "cv": timing["cv"],
                        "sample_count": timing["sample_count"],
                        "mean_ms": timing["mean_ms"],
                        "stdev_ms": timing["stdev_ms"],
                    }
                )

    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for record in records:
        key = (
            record.get("api"),
            record.get("block_scaling_dim"),
            record.get("output_mode"),
            record.get("layout"),
            record.get("num_groups"),
            record.get("cols"),
            record.get("input_dtype"),
            record.get("fp8_dtype"),
        )
        grouped.setdefault(key, []).append(record)

    adjacent_anomalies = []
    adjacent_pair_count = 0
    adjacent_anomaly_count = 0
    for key_records in grouped.values():
        sorted_records = sorted(key_records, key=lambda record: int(record["rows_base"]))
        for prev_record, curr_record in zip(sorted_records, sorted_records[1:]):
            prev_bw = float(prev_record["bandwidth_GBps_actual_bytes"])
            curr_bw = float(curr_record["bandwidth_GBps_actual_bytes"])
            if prev_bw <= 0 or curr_bw <= 0:
                continue
            adjacent_pair_count += 1
            candidate_ratio = curr_bw / prev_bw
            if (
                candidate_ratio < adjacent_drop_ratio_threshold
                or candidate_ratio > adjacent_jump_ratio_threshold
            ):
                adjacent_anomaly_count += 1
                if len(adjacent_anomalies) < max_reported_adjacent_anomalies:
                    prev_baseline_bw = float(prev_record["baseline_bandwidth_GBps_actual_bytes"])
                    curr_baseline_bw = float(curr_record["baseline_bandwidth_GBps_actual_bytes"])
                    baseline_ratio = (
                        curr_baseline_bw / prev_baseline_bw if prev_baseline_bw > 0 else None
                    )
                    adjacent_anomalies.append(
                        {
                            "previous_case_label": prev_record["case_label"],
                            "current_case_label": curr_record["case_label"],
                            "block_scaling_dim": curr_record["block_scaling_dim"],
                            "output_mode": curr_record["output_mode"],
                            "layout": curr_record["layout"],
                            "num_groups": curr_record["num_groups"],
                            "cols": curr_record["cols"],
                            "previous_rows_base": prev_record["rows_base"],
                            "current_rows_base": curr_record["rows_base"],
                            "candidate_bandwidth_ratio_current_over_previous": (
                                candidate_ratio
                            ),
                            "baseline_bandwidth_ratio_current_over_previous": baseline_ratio,
                            "previous_candidate_cv": prev_record["candidate_timing"]["cv"],
                            "current_candidate_cv": curr_record["candidate_timing"]["cv"],
                            "previous_baseline_cv": prev_record["baseline_timing"]["cv"],
                            "current_baseline_cv": curr_record["baseline_timing"]["cv"],
                        }
                    )

    return {
        "stability": {
            "high_cv_threshold": high_cv_threshold,
            "candidate_high_cv_count": sum(cv > high_cv_threshold for cv in candidate_cvs),
            "baseline_high_cv_count": sum(cv > high_cv_threshold for cv in baseline_cvs),
            "monolithic_high_cv_count": sum(cv > high_cv_threshold for cv in monolithic_cvs),
            "max_candidate_cv": _max_or_none(candidate_cvs),
            "max_baseline_cv": _max_or_none(baseline_cvs),
            "max_monolithic_cv": _max_or_none(monolithic_cvs),
            "high_cv_records": high_cv_records[:max_reported_adjacent_anomalies],
            "high_cv_records_truncated": len(high_cv_records)
            > max_reported_adjacent_anomalies,
            "baseline_drift_proxy": (
                "manual-loop baseline CV is used as the first-pass same-session drift "
                "proxy; unexplained high baseline CV is invalid benchmark evidence."
            ),
        },
        "adjacent_size": {
            "grouping_key": [
                "api",
                "block_scaling_dim",
                "output_mode",
                "layout",
                "num_groups",
                "cols",
                "input_dtype",
                "fp8_dtype",
            ],
            "rows_axis": "rows_base",
            "adjacent_pair_count": adjacent_pair_count,
            "drop_ratio_alarm_threshold": adjacent_drop_ratio_threshold,
            "jump_ratio_alarm_threshold": adjacent_jump_ratio_threshold,
            "adjacent_anomaly_count": adjacent_anomaly_count,
            "adjacent_anomalies": adjacent_anomalies,
            "adjacent_anomalies_truncated": adjacent_anomaly_count
            > len(adjacent_anomalies),
        },
        "validity_policy": (
            "Unexplained high variance, large manual-loop baseline drift, large "
            "baseline/candidate path mismatch, invalid roofline fractions above 1.0, "
            "or erratic adjacent-size behavior is a benchmark validity alarm, not a "
            "performance win."
        ),
    }


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
    parser.add_argument(
        "--use-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay preallocated steady-state quantize calls through CUDA graphs.",
    )
    parser.add_argument("--cuda-graph-repetitions", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-case", default="all")
    parser.add_argument("--require-speed-of-light", action="store_true")
    parser.add_argument("--launch-evidence", default="internal")
    parser.add_argument(
        "--shard-across-gpus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard independent benchmark cases across scheduler-allocated visible GPUs.",
    )
    parser.add_argument("--worker-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--include-grouped-linear",
        action="store_true",
        help=(
            "Also run secondary end-to-end GroupedLinear cases. These records include "
            "module/autograd/GEMM overhead and are excluded from the primary grouped "
            "quantize kernel gate."
        ),
    )
    parser.add_argument("--grouped-linear-layouts", default="uniform,jagged")
    parser.add_argument("--grouped-linear-num-groups", default="2,8,16")
    parser.add_argument("--grouped-linear-rows-sweep", default="128,2048,8192")
    parser.add_argument("--grouped-linear-cols", default="1024,4096")
    parser.add_argument("--grouped-linear-out-features", type=int, default=0)
    parser.add_argument("--output", default=os.getenv("ORCHESTRA_BENCHMARK_RAW_REPORT", ""))
    return parser.parse_args()


def _merge_worker_reports(
    reports: List[Dict[str, object]],
    *,
    selected_devices: List[str],
    worker_commands: List[str],
    worker_paths: List[str],
) -> Dict[str, object]:
    if not reports:
        raise RuntimeError("No worker reports were produced.")

    merged = dict(reports[0])
    records = []
    skipped_records = []
    grouped_linear_records = []
    skipped_grouped_linear_records = []
    for report in reports:
        records.extend(report.get("records", []))
        skipped_records.extend(report.get("skipped_records", []))
        grouped_linear_records.extend(report.get("grouped_linear_records", []))
        skipped_grouped_linear_records.extend(report.get("skipped_grouped_linear_records", []))

    requested_cases = sum(int(report["summary"]["num_requested_cases"]) for report in reports)
    requested_grouped_linear_cases = sum(
        int(report["summary"]["num_requested_grouped_linear_cases"]) for report in reports
    )
    observed_cases = len(records) + len(skipped_records)
    observed_grouped_linear_cases = len(grouped_linear_records) + len(
        skipped_grouped_linear_records
    )
    if observed_cases != requested_cases:
        raise RuntimeError(
            f"Merged primary case count mismatch: observed {observed_cases}, "
            f"expected {requested_cases}."
        )
    if observed_grouped_linear_cases != requested_grouped_linear_cases:
        raise RuntimeError(
            "Merged GroupedLinear case count mismatch: observed "
            f"{observed_grouped_linear_cases}, expected {requested_grouped_linear_cases}."
        )

    summary = dict(merged["summary"])
    summary.update(
        {
            "num_cases": len(records),
            "num_requested_cases": requested_cases,
            "num_skipped_cases": len(skipped_records),
            "num_requested_grouped_linear_cases": requested_grouped_linear_cases,
            "num_grouped_linear_cases": len(grouped_linear_records),
            "num_skipped_grouped_linear_cases": len(skipped_grouped_linear_records),
            "benchmark_worker_count": len(reports),
            "selected_devices": selected_devices,
            "merge_validation": (
                "all worker outputs were non-empty and primary/grouped-linear requested "
                "case counts matched merged record plus skipped-record counts"
            ),
            "roofline_invalid_record_count": sum(
                1 for record in records if not record["roofline_fraction_valid"]
            ),
        }
    )
    merged["summary"] = summary
    merged["records"] = records
    merged["skipped_records"] = skipped_records
    merged["grouped_linear_records"] = grouped_linear_records
    merged["skipped_grouped_linear_records"] = skipped_grouped_linear_records
    merged["diagnostics"] = _build_report_diagnostics(records)
    merged["sharding"] = {
        "scheduler_allocated_gpus": os.getenv("SLURM_JOB_GPUS", ""),
        "scheduler_allocated_gpu_count": len(selected_devices),
        "visible_gpu_count_at_parent": torch.cuda.device_count(),
        "selected_devices": selected_devices,
        "worker_count": len(reports),
        "sharding_strategy": "case_index_modulo_scheduler_allocated_gpu",
        "cpu_thread_limits": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "per_worker_commands": worker_commands,
        "per_worker_artifact_paths": worker_paths,
        "merge_validation": summary["merge_validation"],
    }
    return merged


def _run_sharded_parent(args: argparse.Namespace, selected_devices: List[str]) -> None:
    output = args.output
    temp_root = tempfile.mkdtemp(
        prefix="grouped_fp8_block_quantize_",
        dir=os.path.dirname(output) if output and os.path.isdir(os.path.dirname(output)) else None,
    )
    worker_count = len(selected_devices)
    worker_paths = [
        os.path.join(temp_root, f"worker_{worker_idx:02d}.json")
        for worker_idx in range(worker_count)
    ]
    worker_commands = []
    workers = []
    for worker_idx, (device, worker_path) in enumerate(zip(selected_devices, worker_paths)):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            *sys.argv[1:],
            "--no-shard-across-gpus",
            "--worker-index",
            str(worker_idx),
            "--worker-count",
            str(worker_count),
            "--output",
            worker_path,
        ]
        worker_commands.append(" ".join(cmd))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        workers.append((worker_idx, cmd, subprocess.Popen(cmd, env=env)))

    for worker_idx, cmd, proc in workers:
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(
                f"Benchmark worker {worker_idx} failed with return code {returncode}: "
                f"{' '.join(cmd)}"
            )

    reports = []
    for worker_path in worker_paths:
        if not os.path.exists(worker_path) or os.path.getsize(worker_path) == 0:
            raise RuntimeError(f"Worker output is missing or empty: {worker_path}")
        with open(worker_path, "r", encoding="utf-8") as f:
            reports.append(json.load(f))

    merged = _merge_worker_reports(
        reports,
        selected_devices=selected_devices,
        worker_commands=worker_commands,
        worker_paths=worker_paths,
    )
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
            f.write("\n")
    else:
        print(json.dumps(merged, indent=2))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not available:
        raise RuntimeError(f"FP8 block scaling is not available: {reason}")

    selected_devices = _scheduler_selected_devices()
    if (
        args.worker_index < 0
        and args.shard_across_gpus
        and not args.profile
        and len(selected_devices) > 1
    ):
        _run_sharded_parent(args, selected_devices)
        return

    sharding_strategy = "single_worker"
    sharding_rationale = "single worker execution"
    merge_validation = "single worker writes the required raw report directly"
    if args.worker_count > 1:
        sharding_strategy = "case_index_modulo_scheduler_allocated_gpu_worker"
        sharding_rationale = (
            "This worker receives cases whose case index modulo worker_count equals "
            "worker_index; artifacts are isolated per worker and merged by the parent."
        )
        merge_validation = "worker writes one non-empty shard report for parent merge"
    elif args.profile:
        sharding_strategy = "single_worker_profiler_capture"
        sharding_rationale = (
            "Profiler capture is intentionally single-worker so Nsight Systems and Nsight "
            "Compute attribution is not contaminated by concurrent independent workers."
        )
    elif len(selected_devices) <= 1:
        sharding_strategy = "single_scheduler_allocated_gpu"
        sharding_rationale = (
            "No multi-GPU Slurm allocation was detected; the benchmark uses the one "
            "scheduler-allocated or visible GPU available to the job."
        )

    env = _environment(
        " ".join(sys.argv),
        worker_count=args.worker_count,
        worker_index=args.worker_index if args.worker_index >= 0 else None,
        selected_devices=(
            selected_devices if args.worker_count == 1 else _scheduler_selected_devices()
        ),
        merge_validation=merge_validation,
        sharding_strategy=sharding_strategy,
        sharding_rationale=sharding_rationale,
    )
    cases = _build_cases(args)
    if args.worker_count > 1:
        cases = [
            case
            for case_idx, case in enumerate(cases)
            if case_idx % args.worker_count == args.worker_index
        ]
    records = []
    skipped_records = []
    torch.set_grad_enabled(False)
    for case in cases:
        if _has_same_output_manual_loop_baseline(
            case["block_scaling_dim"], case["output_mode"]
        ):
            records.append(_run_case(args, case, env))
        else:
            skipped_records.append(_skipped_primary_case_record(args, case, env))
    grouped_linear_records = []
    skipped_grouped_linear_records = []
    grouped_linear_cases = []
    if args.include_grouped_linear:
        grouped_linear_cases = _build_grouped_linear_cases(args)
        if args.worker_count > 1:
            grouped_linear_cases = [
                case
                for case_idx, case in enumerate(grouped_linear_cases)
                if case_idx % args.worker_count == args.worker_index
            ]
        torch.set_grad_enabled(True)
        for case in grouped_linear_cases:
            if _grouped_linear_invalid_m_split_indices(case["rows"]):
                skipped_grouped_linear_records.append(
                    _skipped_grouped_linear_case_record(args, case, env)
                )
            else:
                grouped_linear_records.append(_run_grouped_linear_case(args, case, env))
        torch.set_grad_enabled(False)

    report = {
        "schema_version": "grouped_fp8_block_quantize_benchmark/v1",
        "summary": {
            "num_cases": len(records),
            "num_requested_cases": len(cases),
            "num_skipped_cases": len(skipped_records),
            "num_requested_grouped_linear_cases": len(grouped_linear_cases),
            "num_grouped_linear_cases": len(grouped_linear_records),
            "num_skipped_grouped_linear_cases": len(skipped_grouped_linear_records),
            "apis": _csv_strings(args.api),
            "dims": _csv_ints(args.dims),
            "output_modes": _csv_strings(args.output_modes),
            "layouts": _csv_strings(args.layouts),
            "primary_record_type": "steady_state_preallocated_grouped_quantize",
            "primary_benchmark_layer": "direct_extension_binding_group_quantize_out",
            "primary_benchmark_excludes": [
                "pytorch_module",
                "grouped_linear",
                "autograd",
                "training_loop",
            ],
            "skipped_record_type": "skipped_primary_case",
            "skipped_grouped_linear_record_type": "skipped_grouped_linear_case",
            "unsupported_primary_case_policy": (
                "dim=2 output_mode=columnwise primary performance cases are skipped "
                "because the same-output-mode non-grouped manual-loop baseline is unsupported."
            ),
            "unsupported_primary_case_reason": _DIM2_COLUMNWISE_ONLY_UNSUPPORTED_BASELINE,
            "unsupported_grouped_linear_case_policy": (
                "Secondary GroupedLinear Float8BlockScaling cases are skipped when any "
                "non-empty m_split is not divisible by 4."
            ),
            "unsupported_grouped_linear_case_reason": (
                _GROUPED_LINEAR_FP8_BLOCK_SCALING_UNSUPPORTED_SPLITS
            ),
            "grouped_linear_required_m_split_multiple": (
                _GROUPED_LINEAR_FP8_BLOCK_SCALING_M_SPLIT_MULTIPLE
            ),
            "primary_measurement_region": "steady_state_preallocated_cuda_graph"
            if args.use_cuda_graph
            else "steady_state_preallocated",
            "grouped_linear_records_are_secondary": True,
            "roofline_byte_model_version": "single_read_requested_outputs_v2",
            "roofline_copy_calibration_accounting": "read_plus_write_global_memory_traffic",
            "roofline_invalid_record_count": sum(
                1 for record in records if not record["roofline_fraction_valid"]
            ),
        },
        "records": records,
        "skipped_records": skipped_records,
        "grouped_linear_records": grouped_linear_records,
        "skipped_grouped_linear_records": skipped_grouped_linear_records,
        "diagnostics": _build_report_diagnostics(records),
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
