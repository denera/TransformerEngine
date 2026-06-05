#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped FP8 block-scaling quantize benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional

import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import Float8BlockQuantizer
import transformer_engine_torch as tex


def _parse_csv_ints(value: str) -> List[int]:
    return [int(part) for part in value.split(",") if part]


def _make_quantizer(block_scaling_dim: int, rowwise: bool, columnwise: bool):
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=block_scaling_dim,
    )


def _assert_same(outputs, references) -> None:
    for got, ref in zip(outputs, references):
        _assert_optional_equal("_rowwise_data", got._rowwise_data, ref._rowwise_data)
        _assert_optional_equal("_columnwise_data", got._columnwise_data, ref._columnwise_data)
        shape = tuple(got.size())
        quantizer = got._quantizer
        _assert_scale_equal(
            "_rowwise_scale_inv",
            got._rowwise_scale_inv,
            ref._rowwise_scale_inv,
            shape,
            quantizer,
            False,
        )
        _assert_scale_equal(
            "_columnwise_scale_inv",
            got._columnwise_scale_inv,
            ref._columnwise_scale_inv,
            shape,
            quantizer,
            True,
        )


def _assert_optional_equal(name: str, got_tensor, ref_tensor) -> None:
    if ref_tensor is None:
        if got_tensor is not None:
            raise AssertionError(f"{name} unexpectedly present")
        return
    if got_tensor is None:
        raise AssertionError(f"{name} missing")
    if got_tensor.shape != ref_tensor.shape:
        raise AssertionError(f"{name} shape mismatch: {got_tensor.shape} != {ref_tensor.shape}")
    torch.testing.assert_close(got_tensor, ref_tensor, rtol=0, atol=0)


def _valid_scale_shape(shape, quantizer, columnwise: bool):
    rows = math.prod(shape[:-1])
    cols = shape[-1] if shape else 1
    block_len = quantizer.block_len
    row_blocks = (rows + block_len - 1) // block_len
    col_blocks = (cols + block_len - 1) // block_len

    if quantizer.block_scaling_dim == 2:
        return (col_blocks, row_blocks) if columnwise else (row_blocks, col_blocks)
    return (row_blocks, cols) if columnwise else (col_blocks, rows)


def _assert_scale_equal(name: str, got_tensor, ref_tensor, shape, quantizer, columnwise: bool) -> None:
    if ref_tensor is None:
        if got_tensor is not None:
            raise AssertionError(f"{name} unexpectedly present")
        return
    if got_tensor is None:
        raise AssertionError(f"{name} missing")
    if got_tensor.shape != ref_tensor.shape:
        raise AssertionError(f"{name} shape mismatch: {got_tensor.shape} != {ref_tensor.shape}")
    valid_rows, valid_cols = _valid_scale_shape(shape, quantizer, columnwise)
    torch.testing.assert_close(
        got_tensor[:valid_rows, :valid_cols],
        ref_tensor[:valid_rows, :valid_cols],
        rtol=0,
        atol=0,
    )


def _case_splits(num_groups: int, rows_per_group: int, jagged: bool) -> List[int]:
    if not jagged:
        return [rows_per_group] * num_groups
    pattern = [512, 1024, 256, 2048, 768, 128, 1536, 640]
    scale = max(1, math.ceil(rows_per_group / 512))
    return [pattern[i % len(pattern)] * scale for i in range(num_groups)]


def _shape_suite(suite: str, num_groups: int, rows_per_group: int, cols: int, jagged: bool):
    if suite == "single":
        return [(num_groups, rows_per_group, cols, jagged)]
    if suite == "work_order":
        return [
            (4, 128, 2048, False),
            (4, 256, 2048, False),
            (8, 256, 4096, False),
            (8, 512, 7168, False),
            (4, 1024, 7168, True),
            (8, 1024, 4096, True),
        ]
    if suite == "profile":
        return [
            (8, 512, 7168, False),
            (8, 1024, 4096, True),
        ]
    raise ValueError(f"Unsupported suite: {suite}")


def _tensor_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _actual_bytes_per_request(inp: torch.Tensor, quantized_parts) -> int:
    total = _tensor_nbytes(inp)
    for part in quantized_parts:
        total += _tensor_nbytes(part._rowwise_data)
        total += _tensor_nbytes(part._columnwise_data)
        total += _tensor_nbytes(part._rowwise_scale_inv)
        total += _tensor_nbytes(part._columnwise_scale_inv)
    return total


def _work_accounting(inp: torch.Tensor, quantized_parts) -> Dict[str, int]:
    rowwise_output_elements = 0
    columnwise_output_elements = 0
    rowwise_scale_elements = 0
    columnwise_scale_elements = 0
    for part in quantized_parts:
        if part._rowwise_data is not None:
            rowwise_output_elements += part._rowwise_data.numel()
        if part._columnwise_data is not None:
            columnwise_output_elements += part._columnwise_data.numel()
        if part._rowwise_scale_inv is not None:
            rowwise_scale_elements += part._rowwise_scale_inv.numel()
        if part._columnwise_scale_inv is not None:
            columnwise_scale_elements += part._columnwise_scale_inv.numel()
    return {
        "input_elements": inp.numel(),
        "rowwise_output_elements": rowwise_output_elements,
        "columnwise_output_elements": columnwise_output_elements,
        "rowwise_scale_elements": rowwise_scale_elements,
        "columnwise_scale_elements": columnwise_scale_elements,
    }


def _bandwidth_gbps(actual_bytes: int, median_ms: float) -> Optional[float]:
    if median_ms <= 0:
        return None
    return actual_bytes / median_ms / 1.0e6


def _throughput_gelements_per_sec(elements: int, median_ms: float) -> Optional[float]:
    if median_ms <= 0:
        return None
    return elements / median_ms / 1.0e6


def _speed_of_light_fraction(
    bandwidth_gbps: Optional[float], peak_bandwidth_gbps: Optional[float]
) -> Optional[float]:
    if bandwidth_gbps is None or peak_bandwidth_gbps is None or peak_bandwidth_gbps <= 0:
        return None
    return bandwidth_gbps / peak_bandwidth_gbps


def _bandwidth_from_clock_bus(clock_mhz: float, bus_width_bits: float) -> float:
    # Peak HBM/GDDR bandwidth from double-data-rate memory clock and bus width.
    return clock_mhz * 1.0e6 * 2.0 * (bus_width_bits / 8.0) / 1.0e9


def _parse_nvidia_smi_peak_bandwidth(output: str) -> Optional[Dict[str, object]]:
    clock_match = re.search(r"Max Memory Clock\s*:\s*([0-9.]+)\s*MHz", output, re.I)
    bus_match = re.search(r"Memory Bus Width\s*:\s*([0-9.]+)\s*bits?", output, re.I)
    if clock_match is None or bus_match is None:
        return None
    clock_mhz = float(clock_match.group(1))
    bus_width_bits = float(bus_match.group(1))
    return {
        "kind": "peak_memory_bandwidth",
        "peak_memory_bandwidth_GBps": _bandwidth_from_clock_bus(clock_mhz, bus_width_bits),
        "source": "nvidia-smi -q Max Memory Clock and Memory Bus Width",
        "clock_mhz": clock_mhz,
        "bus_width_bits": bus_width_bits,
        "formula": "clock_mhz * 1e6 * 2 * (bus_width_bits / 8) / 1e9",
    }


def _known_peak_bandwidth(device_name: str) -> Optional[Dict[str, object]]:
    normalized = device_name.lower()
    known = [
        ("b200", 8000.0, "NVIDIA B200 published peak HBM3e memory bandwidth"),
        ("h200", 4800.0, "NVIDIA H200 published peak HBM3e memory bandwidth"),
        ("h100 nvl", 3900.0, "NVIDIA H100 NVL published peak HBM3 memory bandwidth"),
        ("h100 pcie", 2000.0, "NVIDIA H100 PCIe published peak HBM2e memory bandwidth"),
        ("h100", 3350.0, "NVIDIA H100 SXM published peak HBM3 memory bandwidth"),
        ("a100-sxm", 2039.0, "NVIDIA A100 SXM published peak HBM2e memory bandwidth"),
        ("a100 pcie", 1555.0, "NVIDIA A100 PCIe published peak HBM2e memory bandwidth"),
        ("l40s", 864.0, "NVIDIA L40S published peak GDDR6 memory bandwidth"),
        ("l4", 300.0, "NVIDIA L4 published peak GDDR6 memory bandwidth"),
    ]
    for key, bandwidth_gbps, source in known:
        if key in normalized:
            return {
                "kind": "peak_memory_bandwidth",
                "peak_memory_bandwidth_GBps": bandwidth_gbps,
                "source": source,
                "matched_device_name": device_name,
            }
    return None


def _nvidia_smi_selector_for_torch_device(device: int) -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
        if device < len(tokens):
            return tokens[device]
    return str(device)


def _detect_speed_of_light_denominator(require: bool) -> Dict[str, object]:
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    nvidia_smi_selector = _nvidia_smi_selector_for_torch_device(device)
    details: Dict[str, object] = {
        "status": "unavailable",
        "device": device,
        "device_name": device_name,
        "nvidia_smi_selector": nvidia_smi_selector,
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-i", nvidia_smi_selector],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            parsed = _parse_nvidia_smi_peak_bandwidth(result.stdout)
            if parsed is not None:
                parsed.update(
                    {
                        "status": "detected",
                        "device": device,
                        "device_name": device_name,
                        "nvidia_smi_selector": nvidia_smi_selector,
                    }
                )
                return parsed
            details["nvidia_smi_parse_error"] = "Max Memory Clock or Memory Bus Width not found"
        else:
            details["nvidia_smi_error"] = result.stderr.strip()
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        details["nvidia_smi_error"] = str(exc)

    props = torch.cuda.get_device_properties(device)
    memory_clock_khz = getattr(props, "memory_clock_rate", None)
    memory_bus_width_bits = getattr(props, "memory_bus_width", None)
    if memory_clock_khz and memory_bus_width_bits:
        clock_mhz = float(memory_clock_khz) / 1000.0
        bus_width_bits = float(memory_bus_width_bits)
        return {
            "status": "detected",
            "kind": "peak_memory_bandwidth",
            "peak_memory_bandwidth_GBps": _bandwidth_from_clock_bus(clock_mhz, bus_width_bits),
            "source": "torch.cuda.get_device_properties memory_clock_rate and memory_bus_width",
            "device": device,
            "device_name": device_name,
            "clock_mhz": clock_mhz,
            "bus_width_bits": bus_width_bits,
            "formula": "clock_mhz * 1e6 * 2 * (bus_width_bits / 8) / 1e9",
        }

    known = _known_peak_bandwidth(device_name)
    if known is not None:
        known.update({"status": "detected", "device": device, "device_name": device_name})
        return known

    if require:
        raise RuntimeError(
            "Unable to determine a speed-of-light memory-bandwidth denominator for "
            f"{device_name}. Pass a supported GPU or update the benchmark denominator table."
        )
    return details


def _time_cuda(
    fn: Callable[[], object],
    *,
    samples: int,
    invocations_per_sample: int,
    max_invocations_per_sample: int,
    min_sample_ms: float,
    profile: bool,
) -> Dict[str, object]:
    times_total_ms = []
    times_per_request_ms = []
    actual_invocations_per_sample = []
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for _ in range(samples):
            sample_invocations = invocations_per_sample
            while True:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(sample_invocations):
                    fn()
                end.record()
                end.synchronize()
                elapsed_ms = start.elapsed_time(end)
                if (
                    min_sample_ms <= 0
                    or elapsed_ms >= min_sample_ms
                    or sample_invocations >= max_invocations_per_sample
                ):
                    break
                sample_invocations = min(
                    sample_invocations * 2,
                    max_invocations_per_sample,
                )
            times_total_ms.append(elapsed_ms)
            times_per_request_ms.append(elapsed_ms / sample_invocations)
            actual_invocations_per_sample.append(sample_invocations)
    finally:
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
    return {
        "median_ms": statistics.median(times_per_request_ms),
        "min_ms": min(times_per_request_ms),
        "max_ms": max(times_per_request_ms),
        "samples": samples,
        "requested_invocations_per_sample": invocations_per_sample,
        "max_invocations_per_sample": max_invocations_per_sample,
        "min_sample_ms": min_sample_ms,
        "actual_invocations_per_sample": actual_invocations_per_sample,
        "total_invocations": sum(actual_invocations_per_sample),
        "samples_total_ms": times_total_ms,
        "samples_ms_per_request": times_per_request_ms,
        "profiled": profile,
    }


def _event_name(event) -> str:
    return str(getattr(event, "name", getattr(event, "key", "")))


def _event_is_cuda(event) -> bool:
    device_type = str(getattr(event, "device_type", "")).lower()
    if "cuda" in device_type:
        return True
    return False


def _collect_profiler_evidence(
    fn: Callable[[], object],
    *,
    invocations: int,
    enabled: bool,
) -> Dict[str, object]:
    if not enabled:
        return {"status": "disabled"}
    try:
        activities = [torch.profiler.ProfilerActivity.CUDA]
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=activities) as prof:
            for _ in range(invocations):
                fn()
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - depends on profiler availability
        return {"status": "failed", "error": str(exc)}

    kernel_counts = Counter()
    for event in prof.events():
        if _event_is_cuda(event):
            name = _event_name(event)
            if name:
                kernel_counts[name] += 1

    if not kernel_counts:
        for event in prof.key_averages():
            cuda_time = getattr(event, "cuda_time_total", 0.0)
            if cuda_time <= 0:
                cuda_time = getattr(event, "device_time_total", 0.0)
            name = _event_name(event)
            if cuda_time > 0 and name and "cuda" not in name.lower():
                kernel_counts[name] += int(getattr(event, "count", 1))

    main_hints = ("quantize", "cast", "fp8", "block_scaling")
    main_quantize_counts = {
        name: count
        for name, count in kernel_counts.items()
        if any(hint in name.lower() for hint in main_hints)
    }
    return {
        "status": "collected",
        "profiler": "torch.profiler",
        "invocations": invocations,
        "cuda_kernel_launches": sum(kernel_counts.values()),
        "cuda_kernel_counts": dict(sorted(kernel_counts.items())),
        "main_quantize_kernel_launches": sum(main_quantize_counts.values()),
        "main_quantize_kernel_counts": dict(sorted(main_quantize_counts.items())),
    }


def _make_candidate_fn(
    *,
    api: str,
    inp: torch.Tensor,
    splits: List[int],
    quantizer,
    split_quantizers,
    first_dims: Optional[torch.Tensor],
    output,
):
    if api == "group_quantize":

        def run_group_quantize():
            return tex.group_quantize(inp, quantizer, len(splits), first_dims, output)

        return run_group_quantize

    if api == "split_quantize":

        def run_split_quantize():
            return tex.split_quantize(inp, splits, split_quantizers)

        return run_split_quantize

    raise ValueError(f"Unsupported API: {api}")


def _split_candidate_output(output):
    if hasattr(output, "split_into_quantized_tensors"):
        return output.split_into_quantized_tensors()
    return output


def _run_case(
    *,
    block_scaling_dim: int,
    api: str,
    num_groups: int,
    rows_per_group: int,
    cols: int,
    jagged: bool,
    warmup: int,
    iterations: int,
    invocations_per_sample: int,
    max_invocations_per_sample: int,
    min_sample_ms: float,
    profile_candidate_only: bool,
    collect_launch_evidence: bool,
    evidence_invocations: int,
    peak_bandwidth_gbps: Optional[float],
) -> Dict[str, object]:
    splits = _case_splits(num_groups, rows_per_group, jagged)
    inp = torch.randn(sum(splits), cols, dtype=torch.bfloat16, device="cuda")
    split_views = [part.contiguous() for part in torch.split(inp, splits)]
    quantizer = _make_quantizer(block_scaling_dim, rowwise=True, columnwise=True)
    split_quantizers = [quantizer.copy() for _ in splits]
    use_first_dims = api != "group_quantize" or jagged
    first_dims = (
        torch.tensor(splits, dtype=torch.int64, device=inp.device)
        if use_first_dims and api == "group_quantize"
        else None
    )

    baseline_outputs = [tex.quantize(part, quantizer) for part in split_views]
    grouped_output = None
    if api == "group_quantize":
        grouped_output = tex.group_quantize(inp, quantizer, len(splits), first_dims)

    if api == "split_quantize":
        baseline_mode = "manual_loop_allocating_non_grouped_tex_quantize"
        timing_scope = "end_to_end_api_latency_with_matched_output_allocation"

        def run_baseline():
            return [tex.quantize(part, quantizer) for part in split_views]

    else:
        baseline_mode = "manual_loop_preallocated_non_grouped_tex_quantize"
        timing_scope = "steady_state_kernel_latency_with_preallocated_outputs"

        def run_baseline():
            for part, output in zip(split_views, baseline_outputs):
                tex.quantize(part, quantizer, output)
            return baseline_outputs

    run_candidate = _make_candidate_fn(
        api=api,
        inp=inp,
        splits=splits,
        quantizer=quantizer,
        split_quantizers=split_quantizers,
        first_dims=first_dims,
        output=grouped_output,
    )

    candidate_parts = _split_candidate_output(grouped_output if grouped_output is not None else run_candidate())
    _assert_same(candidate_parts, baseline_outputs)
    actual_bytes = _actual_bytes_per_request(inp, baseline_outputs)
    work = _work_accounting(inp, baseline_outputs)

    for _ in range(warmup):
        if not profile_candidate_only:
            run_baseline()
        run_candidate()
    torch.cuda.synchronize()

    candidate_evidence = _collect_profiler_evidence(
        run_candidate,
        invocations=evidence_invocations,
        enabled=collect_launch_evidence and not profile_candidate_only,
    )
    baseline_evidence = None
    if not profile_candidate_only:
        baseline_evidence = _collect_profiler_evidence(
            run_baseline,
            invocations=evidence_invocations,
            enabled=collect_launch_evidence,
        )

    candidate_timing = _time_cuda(
        run_candidate,
        samples=iterations,
        invocations_per_sample=invocations_per_sample,
        max_invocations_per_sample=max_invocations_per_sample,
        min_sample_ms=min_sample_ms,
        profile=profile_candidate_only,
    )
    candidate_timing["actual_bytes_per_request"] = actual_bytes
    candidate_timing["bandwidth_GBps_actual_bytes"] = _bandwidth_gbps(
        actual_bytes, candidate_timing["median_ms"]
    )
    candidate_timing["input_throughput_Gelements_per_sec"] = _throughput_gelements_per_sec(
        work["input_elements"], candidate_timing["median_ms"]
    )
    candidate_timing["bandwidth_fraction_of_peak_memory"] = _speed_of_light_fraction(
        candidate_timing["bandwidth_GBps_actual_bytes"], peak_bandwidth_gbps
    )

    baseline_timing = None
    if not profile_candidate_only:
        baseline_timing = _time_cuda(
            run_baseline,
            samples=iterations,
            invocations_per_sample=invocations_per_sample,
            max_invocations_per_sample=max_invocations_per_sample,
            min_sample_ms=min_sample_ms,
            profile=False,
        )
        baseline_timing["actual_bytes_per_request"] = actual_bytes
        baseline_timing["bandwidth_GBps_actual_bytes"] = _bandwidth_gbps(
            actual_bytes, baseline_timing["median_ms"]
        )
        baseline_timing["input_throughput_Gelements_per_sec"] = _throughput_gelements_per_sec(
            work["input_elements"], baseline_timing["median_ms"]
        )
        baseline_timing["bandwidth_fraction_of_peak_memory"] = _speed_of_light_fraction(
            baseline_timing["bandwidth_GBps_actual_bytes"], peak_bandwidth_gbps
        )

    speedup = None
    if baseline_timing is not None:
        speedup = baseline_timing["median_ms"] / candidate_timing["median_ms"]
    if api == "split_quantize":
        first_dims_mode = "split_sections"
    elif first_dims is None:
        first_dims_mode = "none_uniform_group_quantize"
    else:
        first_dims_mode = "device_first_dims"

    return {
        "block_scaling_dim": block_scaling_dim,
        "api": api,
        "first_dims_mode": first_dims_mode,
        "num_groups": num_groups,
        "rows_per_group": rows_per_group,
        "cols": cols,
        "splits": splits,
        "jagged": jagged,
        "correctness": {
            "status": "passed",
            "reference": baseline_mode,
            "scale_comparison": "valid_blockwise_scale_regions_only",
        },
        "actual_bytes_per_request": actual_bytes,
        "work_per_request": work,
        "baseline_mode": baseline_mode,
        "timing_scope": timing_scope,
        "candidate": candidate_timing,
        "candidate_output_preallocated": api == "group_quantize",
        "baseline_manual_loop": baseline_timing,
        "speedup_baseline_over_candidate": speedup,
        "throughput": {
            "primary_metric": "candidate_bandwidth_GBps_actual_bytes",
            "candidate_bandwidth_GBps_actual_bytes": candidate_timing[
                "bandwidth_GBps_actual_bytes"
            ],
            "baseline_manual_loop_bandwidth_GBps_actual_bytes": (
                None
                if baseline_timing is None
                else baseline_timing["bandwidth_GBps_actual_bytes"]
            ),
            "candidate_bandwidth_fraction_of_peak_memory": candidate_timing[
                "bandwidth_fraction_of_peak_memory"
            ],
            "baseline_manual_loop_bandwidth_fraction_of_peak_memory": (
                None
                if baseline_timing is None
                else baseline_timing["bandwidth_fraction_of_peak_memory"]
            ),
            "speedup_baseline_over_candidate": speedup,
        },
        "launch_evidence": {
            "candidate": candidate_evidence,
            "baseline_manual_loop": baseline_evidence,
        },
        "expected_candidate_main_quantize_launches_per_request": 1,
        "expected_baseline_main_quantize_launches_per_request": num_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api",
        choices=["group_quantize", "split_quantize", "both"],
        default="group_quantize",
    )
    parser.add_argument("--dims", default="1,2", help="Comma-separated block scaling dims")
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--rows-per-group", type=int, default=512)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--jagged", action="store_true")
    parser.add_argument("--suite", choices=["single", "work_order", "profile"], default="single")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--invocations-per-sample", type=int, default=20)
    parser.add_argument("--max-invocations-per-sample", type=int, default=2048)
    parser.add_argument(
        "--min-sample-ms",
        type=float,
        default=10.0,
        help="Double invocations within a timing sample until the CUDA event window reaches this size.",
    )
    parser.add_argument("--profile-candidate-only", action="store_true")
    parser.add_argument(
        "--require-speed-of-light",
        action="store_true",
        help="Fail if a peak memory-bandwidth denominator cannot be detected for the active GPU.",
    )
    parser.add_argument(
        "--launch-evidence",
        choices=["profiler", "none"],
        default="profiler",
        help="Collect actual CUDA launch evidence with torch.profiler outside timing windows.",
    )
    parser.add_argument("--evidence-invocations", type=int, default=1)
    parser.add_argument(
        "--output",
        default=os.environ.get("ORCHESTRA_BENCHMARK_RAW_REPORT", "grouped_fp8_block_quantize.json"),
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.invocations_per_sample <= 0:
        raise ValueError("--invocations-per-sample must be positive")
    if args.max_invocations_per_sample < args.invocations_per_sample:
        raise ValueError("--max-invocations-per-sample must be at least --invocations-per-sample")
    if args.min_sample_ms < 0:
        raise ValueError("--min-sample-ms must be non-negative")
    if args.evidence_invocations <= 0:
        raise ValueError("--evidence-invocations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not available:
        raise RuntimeError(reason)

    started = time.time()
    speed_of_light = _detect_speed_of_light_denominator(args.require_speed_of_light)
    peak_bandwidth_gbps = speed_of_light.get("peak_memory_bandwidth_GBps")
    apis = ["group_quantize", "split_quantize"] if args.api == "both" else [args.api]
    cases = []
    shape_suite = _shape_suite(
        args.suite,
        args.num_groups,
        args.rows_per_group,
        args.cols,
        args.jagged,
    )
    for api in apis:
        for dim in _parse_csv_ints(args.dims):
            for num_groups, rows_per_group, cols, jagged in shape_suite:
                cases.append(
                    _run_case(
                        block_scaling_dim=dim,
                        api=api,
                        num_groups=num_groups,
                        rows_per_group=rows_per_group,
                        cols=cols,
                        jagged=jagged,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        invocations_per_sample=args.invocations_per_sample,
                        max_invocations_per_sample=args.max_invocations_per_sample,
                        min_sample_ms=args.min_sample_ms,
                        profile_candidate_only=args.profile_candidate_only,
                        collect_launch_evidence=args.launch_evidence == "profiler",
                        evidence_invocations=args.evidence_invocations,
                        peak_bandwidth_gbps=peak_bandwidth_gbps,
                    )
                )

    report = {
        "schema_version": "grouped_fp8_block_quantize_benchmark/v3",
        "command": " ".join(sys.argv),
        "gpu": torch.cuda.get_device_name(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "te_version": getattr(te, "__version__", "unknown"),
        "te_module_path": getattr(te, "__file__", "unknown"),
        "te_build_mode": "debug" if os.environ.get("NVTE_BUILD_DEBUG") else "release_or_default",
        "nvte_framework": os.environ.get("NVTE_FRAMEWORK", "unset"),
        "suite": args.suite,
        "profile_candidate_only": args.profile_candidate_only,
        "profile_after_warmup": args.profile_candidate_only,
        "primary_metric": "candidate_bandwidth_GBps_actual_bytes",
        "metric_unit": "GB/s decimal using actual input/output/scale bytes per request",
        "higher_is_better": True,
        "baseline_comparison": "same_session_manual_non_grouped_tex_quantize_loop",
        "speed_of_light": speed_of_light,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "invocations_per_sample": args.invocations_per_sample,
        "max_invocations_per_sample": args.max_invocations_per_sample,
        "min_sample_ms": args.min_sample_ms,
        "launch_evidence": args.launch_evidence,
        "elapsed_sec": time.time() - started,
        "cases": cases,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
