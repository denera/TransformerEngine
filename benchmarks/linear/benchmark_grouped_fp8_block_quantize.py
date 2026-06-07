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
from typing import Callable, Dict, List, Optional, Tuple

import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import Float8BlockQuantizer
import transformer_engine_torch as tex


BLOCK_LEN = 128
FP8_BLOCK_PHYSICAL_MODEL_VERSION = "fp8_block_group_quantize_physical_bytes/v2"
ROOFLINE_FRACTION_ANOMALY_TOLERANCE = 1.0e-6
THROUGHPUT_TIMING_BASIS = "sustained_mean_ms"
BASELINE_DRIFT_ALARM_FRACTION = 0.10


def _parse_csv_ints(value: str) -> List[int]:
    return [int(part) for part in value.split(",") if part]


def _parse_csv_strings(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _output_mode_to_usage(output_mode: str) -> Tuple[bool, bool]:
    if output_mode == "rowwise":
        return True, False
    if output_mode == "columnwise":
        return False, True
    if output_mode == "both":
        return True, True
    raise ValueError(f"Unsupported output mode: {output_mode}")


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


def _shape_suite(
    suite: str,
    num_groups: int,
    rows_per_group: int,
    cols: int,
    jagged: bool,
    *,
    rows_sweep: Optional[List[int]] = None,
    jagged_scale_sweep: Optional[List[int]] = None,
    layouts: Optional[List[str]] = None,
):
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
    if suite == "plateau":
        rows_sweep = rows_sweep or [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        jagged_scale_sweep = jagged_scale_sweep or [1, 2, 4, 8, 16]
        layouts = layouts or ["uniform", "jagged"]
        shapes = []
        if "uniform" in layouts:
            shapes.extend((num_groups, rows, cols, False) for rows in rows_sweep)
        if "jagged" in layouts:
            shapes.extend((num_groups, 512 * scale, cols, True) for scale in jagged_scale_sweep)
        return shapes
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


def _manual_quantize_one(part: torch.Tensor, quantizer):
    if (
        quantizer.block_scaling_dim == 2
        and quantizer.columnwise_usage
        and not quantizer.rowwise_usage
    ):
        ref_quantizer = _make_quantizer(2, rowwise=True, columnwise=True)
        ref = tex.quantize(part.contiguous(), ref_quantizer)
        ref.update_usage(rowwise_usage=False, columnwise_usage=True)
        return ref
    return tex.quantize(part.contiguous(), quantizer)


def _case_id(
    *,
    api: str,
    block_scaling_dim: int,
    output_mode: str,
    num_groups: int,
    rows_per_group: int,
    cols: int,
    jagged: bool,
) -> str:
    layout = "jagged" if jagged else "uniform"
    return (
        f"{api}_dim{block_scaling_dim}_{output_mode}_{layout}_"
        f"groups{num_groups}_rows{rows_per_group}_cols{cols}"
    )


def _estimate_storage_bytes(splits: List[int], cols: int, quantizer) -> int:
    total_elements = sum(splits) * cols
    total = total_elements * 2
    if quantizer.rowwise_usage:
        total += total_elements
        total += sum(
            math.prod(quantizer.get_scale_shape((rows, cols), False)) * 4 for rows in splits
        )
    if quantizer.columnwise_usage:
        total += total_elements
        total += sum(
            math.prod(quantizer.get_scale_shape((rows, cols), True)) * 4 for rows in splits
        )
    return total


def _metadata_nbytes(tensor: Optional[torch.Tensor]) -> int:
    return _tensor_nbytes(tensor)


def _infer_kernel_path(
    *,
    block_scaling_dim: int,
    splits: List[int],
    cols: int,
    jagged: bool,
    grouped_output,
) -> Dict[str, object]:
    aligned_uniform = not jagged and cols % BLOCK_LEN == 0 and all(
        split % BLOCK_LEN == 0 for split in splits
    )
    row_block_offsets = getattr(grouped_output, "_fp8_row_block_offsets", None)
    aligned_jagged = (
        jagged
        and row_block_offsets is not None
        and cols % BLOCK_LEN == 0
        and sum(splits) % BLOCK_LEN == 0
        and row_block_offsets.shape[1] == sum(splits) // BLOCK_LEN
    )
    if block_scaling_dim == 1:
        if aligned_uniform or aligned_jagged:
            return {
                "name": "group_quantize_fp8_1d_block_scaling_aligned",
                "input_hbm_read_passes": 1,
                "input_global_load_instruction_passes": 1,
                "duplicate_input_read": False,
                "notes": (
                    "The aligned 1D grouped kernel loads input once, emits rowwise output from "
                    "the coalesced load registers, and stages an unpadded swizzled shared tile "
                    "for columnwise output stores."
                ),
            }
        return {
            "name": "group_quantize_fp8_1d_block_scaling",
            "input_hbm_read_passes": 1,
            "input_global_load_instruction_passes": 1,
            "duplicate_input_read": False,
            "notes": "1D grouped kernels stage the input tile and reuse it for output stores.",
        }

    if aligned_uniform or aligned_jagged:
        return {
            "name": "group_quantize_fp8_2d_block_scaling_aligned_register",
            "input_hbm_read_passes": 1,
            "input_global_load_instruction_passes": 2,
            "duplicate_input_read": True,
            "notes": (
                "The aligned 2D register kernel issues a second input load for output stores. "
                "The HBM roofline model charges one HBM input pass because the duplicate tile "
                "load is expected to be served from cache; global-load instruction traffic is "
                "reported separately."
            ),
        }
    return {
        "name": "group_quantize_fp8_2d_block_scaling_shared_tile",
        "input_hbm_read_passes": 1,
        "input_global_load_instruction_passes": 1,
        "duplicate_input_read": False,
        "notes": "The generic 2D grouped kernel stages the input tile in shared memory.",
    }


def _traffic_accounting(
    *,
    inp: torch.Tensor,
    quantized_parts,
    splits: List[int],
    block_scaling_dim: int,
    cols: int,
    jagged: bool,
    grouped_output,
    first_dims: Optional[torch.Tensor],
) -> Dict[str, object]:
    work = _work_accounting(inp, quantized_parts)
    input_bytes = inp.numel() * inp.element_size()
    rowwise_output_bytes = work["rowwise_output_elements"]
    columnwise_output_bytes = work["columnwise_output_elements"]
    rowwise_scale_bytes = work["rowwise_scale_elements"] * 4
    columnwise_scale_bytes = work["columnwise_scale_elements"] * 4
    output_bytes = rowwise_output_bytes + columnwise_output_bytes
    scale_bytes = rowwise_scale_bytes + columnwise_scale_bytes
    ideal_useful_bytes = input_bytes + output_bytes + scale_bytes

    kernel_path = _infer_kernel_path(
        block_scaling_dim=block_scaling_dim,
        splits=splits,
        cols=cols,
        jagged=jagged,
        grouped_output=grouped_output,
    )
    metadata_bytes = _metadata_nbytes(first_dims)
    for attr in (
        "tensor_offsets",
        "_fp8_row_block_offsets",
        "_fp8_rowwise_scale_inv_offsets",
        "_fp8_columnwise_scale_inv_offsets",
    ):
        metadata_bytes += _metadata_nbytes(getattr(grouped_output, attr, None))

    estimated_hbm_bytes = (
        input_bytes * int(kernel_path["input_hbm_read_passes"])
        + output_bytes
        + scale_bytes
        + metadata_bytes
    )
    estimated_global_instruction_bytes = (
        input_bytes * int(kernel_path["input_global_load_instruction_passes"])
        + output_bytes
        + scale_bytes
        + metadata_bytes
    )
    return {
        "physical_model_version": FP8_BLOCK_PHYSICAL_MODEL_VERSION,
        "ideal_useful_bytes": ideal_useful_bytes,
        "allocated_bytes": _actual_bytes_per_request(inp, quantized_parts),
        "estimated_physical_global_bytes": estimated_hbm_bytes,
        "estimated_hbm_global_bytes": estimated_hbm_bytes,
        "estimated_global_memory_instruction_bytes": estimated_global_instruction_bytes,
        "estimated_duplicate_input_cache_read_bytes": (
            estimated_global_instruction_bytes - estimated_hbm_bytes
        ),
        "input_bytes": input_bytes,
        "rowwise_output_bytes": rowwise_output_bytes,
        "columnwise_output_bytes": columnwise_output_bytes,
        "rowwise_scale_bytes": rowwise_scale_bytes,
        "columnwise_scale_bytes": columnwise_scale_bytes,
        "metadata_bytes": metadata_bytes,
        "kernel_path": kernel_path,
        "physical_model_notes": kernel_path["notes"],
    }


def _bandwidth_gbps(actual_bytes: int, ms_per_request: float) -> Optional[float]:
    if ms_per_request <= 0:
        return None
    return actual_bytes / ms_per_request / 1.0e6


def _throughput_gelements_per_sec(elements: int, ms_per_request: float) -> Optional[float]:
    if ms_per_request <= 0:
        return None
    return elements / ms_per_request / 1.0e6


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
    total_invocations = sum(actual_invocations_per_sample)
    sustained_total_ms = sum(times_total_ms)
    sustained_mean_ms = sustained_total_ms / total_invocations
    mean_ms = statistics.mean(times_per_request_ms)
    stdev_ms = (
        statistics.stdev(times_per_request_ms)
        if len(times_per_request_ms) > 1
        else 0.0
    )
    return {
        "timing_basis_for_throughput": THROUGHPUT_TIMING_BASIS,
        "sustained_mean_ms": sustained_mean_ms,
        "sustained_total_ms": sustained_total_ms,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times_per_request_ms),
        "min_ms": min(times_per_request_ms),
        "max_ms": max(times_per_request_ms),
        "stdev_ms": stdev_ms,
        "coefficient_of_variation": stdev_ms / mean_ms if mean_ms > 0 else None,
        "samples": samples,
        "requested_invocations_per_sample": invocations_per_sample,
        "max_invocations_per_sample": max_invocations_per_sample,
        "min_sample_ms": min_sample_ms,
        "actual_invocations_per_sample": actual_invocations_per_sample,
        "total_invocations": total_invocations,
        "samples_total_ms": times_total_ms,
        "samples_ms_per_request": times_per_request_ms,
        "profiled": profile,
    }


def _calibrate_memory_roofline(
    *,
    physical_bytes: int,
    warmup: int,
    samples: int,
    invocations_per_sample: int,
    max_invocations_per_sample: int,
    min_sample_ms: float,
) -> Dict[str, object]:
    copy_payload_bytes = max(1, math.ceil(physical_bytes / 2))
    src = torch.empty(copy_payload_bytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)

    def run_copy():
        dst.copy_(src)

    for _ in range(warmup):
        run_copy()
    torch.cuda.synchronize()

    timing = _time_cuda(
        run_copy,
        samples=samples,
        invocations_per_sample=invocations_per_sample,
        max_invocations_per_sample=max_invocations_per_sample,
        min_sample_ms=min_sample_ms,
        profile=False,
    )
    copy_physical_bytes = copy_payload_bytes * 2
    timing.update(
        {
            "method": "device_to_device_uint8_copy",
            "requested_physical_bytes": physical_bytes,
            "copy_payload_bytes": copy_payload_bytes,
            "copy_physical_bytes": copy_physical_bytes,
            "calibrated_physical_GBps": _bandwidth_gbps(
                copy_physical_bytes, timing["median_ms"]
            ),
        }
    )
    return timing


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
    output_mode: str,
    num_groups: int,
    rows_per_group: int,
    cols: int,
    jagged: bool,
    warmup: int,
    iterations: int,
    calibration_warmup: int,
    calibration_iterations: int,
    invocations_per_sample: int,
    max_invocations_per_sample: int,
    min_sample_ms: float,
    profile_candidate_only: bool,
    collect_launch_evidence: bool,
    evidence_invocations: int,
    peak_bandwidth_gbps: Optional[float],
    memory_fraction_cap: float,
) -> Dict[str, object]:
    case_id = _case_id(
        api=api,
        block_scaling_dim=block_scaling_dim,
        output_mode=output_mode,
        num_groups=num_groups,
        rows_per_group=rows_per_group,
        cols=cols,
        jagged=jagged,
    )
    splits = _case_splits(num_groups, rows_per_group, jagged)
    rowwise, columnwise = _output_mode_to_usage(output_mode)
    quantizer = _make_quantizer(block_scaling_dim, rowwise=rowwise, columnwise=columnwise)
    estimated_storage_bytes = _estimate_storage_bytes(splits, cols, quantizer)
    if memory_fraction_cap > 0:
        free_bytes, _ = torch.cuda.mem_get_info()
        if estimated_storage_bytes > free_bytes * memory_fraction_cap:
            return {
                "case_id": case_id,
                "status": "skipped",
                "skip_reason": "estimated_storage_exceeds_memory_fraction_cap",
                "memory_fraction_cap": memory_fraction_cap,
                "estimated_storage_bytes": estimated_storage_bytes,
                "free_bytes_at_skip_check": free_bytes,
                "block_scaling_dim": block_scaling_dim,
                "api": api,
                "output_mode": output_mode,
                "num_groups": num_groups,
                "rows_per_group": rows_per_group,
                "cols": cols,
                "splits": splits,
                "jagged": jagged,
                "acceptance_eligible": False,
            }
    inp = torch.randn(sum(splits), cols, dtype=torch.bfloat16, device="cuda")
    split_views = [part.contiguous() for part in torch.split(inp, splits)]
    split_quantizers = [quantizer.copy() for _ in splits]
    use_first_dims = api != "group_quantize" or jagged
    first_dims = (
        torch.tensor(splits, dtype=torch.int64, device=inp.device)
        if use_first_dims and api == "group_quantize"
        else None
    )

    baseline_outputs = [_manual_quantize_one(part, quantizer) for part in split_views]
    grouped_output = None
    if api == "group_quantize":
        grouped_output = tex.group_quantize(inp, quantizer, len(splits), first_dims)

    columnwise_only_2d = block_scaling_dim == 2 and columnwise and not rowwise
    if api == "split_quantize" or columnwise_only_2d:
        baseline_mode = "manual_loop_allocating_non_grouped_tex_quantize"
        timing_scope = "end_to_end_api_latency_with_matched_output_allocation"

        def run_baseline():
            return [_manual_quantize_one(part, quantizer) for part in split_views]

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
    traffic = _traffic_accounting(
        inp=inp,
        quantized_parts=baseline_outputs,
        splits=splits,
        block_scaling_dim=block_scaling_dim,
        cols=cols,
        jagged=jagged,
        grouped_output=grouped_output if grouped_output is not None else candidate_parts[0],
        first_dims=first_dims,
    )
    roofline_calibration = _calibrate_memory_roofline(
        physical_bytes=traffic["estimated_physical_global_bytes"],
        warmup=calibration_warmup,
        samples=calibration_iterations,
        invocations_per_sample=invocations_per_sample,
        max_invocations_per_sample=max_invocations_per_sample,
        min_sample_ms=min_sample_ms,
    )

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
    candidate_timing["ideal_useful_bytes_per_request"] = traffic["ideal_useful_bytes"]
    candidate_timing["estimated_physical_global_bytes_per_request"] = traffic[
        "estimated_physical_global_bytes"
    ]
    candidate_timing["estimated_hbm_global_bytes_per_request"] = traffic[
        "estimated_hbm_global_bytes"
    ]
    candidate_timing["estimated_global_memory_instruction_bytes_per_request"] = traffic[
        "estimated_global_memory_instruction_bytes"
    ]
    candidate_throughput_ms = candidate_timing["sustained_mean_ms"]
    candidate_timing["bandwidth_GBps_actual_bytes"] = _bandwidth_gbps(
        actual_bytes, candidate_throughput_ms
    )
    candidate_timing["logical_effective_GBps"] = _bandwidth_gbps(
        traffic["ideal_useful_bytes"], candidate_throughput_ms
    )
    candidate_timing["estimated_physical_GBps"] = _bandwidth_gbps(
        traffic["estimated_physical_global_bytes"], candidate_throughput_ms
    )
    candidate_timing["estimated_hbm_GBps"] = candidate_timing["estimated_physical_GBps"]
    candidate_timing["estimated_global_memory_instruction_GBps"] = _bandwidth_gbps(
        traffic["estimated_global_memory_instruction_bytes"], candidate_throughput_ms
    )
    candidate_timing["input_throughput_Gelements_per_sec"] = _throughput_gelements_per_sec(
        work["input_elements"], candidate_throughput_ms
    )
    candidate_timing["bandwidth_fraction_of_peak_memory"] = _speed_of_light_fraction(
        candidate_timing["bandwidth_GBps_actual_bytes"], peak_bandwidth_gbps
    )
    calibrated_physical_gbps = roofline_calibration["calibrated_physical_GBps"]
    candidate_logical_gbps = candidate_timing["logical_effective_GBps"]
    if (
        calibrated_physical_gbps is None
        or candidate_logical_gbps is None
        or traffic["estimated_physical_global_bytes"] <= 0
    ):
        shape_roofline_logical_gbps = None
        candidate_fraction_of_shape_roofline = None
    else:
        shape_roofline_logical_gbps = (
            calibrated_physical_gbps
            * traffic["ideal_useful_bytes"]
            / traffic["estimated_physical_global_bytes"]
        )
        candidate_fraction_of_shape_roofline = (
            None
            if shape_roofline_logical_gbps <= 0
            else candidate_logical_gbps / shape_roofline_logical_gbps
        )
    candidate_timing["shape_specific_roofline_logical_GBps"] = shape_roofline_logical_gbps
    candidate_timing["candidate_fraction_of_shape_roofline"] = (
        candidate_fraction_of_shape_roofline
    )
    candidate_timing["candidate_fraction_of_shape_roofline_anomaly"] = (
        candidate_fraction_of_shape_roofline is not None
        and candidate_fraction_of_shape_roofline > 1.0 + ROOFLINE_FRACTION_ANOMALY_TOLERANCE
    )
    candidate_timing["shape_specific_roofline_basis"] = (
        "calibrated_device_to_device_copy_over_estimated_hbm_global_bytes"
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
        baseline_timing["ideal_useful_bytes_per_request"] = traffic["ideal_useful_bytes"]
        baseline_timing["estimated_physical_global_bytes_per_request"] = traffic[
            "estimated_physical_global_bytes"
        ]
        baseline_timing["estimated_hbm_global_bytes_per_request"] = traffic[
            "estimated_hbm_global_bytes"
        ]
        baseline_timing["estimated_global_memory_instruction_bytes_per_request"] = traffic[
            "estimated_global_memory_instruction_bytes"
        ]
        baseline_throughput_ms = baseline_timing["sustained_mean_ms"]
        baseline_timing["bandwidth_GBps_actual_bytes"] = _bandwidth_gbps(
            actual_bytes, baseline_throughput_ms
        )
        baseline_timing["logical_effective_GBps"] = _bandwidth_gbps(
            traffic["ideal_useful_bytes"], baseline_throughput_ms
        )
        baseline_timing["estimated_physical_GBps"] = _bandwidth_gbps(
            traffic["estimated_physical_global_bytes"], baseline_throughput_ms
        )
        baseline_timing["estimated_hbm_GBps"] = baseline_timing["estimated_physical_GBps"]
        baseline_timing["estimated_global_memory_instruction_GBps"] = _bandwidth_gbps(
            traffic["estimated_global_memory_instruction_bytes"], baseline_throughput_ms
        )
        baseline_timing["input_throughput_Gelements_per_sec"] = _throughput_gelements_per_sec(
            work["input_elements"], baseline_throughput_ms
        )
        baseline_timing["bandwidth_fraction_of_peak_memory"] = _speed_of_light_fraction(
            baseline_timing["bandwidth_GBps_actual_bytes"], peak_bandwidth_gbps
        )

    speedup = None
    if baseline_timing is not None:
        speedup = baseline_timing["sustained_mean_ms"] / candidate_timing["sustained_mean_ms"]
    if api == "split_quantize":
        first_dims_mode = "split_sections"
    elif first_dims is None:
        first_dims_mode = "none_uniform_group_quantize"
    else:
        first_dims_mode = "device_first_dims"

    return {
        "case_id": case_id,
        "status": "completed",
        "block_scaling_dim": block_scaling_dim,
        "api": api,
        "output_mode": output_mode,
        "first_dims_mode": first_dims_mode,
        "num_groups": num_groups,
        "rows_per_group": rows_per_group,
        "cols": cols,
        "splits": splits,
        "jagged": jagged,
        "estimated_storage_bytes": estimated_storage_bytes,
        "correctness": {
            "status": "passed",
            "reference": baseline_mode,
            "scale_comparison": "valid_blockwise_scale_regions_only",
        },
        "actual_bytes_per_request": actual_bytes,
        "byte_accounting": traffic,
        "roofline_calibration": roofline_calibration,
        "work_per_request": work,
        "baseline_mode": baseline_mode,
        "timing_scope": timing_scope,
        "throughput_timing_basis": THROUGHPUT_TIMING_BASIS,
        "throughput_timing_basis_description": (
            "Bandwidth and throughput fields use aggregate elapsed CUDA-event time divided "
            "by the total number of measured requests across all samples. Median/min/max "
            "per-request timings are diagnostics only."
        ),
        "candidate": candidate_timing,
        "candidate_output_preallocated": api == "group_quantize",
        "baseline_manual_loop": baseline_timing,
        "baseline_kernel_contract": {
            "comparison_baseline": "manual_loop_over_non_grouped_split_quantize",
            "baseline_must_not_use_group_quantize_kernel": True,
            "baseline_must_not_be_modified_for_comparison": True,
            "expected_main_quantize_launches_per_request": num_groups,
            "drift_alarm_fraction": BASELINE_DRIFT_ALARM_FRACTION,
            "drift_alarm_instruction": (
                "Compare sustained baseline throughput for this case against prior "
                "comparable reports before accepting performance conclusions."
            ),
        },
        "speedup_baseline_over_candidate": speedup,
        "throughput": {
            "primary_metric": "candidate_fraction_of_shape_roofline",
            "timing_basis": THROUGHPUT_TIMING_BASIS,
            "candidate_sustained_mean_ms": candidate_timing["sustained_mean_ms"],
            "baseline_manual_loop_sustained_mean_ms": (
                None if baseline_timing is None else baseline_timing["sustained_mean_ms"]
            ),
            "candidate_median_ms_diagnostic": candidate_timing["median_ms"],
            "baseline_manual_loop_median_ms_diagnostic": (
                None if baseline_timing is None else baseline_timing["median_ms"]
            ),
            "candidate_bandwidth_GBps_actual_bytes": candidate_timing[
                "bandwidth_GBps_actual_bytes"
            ],
            "candidate_logical_effective_GBps": candidate_timing["logical_effective_GBps"],
            "candidate_estimated_physical_GBps": candidate_timing["estimated_physical_GBps"],
            "candidate_estimated_hbm_GBps": candidate_timing["estimated_hbm_GBps"],
            "candidate_estimated_global_memory_instruction_GBps": candidate_timing[
                "estimated_global_memory_instruction_GBps"
            ],
            "shape_specific_roofline_logical_GBps": candidate_timing[
                "shape_specific_roofline_logical_GBps"
            ],
            "candidate_fraction_of_shape_roofline": candidate_timing[
                "candidate_fraction_of_shape_roofline"
            ],
            "baseline_manual_loop_bandwidth_GBps_actual_bytes": (
                None
                if baseline_timing is None
                else baseline_timing["bandwidth_GBps_actual_bytes"]
            ),
            "baseline_manual_loop_logical_effective_GBps": (
                None if baseline_timing is None else baseline_timing["logical_effective_GBps"]
            ),
            "baseline_manual_loop_estimated_physical_GBps": (
                None if baseline_timing is None else baseline_timing["estimated_physical_GBps"]
            ),
            "baseline_manual_loop_estimated_hbm_GBps": (
                None if baseline_timing is None else baseline_timing["estimated_hbm_GBps"]
            ),
            "baseline_manual_loop_estimated_global_memory_instruction_GBps": (
                None
                if baseline_timing is None
                else baseline_timing["estimated_global_memory_instruction_GBps"]
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
        "plateau_eligible": False,
        "acceptance_eligible": False,
    }


def _mark_plateau_eligibility(
    cases: List[Dict[str, object]],
    *,
    window: int,
    fraction_of_best: float,
    max_delta_fraction: float,
    success_fraction_threshold: float,
) -> List[Dict[str, object]]:
    for case in cases:
        case.setdefault("plateau_eligible", False)
        case.setdefault("acceptance_eligible", False)
        case["plateau_analysis"] = {
            "window": window,
            "fraction_of_best": fraction_of_best,
            "max_delta_fraction": max_delta_fraction,
            "status": "not_evaluated",
        }

    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for case in cases:
        if case.get("status") != "completed":
            continue
        key = (
            case.get("api"),
            case.get("block_scaling_dim"),
            case.get("output_mode"),
            case.get("jagged"),
        )
        groups.setdefault(key, []).append(case)

    for _, group_cases in groups.items():
        group_cases.sort(
            key=lambda item: item["byte_accounting"]["estimated_physical_global_bytes"]
        )
        bandwidths = [
            item["candidate"].get("estimated_physical_GBps") for item in group_cases
        ]
        valid_bandwidths = [value for value in bandwidths if value is not None and value > 0]
        if len(valid_bandwidths) < window:
            for item in group_cases:
                item["plateau_analysis"]["status"] = "insufficient_valid_points"
            continue
        best = max(valid_bandwidths)
        accepted_windows = []
        for start in range(0, len(group_cases) - window + 1):
            window_cases = group_cases[start : start + window]
            window_values = [
                item["candidate"].get("estimated_physical_GBps") for item in window_cases
            ]
            if any(value is None or value <= 0 for value in window_values):
                continue
            near_best = all(value >= fraction_of_best * best for value in window_values)
            stable_delta = True
            for previous, current in zip(window_values, window_values[1:]):
                if previous <= 0:
                    stable_delta = False
                    break
                if abs(current - previous) / previous > max_delta_fraction:
                    stable_delta = False
                    break
            if near_best and stable_delta:
                accepted_windows.append([item["case_id"] for item in window_cases])
                for item in window_cases:
                    item["plateau_eligible"] = True

        for item in group_cases:
            fraction = item["candidate"].get("candidate_fraction_of_shape_roofline")
            roofline_fraction_anomaly = bool(
                item["candidate"].get("candidate_fraction_of_shape_roofline_anomaly")
            )
            item["acceptance_eligible"] = bool(
                item["plateau_eligible"]
                and item.get("api") == "group_quantize"
                and item.get("output_mode") == "both"
                and fraction is not None
                and fraction >= success_fraction_threshold
                and not roofline_fraction_anomaly
            )
            item["plateau_analysis"].update(
                {
                    "status": "plateau_found" if accepted_windows else "underfilled_or_unstable",
                    "best_candidate_estimated_physical_GBps": best,
                    "best_candidate_estimated_hbm_GBps": best,
                    "accepted_windows": accepted_windows,
                    "underfilled_acceptance_excluded": not item["plateau_eligible"],
                    "roofline_fraction_anomaly": roofline_fraction_anomaly,
                    "roofline_anomaly_acceptance_excluded": roofline_fraction_anomaly,
                    "success_fraction_threshold": success_fraction_threshold,
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api",
        choices=["group_quantize", "split_quantize", "both"],
        default="group_quantize",
    )
    parser.add_argument("--dims", default="1,2", help="Comma-separated block scaling dims")
    parser.add_argument(
        "--output-modes",
        default="both",
        help="Comma-separated output modes: rowwise,columnwise,both",
    )
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--rows-per-group", type=int, default=512)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--jagged", action="store_true")
    parser.add_argument(
        "--suite", choices=["single", "work_order", "profile", "plateau"], default="single"
    )
    parser.add_argument(
        "--rows-sweep",
        default="128,256,512,1024,2048,4096,8192,16384",
        help="Rows per group for uniform plateau sweeps.",
    )
    parser.add_argument(
        "--jagged-scale-sweep",
        default="1,2,4,8,16",
        help="Scale multipliers for the aligned jagged plateau pattern.",
    )
    parser.add_argument(
        "--layouts",
        default="uniform,jagged",
        help="Comma-separated layouts for plateau suite: uniform,jagged",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--calibration-warmup", type=int, default=5)
    parser.add_argument("--calibration-iterations", type=int, default=10)
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
        "--memory-fraction-cap",
        type=float,
        default=0.50,
        help="Skip cases whose estimated benchmark storage exceeds this fraction of free memory.",
    )
    parser.add_argument("--plateau-window", type=int, default=3)
    parser.add_argument("--plateau-fraction-of-best", type=float, default=0.95)
    parser.add_argument("--plateau-max-delta-fraction", type=float, default=0.05)
    parser.add_argument("--success-fraction-threshold", type=float, default=0.80)
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
    if args.calibration_warmup < 0:
        raise ValueError("--calibration-warmup must be non-negative")
    if args.calibration_iterations <= 0:
        raise ValueError("--calibration-iterations must be positive")
    if args.invocations_per_sample <= 0:
        raise ValueError("--invocations-per-sample must be positive")
    if args.max_invocations_per_sample < args.invocations_per_sample:
        raise ValueError("--max-invocations-per-sample must be at least --invocations-per-sample")
    if args.min_sample_ms < 0:
        raise ValueError("--min-sample-ms must be non-negative")
    if args.evidence_invocations <= 0:
        raise ValueError("--evidence-invocations must be positive")
    if args.memory_fraction_cap < 0:
        raise ValueError("--memory-fraction-cap must be non-negative")
    if args.plateau_window <= 0:
        raise ValueError("--plateau-window must be positive")
    if not (0.0 < args.plateau_fraction_of_best <= 1.0):
        raise ValueError("--plateau-fraction-of-best must be in (0, 1]")
    if args.plateau_max_delta_fraction < 0:
        raise ValueError("--plateau-max-delta-fraction must be non-negative")
    if args.success_fraction_threshold <= 0:
        raise ValueError("--success-fraction-threshold must be positive")
    output_modes = _parse_csv_strings(args.output_modes)
    for output_mode in output_modes:
        _output_mode_to_usage(output_mode)
    layouts = _parse_csv_strings(args.layouts)
    for layout in layouts:
        if layout not in ("uniform", "jagged"):
            raise ValueError("--layouts entries must be uniform or jagged")
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
        rows_sweep=_parse_csv_ints(args.rows_sweep),
        jagged_scale_sweep=_parse_csv_ints(args.jagged_scale_sweep),
        layouts=layouts,
    )
    for api in apis:
        for dim in _parse_csv_ints(args.dims):
            for output_mode in output_modes:
                for num_groups, rows_per_group, cols, jagged in shape_suite:
                    cases.append(
                        _run_case(
                            block_scaling_dim=dim,
                            api=api,
                            output_mode=output_mode,
                            num_groups=num_groups,
                            rows_per_group=rows_per_group,
                            cols=cols,
                            jagged=jagged,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            calibration_warmup=args.calibration_warmup,
                            calibration_iterations=args.calibration_iterations,
                            invocations_per_sample=args.invocations_per_sample,
                            max_invocations_per_sample=args.max_invocations_per_sample,
                            min_sample_ms=args.min_sample_ms,
                            profile_candidate_only=args.profile_candidate_only,
                            collect_launch_evidence=args.launch_evidence == "profiler",
                            evidence_invocations=args.evidence_invocations,
                            peak_bandwidth_gbps=peak_bandwidth_gbps,
                            memory_fraction_cap=args.memory_fraction_cap,
                        )
                    )

    _mark_plateau_eligibility(
        cases,
        window=args.plateau_window,
        fraction_of_best=args.plateau_fraction_of_best,
        max_delta_fraction=args.plateau_max_delta_fraction,
        success_fraction_threshold=args.success_fraction_threshold,
    )

    report = {
        "schema_version": "grouped_fp8_block_quantize_benchmark/v4",
        "command": " ".join(sys.argv),
        "gpu": torch.cuda.get_device_name(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "te_version": getattr(te, "__version__", "unknown"),
        "te_module_path": getattr(te, "__file__", "unknown"),
        "te_build_mode": "debug" if os.environ.get("NVTE_BUILD_DEBUG") else "release_or_default",
        "nvte_framework": os.environ.get("NVTE_FRAMEWORK", "unset"),
        "suite": args.suite,
        "output_modes": output_modes,
        "layouts": layouts,
        "physical_model_version": FP8_BLOCK_PHYSICAL_MODEL_VERSION,
        "profile_candidate_only": args.profile_candidate_only,
        "profile_after_warmup": args.profile_candidate_only,
        "primary_metric": "candidate_fraction_of_shape_roofline",
        "metric_unit": "fraction of calibrated shape-specific HBM roofline",
        "higher_is_better": True,
        "baseline_comparison": "same_session_manual_non_grouped_tex_quantize_loop",
        "throughput_timing_basis": THROUGHPUT_TIMING_BASIS,
        "throughput_timing_basis_description": (
            "Reported throughput is sustained aggregate throughput across all measured "
            "kernel launches, not a single launch, minimum latency, or peak sample."
        ),
        "baseline_stability_policy": {
            "baseline_must_not_be_modified_for_comparison": True,
            "baseline_path": "manual loop over non-grouped tex.quantize split tensors",
            "compare_sustained_baseline_against_prior_reports": True,
            "drift_alarm_fraction": BASELINE_DRIFT_ALARM_FRACTION,
            "drift_requires_investigation_before_acceptance": True,
        },
        "success_fraction_threshold": args.success_fraction_threshold,
        "plateau_policy": {
            "window": args.plateau_window,
            "fraction_of_best": args.plateau_fraction_of_best,
            "max_delta_fraction": args.plateau_max_delta_fraction,
            "acceptance_requires_output_mode": "both",
            "acceptance_requires_api": "group_quantize",
        },
        "speed_of_light": speed_of_light,
        "speed_of_light_usage": "metadata_only_not_acceptance_denominator",
        "roofline_accounting": {
            "acceptance_denominator": (
                "shape-specific calibrated device-to-device copy throughput over the same "
                "estimated HBM byte count"
            ),
            "estimated_physical_global_bytes": (
                "estimated HBM/DRAM traffic used for roofline acceptance"
            ),
            "estimated_global_memory_instruction_bytes": (
                "global-memory load/store instruction traffic, including cache-served duplicate "
                "input loads, reported for diagnostics only"
            ),
            "roofline_fraction_upper_bound": 1.0,
            "roofline_fraction_anomaly_tolerance": ROOFLINE_FRACTION_ANOMALY_TOLERANCE,
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "calibration_warmup": args.calibration_warmup,
        "calibration_iterations": args.calibration_iterations,
        "invocations_per_sample": args.invocations_per_sample,
        "max_invocations_per_sample": args.max_invocations_per_sample,
        "min_sample_ms": args.min_sample_ms,
        "memory_fraction_cap": args.memory_fraction_cap,
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
