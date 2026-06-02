#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped FP8 block-scaling quantize benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import statistics
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
        for name in (
            "_rowwise_data",
            "_columnwise_data",
            "_rowwise_scale_inv",
            "_columnwise_scale_inv",
        ):
            got_tensor = getattr(got, name)
            ref_tensor = getattr(ref, name)
            if ref_tensor is None:
                if got_tensor is not None:
                    raise AssertionError(f"{name} unexpectedly present")
                continue
            if got_tensor is None:
                raise AssertionError(f"{name} missing")
            torch.testing.assert_close(got_tensor, ref_tensor, rtol=0, atol=0)


def _case_splits(num_groups: int, rows_per_group: int, jagged: bool) -> List[int]:
    if not jagged:
        return [rows_per_group] * num_groups
    pattern = [512, 1024, 256, 2048, 768, 128, 1536, 640]
    return [pattern[i % len(pattern)] for i in range(num_groups)]


def _shape_suite(suite: str, num_groups: int, rows_per_group: int, cols: int, jagged: bool):
    if suite == "single":
        return [(num_groups, rows_per_group, cols, jagged)]
    if suite == "work_order":
        return [
            (4, 64, 2048, False),
            (4, 128, 2048, False),
            (8, 128, 4096, False),
            (4, 512, 7168, True),
            (8, 512, 4096, True),
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


def _bandwidth_gbps(actual_bytes: int, median_ms: float) -> Optional[float]:
    if median_ms <= 0:
        return None
    return actual_bytes / median_ms / 1.0e6


def _time_cuda(
    fn: Callable[[], object],
    *,
    samples: int,
    invocations_per_sample: int,
    profile: bool,
) -> Dict[str, object]:
    times_total_ms = []
    times_per_request_ms = []
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(invocations_per_sample):
                fn()
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)
            times_total_ms.append(elapsed_ms)
            times_per_request_ms.append(elapsed_ms / invocations_per_sample)
    finally:
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
    return {
        "median_ms": statistics.median(times_per_request_ms),
        "min_ms": min(times_per_request_ms),
        "max_ms": max(times_per_request_ms),
        "samples": samples,
        "invocations_per_sample": invocations_per_sample,
        "total_invocations": samples * invocations_per_sample,
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
    profile_candidate_only: bool,
    collect_launch_evidence: bool,
    evidence_invocations: int,
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
        profile=profile_candidate_only,
    )
    candidate_timing["actual_bytes_per_request"] = actual_bytes
    candidate_timing["bandwidth_GBps_actual_bytes"] = _bandwidth_gbps(
        actual_bytes, candidate_timing["median_ms"]
    )

    baseline_timing = None
    if not profile_candidate_only:
        baseline_timing = _time_cuda(
            run_baseline,
            samples=iterations,
            invocations_per_sample=invocations_per_sample,
            profile=False,
        )
        baseline_timing["actual_bytes_per_request"] = actual_bytes
        baseline_timing["bandwidth_GBps_actual_bytes"] = _bandwidth_gbps(
            actual_bytes, baseline_timing["median_ms"]
        )

    speedup = None
    if baseline_timing is not None:
        speedup = baseline_timing["median_ms"] / candidate_timing["median_ms"]

    return {
        "block_scaling_dim": block_scaling_dim,
        "api": api,
        "first_dims_mode": (
            "none_uniform_group_quantize"
            if api == "group_quantize" and first_dims is None
            else "device_first_dims"
        ),
        "num_groups": num_groups,
        "rows_per_group": rows_per_group,
        "cols": cols,
        "splits": splits,
        "jagged": jagged,
        "correctness": {
            "status": "passed",
            "reference": "manual_loop_preallocated_non_grouped_tex_quantize",
        },
        "actual_bytes_per_request": actual_bytes,
        "candidate": candidate_timing,
        "candidate_output_preallocated": api == "group_quantize",
        "baseline_manual_loop": baseline_timing,
        "speedup_baseline_over_candidate": speedup,
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
    parser.add_argument("--suite", choices=["single", "work_order"], default="single")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--invocations-per-sample", type=int, default=20)
    parser.add_argument("--profile-candidate-only", action="store_true")
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
    if args.evidence_invocations <= 0:
        raise ValueError("--evidence-invocations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not available:
        raise RuntimeError(reason)

    started = time.time()
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
                        profile_candidate_only=args.profile_candidate_only,
                        collect_launch_evidence=args.launch_evidence == "profiler",
                        evidence_invocations=args.evidence_invocations,
                    )
                )

    report = {
        "schema_version": "grouped_fp8_block_quantize_benchmark/v2",
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
        "warmup": args.warmup,
        "iterations": args.iterations,
        "invocations_per_sample": args.invocations_per_sample,
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
