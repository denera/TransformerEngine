#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped FP8 block-scaling quantize benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Dict, List

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


def _manual_loop(inp: torch.Tensor, splits: List[int], quantizer):
    return [tex.quantize(part.contiguous(), quantizer) for part in torch.split(inp, splits)]


def _candidate(inp: torch.Tensor, splits: List[int], quantizer, api: str):
    if api == "group_quantize":
        first_dims = torch.tensor(splits, dtype=torch.int64, device=inp.device)
        grouped = tex.group_quantize(inp, quantizer, len(splits), first_dims)
        return grouped.split_into_quantized_tensors()
    if api == "split_quantize":
        quantizers = [quantizer.copy() for _ in splits]
        return tex.split_quantize(inp, splits, quantizers)
    raise ValueError(f"Unsupported API: {api}")


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


def _time_cuda(fn, iterations: int, profile: bool = False) -> Dict[str, object]:
    times_ms = []
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    return {
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "iterations": iterations,
        "samples_ms": times_ms,
    }


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
            (4, 128, 2048, False),
            (8, 128, 4096, False),
            (4, 512, 7168, True),
            (8, 512, 4096, True),
        ]
    raise ValueError(f"Unsupported suite: {suite}")


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
    profile_candidate_only: bool,
) -> Dict[str, object]:
    splits = _case_splits(num_groups, rows_per_group, jagged)
    inp = torch.randn(sum(splits), cols, dtype=torch.bfloat16, device="cuda")
    quantizer = _make_quantizer(block_scaling_dim, rowwise=True, columnwise=True)

    reference = _manual_loop(inp, splits, quantizer)
    candidate = _candidate(inp, splits, quantizer, api)
    _assert_same(candidate, reference)

    for _ in range(warmup):
        if not profile_candidate_only:
            _manual_loop(inp, splits, quantizer)
        _candidate(inp, splits, quantizer, api)
    torch.cuda.synchronize()

    candidate_timing = _time_cuda(
        lambda: _candidate(inp, splits, quantizer, api),
        iterations,
        profile=profile_candidate_only,
    )
    baseline_timing = None
    if not profile_candidate_only:
        baseline_timing = _time_cuda(lambda: _manual_loop(inp, splits, quantizer), iterations)

    speedup = None
    if baseline_timing is not None:
        speedup = baseline_timing["median_ms"] / candidate_timing["median_ms"]

    return {
        "block_scaling_dim": block_scaling_dim,
        "api": api,
        "num_groups": num_groups,
        "rows_per_group": rows_per_group,
        "cols": cols,
        "splits": splits,
        "jagged": jagged,
        "correctness": "passed",
        "candidate": candidate_timing,
        "baseline_manual_loop": baseline_timing,
        "speedup_baseline_over_candidate": speedup,
        "expected_candidate_main_quantize_launches": 1,
        "baseline_main_quantize_launches": num_groups,
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
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-candidate-only", action="store_true")
    parser.add_argument(
        "--output",
        default=os.environ.get("ORCHESTRA_BENCHMARK_RAW_REPORT", "grouped_fp8_block_quantize.json"),
    )
    args = parser.parse_args()

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
                        profile_candidate_only=args.profile_candidate_only,
                    )
                )

    report = {
        "schema_version": "grouped_fp8_block_quantize_benchmark/v1",
        "gpu": torch.cuda.get_device_name(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "te_version": getattr(te, "__version__", "unknown"),
        "te_module_path": getattr(te, "__file__", "unknown"),
        "te_build_mode": "debug" if os.environ.get("NVTE_BUILD_DEBUG") else "release_or_default",
        "nvte_framework": os.environ.get("NVTE_FRAMEWORK", "unset"),
        "suite": args.suite,
        "profile_candidate_only": args.profile_candidate_only,
        "elapsed_sec": time.time() - started,
        "cases": cases,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
