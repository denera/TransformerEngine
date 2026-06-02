# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 2D block-scaling quantization against a looped baseline."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch

import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.pytorch import Float8BlockQuantizer
import transformer_engine_torch as tex


@dataclass(frozen=True)
class BenchCase:
    name: str
    split_sections: tuple[int, ...]
    k_dim: int


DEFAULT_CASES = (
    BenchCase("g4_small_uniform_k2048", (128, 128, 128, 128), 2048),
    BenchCase("g4_small_jagged_k2048", (64, 193, 0, 319), 2048),
    BenchCase("g8_medium_uniform_k4096", (256,) * 8, 4096),
    BenchCase("g8_medium_jagged_k4096", (64, 512, 128, 320, 0, 768, 192, 384), 4096),
    BenchCase("g8_large_uniform_k7168", (1024,) * 8, 7168),
)


def _make_quantizer(rowwise: bool, columnwise: bool) -> Float8BlockQuantizer:
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=2,
    )


def _sync_time(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _looped_baseline(x: torch.Tensor, split_sections: tuple[int, ...], rowwise: bool, columnwise: bool):
    chunks = torch.split(x, list(split_sections))
    return [_make_quantizer(rowwise, columnwise)(chunk.contiguous()) for chunk in chunks]


def _candidate_group_quantize(
    x: torch.Tensor, split_sections: tuple[int, ...], rowwise: bool, columnwise: bool
):
    first_dims = torch.tensor(split_sections, dtype=torch.int64, device=x.device)
    return tex.group_quantize(x, _make_quantizer(rowwise, columnwise), len(split_sections), first_dims)


def _candidate_split_quantize(
    x: torch.Tensor, split_sections: tuple[int, ...], rowwise: bool, columnwise: bool
):
    quantizers = [_make_quantizer(rowwise, columnwise) for _ in split_sections]
    return tex.split_quantize(x, list(split_sections), quantizers)


def _members(candidate: Any) -> list[Any]:
    if hasattr(candidate, "split_into_quantized_tensors"):
        return candidate.split_into_quantized_tensors()
    return list(candidate)


def _assert_tensor_equal(got, ref) -> None:
    if ref is None:
        if got is not None:
            raise AssertionError("Expected None tensor")
        return
    if got is None:
        raise AssertionError("Expected allocated tensor")
    if got.shape != ref.shape or got.dtype != ref.dtype:
        raise AssertionError(f"Shape/dtype mismatch: got {got.shape}/{got.dtype}, ref {ref.shape}/{ref.dtype}")
    if ref.numel() > 0:
        torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)


def _check_correctness(refs: list[Any], candidate: Any) -> dict[str, Any]:
    outputs = _members(candidate)
    if len(outputs) != len(refs):
        raise AssertionError(f"Expected {len(refs)} outputs, got {len(outputs)}")
    for got, ref in zip(outputs, refs):
        _assert_tensor_equal(got._rowwise_data, ref._rowwise_data)
        _assert_tensor_equal(got._columnwise_data, ref._columnwise_data)
        _assert_tensor_equal(got._rowwise_scale_inv, ref._rowwise_scale_inv)
        _assert_tensor_equal(got._columnwise_scale_inv, ref._columnwise_scale_inv)
    return {
        "passed": True,
        "checked_fields": [
            "rowwise_data",
            "columnwise_data",
            "rowwise_scale_inv",
            "columnwise_scale_inv",
            "split_views",
        ],
    }


def _environment() -> dict[str, Any]:
    return {
        "gpu_model": torch.cuda.get_device_name(),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "transformer_engine_version": getattr(transformer_engine, "__version__", "unknown"),
        "te_build_debug": os.environ.get("NVTE_BUILD_DEBUG", "0"),
        "command": " ".join(sys.argv),
    }


def run_case(args: argparse.Namespace, case: BenchCase) -> dict[str, Any]:
    rowwise = not args.columnwise_only
    columnwise = args.columnwise or args.columnwise_only
    m_dim = sum(case.split_sections)
    x = torch.randn((m_dim, case.k_dim), dtype=getattr(torch, args.dtype), device="cuda")

    candidate_fn = (
        _candidate_split_quantize if args.api == "split_quantize" else _candidate_group_quantize
    )

    refs = _looped_baseline(x, case.split_sections, rowwise, columnwise)
    correctness = _check_correctness(refs, candidate_fn(x, case.split_sections, rowwise, columnwise))

    for _ in range(args.warmup):
        _looped_baseline(x, case.split_sections, rowwise, columnwise)
        candidate_fn(x, case.split_sections, rowwise, columnwise)
    torch.cuda.synchronize()

    baseline_times = [
        _sync_time(lambda: _looped_baseline(x, case.split_sections, rowwise, columnwise))
        for _ in range(args.iterations)
    ]

    candidate_times = []
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for _ in range(args.iterations):
            candidate_times.append(
                _sync_time(lambda: candidate_fn(x, case.split_sections, rowwise, columnwise))
            )
    finally:
        if args.profile:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

    baseline_median = statistics.median(baseline_times)
    candidate_median = statistics.median(candidate_times)
    return {
        "case": case.name,
        "api": args.api,
        "dtype": args.dtype,
        "num_groups": len(case.split_sections),
        "split_sections": list(case.split_sections),
        "m_dim": m_dim,
        "k_dim": case.k_dim,
        "rowwise": rowwise,
        "columnwise": columnwise,
        "baseline": {
            "path": "manual_loop_non_grouped_fp8_block_scaling_2d",
            "median_sec": baseline_median,
            "times_sec": baseline_times,
            "main_quantize_launches_expected": len(case.split_sections),
        },
        "candidate": {
            "path": f"{args.api}_grouped_fp8_block_scaling_2d",
            "median_sec": candidate_median,
            "times_sec": candidate_times,
            "main_quantize_launches_expected": 1,
        },
        "speedup": baseline_median / candidate_median if candidate_median > 0 else None,
        "candidate_over_baseline_ratio": candidate_median / baseline_median
        if baseline_median > 0
        else None,
        "correctness": correctness,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", choices=("group_quantize", "split_quantize"), default="group_quantize")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--columnwise", action="store_true", help="Produce rowwise and columnwise outputs")
    parser.add_argument("--columnwise-only", action="store_true", help="Produce only columnwise outputs")
    parser.add_argument("--profile", action="store_true", help="Use CUDA profiler API around measured candidate iterations")
    parser.add_argument(
        "--output",
        default=os.environ.get("ORCHESTRA_BENCHMARK_RAW_REPORT", "grouped_fp8_block_quantize.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not available:
        raise RuntimeError(f"FP8 block scaling is not available: {reason}")
    if args.columnwise_only:
        args.columnwise = True

    results = {
        "schema_version": "grouped_fp8_block_quantize_benchmark/v1",
        "environment": _environment(),
        "profile_mode": args.profile,
        "profile_after_warmup": args.profile,
        "cases": [run_case(args, case) for case in DEFAULT_CASES],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"output": args.output, "num_cases": len(results["cases"])}, indent=2))


if __name__ == "__main__":
    main()
