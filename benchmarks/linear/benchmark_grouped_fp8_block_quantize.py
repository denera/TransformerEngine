# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 2D block-scaling quantization."""

import argparse
import json
import os
import statistics
import time
from typing import List

import torch

from transformer_engine.pytorch import Float8BlockQuantizer
import transformer_engine_torch as tex


def make_quantizer(rowwise: bool, columnwise: bool) -> Float8BlockQuantizer:
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=2,
    )


def ceildiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def valid_scale_tiles(scale: torch.Tensor, shape: torch.Size, columnwise: bool) -> torch.Tensor:
    row_tiles = ceildiv(shape[0], 128)
    col_tiles = ceildiv(shape[1], 128)
    if columnwise:
        return scale[:col_tiles, :row_tiles]
    return scale[:row_tiles, :col_tiles]


def quantized_equal(lhs, rhs) -> bool:
    for name in ("_rowwise_data", "_columnwise_data"):
        lhs_tensor = getattr(lhs, name, None)
        rhs_tensor = getattr(rhs, name, None)
        if lhs_tensor is None or rhs_tensor is None:
            if lhs_tensor is not None or rhs_tensor is not None:
                return False
            continue
        if lhs_tensor.shape != rhs_tensor.shape or not torch.equal(lhs_tensor, rhs_tensor):
            return False

    for name, columnwise in (
        ("_rowwise_scale_inv", False),
        ("_columnwise_scale_inv", True),
    ):
        lhs_tensor = getattr(lhs, name, None)
        rhs_tensor = getattr(rhs, name, None)
        if lhs_tensor is None or rhs_tensor is None:
            if lhs_tensor is not None or rhs_tensor is not None:
                return False
            continue
        if lhs_tensor.shape != rhs_tensor.shape:
            return False
        if not torch.equal(
            valid_scale_tiles(lhs_tensor, lhs.shape, columnwise),
            valid_scale_tiles(rhs_tensor, rhs.shape, columnwise),
        ):
            return False
    return True


def timed_ms(fn, warmup: int, iters: int, profile: bool) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if profile:
        torch.cuda.cudart().cudaProfilerStart()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    return times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--jagged", action="store_true")
    parser.add_argument("--rowwise", action="store_true", default=True)
    parser.add_argument("--no-columnwise", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-target", choices=("baseline", "grouped", "both"), default="grouped"
    )
    parser.add_argument("--output", default=os.environ.get("ORCHESTRA_BENCHMARK_RAW_REPORT", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    if args.jagged:
        row_delta = max(1, args.rows // 4)
        splits = [
            max(1, args.rows + ((i % 5) - 2) * row_delta) for i in range(args.num_groups)
        ]
    else:
        splits = [args.rows] * args.num_groups
    cols = args.cols
    x = torch.randn(sum(splits), cols, device="cuda", dtype=torch.bfloat16)
    quantizers = [
        make_quantizer(rowwise=args.rowwise, columnwise=not args.no_columnwise) for _ in splits
    ]

    def baseline():
        return [
            tex.quantize(split_x, quantizer)
            for split_x, quantizer in zip(torch.split(x, splits), quantizers)
        ]

    def grouped():
        return tex.split_quantize(x, splits, quantizers)

    baseline_out = baseline()
    grouped_out = grouped()
    correctness = all(quantized_equal(a, b) for a, b in zip(grouped_out, baseline_out))
    if not correctness:
        raise RuntimeError("Grouped FP8 2D quantize output does not match the baseline loop")

    profile_baseline = args.profile and args.profile_target in ("baseline", "both")
    profile_grouped = args.profile and args.profile_target in ("grouped", "both")
    baseline_ms = timed_ms(baseline, args.warmup, args.iters, profile_baseline)
    grouped_ms = timed_ms(grouped, args.warmup, args.iters, profile_grouped)

    report = {
        "benchmark": "grouped_fp8_block_quantize",
        "timestamp_unix": time.time(),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "shape": {
            "num_groups": args.num_groups,
            "splits": splits,
            "cols": cols,
            "rowwise": args.rowwise,
            "columnwise": not args.no_columnwise,
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "profile_target": args.profile_target if args.profile else None,
        "correctness": correctness,
        "baseline_loop_ms": {
            "median": statistics.median(baseline_ms),
            "min": min(baseline_ms),
            "max": max(baseline_ms),
        },
        "grouped_ms": {
            "median": statistics.median(grouped_ms),
            "min": min(grouped_ms),
            "max": max(grouped_ms),
        },
    }
    report["speedup"] = report["baseline_loop_ms"]["median"] / report["grouped_ms"]["median"]

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    print(text)


if __name__ == "__main__":
    main()
