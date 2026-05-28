# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import json
import os
import statistics
import time

import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import Float8BlockQuantizer
import transformer_engine_torch as tex


def make_quantizer():
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=2,
    )


def check_equal_outputs(ref_outputs, test_outputs):
    if len(ref_outputs) != len(test_outputs):
        return False
    for ref, got in zip(ref_outputs, test_outputs):
        if not torch.equal(ref._rowwise_data, got._rowwise_data):
            return False
        if not torch.equal(ref._rowwise_scale_inv, got._rowwise_scale_inv):
            return False
        if not torch.equal(ref._columnwise_data, got._columnwise_data):
            return False
        if not torch.equal(ref._columnwise_scale_inv, got._columnwise_scale_inv):
            return False
    return True


def run_baseline(x, splits):
    split_tensors = torch.split(x, splits, dim=0)
    outputs = []
    for t in split_tensors:
        outputs.append(make_quantizer()(t))
    return outputs


def run_grouped(x, splits):
    quantizers = [make_quantizer() for _ in splits]
    return tex.split_quantize(x, splits, quantizers)


def time_op(op, warmup_iters, timed_iters):
    for _ in range(warmup_iters):
        _ = op()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(timed_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = op()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def profile_op(op, warmup_iters, profile_iters):
    for _ in range(warmup_iters):
        _ = op()
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    for _ in range(profile_iters):
        _ = op()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()


def parse_cases():
    # Shape tuples are (splits, k_dim). M is sum(splits).
    return [
        ([512, 512, 512, 512], 2048),
        ([1024, 1024, 1024, 1024], 4096),
        ([2048, 2048, 2048, 2048], 7168),
        ([129, 255, 303, 337], 2048),
        ([65, 127, 191, 255, 319, 383, 447, 511], 4096),
        ([768, 1024, 1280, 1536, 1792, 2048, 2304, 2560], 7168),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--timed-iters", type=int, default=80)
    parser.add_argument("--profile", action="store_true", help="Enable CUDA profiler capture mode.")
    parser.add_argument(
        "--profile-target",
        type=str,
        default="grouped",
        choices=["baseline", "grouped"],
        help="Target op to profile when --profile is set.",
    )
    parser.add_argument("--profile-iters", type=int, default=40)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=os.environ.get("ORCHESTRA_BENCHMARK_RAW_REPORT", "benchmark_raw_report.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not available:
        raise RuntimeError(f"FP8 block scaling is unavailable: {reason}")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    case_results = []
    start_ts = time.time()
    for splits, k_dim in parse_cases():
        m_dim = sum(splits)
        x = torch.randn((m_dim, k_dim), dtype=dtype, device="cuda")

        baseline_op = lambda: run_baseline(x, splits)
        grouped_op = lambda: run_grouped(x, splits)

        baseline_ref = baseline_op()
        grouped_ref = grouped_op()
        outputs_match = check_equal_outputs(baseline_ref, grouped_ref)

        baseline_times = time_op(baseline_op, args.warmup_iters, args.timed_iters)
        grouped_times = time_op(grouped_op, args.warmup_iters, args.timed_iters)
        baseline_p50 = statistics.median(baseline_times)
        grouped_p50 = statistics.median(grouped_times)
        speedup = baseline_p50 / grouped_p50 if grouped_p50 > 0.0 else float("inf")

        case_results.append(
            {
                "splits": splits,
                "m": m_dim,
                "k": k_dim,
                "num_groups": len(splits),
                "outputs_match": outputs_match,
                "baseline_p50_ms": baseline_p50,
                "grouped_p50_ms": grouped_p50,
                "speedup_ratio_baseline_over_grouped": speedup,
            }
        )

        if args.profile:
            if args.profile_target == "baseline":
                profile_op(baseline_op, args.warmup_iters, args.profile_iters)
            else:
                profile_op(grouped_op, args.warmup_iters, args.profile_iters)

    output = {
        "benchmark": "grouped_fp8_block_scaling_split_quantize",
        "timestamp_unix_sec": start_ts,
        "dtype": args.dtype,
        "warmup_iters": args.warmup_iters,
        "timed_iters": args.timed_iters,
        "profile_enabled": args.profile,
        "profile_target": args.profile_target if args.profile else None,
        "profile_iters": args.profile_iters if args.profile else None,
        "cases": case_results,
    }

    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
