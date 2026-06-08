#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GroupedLinear FP8 block-scaling end-to-end benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Callable, Dict, List

import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling
from transformer_engine.pytorch import Float8BlockQuantizer, autocast
import transformer_engine.pytorch.module.grouped_linear as grouped_linear_module
from transformer_engine.pytorch.module import GroupedLinear
import transformer_engine_torch as tex


def _parse_csv_ints(value: str) -> List[int]:
    return [int(part) for part in value.split(",") if part]


def _parse_csv_strings(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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
                sample_invocations = min(sample_invocations * 2, max_invocations_per_sample)
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


def _make_splits(total_m: int, num_gemms: int, layout: str) -> List[int]:
    if layout == "uniform":
        if total_m % num_gemms != 0:
            raise ValueError("Uniform layout requires total M divisible by num_gemms")
        return [total_m // num_gemms] * num_gemms
    if layout != "jagged":
        raise ValueError(f"Unsupported layout: {layout}")
    pattern = [512, 1024, 256, 2048, 768, 128, 1536, 640]
    base = [pattern[i % len(pattern)] for i in range(num_gemms)]
    scale = max(1, total_m // sum(base))
    splits = [value * scale for value in base]
    splits[-1] += total_m - sum(splits)
    if any(split <= 0 or split % 4 != 0 for split in splits):
        raise ValueError(f"Jagged splits must be positive and divisible by 4, got {splits}")
    return splits


def _make_block_weight_quantizer() -> Float8BlockQuantizer:
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=2,
    )


def _run_step(layer, x, gradient, m_splits, mode: str, recipe):
    layer.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    context = torch.no_grad() if mode == "fwd_only" else torch.enable_grad()
    with context, autocast(enabled=True, recipe=recipe):
        y = layer(x, m_splits, is_first_microbatch=True)
        if mode == "fwd_bwd":
            y.backward(gradient)
    return y


def _time_full_step(
    *,
    layer,
    x,
    gradient,
    m_splits,
    mode: str,
    recipe,
    disable_grouped_weight_update: bool,
    warmup: int,
    samples: int,
    invocations_per_sample: int,
    max_invocations_per_sample: int,
    min_sample_ms: float,
    profile: bool,
) -> Dict[str, object]:
    original_helper = grouped_linear_module._try_group_quantize_fp8_block_weights
    if disable_grouped_weight_update:
        grouped_linear_module._try_group_quantize_fp8_block_weights = lambda *args, **kwargs: None
    try:
        for _ in range(warmup):
            _run_step(layer, x, gradient, m_splits, mode, recipe)
        torch.cuda.synchronize()
        timing = _time_cuda(
            lambda: _run_step(layer, x, gradient, m_splits, mode, recipe),
            samples=samples,
            invocations_per_sample=invocations_per_sample,
            max_invocations_per_sample=max_invocations_per_sample,
            min_sample_ms=min_sample_ms,
            profile=profile,
        )
    finally:
        grouped_linear_module._try_group_quantize_fp8_block_weights = original_helper
    timing["grouped_weight_update_enabled"] = not disable_grouped_weight_update
    return timing


def _time_weight_update_components(
    *,
    layer,
    m_splits,
    warmup: int,
    samples: int,
    invocations_per_sample: int,
    max_invocations_per_sample: int,
    min_sample_ms: float,
) -> Dict[str, object]:
    weights = tuple(getattr(layer, f"weight{i}") for i in range(layer.num_gemms))
    quantizer = _make_block_weight_quantizer()
    weight_quantizers = [quantizer.copy() for _ in weights]
    packed = None

    def pack_weights():
        nonlocal packed
        packed = torch.cat(
            [weight if weight.is_contiguous() else weight.contiguous() for weight in weights],
            dim=0,
        )
        return packed

    for _ in range(warmup):
        pack_weights()
    torch.cuda.synchronize()
    pack_timing = _time_cuda(
        pack_weights,
        samples=samples,
        invocations_per_sample=invocations_per_sample,
        max_invocations_per_sample=max_invocations_per_sample,
        min_sample_ms=min_sample_ms,
        profile=False,
    )
    if packed is None:
        packed = pack_weights()
    grouped_output = tex.group_quantize(packed, quantizer, len(weights))

    def group_quantize_weights():
        return tex.group_quantize(packed, quantizer, len(weights), None, grouped_output)

    for _ in range(warmup):
        group_quantize_weights()
    torch.cuda.synchronize()
    group_quantize_timing = _time_cuda(
        group_quantize_weights,
        samples=samples,
        invocations_per_sample=invocations_per_sample,
        max_invocations_per_sample=max_invocations_per_sample,
        min_sample_ms=min_sample_ms,
        profile=False,
    )

    def helper_update():
        return grouped_linear_module._try_group_quantize_fp8_block_weights(
            weights,
            weight_quantizers,
            m_splits=m_splits,
            debug=False,
            cache_weight=False,
            weight_workspaces=[None] * len(weights),
            skip_fp8_weight_update=None,
        )

    for _ in range(warmup):
        helper_update()
    torch.cuda.synchronize()
    helper_timing = _time_cuda(
        helper_update,
        samples=samples,
        invocations_per_sample=invocations_per_sample,
        max_invocations_per_sample=max_invocations_per_sample,
        min_sample_ms=min_sample_ms,
        profile=False,
    )
    return {
        "weight_pack_torch_cat": pack_timing,
        "weight_group_quantize_prepacked": group_quantize_timing,
        "weight_update_helper_pack_plus_group_quantize": helper_timing,
        "weight_update_helper_eligible_for_m_splits": all(
            split == m_splits[0] for split in m_splits
        ),
        "packed_weight_bytes": packed.numel() * packed.element_size(),
    }


def _run_case(
    *,
    total_m: int,
    k: int,
    n: int,
    num_gemms: int,
    layout: str,
    mode: str,
    warmup: int,
    iterations: int,
    invocations_per_sample: int,
    max_invocations_per_sample: int,
    min_sample_ms: float,
    profile_candidate_only: bool,
) -> Dict[str, object]:
    torch.manual_seed(1234)
    m_splits = _make_splits(total_m, num_gemms, layout)
    x = torch.randn((total_m, k), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    gradient = torch.ones((total_m, n), dtype=torch.bfloat16, device="cuda")
    layer = GroupedLinear(
        num_gemms,
        k,
        n,
        bias=False,
        params_dtype=torch.bfloat16,
        fuse_wgrad_accumulation=False,
        device="cuda",
    )
    recipe = Float8BlockScaling()
    grouped_weight_update_eligible = all(split == m_splits[0] for split in m_splits)

    enabled_timing = _time_full_step(
        layer=layer,
        x=x,
        gradient=gradient,
        m_splits=m_splits,
        mode=mode,
        recipe=recipe,
        disable_grouped_weight_update=False,
        warmup=warmup,
        samples=iterations,
        invocations_per_sample=invocations_per_sample,
        max_invocations_per_sample=max_invocations_per_sample,
        min_sample_ms=min_sample_ms,
        profile=profile_candidate_only,
    )
    enabled_timing["grouped_weight_update_eligible_for_m_splits"] = grouped_weight_update_eligible
    fallback_timing = _time_full_step(
        layer=layer,
        x=x,
        gradient=gradient,
        m_splits=m_splits,
        mode=mode,
        recipe=recipe,
        disable_grouped_weight_update=True,
        warmup=warmup,
        samples=iterations,
        invocations_per_sample=invocations_per_sample,
        max_invocations_per_sample=max_invocations_per_sample,
        min_sample_ms=min_sample_ms,
        profile=False,
    )
    component_timings = _time_weight_update_components(
        layer=layer,
        m_splits=m_splits,
        warmup=max(1, warmup),
        samples=iterations,
        invocations_per_sample=invocations_per_sample,
        max_invocations_per_sample=max_invocations_per_sample,
        min_sample_ms=min_sample_ms,
    )
    speedup = fallback_timing["median_ms"] / enabled_timing["median_ms"]
    return {
        "case_id": f"grouped_linear_{mode}_{layout}_g{num_gemms}_m{total_m}_k{k}_n{n}",
        "status": "completed",
        "mode": mode,
        "layout": layout,
        "num_gemms": num_gemms,
        "m": total_m,
        "k": k,
        "n": n,
        "m_splits": m_splits,
        "grouped_weight_update_eligible_for_m_splits": grouped_weight_update_eligible,
        "candidate_grouped_weight_update": enabled_timing,
        "baseline_per_weight_update_loop": fallback_timing,
        "speedup_baseline_over_candidate": speedup,
        "component_timings": component_timings,
        "primary_metric": "full_step_speedup_baseline_over_candidate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", default="65536,98304")
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--num-gemms-values", default="4,8")
    parser.add_argument("--layouts", default="uniform,jagged")
    parser.add_argument("--mode", choices=["fwd_only", "fwd_bwd"], default="fwd_bwd")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--invocations-per-sample", type=int, default=1)
    parser.add_argument("--max-invocations-per-sample", type=int, default=16)
    parser.add_argument("--min-sample-ms", type=float, default=0.0)
    parser.add_argument("--profile-candidate-only", action="store_true")
    parser.add_argument(
        "--output",
        default=os.environ.get(
            "ORCHESTRA_GROUPED_LINEAR_E2E_REPORT", "grouped_linear_fp8_block_e2e.json"
        ),
    )
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.invocations_per_sample <= 0:
        raise ValueError("--invocations-per-sample must be positive")
    if args.max_invocations_per_sample < args.invocations_per_sample:
        raise ValueError("--max-invocations-per-sample must be at least --invocations-per-sample")
    if args.min_sample_ms < 0:
        raise ValueError("--min-sample-ms must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not available:
        raise RuntimeError(reason)

    started = time.time()
    cases = []
    for num_gemms in _parse_csv_ints(args.num_gemms_values):
        for total_m in _parse_csv_ints(args.m_values):
            for layout in _parse_csv_strings(args.layouts):
                cases.append(
                    _run_case(
                        total_m=total_m,
                        k=args.k,
                        n=args.n,
                        num_gemms=num_gemms,
                        layout=layout,
                        mode=args.mode,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        invocations_per_sample=args.invocations_per_sample,
                        max_invocations_per_sample=args.max_invocations_per_sample,
                        min_sample_ms=args.min_sample_ms,
                        profile_candidate_only=args.profile_candidate_only,
                    )
                )

    report = {
        "schema_version": "grouped_linear_fp8_block_e2e_benchmark/v1",
        "command": " ".join(sys.argv),
        "gpu": torch.cuda.get_device_name(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "te_version": getattr(te, "__version__", "unknown"),
        "te_module_path": getattr(te, "__file__", "unknown"),
        "profile_candidate_only": args.profile_candidate_only,
        "profile_after_warmup": args.profile_candidate_only,
        "primary_metric": "full_step_speedup_baseline_over_candidate",
        "higher_is_better": True,
        "baseline_comparison": "same_session_grouped_weight_helper_disabled",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "invocations_per_sample": args.invocations_per_sample,
        "max_invocations_per_sample": args.max_invocations_per_sample,
        "min_sample_ms": args.min_sample_ms,
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
