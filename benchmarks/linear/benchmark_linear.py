# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

from transformer_engine.common.recipe import Float8BlockScaling, MXFP8BlockScaling, NVFP4BlockScaling
from transformer_engine.pytorch.module import Linear as TELinear
from transformer_engine.pytorch.quantization import FP8GlobalStateManager, autocast

"""
# Profile BF16 recipe with Nsight Systems
nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --output=./benchmarks/linear/b200_linear_bf16 \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_linear.py --profile --recipe bf16

# Profile FP8 sub-channel recipe with Nsight Systems
nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --output=./benchmarks/linear/b200_linear_fp8_sub_channel \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_linear.py --profile --recipe fp8_sub_channel

# Profile MXFP8 recipe with Nsight Systems
nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --output=./benchmarks/linear/b200_linear_mxfp8 \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_linear.py --profile --recipe mxfp8

# Profile NVFP4 recipe with Nsight Systems
nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --output=./benchmarks/linear/b200_linear_nvfp4_rht_cast_fusion \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_linear.py --profile --recipe nvfp4

# Example to look at a single kernel target with NCU, like the fused hadamard amax kernel for NVFP4 recipe
ncu -f -o ./benchmarks/linear/ncu_b200_linear_nvfp4_rht_cast_fusion \
    --set=full \
    --kernel-name "row_col_rht_gemm_device" \
    -s 5 -c 5 \
    python benchmarks/linear/benchmark_linear.py --profile --recipe nvfp4
"""

RECIPES = {
    "bf16": None,
    "fp8_sub_channel": Float8BlockScaling(),
    "mxfp8": MXFP8BlockScaling(),
    "nvfp4": NVFP4BlockScaling(),
}

mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
    FP8GlobalStateManager.is_fp8_block_scaling_available()
)
nvfp4_available, reason_for_no_nvfp4 = FP8GlobalStateManager.is_nvfp4_available()


def recipe_skip_reason(recipe_name):
    if recipe_name == "mxfp8" and not mxfp8_available:
        return reason_for_no_mxfp8
    if recipe_name == "fp8_sub_channel" and not fp8_block_scaling_available:
        return reason_for_no_fp8_block_scaling
    if recipe_name == "nvfp4" and not nvfp4_available:
        return reason_for_no_nvfp4
    return ""


def run_linear_multiple_steps(layer, x, mode, gradient, run_num_steps=1, recipe=None):
    assert mode in ["fwd_only", "fwd_bwd"]
    quantization_context = autocast(enabled=True, recipe=recipe) if recipe is not None else nullcontext()

    if mode == "fwd_only":
        with torch.no_grad(), quantization_context:
            for i in range(run_num_steps):
                y_q = layer.forward(
                    x,
                    is_first_microbatch=(i == 0),
                )
        return y_q

    layer.zero_grad()
    x.grad = None

    with quantization_context:
        for i in range(run_num_steps):
            label = f"step_{i}"
            torch.cuda.nvtx.range_push(label)
            y_q = layer.forward(
                x,
                is_first_microbatch=(i == 0),
            )
            y_q.backward(gradient)
            torch.cuda.nvtx.range_pop()

    grads_q = [x.grad]
    for p in layer.parameters():
        if p.requires_grad:
            grads_q.append(p.grad)

    return y_q, grads_q


def build_linear_case(m, k, n, bias):
    device = "cuda"
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device, requires_grad=True)
    w = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    gradient = torch.ones((m, n), dtype=torch.bfloat16, device=device)

    layer = TELinear(
        k,
        n,
        bias=bias,
        params_dtype=torch.bfloat16,
    ).to(device)

    with torch.no_grad():
        layer.weight.copy_(w)

    return layer, x, gradient


def benchmark_linear(layer, x, mode, gradient, recipe_name, num_microbatches=32, min_run_time=10.0):
    recipe = RECIPES[recipe_name]
    label = f"{recipe_name}_linear"
    torch.cuda.nvtx.range_push(label)
    timing = benchmark.Timer(
        stmt="run_linear_multiple_steps(layer, x, mode, gradient, num_microbatches, recipe)",
        globals={
            "run_linear_multiple_steps": run_linear_multiple_steps,
            "layer": layer,
            "x": x,
            "mode": mode,
            "gradient": gradient,
            "num_microbatches": num_microbatches,
            "recipe": recipe,
        },
        num_threads=1,
    ).blocked_autorange(min_run_time=min_run_time)
    torch.cuda.nvtx.range_pop()
    print(f"{recipe_name}: {timing}\n")
    return timing.median * 1000 / num_microbatches


def profile_linear(layer, x, mode, gradient, recipe_name, warmup_steps, profile_steps):
    if profile_steps < 1:
        raise ValueError("--profile-steps must be >= 1")

    recipe = RECIPES[recipe_name]
    if warmup_steps > 0:
        torch.cuda.nvtx.range_push(f"{recipe_name}_profile_warmup")
        run_linear_multiple_steps(layer, x, mode, gradient, warmup_steps, recipe)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    cudart = torch.cuda.cudart()

    torch.cuda.nvtx.range_push(f"{recipe_name}_profile_capture")
    cudart.cudaProfilerStart()
    start_event.record()
    run_linear_multiple_steps(layer, x, mode, gradient, profile_steps, recipe)
    end_event.record()
    cudart.cudaProfilerStop()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms / profile_steps


def run_benchmark_linear(
    mkns,
    recipe_name,
    use_bias,
    fwd_only=False,
    profile=False,
    profile_warmup_steps=5,
    profile_steps=20,
    benchmark_microbatches=32,
    benchmark_min_run_time=10.0,
):
    data = []
    assert not use_bias, "Bias is not supported in this benchmark script"

    metric_name = "linear_fwd_time_ms" if fwd_only else "linear_fwd_bwd_time_ms"
    if profile:
        metric_name = "linear_profile_step_time_ms"

    print(f"========== Benchmarking {recipe_name} ==========")
    for m, k, n in mkns:
        layer, x, gradient = build_linear_case(m, k, n, bias=False)
        print(f"fwd_m={m}, fwd_k={k}, fwd_n={n}")
        print(f"fwd_only: {fwd_only}")

        mode = "fwd_only" if fwd_only else "fwd_bwd"
        if profile:
            timing_ms = profile_linear(
                layer,
                x,
                mode,
                gradient,
                recipe_name,
                profile_warmup_steps,
                profile_steps,
            )
        else:
            timing_ms = benchmark_linear(
                layer,
                x,
                mode,
                gradient,
                recipe_name,
                num_microbatches=benchmark_microbatches,
                min_run_time=benchmark_min_run_time,
            )

        data.append([m, k, n, recipe_name, timing_ms])

    df = pd.DataFrame(data=data, columns=["m", "k", "n", "recipe", metric_name])
    print(df, "\n")
    return df


def output_csv_path(output_dir, output_file, profile, recipe, fwd_only):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    mode = "fwd" if fwd_only else "fwd_bwd"
    run_kind = "profile" if profile else "benchmark"
    return out_dir / f"benchmark_linear_{run_kind}_{recipe}_{mode}.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_output/",
        help="Output path for benchmark report",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional explicit CSV path. Overrides the default file naming.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="bf16",
        help="Recipe to use, options are fp8_sub_channel, mxfp8, nvfp4, bf16, or all",
    )
    parser.add_argument(
        "--token-dim",
        type=int,
        default=None,
        help="Token dimension to use, calculated by SEQ_LEN * MBS / TP_SIZE",
    )
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden dimension to use")
    parser.add_argument("--output-dim", type=int, default=None, help="Output dimension to use")
    parser.add_argument(
        "--fwd-only",
        action="store_true",
        default=False,
        help="Run forward pass only, default is both forward and backward passes",
    )
    parser.add_argument(
        "--benchmark-min-run-time",
        type=float,
        default=10.0,
        help="Min run time in seconds for torch benchmark.Timer",
    )
    parser.add_argument(
        "--benchmark-microbatches",
        type=int,
        default=32,
        help="Number of microbatches per timed call in benchmark mode",
    )
    parser.add_argument(
        "--profile-warmup-steps",
        type=int,
        default=5,
        help="Warmup steps before cudaProfilerStart in profile mode",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=20,
        help="Profiled steps between cudaProfilerStart/Stop in profile mode",
    )
    parser.add_argument(
        "--strict-recipe",
        action="store_true",
        help="Fail when requested recipe is unsupported on the current platform",
    )
    args = parser.parse_args()

    use_bias = False

    token_dim_list = [16384]
    hidden_dim_list = [4096]
    output_dim_list = [4096]

    if args.token_dim is not None:
        token_dim_list = [args.token_dim]
    if args.hidden_dim is not None:
        hidden_dim_list = [args.hidden_dim]
    if args.output_dim is not None:
        output_dim_list = [args.output_dim]

    mkns = [(m, k, n) for m in token_dim_list for k in hidden_dim_list for n in output_dim_list]

    if args.recipe == "all":
        recipe_list = ["bf16", "fp8_sub_channel", "mxfp8", "nvfp4"]
    else:
        recipe_list = [args.recipe]

    if args.profile:
        hidden_dim_to_profile = 4096 if args.hidden_dim is None else args.hidden_dim
        output_dim_to_profile = 4096 if args.output_dim is None else args.output_dim
        token_dim_to_profile = 16384 if args.token_dim is None else args.token_dim
        mkns = [(token_dim_to_profile, hidden_dim_to_profile, output_dim_to_profile)]
        assert args.recipe != "all", (
            "In profile mode, only one recipe can be specified, please specify the recipe as"
            " fp8_sub_channel, mxfp8, nvfp4, or bf16"
        )
        recipe_list = [args.recipe]

    if args.profile_warmup_steps < 0:
        raise ValueError("--profile-warmup-steps must be >= 0")
    if args.benchmark_microbatches < 1:
        raise ValueError("--benchmark-microbatches must be >= 1")

    df_linears = pd.DataFrame()

    for recipe_name in recipe_list:
        assert recipe_name in RECIPES, (
            "Recipe must be one of bf16, fp8_sub_channel, mxfp8, nvfp4, or all"
        )
        skip_reason = recipe_skip_reason(recipe_name)
        if skip_reason:
            print(f"RECIPE_STATUS:{recipe_name}:SKIP_UNSUPPORTED:{skip_reason}")
            if args.strict_recipe:
                raise RuntimeError(f"Recipe {recipe_name} is unsupported: {skip_reason}")
            continue

        print(f"RECIPE_STATUS:{recipe_name}:PASS")
        df = run_benchmark_linear(
            mkns,
            recipe_name,
            use_bias,
            fwd_only=args.fwd_only,
            profile=args.profile,
            profile_warmup_steps=args.profile_warmup_steps,
            profile_steps=args.profile_steps,
            benchmark_microbatches=args.benchmark_microbatches,
            benchmark_min_run_time=args.benchmark_min_run_time,
        )
        df_linears = pd.concat([df_linears, df])

    print(df_linears)

    csv_path = output_csv_path(
        args.output_dir,
        args.output_file,
        profile=args.profile,
        recipe=args.recipe,
        fwd_only=args.fwd_only,
    )
    df_linears.to_csv(csv_path, index=False)
    print(f"RESULT_CSV:{csv_path}")
