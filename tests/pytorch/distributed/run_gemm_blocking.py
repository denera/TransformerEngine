#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import socket
import warnings
import subprocess
import argparse
import operator
from functools import partial, reduce

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.fp8 import _default_sf_compute

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch_dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def _mapped_argtype(opt, typemap):
    if str(opt).lower() not in typemap.keys():
        raise TypeError(f"Unrecognized option! Please choose from: {typemap.keys()}")
    return typemap[str(opt).lower()]


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Test comm+GEMM overlap with Userbuffers.")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Input sequence length.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=64, help="Number of attention heads."
    )
    parser.add_argument(
        "-d", "--head-dim", type=int, default=128, help="Dimension of each attention head."
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument(
        "--fp8", action="store_true", default=False, help="Enables the te.fp8_autocast() context."
    )
    parser.add_argument(
        "--comm-type",
        type=str.upper,
        default="AG",
        choices=["AG", "RS"],
        help="Comm type to overlap.",
    )
    parser.add_argument(
        "--check-numerics",
        action="store_true",
        default=False,
        help="Test numerical result against torch.matmul(...)",
    )
    parser.add_argument(
        "--scale", type=float, default=1e-2, help="Set scaling factor for input and weight tensors."
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Run some warmup iterations of the comm+GEMM overlap before " + "the timing runs.",
    )
    parser.add_argument(
        "--timing-iters",
        type=int,
        default=1,
        help="Benchmark the comm+GEMM overlap as an average of many iterations.",
    )
    parser.add_argument(
        "--clock-speed",
        type=int,
        default=-1,
        help="Set device clock speed to a fixed value via `nvidia-smi`.",
    )
    parser.add_argument(
        "--tcp-init",
        action="store_true",
        default=False,
        help="Initialize torch.distributed with TcpStore.",
    )
    parser.add_argument(
        "--init-method", type=str, default=None, help="Set the torch.distributed init method."
    )
    parser.add_argument(
        "--bind-to-device",
        action="store_true",
        default=False,
        help=(
            "Initialize torch.distributed with 'device_id' argument to bind each rank to 1 device."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Verbose info messages."
    )
    opts = parser.parse_args(argv, namespace)

    return opts


@record
def _main(opts):
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        # Execution with `mpirun -np N`
        WORLD_RANK = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
        WORLD_SIZE = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))
        opts.tcp_init = True
        opts.bootstrap_backend = "mpi"
    elif "TORCHELASTIC_RUN_ID" in os.environ:
        WORLD_RANK = int(os.getenv("RANK", "0"))
        WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    else:
        raise RuntimeError(f"{__file__} must be launched with either `mpirun` or `torchrun`!")
    assert WORLD_SIZE == LOCAL_SIZE  # this test supports only 1 node
    assert LOCAL_SIZE <= torch.cuda.device_count()

    # Fix clock speed
    torch.cuda.set_device(LOCAL_RANK)
    if opts.clock_speed > 0:
        subprocess.run(
            ["nvidia-smi", "-pm", "ENABLED", "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result = subprocess.run(
            ["nvidia-smi", "-lgc", str(opts.clock_speed), "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        msg = result.stdout.decode("utf-8").splitlines()[0]
        print(f"[rank:{LOCAL_RANK}] {msg}\n", end="", flush=True)

    # Info printout
    def dist_print(msg, src=None, debug=False, section=False, group=None):
        rank = dist.get_rank(group)
        if debug or opts.verbose:
            if section:
                if rank == (0 if src is None else src):
                    print("\n", end="", flush=True)
                dist.barrier(group)
            if src is None or rank == src:
                prefix = "[GLOBAL] " if src is not None else f"[rank:{rank}] "
                lines = msg.splitlines()
                msg = "\n".join(
                    [prefix + lines[0]] + [(" " * len(prefix)) + line for line in lines[1:]]
                )
                print(msg + "\n", end="", flush=True)

    # Initialize torch.distributed global process group and get TP group
    dist_init_kwargs = {
        "backend": "nccl",
        "rank": WORLD_RANK,
        "world_size": WORLD_SIZE,
    }
    if opts.tcp_init:
        if opts.init_method is not None:
            assert opts.init_method.startswith("tcp://")
            init_method = opts.init_method
        else:
            MASTER_ADDR = os.getenv("MASTER_ADDR", socket.gethostname())
            MASTER_PORT = os.getenv("MASTER_PORT", "1234")
            init_method = f"tcp://{MASTER_ADDR}:{MASTER_PORT}"
        dist_init_kwargs["init_method"] = init_method
    elif opts.init_method is not None:
        assert (
            opts.init_method.startswith("env://")
            or opts.init_method.startswith("file://")
            or opts.init_method.startswith("tcp://")
        )
        dist_init_kwargs["init_method"] = opts.init_method
    if opts.bind_to_device:
        dist_init_kwargs["device_id"] = torch.device(f"cuda:{LOCAL_RANK}")
    assert dist.is_nccl_available()
    dist.init_process_group(**dist_init_kwargs)
    tp_group = dist.new_group(backend="nccl")
    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    dist_print(
        f"Initialized default NCCL process group with {tp_size} GPUs",
        src=0,
        section=True,
        debug=True,
        group=tp_group,
    )

    # Initialize userbuffers with (M, N) buffer
    # M = sequence * batch
    # N = hidden size
    hidden_size = opts.num_heads * opts.head_dim
    inp_shape = (opts.seq_length, opts.batch_size, hidden_size)
    outer_size = reduce(operator.mul, inp_shape[:-1], 1)

    # Figure out problem sizing:
    # M = sequence * batch
    # N = hidden size
    # K = MLP intermediate size (usually 4x hidden size)
    # P = number of devices for sequence/tensor parallelism
    # NOTE: TE-GEMM is set up to work with a transposed kernels and  non-transposed inputs.
    ffn_hidden_size = 4 * hidden_size
    if opts.comm_type == "AG":
        # (M/P, N) -> overlapped AG -> (M, N) x (K/P, N)^T = (M, K/P)
        local_kernel_t_shape = (ffn_hidden_size // tp_size, hidden_size)
        local_inp_shape = (outer_size // tp_size, hidden_size)
    else:
        # (M, K/P) x (N, K/P)^T = (M, N) -> overlapped RS -> (M/P, N)
        local_kernel_t_shape = (hidden_size, ffn_hidden_size // tp_size)
        local_inp_shape = (outer_size, ffn_hidden_size // tp_size)

    # Initialize distributed input tensor and GEMM kernels
    torch.manual_seed(opts.seed + tp_rank)
    torch.cuda.manual_seed(opts.seed + tp_rank)
    inp = torch.mul(torch.rand(local_inp_shape, dtype=torch.bfloat16, device="cuda"), opts.scale)
    kernel_t = torch.mul(
        torch.rand(local_kernel_t_shape, dtype=torch.bfloat16, device="cuda"), opts.scale
    )

    # Gather global tensors and calculate reference result (need these first for Fp8 scales)
    if opts.comm_type == "AG":
        # AG Kernel: (K/P, N) -> gather -> (K, N) -> T -> (N, K)
        ker_g = torch.transpose(
            te.distributed.gather_along_first_dim(kernel_t, tp_group)[0], 0, 1
        )
        # AG Input: (M/P, N) -> gather -> (M, N)
        inp_g = te.distributed.gather_along_first_dim(inp, tp_group)[0]
    else:
        # RS Kernel: (N, K/P) -> T -> (K/P, N) -> gather -> (K, N)
        ker_g = te.distributed.gather_along_first_dim(
            torch.transpose(kernel_t, 0, 1), tp_group
        )[0]
        # RS Input: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
        inp_g = torch.transpose(
            te.distributed.gather_along_first_dim(torch.transpose(inp, 0, 1), tp_group)[0], 0, 1
        )

    ref_g = torch.matmul(inp_g, ker_g)

    if opts.fp8:
        fp8_formats = {
            tex.DType.kFloat8E4M3: Format.E4M3,
            tex.DType.kFloat8E5M2: Format.E5M2,
        }

        # Structure to maintain amax and scale/scale_inv information for the kernel and input
        fp8_dtype = tex.DType.kFloat8E4M3
        fp8_meta = tex.FP8TensorMeta()
        fp8_meta.amax_history = torch.zeros((2, 3), dtype=torch.float, device="cuda")
        fp8_meta.scale = torch.ones(3, dtype=torch.float, device="cuda")
        fp8_meta.scale_inv = torch.ones(3, dtype=torch.float, device="cuda")

        # Compute initial amaxes and scales
        inp_amax = torch.max(torch.abs(inp_g))
        fp8_meta.amax_history[1][tex.FP8FwdTensors.GEMM1_INPUT].copy_(inp_amax)
        ker_amax = torch.max(torch.abs(ker_g))
        fp8_meta.amax_history[1][tex.FP8FwdTensors.GEMM1_WEIGHT].copy_(ker_amax)
        ref_amax = torch.max(torch.abs(ref_g))
        fp8_meta.amax_history[1][tex.FP8FwdTensors.GEMM1_OUTPUT].copy_(ref_amax)
        fp8_meta.scale = _default_sf_compute(
            fp8_meta.amax_history[1], fp8_meta.scale, fp8_formats[fp8_dtype].value.max_fwd, 1
        )
        fp8_meta.scale_inv = torch.reciprocal(fp8_meta.scale)

        # Cast input to Float8Tensor
        inp_fp8 = tex.cast_to_fp8(inp, fp8_meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_dtype)

        # Cast kernel to Float8Tensor
        kernel_t_fp8 = tex.cast_to_fp8(
            kernel_t, fp8_meta, tex.FP8FwdTensors.GEMM1_WEIGHT, fp8_dtype
        )

        # Set GEMM input to fp8
        pre_comm_inp = inp_fp8
    else:
        pre_comm_inp = inp

    # Trigger GEMM
    total_iters = opts.warmup_iters + opts.timing_iters
    total_start = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    total_end = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    comm_start = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    comm_end = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    gemm_start = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    gemm_end = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    torch.cuda.synchronize()

    for i in range(total_iters):
        total_start[i].record()

        if opts.comm_type == "AG":
            # Blocking all-gather before GEMM
            comm_start[i].record()
            gemm_inp, *_ = te.distributed.gather_along_first_dim(pre_comm_inp, tp_group)
            comm_end[i].record()
        else:
            # No pre-GEMM comm for reduce-scatter
            gemm_inp = pre_comm_inp

        gemm_start[i].record()
        if opts.fp8:
            gemm_out, *_ = tex.fp8_gemm(
                kernel_t_fp8,
                fp8_meta.scale_inv,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype,
                gemm_inp,
                fp8_meta.scale_inv,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype,
                torch.bfloat16,
                te.module.base.get_workspace(),
                bias=None,
                use_bias=False,
                gelu=False,
                use_split_accumulator=te.module.base._2X_ACC_FPROP,
            )
        else:
            gemm_out, *_ = tex.gemm(
                kernel_t,
                gemm_inp,
                torch.bfloat16,
                te.module.base.get_workspace(),
                bias=None,
                use_bias=False,
                gelu=False,
            )
        gemm_end[i].record()

        if opts.comm_type == "AG":
            # No post-GEMM comm for all-gather
            final_out = gemm_out
        else:
            # Blocking reduce-scatter after GEMM
            comm_start[i].record()
            final_out, *_ = te.distributed.reduce_scatter_along_first_dim(gemm_out, tp_group)
            comm_end[i].record()

        total_end[i].record()

    torch.cuda.synchronize()
    gpu_times = [
        s.elapsed_time(e)
        for s, e in zip(total_start[opts.warmup_iters :], total_end[opts.warmup_iters :])
    ]
    comm_times = [
        s.elapsed_time(e)
        for s, e in zip(comm_start[opts.warmup_iters :], comm_end[opts.warmup_iters :])
    ]
    gemm_times = [
        s.elapsed_time(e)
        for s, e in zip(gemm_start[opts.warmup_iters :], gemm_end[opts.warmup_iters :])
    ]
    avg_gpu_time = sum(gpu_times) / opts.timing_iters
    avg_comm_time = sum(comm_times) / opts.timing_iters
    avg_gemm_time = sum(gemm_times) / opts.timing_iters
    gemm_name = "".join(
        [
            "all-gather + " if opts.comm_type == "AG" else "",
            "GEMM",
            (" + reduce-scatter" if opts.comm_type == "RS" else ""),
        ]
    )
    total_prefix = f"Avg. GPU time for {gemm_name}:"
    timing_info = (
        f"{total_prefix} {avg_gpu_time} ms "
        + f"({opts.warmup_iters} warmup + {opts.timing_iters} timing runs)\n"
    )
    comm_name = "all-gather" if opts.comm_type == "AG" else "reduce-scatter"
    comm_prefix = f"Avg. GPU time for {comm_name}:"
    comm_space = ' ' * (len(total_prefix) - len(comm_prefix))
    gemm_prefix = "Avg. GPU time for GEMM:"
    gemm_space = ' ' * (len(total_prefix) - len(gemm_prefix))
    timing_info += (
        f"{comm_prefix}{comm_space} {avg_comm_time} ms\n"
        + f"{gemm_prefix}{gemm_space} {avg_gemm_time} ms"
    )
    dist_print(timing_info, section=True, debug=True, group=tp_group)

    numerics_failed = False
    if opts.check_numerics:
        if opts.comm_type == "AG":
            test_out = torch.transpose(
                te.distributed.gather_along_first_dim(
                    torch.transpose(final_out, 0, 1), tp_group
                )[0],
                0,
                1,
            )
        else:
            test_out = te.distributed.gather_along_first_dim(final_out, tp_group)[0]

        test_nonzeros = torch.count_nonzero(test_out)
        ref_nonzeros = torch.count_nonzero(ref_g)
        nonzero_info = (
            f"output nonzeros = {test_nonzeros} " + f"| reference count = {ref_nonzeros}"
        )
        dist_print(nonzero_info, src=0, section=True, group=tp_group)

        sizing_info = (
            f"input: {list(inp.shape)} " + f"| GEMM1 weights: {list(kernel_t.shape)[::-1]} "
        )
        sizing_info += f"| output: {list(final_out.shape)}\n"
        dist_print(sizing_info, section=True, group=tp_group)

        sizing_info_g = (
            f"input: {list(inp_g.shape)} " + f"| GEMM1 weights: {list(ker_g.shape)} "
        )
        sizing_info_g += (
            f"| output: {list(test_out.shape)} " + f"| reference: {list(ref_g.shape)}\n"
        )
        dist_print(sizing_info_g, src=0, group=tp_group)

        torch.cuda.synchronize()
        dist.barrier(tp_group)
        test_out = test_out.to(dtype=torch.float32)
        ref_g = ref_g.to(dtype=torch.float32)
        error_below_tol = torch.allclose(
            test_out,
            ref_g,
            rtol=0.125 if opts.fp8 else 0.02,
            atol=0.0675 if opts.fp8 else 0.001,
        )
        diff = torch.abs(test_out - ref_g).flatten()
        m = torch.argmax(diff)
        abs_err = diff[m].item()
        rel_err = abs_err / (ref_g.flatten()[m].item() + 1e-5)
        if not error_below_tol:
            numerics_failed = True
            numerics_info = (
                "NUMERICAL CHECK FAILED: "
                + f"Outputs not close enough at index {m.item()} "
                + f"with {test_out.flatten()[m].item()} vs {ref_g.flatten()[m].item()} "
                + f"(abs error = {abs_err} | rel error = {rel_err})."
            )
        else:
            numerics_info = f"NUMERICAL CHECK PASSED: abs error = {abs_err} | rel error = {rel_err}"

        dist_print(numerics_info, src=0, section=True, debug=True, group=tp_group)

    dist.barrier(tp_group)
    if LOCAL_RANK == 0:
        print("\n", end="", flush=True)

    dist.destroy_process_group()

    # Reset clock speeds
    if opts.clock_speed > 0:
        subprocess.run(
            ["nvidia-smi", "-pm", "ENABLED", "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result = subprocess.run(
            ["nvidia-smi", "-rgc", "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return int(numerics_failed)


if __name__ == "__main__":
    sys.exit(_main(_parse_args()))