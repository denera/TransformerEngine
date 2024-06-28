#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import subprocess
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


def parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Test a te.LayerNormMLP module with GEMM+comm overlap via Userbuffers."
    )
    parser.add_argument(
        "-i", "--num-iters", type=int, default=5, help="Number of dummy 'training' iterations."
    )
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Input sequence length.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=64, help="Number of attention heads."
    )
    parser.add_argument(
        "-d", "--head-dim", type=int, default=128, help="Dimension of each attention head."
    )
    parser.add_argument(
        "--mlp-expansion-factor",
        type=int,
        default=4,
        help="MLP block intermediate size as a factor of hidden dimension.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument(
        "--fp8", action="store_true", default=False, help="Enables the te.fp8_autocast() context."
    )
    parser.add_argument(
        "--no-comm-overlap",
        action="store_true",
        default=False,
        help="Disable the comm+GEMM overlap.",
    )
    parser.add_argument(
        "--num-replicas", type=int, default=1, help="Number of model replicas for data parallelism."
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    return parser.parse_args(argv, namespace)


def train(opts):
    WORLD_RANK = int(os.getenv("RANK", '0'))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", '1'))

    def dist_print(msg, end="\n", group=None):
        group_rank = dist.get_rank(group)
        group_size = dist.get_world_size(group)
        all_ranks = dist.get_process_group_ranks(group)
        ranks_skip = all_ranks[1] - all_ranks[0] > 1
        group_id = WORLD_RANK % group_size if ranks_skip else WORLD_RANK // group_size
        if group_rank == 0 or opts.verbose:
            print(f"[{group_id}:{group_rank}] {msg}{end}", end='', flush=True)
        dist.barrier()

    # Seed RNG
    torch.cuda.set_device(WORLD_RANK)
    torch.manual_seed(opts.seed + WORLD_RANK)
    torch.cuda.manual_seed(opts.seed + WORLD_RANK)

    # Initialize torch.distributed global process group and get DP/TP groups
    dist.init_process_group(
        backend="nccl",
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
        device_id=torch.device(f"cuda:{WORLD_RANK}"),
    )
    world_group = dist.new_group(backend="nccl")
    if opts.num_replicas > 1:
        TP_SIZE = WORLD_SIZE // opts.num_replicas
        mesh2d = dist.device_mesh.init_device_mesh(
            "cuda",
            (opts.num_replicas, TP_SIZE),
            mesh_dim_names=('data', 'model'),
        )
        dp_group, tp_group = mesh2d.get_group()
        dist_print(
            f"|-- Initialized data-parallel group: {dist.get_process_group_ranks(dp_group)}",
            group=dp_group
        )
    else:
        dp_group = None
        tp_group = world_group
        tp_size = dist.get_world_size(tp_group)
    dist_print(f"|-- Initialized tensor-parallel group: {dist.get_process_group_ranks(tp_group)}",
               group=tp_group)

    # Intialize userbuffers
    ag_cfg = {  # Ring-exchange All-Gather overlap for fc1_fprop and fc2_dgrad
        "method": "ring_exchange",
        "num_sm": 1,
        "set_sm_margin": False,
    }
    rs_cfg = {  # Reduce-scatter overlap for fc1_dgrad and fc2_fprop
        "method": "ring_exchange",
        "num_sm": 1,
        "set_sm_margin": True,
    }
    hidden_size = opts.num_heads * opts.head_dim
    batched_size = opts.seq_length * opts.batch_size
    if not opts.no_comm_overlap:
        te.module.base.initialize_ub(
            [batched_size, hidden_size],
            tp_group,
            dp_group=dp_group,
            use_fp8=opts.fp8,
            dtype=torch.bfloat16,
            ub_cfgs={
                "fc1_fprop": ag_cfg,
                "fc1_dgrad": rs_cfg,
                "fc2_fprop": rs_cfg,
                "fc2_dgrad": ag_cfg,
            },
        )

    # Initialize the fused LayerNorm + Multi-layer Perceptron module
    model = te.LayerNormMLP(
        hidden_size,
        opts.mlp_expansion_factor * hidden_size,
        params_dtype=torch.bfloat16,
        device="cuda",
        tp_group=tp_group,
        tp_size=TP_SIZE,
        set_parallel_mode=True,
        sequence_parallel=True,  # this is required for comm+GEMM overlap
        seq_length=opts.seq_length,
        micro_batch_size=opts.batch_size,
        ub_overlap_rs_dgrad=not opts.no_comm_overlap,
        ub_overlap_rs=not opts.no_comm_overlap,
        ub_overlap_ag=not opts.no_comm_overlap,
    )
    if dp_group is not None:
        model = DistributedDataParallel(model, process_group=dp_group)

    # Initialize optimizer with model parameters
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Fp8 recipe setup
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")

    # Start dummy "training" iterations
    for i in range(opts.num_iters):
        dist_print(f"Iter {i+1}", group=tp_group)

        dist_print("|-- Generate random input batch", group=tp_group)
        x = torch.rand(
            (opts.seq_length // TP_SIZE, opts.batch_size, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )

        dist_print("|-- Forward pass", group=tp_group)
        with te.fp8_autocast(enabled=opts.fp8, fp8_recipe=fp8_recipe, fp8_group=world_group):
            y = model(x)
            dist_print("|-- Compute loss", group=tp_group)
            loss = y.flatten().sum()

        dist_print("|-- Backward pass", group=tp_group)
        loss.backward()

        dist_print("|-- Optimizer step", group=tp_group)
        optim.step()

    te.module.base.destroy_ub()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "TORCHELASTIC_RUN_ID" in os.environ.keys():
        args = parse_args()
        train(args)
    else:
        subprocess.run(
            ["torchrun", f"--nproc-per-node={torch.cuda.device_count()}", *sys.argv],
            env=os.environ,
            check=True,
        )
    os._exit(0)
