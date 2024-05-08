# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse

from mpi4py import MPI

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

comm = MPI.COMM_WORLD
LOCAL_RANK = comm.Get_rank()
WORLD_SIZE = comm.Get_size()

def torch_dtype(opt):
    typemap = {
        'fp32' : torch.float32,
        'float32' : torch.float32,
        'fp16' : torch.float16,
        'float16' : torch.float16,
        'bf16' : torch.bfloat16,
        'bfloat16' : torch.bfloat16
    }
    if str(opt).lower() not in typemap.keys():
        raise TypeError
    return typemap[str(opt).lower()]

def dist_print(msg, end='\n', all_ranks=False):
    if LOCAL_RANK == 0 or all_ranks:
        print(f"[RANK-{LOCAL_RANK}] {msg}", end=end)

def train(args):
    # Initialize torch.distributed global process group and get TP group
    dist.init_process_group(backend="nccl",
                            rank=LOCAL_RANK,
                            world_size=WORLD_SIZE)
    tp_group = dist.new_group(backend="nccl")
    tp_size = dist.get_world_size(tp_group)

    torch.cuda.set_device(LOCAL_RANK)
    torch.manual_seed(args.seed+LOCAL_RANK)
    torch.cuda.manual_seed(args.seed+LOCAL_RANK)

    # Intialize userbuffers
    ag_cfg = {  # Ring-exchange All-Gather overlap for fc1_fprop and fc2_dgrad
        'method': 'ring_exchange',
        'num_splits' : 8,
        'num_sm' : 1,
        'set_sm_margin' : False,
    }
    rs_cfg = {  # Reduce-scatter overlap for fc1_dgrad and fc2_fprop
        'num_splits' : 4,
        'num_sm' : 1,
        'set_sm_margin' : True,
    }
    hidden_size = args.num_heads * args.head_dim
    if args.comm_overlap:
        te.module.base.initialize_ub(
            [args.seq_length * args.batch_size, hidden_size],
            tp_size,
            use_fp8 = args.fp8,
            dtype = args.dtype,
            ub_cfgs = {
                'fc1_fprop': ag_cfg,
                'fc1_dgrad': rs_cfg,
                'fc2_fprop': rs_cfg,
                'fc2_dgrad': ag_cfg,
            },
        )

    # Initialize TE model
    ln_mlp = te.LayerNormMLP(
        hidden_size, args.mlp_expansion_factor * hidden_size,
        params_dtype = args.dtype,
        device = 'cuda',
        tp_group = tp_group,
        tp_size = tp_size,
        set_parallel_mode = True,
        sequence_parallel = True,
        micro_batch_size = args.batch_size,
        ub_overlap_rs_dgrad = args.comm_overlap,
        ub_overlap_rs = args.comm_overlap,
        ub_overlap_ag = args.comm_overlap,
    )

    # Fp8 recipe setup
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32,
                                amax_compute_algo="max")

    # Optimizer must be created after the model is wrapped in FSDP and the parameters are sharded
    optim = torch.optim.Adam(ln_mlp.parameters(), lr=0.0001)

    # Start dummy "training" iterations
    for i in range(args.num_iters):
        dist_print(f"Iter {i}", all_ranks=args.verbose)

        dist_print("|-- Generate random input batch", all_ranks=args.verbose)
        x = torch.rand((args.seq_length * args.batch_size // tp_size, hidden_size),
                       dtype=args.dtype, device='cuda')

        dist_print("|-- Forward pass", all_ranks=args.verbose)
        with te.fp8_autocast(enabled=args.fp8, fp8_recipe=fp8_recipe, fp8_group=tp_group):
            y = ln_mlp(x)
            loss = y.flatten().sum()

        dist_print("|-- Backward pass", all_ranks=args.verbose)
        loss.backward()

        dist_print("|-- Optimizer step", all_ranks=args.verbose)
        optim.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a te.LayerNormMLP module with "
                                                 "GEMM+comm overlap via Userbuffers.")
    parser.add_argument('-i', "--num-iters", type=int, default=5,
                        help="Number of dummy 'training' iterations.")
    parser.add_argument('-b', "--batch-size", type=int, default=2,
                        help="Input batch size.")
    parser.add_argument('-s', "--seq-length", type=int, default=2048,
                        help="Input sequence length.")
    parser.add_argument('-n', "--num-heads", type=int, default=64,
                        help="Number of attention heads.")
    parser.add_argument('-d', "--head-dim", type=int, default=128,
                        help="Dimension of each attention head.")
    parser.add_argument('-m', "--mlp-expansion-factor", type=int, default=4,
                        help="MLP block intermediate size as a factor of hidden dimension.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="RNG seed.")
    parser.add_argument("--fp8", action="store_true", default=False,
                        help="Enables the te.fp8_autocast() context.")
    parser.add_argument("--comm-overlap", action="store_true", default=False,
                        help="Enable GEMM+comm overlap via Userbuffers"
                             "(requires launching with mpiexec).")
    parser.add_argument("--dtype", type=torch_dtype, default=torch.bfloat16,
                        help="Data type for input tensor and Transformer Engine module parameters.")
    parser.add_argument('-v', "--verbose", action="store_true", default=False)
    args = parser.parse_args()
    train(args)
