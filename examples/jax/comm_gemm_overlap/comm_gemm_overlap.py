# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Comm+GEMM Overlap with TE/JAX"""

import argparse
import numpy as np
from functools import partial
from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils

import transformer_engine_jax as tex
import transformer_engine.jax as te
from transformer_engine.jax.sharding import get_padded_spec
from transformer_engine.jax.cpp_extensions import (
    te_gemm,
    create_comm_overlap_buffer,
    destroy_all_comm_overlaps,
)

jax.clear_caches()

# This script needs to be launched via `mpirun` with 1 process per GPU
myrank = MPI.COMM_WORLD.Get_rank()
numranks = MPI.COMM_WORLD.Get_size()
jax.distributed.initialize(cluster_detection_method="mpi4py")

parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--dp-size", type=int, default=1)
parser.add_argument("-zp", "--fsdp-size", type=int, default=2)
parser.add_argument("-tp", "--tp-size", type=int, default=4)
parser.add_argument("-np", "--num-gpus", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--seq-length", type=int, default=8192)
parser.add_argument("--hidden-size", type=int, default=16384)
parser.add_argument("--activation-size", type=int, default=53248)
parser.add_argument("--no-batch", action="store_true")
parser.add_argument("--no-fsdp", action="store_true")
parser.add_argument("--comm-type", type=str.upper, default="AG", choices=["AG", "RS"])
parser.add_argument("--check-result", action="store_true")
args = parser.parse_args()

# Operand shapes
dtype = jnp.bfloat16
lhs_shape = (
    [args.seq_length, args.hidden_size]
    if args.comm_type == "AG"
    else [args.seq_length, args.activation_size]
)
rhs_shape = (
    [args.hidden_size, args.activation_size]
    if args.comm_type == "AG"
    else [args.activation_size, args.hidden_size]
)

# Operand partitioning
batched = not args.no_batch
fsdp = not args.no_fsdp
input_specs = [ None ] * len(lhs_shape)
weight_specs = [ None ] * len(rhs_shape)
weight_no_fsdp = weight_specs.copy()
if batched:
    lhs_shape = [args.batch_size] + lhs_shape
    if fsdp:
        mesh_shape = {"dp": args.dp_size, "zp": args.fsdp_size, "tp": args.tp_size}
        mesh_resource = te.MeshResource(
            dp_resource="dp", tp_resource="tp", cp_resource="tp", fsdp_resource="zp"
        )
        if args.comm_type == "AG":
            input_specs = [("dp", "zp"), "tp", None]
            weight_specs = ["zp", "tp"]
            weight_no_fsdp = [None, "tp"]
        elif args.comm_type == "RS":
            input_specs = [("dp", "zp"), None, "tp"]
            weight_specs = ["tp", "zp"]
            weight_no_fsdp = ["tp", None]
    else:
        mesh_shape = {"dp": args.dp_size, "tp": args.tp_size}
        mesh_resource = te.MeshResource(
            dp_resource="dp",
            tp_resource="tp",
            cp_resource="tp",
        )
        if args.comm_type == "AG":
            input_specs = ["dp", "tp", None]
            weight_specs = [None, "tp"]
        elif args.comm_type == "RS":
            input_specs = ["dp", None, "tp"]
            weight_specs = ["tp", None]
        weight_no_fsdp = weight_specs
else:
    if fsdp:
        mesh_shape = {"zp": args.fsdp_size, "tp": args.tp_size}
        mesh_resource = te.MeshResource(fsdp_resource="zp", tp_resource="tp", cp_resource="cp")
        if args.comm_type == "AG":
            input_specs = ["tp", None]
            weight_specs = ["zp", "tp"]
        elif args.comm_type == "RS":
            input_specs = [None, "tp"]
            weight_specs = ["tp", "zp"]
        weight_no_fsdp = ["tp", None]
    else:
        mesh_shape = {"tp": args.tp_size}
        mesh_resource = te.MeshResource(tp_resource="tp", cp_resource="cp")
        if args.comm_type == "AG":
            input_specs = ["tp", None]
            weight_specs = [None, "tp"]
        elif args.comm_type == "RS":
            input_specs = [None, "tp"]
            weight_specs = ["tp", None]
        weight_no_fsdp = weight_specs

# Mesh setup and sharding definitions
devices = mesh_utils.create_device_mesh((args.num_gpus,), devices=jax.devices()[: args.num_gpus])
mesh = Mesh(np.array(devices).reshape(tuple(mesh_shape.values())), tuple(mesh_shape.keys()))
no_sharding = NamedSharding(mesh, PartitionSpec(None))
input_sharding = NamedSharding(mesh, PartitionSpec(*input_specs))
weight_sharding = NamedSharding(mesh, PartitionSpec(*weight_specs))
weight_no_fsdp_sharding = NamedSharding(mesh, PartitionSpec(*weight_no_fsdp))

# Operand initialization
key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key, 2)
lhs_data = jax.random.normal(key1, lhs_shape, dtype=dtype)
rhs_data = jax.random.normal(key2, rhs_shape, dtype=dtype)
lhs = jax.device_put(lhs_data, input_sharding)
rhs = jax.device_put(rhs_data, weight_sharding)

# Name of comm+GEMM overlap layer
overlap_method = tex.CommOverlapMethod.RING_EXCHANGE
comm_type = tex.CommOverlapType.AG if args.comm_type == "AG" else tex.CommOverlapType.RS

# Bootstrap Userbuffers communicators and communication buffers
overlap_config = create_comm_overlap_buffer(
    lhs_shape,
    dtype,
    mesh,
    "tp",
    overlap_method,
    comm_type,
    num_splits=args.tp_size,
    num_max_streams=3,
    comm_cga_size=1,
    comm_priority=0,
    gemm_priority=0,
    num_comm_sm=1,
    set_sm_margin=False,
    atomic_gemm=False,
    use_ce=True,
    rs_overlap_first_gemm=False,
    aggregate_ag=False,
)

if myrank == 0:
    print(
        f"{myrank}: INPUTS {lhs.shape} x {rhs.shape}\n"
        + f"{myrank}:    LHS sharding: {lhs.sharding.spec}\n"
        + f"{myrank}:    RHS sharding: {rhs.sharding.spec}\n",
        flush=True,
    )

jitted_te_gemm = jax.jit(partial(te_gemm, comm_overlap_config=overlap_config),
                         static_argnums=(3, 4, 5, 6, 7))
with te.sharding.global_shard_guard(mesh_resource):
    with jax.sharding.with_sharding_constraint(rhs, weight_no_fsdp):
        output = jitted_te_gemm(lhs, rhs, None)

if myrank == 0:
    print(
        f"{myrank}: {'AG -> GEMM' if args.comm_type == 'AG' else 'GEMM -> RS'} OUTPUT "
        + f"{output.shape}\n"
        + f"{myrank}:    Sharding: {get_padded_spec(output.sharding.spec, output.ndim)}\n",
        flush=True,
    )

if args.check_result:
    ref_global = jnp.matmul(
        jax.device_put(lhs_data, no_sharding), jax.device_put(rhs_data, no_sharding)
    )
    if myrank == 0:
        print(f"{myrank}: Global reference: {ref_global}\n", flush=True)

    output_global = jax.lax.with_sharding_constraint(output, no_sharding)
    if myrank == 0:
        print(f"{myrank}: Global output: {output_global}\n", flush=True)

    diff = jnp.abs(ref_global - output_global).flatten()
    if myrank == 0:
        print(f"{myrank}: Global difference: {diff}\n", flush=True)

    m = jnp.argmax(diff).item()
    abs_err = diff[m].item()
    rel_err = abs_err / max(abs(ref_global.flatten()[m]), 1e-5)

    rtol = 0.02
    atol = 0.001
    numerics_failed = False
    if rel_err > rtol and abs_err > atol:
        numerics_failed = True
        numerics_info = (
            "NUMERICAL CHECK FAILED: "
            + f"Outputs not close enough at index {m} "
            + f"with {output.flatten()[m].item()} vs {ref_global.flatten()[m].item()} | "
            + f"rel. error = {rel_err} (tol = {rtol}) | "
            + f"abs. error = {abs_err} (tol = {atol})"
        )
    else:
        numerics_info = "NUMERICAL CHECK PASSED: "
        if rel_err <= rtol:
            numerics_info += f"rel. error = {rel_err} (tol = {rtol})" + (
                " | " if abs_err < atol else ""
            )
        if abs_err <= atol:
            numerics_info += f"abs. error = {abs_err} (tol = {atol})"

    if myrank == 0:
        print(numerics_info + "\n", end="", flush=True)

destroy_all_comm_overlaps()
