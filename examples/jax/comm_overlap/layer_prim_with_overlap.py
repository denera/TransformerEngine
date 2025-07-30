# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Comm+GEMM Overlap with TE/JAX"""
import os
import argparse
from functools import partial

from mpi4py import MPI

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils

import flax
from flax import linen as nn

import transformer_engine.jax as te
from transformer_engine.common import recipe
from transformer_engine.jax.sharding import (
    MeshResource,
    global_shard_guard,
    generate_pspec,
    get_sharding_map_logic_axis_to_mesh_axis,
    BATCH_AXES,
    SEQLEN_AXES,
    SEQLEN_TP_AXES,
    HIDDEN_AXES,
    HIDDEN_TP_AXES,
    W_FSDP_AXES,
    W_NO_SHARD_AXES,
    W_JOINED_AXES,
    W_TP_AXES,
)
from transformer_engine.jax.dense import dense
from transformer_engine.jax.layernorm_dense import layernorm_dense
from transformer_engine.jax.layernorm_mlp import layernorm_mlp
from transformer_engine.jax.cpp_extensions import CommOverlapHelper, CommOverlapHelperSet

import transformer_engine_jax as tex

# This script needs to be launched via `mpirun` with 1 process per GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
myrank = MPI.COMM_WORLD.Get_rank()
numranks = MPI.COMM_WORLD.Get_size()
jax.clear_caches()
jax.distributed.initialize(cluster_detection_method="mpi4py")
assert (
    jax.local_device_count() == 1
), f"[{myrank}|{numranks}] Expected 1 GPU per process, found {jax.local_device_count()}"

# Parse script arguments
_supported_prims = (dense, layernorm_dense, layernorm_mlp)
_prim_map = dict((prim.__name__.lower(), prim) for prim in _supported_prims)


def _te_layer_prim(prim_name):
    assert isinstance(prim_name, str) and prim_name.lower() in _prim_map
    return _prim_map[prim_name.lower()]


parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--dp-size", type=int, default=2)
parser.add_argument("-tp", "--tp-size", type=int, default=numranks // 2)
parser.add_argument("-np", "--num-gpus", type=int, default=numranks)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--seq-length", type=int, default=8192)
parser.add_argument("--hidden-size", type=int, default=16384)
parser.add_argument("--activation-size", type=int, default=53248)
parser.add_argument("--no-batch", action="store_true")
parser.add_argument("--no-fsdp", action="store_true")
parser.add_argument("--layer-type", type=_te_layer_prim, default=dense, choices=_supported_prims)
parser.add_argument(
    "--fp8-recipe", type=str.lower, default="none", choices=["none", "current", "delayed", "mxfp8"]
)
parser.add_argument("--check-result", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# FP8 recipe
fp8_recipe = None
match args.fp8_recipe:
    case "current":
        fp8_recipe = recipe.Float8CurrentScaling()
    case "delayed":
        fp8_recipe = recipe.DelayedScaling()
    case "mxfp8":
        fp8_recipe = recipe.MXFP8BlockScaling()
    case _:
        fp8_recipe = None

# Declare inputs
dtype = jnp.bfloat16
input_shape = (args.seq_length, args.hidden_size)
if not args.no_batch:
    input_shape = (args.batch_size,) + input_shape
features = args.hidden_size  # post-attention projection
if args.layer_type is layernorm_dense:
    features *= 3  # QKV projection
kernel_shape = (
    (args.hidden_size, 1, args.activation_size)  # MLP FFN1
    if args.layer_type is layernorm_mlp
    else (args.hidden_size, features)
)
bias_shape = (1, args.activation_size) if args.layer_type is layernorm_mlp else (features,)

rng = jax.random.PRNGKey(args.seed)
rng, params_rng = jax.random.split(rng)
params_rng, kernel_rng = jax.random.split(params_rng)
params_rng, bias_rng = jax.random.split(params_rng)
x = jax.random.normal(rng, input_shape, dtype=dtype)

gamma = beta = None
if args.layer_type in (layernorm_dense, layernorm_mlp):
    params_rng, gamma_rng = jax.random.split(params_rng)
    gamma = jax.random.normal(gamma_rng, (args.hidden_size,), dtype=dtype)
    params_rng, beta_rng = jax.random.split(params_rng)
    beta = jax.random.normal(beta_rng, (args.hidden_size,), dtype=dtype)

kernel_1 = jax.random.normal(kernel_rng, kernel_shape, dtype=dtype)
bias_1 = jax.random.normal(bias_rng, bias_shape, dtype=dtype)

kernel_2 = bias_2 = None
if args.layer_type is layernorm_mlp:
    kernel_rng, kernel_2_rng = jax.random.split(kernel_rng)
    kernel_2 = jax.random.normal(
        kernel_2_rng, (args.activation_size, args.hidden_size), dtype=dtype
    )
    bias_rng, bias_2_rng = jax.random.split(bias_rng)
    bias_2 = jax.random.normal(bias_2_rng, (args.hidden_size,), dtype=dtype)

if myrank == 0:
    print(
        f"[{myrank}|{numranks}] {args.layer_type.__name__} inputs:\n"
        + f"  x:        {x.shape}\n"
        + f"  gamma:    {gamma.shape if gamma is not None else None}\n"
        + f"  beta:     {beta.shape if beta is not None else None}\n"
        + f"  kernel_1: {kernel_1.shape}\n"
        + f"  bias_1:   {bias_1.shape}\n"
        + f"  kernel_2: {kernel_2.shape if kernel_2 is not None else None}\n"
        + f"  bias_2:   {bias_2.shape if bias_2 is not None else None}\n"
    )


# Single GPU evaluation
def _eval_layer_serial(layer_type_, x_, gamma_, beta_, kernel_1_, bias_1_, kernel_2_, bias_2_):
    layer_args = []
    layer_kwargs = {}

    if layer_type_ is dense:
        layer_args = (x_, kernel_1_, bias_1_)
        layer_kwargs = {"contracting_dims": ((x.ndim - 1,), (0,))}

    elif layer_type_ is layernorm_dense:
        layer_args = (x_, kernel_1_, gamma_, beta_, bias_1_)

    elif layer_type_ is layernorm_mlp:
        layer_args = (x_, gamma_, beta_, (kernel_1_, kernel_2_), (bias_1_, bias_2_))

    output = layer_type_(*layer_args, **layer_kwargs)
    return jnp.mean(output)


with te.fp8_autocast(enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe):
    fwd_bwd_serial = jax.jit(
        jax.value_and_grad(
            partial(_eval_layer_serial, args.layer_type), argnums=range(7)
        )
    )
    loss_serial, grads_serial = fwd_bwd_serial(
        x, gamma, beta, kernel_1, bias_1, kernel_2, bias_2
    )

# Device mesh and logical axis resources
DEVICE_DP_AXIS = "dp"
DEVICE_TP_AXIS = "tp"
mesh_shape = {DEVICE_TP_AXIS: args.tp_size}
if not args.no_batch:
    mesh_shape[DEVICE_DP_AXIS] = args.dp_size
devices = mesh_utils.create_device_mesh((args.num_gpus,), devices=jax.devices()[: args.num_gpus])
mesh = Mesh(np.array(devices).reshape(tuple(mesh_shape.values())), tuple(mesh_shape.keys()))
mesh_resource = MeshResource(
    dp_resource=None if args.no_batch else DEVICE_DP_AXIS,
    fsdp_resource=None if args.no_fsdp else DEVICE_DP_AXIS,
    tp_resource=DEVICE_TP_AXIS,
)
if myrank == 0:
    print(f"[{myrank}|{numranks}] Device mesh: {mesh}\n")


# Logical axes
INPUT_AXES = (
    SEQLEN_AXES if args.layer_type is dense else SEQLEN_TP_AXES,
    HIDDEN_TP_AXES if args.layer_type is dense else HIDDEN_AXES,
)
INTERMEDIATE_AXES = (SEQLEN_AXES, HIDDEN_TP_AXES)
if not args.no_batch:
    INPUT_AXES = (BATCH_AXES,) + INPUT_AXES
    INTERMEDIATE_AXES = (BATCH_AXES,) + INTERMEDIATE_AXES

LN_SCALE_AXES = LN_BIAS_AXES = (W_NO_SHARD_AXES,)

KERNEL_AXES_ROW_PARALLEL = (W_TP_AXES, W_FSDP_AXES)
BIAS_AXES_ROW_PARALLEL = (W_FSDP_AXES,)
KERNEL_AXES_COL_PARALLEL = (W_FSDP_AXES, W_TP_AXES)
BIAS_AXES_COL_PARALLEL = (W_TP_AXES,)
if args.layer_type is layernorm_mlp:
    KERNEL_AXES_COL_PARALLEL = (W_FSDP_AXES, W_JOINED_AXES, W_TP_AXES)
    BIAS_AXES_COL_PARALLEL = (W_JOINED_AXES, W_TP_AXES)

KERNEL_1_AXES = KERNEL_AXES_ROW_PARALLEL if args.layer_type is dense else KERNEL_AXES_COL_PARALLEL
BIAS_1_AXES = BIAS_AXES_ROW_PARALLEL if args.layer_type is dense else BIAS_AXES_COL_PARALLEL
KERNEL_2_AXES = KERNEL_AXES_ROW_PARALLEL if args.layer_type is layernorm_mlp else None
BIAS_2_AXES = BIAS_AXES_ROW_PARALLEL if args.layer_type is layernorm_mlp else None


# Multi GPU evaluation
def _eval_layer_sharded(
    layer_type_,
    comm_overlaps_,
    x_,
    gamma_,
    beta_,
    kernel_1_,
    bias_1_,
    kernel_2_,
    bias_2_,
):
    layer_args = []
    layer_kwargs = {}

    if layer_type_ is dense:
        layer_args = (x_, kernel_1_, bias_1_)
        layer_kwargs = {
            "input_axes": INPUT_AXES,
            "kernel_axes": KERNEL_AXES_ROW_PARALLEL,
            "comm_overlaps": comm_overlaps_[0],
            "contracting_dims": ((x.ndim - 1,), (0,)),
            "sequence_parallel_output": True,
        }

    elif layer_type_ is layernorm_dense:
        layer_args = (x_, kernel_1_, gamma_, beta_, bias_1_)
        layer_kwargs = {
            "layernorm_input_axes": INPUT_AXES,
            "dot_input_axes": INPUT_AXES,
            "kernel_axes": KERNEL_AXES_COL_PARALLEL,
            "comm_overlaps": comm_overlaps_[0],
        }

    elif layer_type_ is layernorm_mlp:
        layer_args = (x_, gamma_, beta_, (kernel_1_, kernel_2_), (bias_1_, bias_2_))
        layer_kwargs = {
            "norm_input_axes": INPUT_AXES,
            "dot_1_input_axes": INPUT_AXES,
            "kernel_1_axes": KERNEL_AXES_COL_PARALLEL,
            "dot_2_input_axes": INTERMEDIATE_AXES,
            "kernel_2_axes": KERNEL_AXES_ROW_PARALLEL,
            "ffn1_comm_overlaps": comm_overlaps_[0],
            "ffn2_comm_overlaps": comm_overlaps_[1],
        }

    output = layer_type_(*layer_args, **layer_kwargs)
    return jnp.mean(output)


with (
    mesh,
    global_shard_guard(mesh_resource),
    te.fp8_autocast(
        enabled=fp8_recipe is not None,
        fp8_recipe=fp8_recipe,
        mesh_resource=mesh_resource,
    ),
):
    nn.set_logical_axis_rules(get_sharding_map_logic_axis_to_mesh_axis(as_tuple=True))

    # Comm+GEMM overlap configs
    # NOTE: Need to set `tp_resource=` kwarg when *not* initializing under a `global_shard_guard()`.
    #       Also need `logical_tp_axis=` and `logical_sp_axis=` kwargs if they differ from TE's
    #       built-in logical axis names.
    buffer_shape = list(input_shape).copy()
    if not args.no_batch:
        buffer_shape[0] = buffer_shape[0] // args.dp_size
    fprop_1_overlap = CommOverlapHelper(
        comm_type=tex.CommOverlapType.RS if args.layer_type is dense else tex.CommOverlapType.AG,
        method=tex.CommOverlapMethod.RING_EXCHANGE,
        buffer_shape=buffer_shape,
    )
    comm_overlaps = [CommOverlapHelperSet(fprop=fprop_1_overlap)]
    if args.layer_type is layernorm_mlp:
        fprop_2_overlap = CommOverlapHelper(
            comm_type=tex.CommOverlapType.RS,
            method=tex.CommOverlapMethod.RING_EXCHANGE,
            buffer_shape=buffer_shape,
        )
        comm_overlaps.append(CommOverlapHelperSet(fprop=fprop_2_overlap))

    x_spec = generate_pspec(INPUT_AXES, with_flax_rules=True, padded=True)
    x_sharding = NamedSharding(mesh, x_spec)
    x = jax.device_put(x, x_sharding)

    gamma_sharding = beta_sharding = None
    if gamma is not None:
        gamma_spec = generate_pspec(LN_SCALE_AXES, with_flax_rules=True, padded=True)
        gamma_sharding = NamedSharding(mesh, gamma_spec)
        gamma = jax.device_put(gamma, gamma_sharding)
    if beta is not None:
        beta_spec = generate_pspec(LN_BIAS_AXES, with_flax_rules=True, padded=True)
        beta_sharding = NamedSharding(mesh, beta_spec)
        beta = jax.device_put(beta, beta_sharding)

    kernel_1_spec = generate_pspec(KERNEL_1_AXES, with_flax_rules=True, padded=True)
    kernel_1_sharding = NamedSharding(mesh, kernel_1_spec)
    bias_1_spec = generate_pspec(BIAS_1_AXES, with_flax_rules=True, padded=True)
    bias_1_sharding = NamedSharding(mesh, bias_1_spec)

    kernel_2_spec = None
    kernel_2_sharding = bias_2_sharding = None
    if kernel_2 is not None:
        kernel_2_spec = generate_pspec(KERNEL_2_AXES, with_flax_rules=True, padded=True)
        kernel_2_sharding = NamedSharding(mesh, kernel_2_spec)
        kernel_2 = jax.device_put(kernel_2, kernel_2_sharding)
    if bias_2 is not None:
        bias_2_spec = generate_pspec(BIAS_2_AXES, with_flax_rules=True, padded=True)
        bias_2_sharding = NamedSharding(mesh, bias_2_spec)
        bias_2 = jax.device_put(bias_2, bias_2_sharding)

    input_shardings = (
        x_sharding,
        gamma_sharding,
        beta_sharding,
        kernel_1_sharding,
        bias_1_sharding,
        kernel_2_sharding,
        bias_2_sharding,
    )
    output_shardings = (
        NamedSharding(mesh, PartitionSpec()),
        input_shardings,
    )

    value_and_grad_sharded = jax.jit(
        jax.value_and_grad(
            partial(_eval_layer_sharded, args.layer_type, comm_overlaps),
            argnums=range(7),
        ),
        in_shardings=input_shardings,
        out_shardings=output_shardings,
    )

    loss_sharded, grads_sharded = value_and_grad_sharded(
        x, gamma, beta, kernel_1, bias_1, kernel_2, bias_2
    )

if args.check_result:
    eps = np.finfo(dtype).eps
    rtol = 1.6e-2
    atol = 1e-5

    diff = jnp.abs(loss_serial - loss_sharded)
    if myrank == 0:
        print(
            f"[{myrank}|{numranks}] Loss: serial = {loss_serial} | sharded = {loss_sharded}"
        )
    rel_err = diff / max(abs(diff), eps)
    if rel_err > rtol and diff > atol:
        if myrank == 0:
            print("NUMERICAL CHECK_FAILED: Loss not close enough!\n")
    else:
        if myrank == 0:
            print("NUMERICAL CHECK PASSED\n")

    labels = ("dX", "dGamma", "dBeta", "dKernel_1", "dBias_1", "dKernel_2", "dBias_2")
    for i, (serial, sharded) in enumerate(zip(grads_serial, grads_sharded)):
        if serial is not None and sharded is not None:
            if myrank == 0:
                print(
                    f"[{myrank}|{numranks}] {labels[i]} : {sharded.shape}\n"
                    + f"  Sharding: {sharded.sharding.spec}\n"
                )

            sharded = jax.lax.with_sharding_constraint(
                sharded, NamedSharding(mesh, PartitionSpec(None))
            )
            jax.block_until_ready(sharded)
            diff = jnp.abs(serial - sharded).flatten()
            if myrank == 0:
                print(f"{myrank}: Global {labels[i]} difference: {diff}\n", flush=True)

            m = jnp.argmax(diff).item()
            abs_err = diff[m].item()
            rel_err = abs_err / max(abs(serial.flatten()[m]), eps)

            if rel_err > rtol and abs_err > atol:
                numerics_info = (
                    "NUMERICAL CHECK FAILED: "
                    + f"{labels[i]} not close enough at index {m} "
                    + f"with {sharded.flatten()[m].item()} vs {serial.flatten()[m].item()} "
                    + f"| rel. error = {rel_err} (tol = {rtol}) "
                    + f"| abs. error = {abs_err} (tol = {atol})"
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

tex.destroy_all_comm_overlap_buffers()
