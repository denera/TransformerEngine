# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

import operator
from typing import Tuple, Sequence, Union
from functools import partial, reduce

import jax
import jax.numpy as jnp
from jax import dtypes
from jax.sharding import NamedSharding, PartitionSpec
from jax.experimental.custom_partitioning import SdyShardingRule

import transformer_engine_jax as tex

from .base import BasePrimitive, register_primitive
from ..quantize import (
    ScaledTensor,
    ScaledTensor2x,
    ScalingMode,
    Quantizer,
    QuantizeConfig,
    is_fp8_gemm_with_all_layouts_supported,
    apply_padding_to_scale_inv,
)
from .misc import (
    get_padded_spec,
    get_cublas_workspace_size_bytes,
    sanitize_dims,
    transpose_dims,
)
from .comm_gemm_overlap import CommOverlapHelper


__all__ = [
    "gemm",
    "gemm_uses_jax_dot",
]


def _compatible_fp8_gemm_dtypes(lhs_dtype, rhs_dtype) -> bool:
    lhs, rhs, e4m3, e5m2 = map(
        dtypes.canonicalize_dtype,
        (
            lhs_dtype,
            rhs_dtype,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
        ),
    )

    # FP8 GEMM supports (e4m3 x e4m3), (e4m3 x e5m2) and (e5m2 x e4m3)
    if (lhs is e4m3 and rhs in (e4m3, e5m2)) or (lhs in (e4m3, e5m2) and rhs is e4m3):
        return True

    # Any other combination of data types is not supported
    return False


def _get_gemm_layout(
    operand_ndims: Tuple[int, int], contracting_dims: Tuple[Sequence[int], Sequence[int]]
) -> Tuple[bool, bool]:
    lhs_contracting, rhs_contracting = map(sanitize_dims, operand_ndims, contracting_dims)
    lhs_is_transposed = operand_ndims[0] - 1 not in lhs_contracting
    rhs_is_transposed = operand_ndims[1] - 1 in rhs_contracting
    return lhs_is_transposed, rhs_is_transposed


def _quantize_gemm_operands(lhs, rhs, lhs_quantizer, rhs_quantizer, contracting_dims):
    lhs_q = lhs
    rhs_q = rhs

    if not isinstance(lhs, ScaledTensor) and lhs_quantizer is not None:
        lhs_cdims = sanitize_dims(lhs.ndim, contracting_dims[0])
        lhs_is_transposed = lhs.ndim - 1 not in lhs_cdims
        need_lhs_colwise = lhs_is_transposed and (
            lhs_quantizer.scaling_mode.is_1d_block_scaling()
            or not is_fp8_gemm_with_all_layouts_supported()
        )
        flatten_axis = max(lhs_cdims) + 1 if lhs_is_transposed else min(lhs_cdims)
        lhs_q = lhs_quantizer.quantize(
            lhs,
            is_rowwise=not need_lhs_colwise,
            is_colwise=need_lhs_colwise,
            flatten_axis=flatten_axis,
        )

    if not isinstance(rhs, ScaledTensor) and rhs_quantizer is not None:
        rhs_cdims = sanitize_dims(rhs.ndim, contracting_dims[1])
        rhs_is_transposed = rhs.ndim - 1 in rhs_cdims
        need_rhs_colwise = not rhs_is_transposed and (
            rhs_quantizer.scaling_mode.is_1d_block_scaling()
            or not is_fp8_gemm_with_all_layouts_supported()
        )
        flatten_axis = min(rhs_cdims) if rhs_is_transposed else max(rhs_cdims) + 1
        rhs_q = rhs_quantizer.quantize(
            rhs,
            is_rowwise=not need_rhs_colwise,
            is_colwise=need_rhs_colwise,
            flatten_axis=flatten_axis,
        )

    assert not isinstance(lhs_q, ScaledTensor2x)
    assert not isinstance(rhs_q, ScaledTensor2x)

    return lhs_q, rhs_q


class GemmPrimitive(BasePrimitive):
    """
    Primitive for cuBLAS GEMM
    """

    name = "te_gemm_ffi"
    multiple_results = True
    impl_static_args = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        gelu_input,
        aux_in,
        out_dtype,
        contracting_dims,
        batched_dims,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        sequence_parallel_output,
        sequence_dim,
        comm_overlap,
    ):
        del lhs_quantized_colwise, rhs_quantized_colwise, use_split_accumulator
        del (
            sequence_parallel_output,
            sequence_dim,
        )

        def _dims_are_consecutive(dims):
            if len(dims) <= 1:
                return True
            return sorted(dims) == list(range(min(dims), max(dims) + 1))

        # Sanity-check operand layouts and types
        operand_ndims = (lhs.ndim, rhs.ndim)

        (
            lhs_contracting_dims,
            rhs_contracting_dims,
        ) = map(sanitize_dims, operand_ndims, contracting_dims)
        assert _dims_are_consecutive(lhs_contracting_dims), (
            "cuBLAS GEMM expected consecutive contracting dimensions for LHS operand, but got "
            f"{lhs_contracting_dims}."
        )
        assert _dims_are_consecutive(rhs_contracting_dims), (
            "cuBLAS GEMM expected consecutive contracting dimensions for RHS operand, but got "
            f"{rhs_contracting_dims}."
        )

        (
            lhs_batch_dims,
            rhs_batch_dims,
        ) = map(sanitize_dims, operand_ndims, batched_dims)
        assert _dims_are_consecutive(lhs_batch_dims), (
            "cuBLAS GEMM expected consecutive batch dimensions for LHS operand, but got "
            f"{lhs_batch_dims}."
        )
        assert _dims_are_consecutive(rhs_batch_dims), (
            "cuBLAS GEMM expected consecutive batch dimensions for RHS operand, but got "
            f"{rhs_batch_dims}."
        )
        if len(lhs_batch_dims) == 0:
            assert (
                len(rhs_batch_dims) == 0
            ), "cuBLAS GEMM RHS operand cannot be batched if LHS operand is not batched."
        elif len(rhs_batch_dims) != 0:
            assert all(bdim in lhs_contracting_dims for bdim in lhs_batch_dims) and all(
                bdim in rhs_contracting_dims for bdim in rhs_batch_dims
            ), "cuBLAS GEMM batched dimensions must be contracting when both operands are batched."

        lhs_contracting_size, rhs_contracting_size = map(
            lambda shape, dims: reduce(operator.mul, [shape[dim] for dim in dims]),
            (lhs.shape, rhs.shape),
            (lhs_contracting_dims, rhs_contracting_dims),
        )
        assert lhs_contracting_size == rhs_contracting_size, (
            "cuBLAS GEMM operands have incompatible contracting dimensions: "
            f"{lhs.shape} @ idx {lhs_contracting_dims} X {rhs.shape} @ idx {rhs_contracting_dims}."
        )

        lhs_is_transposed, rhs_is_transposed = _get_gemm_layout(operand_ndims, contracting_dims)
        if scaling_mode != ScalingMode.NO_SCALING:
            assert _compatible_fp8_gemm_dtypes(lhs.dtype, rhs.dtype), (
                "cuBLAS GEMM quantized operands have incompatible data types: "
                f"{lhs.dtype} x {rhs.dtype}."
            )
            assert (
                lhs_scale_inv.size > 0 and rhs_scale_inv.size > 0
            ), "Quantized cuBLAS GEMM requires inverse scaling factors for both operands."
            if (
                scaling_mode != ScalingMode.MXFP8_1D_SCALING
                and not tex.is_non_nt_fp8_gemm_supported()
            ):
                assert not lhs_is_transposed and rhs_is_transposed, (
                    "cuBLAS FP8 GEMM on devices with compute capability < 10.0 (Hopper) "
                    "require non-transposed LHS and transposed RHS operands "
                    "(`contracting_dims=((-1, ), (-1, ))`)."
                )

        # Determine output shape and dtype
        assert (
            dtypes.canonicalize_dtype(out_dtype).itemsize > 1
        ), "cuBLAS GEMM custom op does not support 8-bit quantized output types."
        lhs_non_contracting_shape, rhs_non_contracting_shape = map(
            lambda shape, dims: [shape[dim] for dim in range(len(shape)) if dim not in dims],
            (lhs.shape, rhs.shape),
            (lhs_contracting_dims, rhs_contracting_dims),
        )
        out_shape = [*lhs_non_contracting_shape, *rhs_non_contracting_shape]
        output = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)

        # Auxiliary output for comm+GEMM overlap
        aux_out_shape = (0,)
        aux_out_dtype = jnp.bfloat16
        if comm_overlap.is_enabled:
            if comm_overlap.is_bulk():
                # Bulk overlap will all-gather or reduce-scatter the tensor in the auxiliary input
                # and return the result of the collective in the auxiliary output
                assert (
                    aux_in.size > 0
                ), "cuBLAS GEMM w/ bulk collective overlap requires an auxiliary input."
                assert aux_in.ndim > 1, (
                    "cuBLAS GEMM w/ bulk collective overlap only supports multidimensional "
                    "auxiliary inputs."
                )

                aux_out_shape = list(aux_in.shape).copy()
                aux_out_dtype = aux_in.dtype
                if comm_overlap.sharded_impl:
                    if comm_overlap["comm_type"] == tex.CommOverlapType.AG:
                        aux_out_shape[comm_overlap.gather_dim] *= comm_overlap.tp_size
                    else:
                        assert aux_in.shape[comm_overlap.scatter_dim] % comm_overlap.tp_size, (
                            "cuBLAS GEMM w/ bulk reduce-scatter overlap requires the auxiliary "
                            "input to be divisible by tensor-parallel size in dimension index "
                            f"{comm_overlap.scatter_dim}."
                        )
                        aux_out_shape[comm_overlap.scatter_dim] = (
                            aux_out_shape[comm_overlap.scatter_dim] // comm_overlap.tp_size
                        )

            elif comm_overlap.is_all_gather():
                # Sharded abstract multiplies gathered dimension by TP size
                if comm_overlap.sharded_impl:
                    out_shape[comm_overlap.gather_dim] *= comm_overlap.tp_size
                    output = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)

                # AG->GEMM overlap can copy all-gathered LHS into the auxiliary buffer
                if comm_overlap.output_all_gathered_lhs:
                    aux_out_shape = list(lhs.shape).copy()
                    aux_out_dtype = lhs.dtype

                    # Sharded abstract multiplies gathered dimension by TP size
                    if comm_overlap.sharded_impl:
                        aux_out_shape[comm_overlap.gather_dim] *= comm_overlap.tp_size
            elif comm_overlap.is_reduce_scatter():
                # GEMM->RS auxiliary output is the reduce-scattered output
                rs_out_shape = list(out_shape).copy()

                # Sharded abstract divides scattered dimension by TP size
                if comm_overlap.sharded_impl:
                    rs_out_shape[comm_overlap.scatter_dim] = (
                        rs_out_shape[comm_overlap.scatter_dim] // comm_overlap.tp_size
                    )

                output = jax.core.ShapedArray(shape=rs_out_shape, dtype=out_dtype)

        aux_out = jax.core.ShapedArray(shape=aux_out_shape, dtype=aux_out_dtype)

        # Validate bias
        bias_shape = (0,)
        bias_dtype = out_dtype
        if fuse_bias:
            expected_bias_size = reduce(operator.mul, rhs_non_contracting_shape)
            if not grad:
                assert bias.size == expected_bias_size, (
                    "cuBLAS GEMM bias tensor has incorrect shape, "
                    f"expected ({expected_bias_size}, ) but found {bias.shape}."
                )
                assert bias.dtype == out_dtype, (
                    "cuBLAS GEMM bias tensor has incorrect data type, "
                    f"expected {bias_dtype} but found {bias.dtype}."
                )
                bias_shape = bias.shape
            else:
                bias_shape = rhs_non_contracting_shape
        bias_grad = jax.core.ShapedArray(shape=bias_shape, dtype=bias_dtype)

        # Validate pre-GeLU
        pre_gelu_shape = (0,)
        pre_gelu_dtype = out_dtype
        if fuse_gelu:
            pre_gelu_shape = out_shape
            if grad:
                pre_gelu_ndim = len(pre_gelu_shape)
                assert gelu_input.ndim == pre_gelu_shape and all(
                    gelu_input.shape[i] == pre_gelu_shape[i] for i in range(pre_gelu_ndim)
                ), (
                    "cuBLAS GEMM pre-GeLU tensor has incorrect shape, "
                    f"expected {pre_gelu_shape} but found {gelu_input.shape}."
                )
                assert gelu_input.dtype == out_dtype, (
                    "cuBLAS GEMM pre-GeLU tensor has incorrect data type, "
                    f"expected {pre_gelu_dtype} but found {gelu_input.dtype}."
                )
        pre_gelu_out = jax.core.ShapedArray(shape=pre_gelu_shape, dtype=pre_gelu_dtype)

        # Need extra workspace for swizzled scale factors
        lhs_swizzle_size = 0
        rhs_swizzle_size = 0
        swizzle_dtype = jnp.uint8
        if scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            lhs_swizzle_size = lhs_scale_inv.size
            rhs_swizzle_size = rhs_scale_inv.size
        lhs_swizzle = jax.core.ShapedArray(shape=(lhs_swizzle_size,), dtype=swizzle_dtype)
        rhs_swizzle = jax.core.ShapedArray(shape=(rhs_swizzle_size,), dtype=swizzle_dtype)

        # Size cuBLAS workspace -- multiplied by number of comm+GEMM overlap compute streams
        workspace_size = get_cublas_workspace_size_bytes()
        if comm_overlap.is_enabled:
            workspace_size *= comm_overlap.num_max_streams

        # cuBLAS requires workspace pointers aligned to 256 bytes but XLA does not guarantee that
        # so we add to the size here and align the pointer in the C++ custom call.
        workspace_size += 256
        workspace = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)

        return output, bias_grad, pre_gelu_out, aux_out, lhs_swizzle, rhs_swizzle, workspace

    @staticmethod
    def outer_abstract(*args, **kwargs):
        outputs = GemmPrimitive.abstract(*args, **kwargs)
        return outputs[:-3]  # discard workspace arrays

    @staticmethod
    def lowering(
        ctx,
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        gelu_input,
        aux_in,
        out_dtype,
        contracting_dims,
        batched_dims,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        sequence_parallel_output,
        sequence_dim,
        comm_overlap,
    ):
        del batched_dims, lhs_quantized_colwise, rhs_quantized_colwise, out_dtype
        del sequence_parallel_output, sequence_dim

        lhs_aval, _, rhs_aval, *_ = ctx.avals_in
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs_aval.ndim, rhs_aval.ndim), contracting_dims)
        lhs_transposed, rhs_transposed = _get_gemm_layout(
            (lhs_aval.ndim, rhs_aval.ndim), (lhs_cdims, rhs_cdims)
        )

        args = (lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, aux_in)
        kwargs = {
            "scaling_mode": int(scaling_mode.value),
            "lhs_axis_boundary": max(lhs_cdims) + 1 if lhs_transposed else min(lhs_cdims),
            "rhs_axis_boundary": min(rhs_cdims) if rhs_transposed else max(rhs_cdims) + 1,
            "lhs_transposed": lhs_transposed,
            "rhs_transposed": rhs_transposed,
            "fuse_bias": fuse_bias,
            "fuse_gelu": fuse_gelu,
            "grad": grad,
            "use_split_accumulator": use_split_accumulator,
        }
        kwargs.update(comm_overlap.get_lowering_kwargs())

        operand_output_aliases = {}
        if fuse_bias and not grad:
            operand_output_aliases.update({4: 1})  # bias <-> bias_grad
        if fuse_gelu and grad:
            operand_output_aliases.update({5: 2})  # gelu_input <-> pre_gelu_out

        return jax.ffi.ffi_lowering(
            GemmPrimitive.name,
            operand_output_aliases=operand_output_aliases,
        )(ctx, *args, **kwargs)

    @staticmethod
    def impl(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        gelu_input,
        aux_in,
        out_dtype,
        contracting_dims,
        batched_dims,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        sequence_parallel_output,
        sequence_dim,
        comm_overlap,
    ):
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs.ndim, rhs.ndim), contracting_dims)
        lhs_transposed, rhs_transposed = _get_gemm_layout(
            (lhs.ndim, rhs.ndim), (lhs_cdims, rhs_cdims)
        )
        lhs_scale_inv = apply_padding_to_scale_inv(
            lhs_scale_inv,
            scaling_mode,
            lhs.shape,
            is_colwise=lhs_quantized_colwise,
            flatten_axis=max(lhs_cdims) + 1 if lhs_transposed else min(lhs_cdims),
        )
        rhs_scale_inv = apply_padding_to_scale_inv(
            rhs_scale_inv,
            scaling_mode,
            rhs.shape,
            is_colwise=rhs_quantized_colwise,
            flatten_axis=min(rhs_cdims) if rhs_transposed else max(rhs_cdims) + 1,
        )

        outputs = GemmPrimitive.inner_primitive.bind(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_input,
            aux_in,
            out_dtype=out_dtype,
            contracting_dims=contracting_dims,
            batched_dims=batched_dims,
            lhs_quantized_colwise=lhs_quantized_colwise,
            rhs_quantized_colwise=rhs_quantized_colwise,
            scaling_mode=scaling_mode,
            fuse_bias=fuse_bias,
            fuse_gelu=fuse_gelu,
            grad=grad,
            use_split_accumulator=use_split_accumulator,
            sequence_parallel_output=sequence_parallel_output,
            sequence_dim=sequence_dim,
            comm_overlap=comm_overlap,
        )
        return outputs[:-3]  # discard workspace arrays

    @staticmethod
    def batcher(
        batched_args,
        jax_batch_dims,
        out_dtype,
        contracting_dims,
        batched_dims,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        sequence_parallel_output,
        sequence_dim,
        comm_overlap,
    ):
        assert GemmPrimitive.outer_primitive is not None
        lhs, _, rhs, *_ = batched_args
        lhs_bdims, _, rhs_bdims, *_, aux_in_bdims = jax_batch_dims
        arg_lhs_bdims, arg_rhs_bdims = map(sanitize_dims, (lhs.ndim, rhs.ndim), batched_dims)
        arg_lhs_bdims = (None,) if len(arg_lhs_bdims) == 0 else arg_lhs_bdims
        assert all(bdim == arg_bdim for bdim, arg_bdim in zip(lhs_bdims, arg_lhs_bdims)), (
            "User-specified batch dimension(s) for cuBLAS GEMM LHS operand does not match batch "
            f"dimensions inferred by JAX/XLA, expected {lhs_bdims} but got {arg_lhs_bdims}."
        )
        arg_rhs_bdims = (None,) if len(arg_rhs_bdims) == 0 else arg_rhs_bdims
        assert all(bdim == arg_bdim for bdim, arg_bdim in zip(rhs_bdims, arg_rhs_bdims)), (
            "User-specified batch dimension(s) for cuBLAS GEMM RHS operand does not match batch "
            f"dimensions inferred by JAX/XLA, expected {lhs_bdims} but got {arg_lhs_bdims}."
        )

        # Output is batched like the non-contracting batch dimensions of the LHS operand
        lhs_cdims = sanitize_dims(lhs.ndim, contracting_dims)
        lhs_non_contracting_bdims = tuple(dim for dim in lhs_bdims if dim not in lhs_cdims)
        out_bdims = (None,) if len(lhs_non_contracting_bdims) == 0 else lhs_non_contracting_bdims

        # Bias gradient is never batched
        bias_bdims = (None,)

        # Pre-GeLU output, if exists, is batched like GEMM output
        pre_gelu_bdims = (None,)
        if fuse_gelu and not grad:
            pre_gelu_bdims = out_bdims

        aux_out_bdims = (None,)
        if comm_overlap.is_enabled:
            if comm_overlap.is_bulk():
                # Bulk overlap auxiliary output must have the same batch dims as the auxiliary input
                aux_out_bdims = aux_in_bdims
            elif comm_overlap.is_all_gather() and comm_overlap.output_all_gathered_lhs:
                # AG->GEMM overlap with all-gathered LHS output must have same batch dims as
                # sharded LHS input
                aux_out_bdims = arg_lhs_bdims

        return (
            GemmPrimitive.outer_primitive.bind(
                *batched_args,
                out_dtype=out_dtype,
                contracting_dims=contracting_dims,
                batched_dims=batched_dims,
                lhs_quantized_colwise=lhs_quantized_colwise,
                rhs_quantized_colwise=rhs_quantized_colwise,
                scaling_mode=scaling_mode,
                fuse_bias=fuse_bias,
                fuse_gelu=fuse_gelu,
                grad=grad,
                use_split_accumulator=use_split_accumulator,
                sequence_parallel_output=sequence_parallel_output,
                sequence_dim=sequence_dim,
                comm_overlap=comm_overlap,
            ),
            (out_bdims, bias_bdims, pre_gelu_bdims, aux_out_bdims),
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype,
        contracting_dims,
        batched_dims,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        sequence_parallel_output,
        sequence_dim,
        comm_overlap,
        mesh,
        arg_infos,
        result_infos,
    ):
        del (
            out_dtype,
            lhs_quantized_colwise,
            rhs_quantized_colwise,
            scaling_mode,
            grad,
        )
        del use_split_accumulator, result_infos

        lhs_specs, _, rhs_specs, *_, aux_in_specs = map(get_padded_spec, arg_infos)
        (
            _, (out_specs, bias_grad_specs, pre_gelu_specs, aux_out_specs), *_
        ) = comm_overlap.get_partitioning_rules(
            lhs_specs,
            rhs_specs,
            aux_in_specs,
            contracting_dims,
            batched_dims,
            sequence_parallel_output,
            sequence_dim,
        )

        # Discard bias gradient and pre-GeLU output specs based on fusion choices
        if not fuse_bias:
            bias_grad_specs = (None,)
        if not fuse_gelu:
            pre_gelu_specs = (None,)

        # Assemble output shardings
        out_shardings = list(
            map(
                lambda specs: NamedSharding(mesh, PartitionSpec(*specs)),
                (out_specs, bias_grad_specs, pre_gelu_specs, aux_out_specs),
            )
        )

        return out_shardings

    @staticmethod
    def partition(
        out_dtype,
        contracting_dims,
        batched_dims,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        sequence_parallel_output,
        sequence_dim,
        comm_overlap,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos

        lhs_specs, _, rhs_specs, *_, aux_in_specs = map(get_padded_spec, arg_infos)
        (
            (lhs_specs, rhs_specs, bias_specs, gelu_input_specs, aux_in_specs),
            (out_specs, bias_grad_specs, pre_gelu_specs, aux_out_specs),
            (reduce_spec, scatter_dim),
        ) = comm_overlap.get_partitioning_rules(
            lhs_specs,
            rhs_specs,
            aux_in_specs,
            contracting_dims,
            batched_dims,
            sequence_parallel_output,
            sequence_dim,
        )

        # Block scale inverses match their operands, but tensor scale inverses are unsharded.
        lhs_scale_specs = (None,)
        rhs_scale_specs = (None,)
        if scaling_mode.is_1d_block_scaling() and not comm_overlap.is_enabled:
            lhs_scale_specs = lhs_specs
            rhs_scale_specs = rhs_specs

        # Discard bias and pre-GeLU specs based on fusion choices
        if not fuse_bias:
            bias_specs = (None,)
            bias_grad_specs = (None,)
        if not fuse_gelu:
            gelu_input_specs = (None,)
            pre_gelu_specs = (None,)

        # Assemble argument shardings
        arg_shardings = tuple(
            map(
                lambda specs: NamedSharding(mesh, PartitionSpec(*specs)),
                (
                    lhs_specs,
                    lhs_scale_specs,
                    rhs_specs,
                    rhs_scale_specs,
                    bias_specs,
                    gelu_input_specs,
                    aux_in_specs,
                ),
            )
        )

        # Assemble output shardings
        out_shardings = list(
            map(
                lambda specs: NamedSharding(mesh, PartitionSpec(*specs)),
                (out_specs, bias_grad_specs, pre_gelu_specs, aux_out_specs),
            )
        )

        def _sharded_impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, aux_in):
            comm_overlap._set_sharded_impl(True)
            outputs = GemmPrimitive.impl(
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                gelu_input,
                aux_in,
                out_dtype=out_dtype,
                contracting_dims=contracting_dims,
                batched_dims=batched_dims,
                lhs_quantized_colwise=lhs_quantized_colwise,
                rhs_quantized_colwise=rhs_quantized_colwise,
                scaling_mode=scaling_mode,
                fuse_bias=fuse_bias,
                fuse_gelu=fuse_gelu,
                grad=grad,
                use_split_accumulator=use_split_accumulator,
                sequence_parallel_output=sequence_parallel_output,
                sequence_dim=sequence_dim,
                comm_overlap=comm_overlap,
            )
            comm_overlap._set_sharded_impl(False)

            # All-Reduce/Reduce-Scatter GEMM output
            if reduce_spec is not None:
                if scatter_dim is None:
                    outputs[0] = jax.lax.psum(outputs[0], reduce_spec)
                else:
                    outputs[0] = jax.lax.psum_scatter(
                        outputs[0], reduce_spec, scatter_dimension=scatter_dim, tiled=True
                    )

            return outputs

        return mesh, _sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(
        out_dtype,
        contracting_dims,
        batched_dims,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        sequence_parallel_output,
        sequence_dim,
        comm_overlap,
        mesh,
        operand_types,
        result_types,
    ):
        del (
            lhs_quantized_colwise,
            rhs_quantized_colwise,
            out_dtype,
            grad,
            use_split_accumulator,
            sequence_parallel_output,
            sequence_dim,
            comm_overlap,
        )
        del mesh, result_types

        prefix = "GemmPrimitive_"

        def _generate_operand_rules(name, ndim, cdims, bdims):
            specs = []
            ldims = tuple(i for i in range(ndim) if i not in bdims + cdims)
            for i in range(ndim):
                dim_name = None
                if i in bdims:
                    dim_idx = bdims.index(i) if len(bdims) > 1 else ""
                    dim_name = f"b{dim_idx}"
                elif i in cdims:
                    dim_idx = cdims.index(i) if len(cdims) > 1 else ""
                    dim_name = f"k{dim_idx}"
                else:
                    dim_idx = ldims.index(i) if len(ldims) > 1 else ""
                    dim_name = f"{name}_l{dim_idx}"
                specs.append(prefix + dim_name)
            return specs

        lhs, _, rhs, *_ = operand_types
        operand_ndims = (len(lhs.shape), len(rhs.shape))
        (lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims) = map(
            lambda dims: map(sanitize_dims, operand_ndims, dims),
            (contracting_dims, batched_dims),
        )
        lhs_specs, rhs_specs = map(
            _generate_operand_rules,
            ("lhs", "rhs"),
            operand_ndims,
            (lhs_cdims, rhs_cdims),
            (lhs_bdims, rhs_bdims),
        )
        lhs_scale_specs = ("…1",)
        rhs_scale_specs = ("…2",)
        if scaling_mode.is_1d_block_scaling():
            # Shardy rules for MXFP8 scales cannot be related to the operands because of the
            # global-unpadding and local-padding workflow. This can potentially insert expensive
            # re-shards in the partition call later if the scales are not already sharded correctly.
            lhs_scale_specs, rhs_scale_specs = map(
                lambda specs: tuple(spec.replace(prefix, prefix + "scale_inv_") for spec in specs),
                (lhs_specs, rhs_specs),
            )

        lhs_non_cspec = tuple(lhs_specs[i] for i in range(operand_ndims[0]) if i not in lhs_cdims)
        rhs_non_cspec = tuple(rhs_specs[i] for i in range(operand_ndims[1]) if i not in rhs_cdims)
        out_spec = (*lhs_non_cspec, *rhs_non_cspec)
        bias_spec = rhs_non_cspec if fuse_bias else ("…4",)
        gelu_spec = out_spec if fuse_gelu else ("…5",)
        aux_spec = ("…6",)

        return SdyShardingRule(
            operand_mappings=(
                lhs_specs,
                lhs_scale_specs,
                rhs_specs,
                rhs_scale_specs,
                bias_spec,
                gelu_spec,
                aux_spec,
            ),
            result_mappings=(
                out_spec,
                bias_spec,
                gelu_spec,
                aux_spec,
            ),
        )


register_primitive(GemmPrimitive)


def gemm_uses_jax_dot() -> bool:
    """Check if the GEMM call directs to the TE custom cuBLAS call or native JAX dot."""
    return not GemmPrimitive.enabled()


def _te_gemm(
    lhs: Union[jax.Array, ScaledTensor],
    rhs: Union[jax.Array, ScaledTensor],
    bias: jax.Array = None,
    gelu_input: jax.Array = None,
    aux_in: jax.Array = None,
    lhs_quantizer: Quantizer = None,
    rhs_quantizer: Quantizer = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (0,)),
    batched_dims: Tuple[Sequence[int], Sequence[int]] = ((), ()),
    fuse_bias: bool = False,
    fuse_gelu: bool = False,
    grad: bool = False,
    use_split_accumulator: bool = QuantizeConfig.FP8_2X_ACC_FPROP,
    sequence_parallel_output: bool = False,
    sequence_dim: int = None,
    comm_overlap: CommOverlapHelper = CommOverlapHelper(),
) -> Tuple[jax.Array, ...]:

    # Prepare non-quantized GEMM operands
    lhs_data = lhs
    rhs_data = rhs
    lhs_scale_inv = jnp.empty(0, dtype=jnp.float32)
    rhs_scale_inv = jnp.empty(0, dtype=jnp.float32)
    scaling_mode = ScalingMode.NO_SCALING
    lhs_is_transposed, rhs_is_transposed = _get_gemm_layout((lhs.ndim, rhs.ndim), contracting_dims)
    lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs.ndim, rhs.ndim), contracting_dims)
    lhs_bdims, rhs_bdims = map(sanitize_dims, (lhs.ndim, rhs.ndim), batched_dims)

    # Quantize operands (if necessary)
    lhs_q, rhs_q = _quantize_gemm_operands(lhs, rhs, lhs_quantizer, rhs_quantizer, contracting_dims)

    # Extract GEMM custom op inputs from quantized operands
    if isinstance(lhs_q, ScaledTensor):
        assert isinstance(rhs_q, ScaledTensor) or rhs_quantizer is not None, (
            "cuBLAS GEMM with quantized LHS and non-quantized RHS operands requires a valid "
            "`Quantizer` object to quantize the RHS operand."
        )
        if isinstance(lhs_q, ScaledTensor2x):
            # Choose the quantization of the contracting dimension(s)
            lhs_q = lhs_q.get_colwise_tensor() if lhs_is_transposed else lhs_q.get_rowwise_tensor()
        scaling_mode = lhs_q.scaling_mode
        lhs_data = lhs_q.data
        lhs_scale_inv = lhs_q.scale_inv
        if lhs_q.data_layout == "T":
            lhs_cdims = transpose_dims(lhs_q.ndim, lhs_cdims, flatten_axis=lhs_q.flatten_axis)
            lhs_bdims = transpose_dims(lhs_q.ndim, lhs_bdims, flatten_axis=lhs_q.flatten_axis)

    if isinstance(rhs_q, ScaledTensor):
        assert isinstance(lhs_q, ScaledTensor) or lhs_quantizer is not None, (
            "cuBLAS GEMM with non-quantized LHS and quantized RHS operands requires a valid "
            "`Quantizer` object to quantize the LHS operand."
        )
        if isinstance(rhs_q, ScaledTensor2x):
            # Choose the quantization of the contracting dimension(s)
            rhs_q = rhs_q.get_rowwise_tensor() if rhs_is_transposed else rhs_q.get_colwise_tensor()
        assert rhs_q.scaling_mode == lhs_q.scaling_mode, (
            "cuBLAS GEMM quantized operands have mismatched scaling types, "
            f"LHS:{lhs_q.scaling_mode} x RHS:{rhs_q.scaling_mode}."
        )
        rhs_data = rhs_q.data
        rhs_scale_inv = rhs_q.scale_inv
        if rhs_q.data_layout == "T":
            rhs_cdims = transpose_dims(rhs_q.ndim, rhs_cdims, flatten_axis=rhs_q.flatten_axis)
            rhs_bdims = transpose_dims(rhs_q.ndim, rhs_bdims, flatten_axis=rhs_q.flatten_axis)

    # Dummy empties for bias and gelu
    out_dtype = lhs_q.dq_dtype if isinstance(lhs_q, ScaledTensor) else lhs_data.dtype
    if bias is None or not (fuse_bias and not grad):
        bias = jnp.empty(0, dtype=out_dtype)
    if gelu_input is None or not (fuse_gelu and grad):
        gelu_input = jnp.empty(0, dtype=out_dtype)
    if aux_in is None:
        aux_in = jnp.empty(0, dtype=jnp.bfloat16)

    return GemmPrimitive.outer_primitive.bind(
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        gelu_input,
        aux_in,
        out_dtype=out_dtype,
        contracting_dims=(lhs_cdims, rhs_cdims),
        batched_dims=(lhs_bdims, rhs_bdims),
        lhs_quantized_colwise=lhs_q.is_colwise if isinstance(lhs_q, ScaledTensor) else False,
        rhs_quantized_colwise=rhs_q.is_colwise if isinstance(rhs_q, ScaledTensor) else False,
        scaling_mode=scaling_mode,
        fuse_bias=fuse_bias,
        fuse_gelu=fuse_gelu,
        grad=grad,
        use_split_accumulator=use_split_accumulator,
        sequence_parallel_output=sequence_parallel_output,
        sequence_dim=sequence_dim,
        comm_overlap=comm_overlap,
    )


def _shape_normalization(x, dimension_numbers, already_transposed: bool = False):
    orig_order = list(range(x.ndim))
    contracting_dims, batch_dims = dimension_numbers
    contracting_order = [d for d in orig_order if d in contracting_dims]
    batch_order = [d for d in orig_order if d in batch_dims]
    non_contracting_order = [
        d for d in orig_order if d not in contracting_dims and d not in batch_dims
    ]
    batch_shape = [x.shape[d] for d in batch_order]
    rows_shape = [x.shape[d] for d in non_contracting_order]
    cols_shape = [x.shape[d] for d in contracting_order]
    new_order = batch_order + non_contracting_order + contracting_order
    rows, cols, batches = (
        reduce(operator.mul, rows_shape, 1),
        reduce(operator.mul, cols_shape, 1),
        reduce(operator.mul, batch_shape, 1),
    )
    # Remove this transpose when non-TN dot is supported
    if not already_transposed:
        t = jnp.transpose(x, new_order)
    else:
        t = x
    return jnp.reshape(t, (batches, rows, cols))


# Apply jit to guarantee correctness of FP8 GEMM.
@partial(jax.jit, static_argnums=(2, 3))
def _jax_gemm_tensor_scaling_fp8(lhs, rhs, dim_nums, precision):
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
    if lhs.data_layout == "T":
        lhs_contract = transpose_dims(lhs.data.ndim, lhs_contract, flatten_axis=lhs.flatten_axis)
        lhs_batch = transpose_dims(lhs.data.ndim, lhs_batch, flatten_axis=lhs.flatten_axis)
    if rhs.data_layout == "T":
        rhs_contract = transpose_dims(rhs.data.ndim, rhs_contract, flatten_axis=rhs.flatten_axis)
        rhs_batch = transpose_dims(rhs.data.ndim, rhs_batch, flatten_axis=rhs.flatten_axis)

    dim_nums = (lhs_contract, rhs_contract), (lhs_batch, rhs_batch)

    out_fp8 = jax.lax.dot_general(
        lhs.data, rhs.data, dim_nums, precision=precision, preferred_element_type=lhs.dq_dtype
    )
    scale_inv = lhs.scale_inv * rhs.scale_inv
    out = (out_fp8 * scale_inv).astype(lhs.dq_dtype)

    return out


@partial(jax.jit, static_argnums=(2,))
def _jax_gemm_mxfp8_1d(
    lhs: ScaledTensor, rhs: ScaledTensor, dim_nums: Tuple[Tuple[Sequence[int], Sequence[int]]]
):
    """
    JAX GEMM for MXFP8 via scaled_matmul
    """
    assert (
        rhs.scaling_mode == ScalingMode.MXFP8_1D_SCALING
    ), "rhs does not have MXFP8 1D scaling mode"

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums

    expected_lhs_is_colwise = lhs_contract[-1] != lhs.data.ndim - 1
    expected_rhs_is_colwise = rhs_contract[-1] != rhs.data.ndim - 1
    assert lhs.is_colwise is expected_lhs_is_colwise, (
        f"LHS with unexpected quantize dimension.\nExpect is_colwise={expected_lhs_is_colwise}, got"
        f" {lhs.is_colwise}"
    )
    assert rhs.is_colwise is expected_rhs_is_colwise, (
        f"RHS with unexpected quantize dimension.\nExpect is_colwise={expected_rhs_is_colwise}, got"
        f" {rhs.is_colwise}"
    )

    # Reshape + Transpose (if needed)
    # [..., M, K] -> [1, reduce(..., M), K]
    # [..., K, M] -> [1, reduce(..., M), K]
    lhs_3d = _shape_normalization(lhs.data, (lhs_contract, lhs_batch))
    rhs_3d = _shape_normalization(rhs.data, (rhs_contract, rhs_batch))
    lhs_scale_3d = _shape_normalization(lhs.scale_inv, (lhs_contract, lhs_batch))
    rhs_scale_3d = _shape_normalization(rhs.scale_inv, (rhs_contract, rhs_batch))

    # JAX scaled_matmul only supports NT now (TN-gemm)
    # * Expected shape:
    # * lhs_data  (B, M, K)           * rhs_data  (B, N, K)
    # * lhs_scale (B, M, K_block)     * rhs_scale (B, N, K_block)
    out_3d = jax.nn.scaled_matmul(
        lhs_3d, rhs_3d, lhs_scale_3d, rhs_scale_3d, preferred_element_type=lhs.dq_dtype
    )
    # Reshape [1, reduce(..., M), N] -> [..., M, N]
    lhs_remain_shape = tuple(
        lhs.data.shape[dim] for dim in range(len(lhs.data.shape)) if dim not in lhs_contract
    )
    rhs_remain_shape = tuple(
        rhs.data.shape[dim] for dim in range(len(rhs.data.shape)) if dim not in rhs_contract
    )
    out = out_3d.reshape(*lhs_remain_shape, *rhs_remain_shape)
    return out


def _jax_gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    lhs_quantizer: Quantizer = None,
    rhs_quantizer: Quantizer = None,
) -> jnp.ndarray:
    """
    FP8 GEMM via JAX
    """
    dim_nums = (contracting_dims, ((), ()))

    def _jax_gemm_fp8_impl(lhs, rhs):
        if lhs.scaling_mode.is_tensor_scaling():
            assert (
                rhs.scaling_mode == lhs.scaling_mode
            ), f"rhs.scaling_mode={rhs.scaling_mode} != lhs.scaling_mode={lhs.scaling_mode}"
            precision = (
                jax.lax.Precision.HIGHEST
                if QuantizeConfig.FP8_2X_ACC_FPROP
                else jax.lax.Precision.DEFAULT
            )
            return _jax_gemm_tensor_scaling_fp8(lhs, rhs, dim_nums, precision)

        if lhs.scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            return _jax_gemm_mxfp8_1d(lhs, rhs, dim_nums)

        raise NotImplementedError("Unsupported ScalingMode: {lhs.scaling_mode}")

    lhs_q, rhs_q = _quantize_gemm_operands(lhs, rhs, lhs_quantizer, rhs_quantizer, contracting_dims)

    if isinstance(lhs_q, ScaledTensor) and isinstance(rhs_q, ScaledTensor):
        return _jax_gemm_fp8_impl(lhs_q, rhs_q)

    if (
        isinstance(lhs, jnp.ndarray)
        and isinstance(rhs, jnp.ndarray)
        and lhs_quantizer is None
        and rhs_quantizer is None
    ):
        return jax.lax.dot_general(lhs, rhs, dim_nums, preferred_element_type=lhs.dtype)

    raise NotImplementedError("Not supporting multiplication of ScaledTensor and jnp.array")


def gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (0,)),
    batched_dims: Tuple[Sequence[int], Sequence[int]] = ((), ()),
    lhs_quantizer: Quantizer = None,
    rhs_quantizer: Quantizer = None,
    **kwargs,
) -> Tuple[jnp.ndarray, ...]:
    r"""General matrix multiplication with optional quantization.

    Parameters
    ----------
    lhs: Union[jax.Array, ScaledTensor]
        Left-hand side operand in the matrix multiplication.
    rhs: Union[jax.Array, ScaledTensor]
        Right-hand side operand in the matrix multiplication.
    lhs_quantizer: Quantizer, default = None
        Object for down-casting the LHS operand for quantized GEMM.
    rhs_quantizer: Quantizer, default = None
        Object for down-casting the RHS operand for quantized GEMM.
    contracting_dims: Tuple[Sequence[int], Sequence[int]], default = ((-1, ), (0, ))
        Tuple of sequences representing the contracting dimensions of the operands.
    batched_dims: Tuple[Sequence[int], Sequence[int]], default = ((), ()),
        Tuple of sequences representing the batched dimensions of the operands. This is *not* used
        to perform a batched matrix multiplication, but it is required for TE's custom cuBLAS GEMM
        call to avoid a potentially undesirable reduction in any batched contracting dimensions
        when invoked with sharded operands (e.g. when computing weight gradients in a Flax module).
    bias: jax.Array, default = None
        Optional additive bias term, required for forward GEMM with bias fusion. Only supported
        with TE's custom call to cuBLAS GEMM.
    gelu_input: jax.Array, default = None
        Pre-GeLU output from forward GEMM, required for backward/grad GEMM with dGeLU fusion. Only
        supported with TE's custom call to cuBLAS GEMM.
    fuse_bias: bool, default = False
        Enable bias addition in forward GEMM or bias gradient in backward GEMM. Only supported with
        TE's custom call to cuBLAS GEMM.
    fuse_gelu: bool, default = False
        Enable GeLU activation in forward GEMM or GeLU gradient in backward GEMM. Only supported
        with TE's custom call to cuBLAS GEMM.
    grad: bool, default = False
        Flag for switching bias and GeLU fusions from forward to backward mode. Only supported with
        TE's custom call to cuBLAS GEMM.
    use_split_accumulator: bool, default = True
        Enable promoting some intermediate sums to higher precision when accumulating the result in
        the cuBLAS GEMM kernel. Disabling this trades off numerical accuracy for speed. Only
        supported with TE's custom call to cuBLAS GEMM.
    sequence_parallel_output: bool, default = False
        Produces an output with the first non-batched non-contracting dimension sharded with the
        same spec as operand contracting dimensions. This effectively converts the `jax.lax.psum`
        for the GEMM output into a `jax.lax.psum_scatter`. Only supported with TE's custom call to
        cuBLAS GEMM.
    sequence_dim: int, default = None
        Index of the sequence dimension for the LHS operand. This controls which dimension of the
        GEMM output is scattered when `sequence_parallel_output=True`. When `None`, the first
        non-batched non-contracting dimension is assumed to be the sequence dimension.
    comm_overlap: CommOverlapHelper, default = None
        Helper object that manages comm+GEMM overlap options.
    aux_in: jax.Array, default = None
        Auxiliary input to be all-gathered or reduce-scattered when GEMM is bulk-overlapped with
        an independent collective.

    Returns
    -------
    jax.Array:
        Result of the operation. For TE's custom call to cuBLAS GEMM, this result can include the
        GeLU application when `fuse_gelu=True` and `grad=False`, the GeLU gradient contribution
        when `fuse_gelu=True` and `grad=True`, and the additive bias when `fuse_bias=True` and
        `grad=False`.
    Optional[jax.Array]:
        Bias gradient when `fuse_bias=True` and `grad=True`. Only supported with TE's custom call
        to cuBLAS GEMM.
    Optional[jax.Array]:
        Pre-GeLU GEMM output when `fuse_gelu=True` and `grad=False`. This is required as an input
        to `_te_gemm()` with `fuse_gelu=True` and `grad=True` in the backward pass in order to
        compute the GeLU contribution to the gradient. Only supported with TE's custom call to
        cuBLAS GEMM.
    Optional[jax.Array]:
        Auxiliary output from comm+GEMM overlap, used only for overlapping bulk-collectives or
        returning an all-gathered copy of the LHS operand.
    """
    # Try to get LHS and RHS quantizers from a quantizer set for backward compatibility
    if lhs_quantizer is None or rhs_quantizer is None:
        quantizer_set = kwargs.get("quantizer_set", None)
        if quantizer_set is not None:
            lhs_quantizer = quantizer_set.x
            rhs_quantizer = quantizer_set.kernel

    # Fall back on a native JAX implementation when the custom call to cuBLAS GEMM is disabled
    fuse_bias = kwargs.get("fuse_bias", False)
    fuse_gelu = kwargs.get("fuse_gelu", False)
    if not GemmPrimitive.enabled():
        assert kwargs.get("bias", None) is None and not fuse_gelu, (
            "TE GEMM was invoked with bias fusion options that are not supported by the "
            "`jax.lax.dot_general` and `jax.nn.scaled_matmul` backends used when the custom cuBLAS "
            "GEMM primitive is disabled."
        )
        assert kwargs.get("gelu_input", None) is None and not fuse_bias, (
            "TE GEMM was invoked with GeLU fusion options that are not supported by the "
            "`jax.lax.dot_general` and `jax.nn.scaled_matmul` backends used when the custom cuBLAS "
            "GEMM primitive is disabled."
        )
        assert (
            not kwargs.get("sequence_parallel_output", False)
            and kwargs.get("sequence_dim", None) is None
        ), (
            "TE GEMM was invoked with sequence-parallelism options that are not supported by the "
            "`jax.lax.dot_general` and `jax.nn.scaled_matmul` backends used when the custom cuBLAS "
            "GEMM primitive is disabled."
        )
        assert (
            kwargs.get("comm_overlap", None) is None
            and kwargs.get("aux_in", None) is None
        ), (
            "TE GEMM was invoked with communication overlap options that are not supported by the "
            "`jax.lax.dot_general` and `jax.nn.scaled_matmul` backends used when the custom cuBLAS "
            "GEMM primitive is disabled."
        )
        return _jax_gemm(lhs, rhs, contracting_dims, lhs_quantizer, rhs_quantizer)

    outputs = _te_gemm(
        lhs,
        rhs,
        lhs_quantizer=lhs_quantizer,
        rhs_quantizer=rhs_quantizer,
        contracting_dims=contracting_dims,
        batched_dims=batched_dims,
        **kwargs,
    )

    # Discard empty outputs
    grad = kwargs.get("grad", False)
    comm_overlap = kwargs.get("comm_overlap", CommOverlapHelper())
    clean_outputs = outputs[0]  # first output is the final result and is never empty
    if (fuse_bias and grad) or (fuse_gelu and not grad):
        clean_outputs = (outputs[0],)
        if fuse_bias and grad:  # only return bias gradient if it exists
            clean_outputs += (outputs[1],)
        if fuse_gelu and not grad:  # only return pre-GeLU output if it exists
            clean_outputs += (outputs[2],)
        if comm_overlap.has_aux_output():
            # only return aux output for bulk overlap or non-bulk all-gather overlap
            # with gathered LHS output
            clean_outputs += (outputs[3],)
    return clean_outputs
