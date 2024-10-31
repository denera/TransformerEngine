# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for cuBlasLt GEMM"""
import warnings
import operator
from functools import reduce
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import dtypes
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax.sharding import PartitionSpec, NamedSharding
from jax.extend import ffi
from jax.typing import ArrayLike

from transformer_engine import transformer_engine_jax as tex
from .base import BasePrimitive, register_primitive
from .custom_call import custom_caller, CustomCallArgsWrapper
from .misc import (
    check_valid_batch_dims,
    jax_dtype_to_te_dtype,
    get_padded_spec,
    is_ffi_enabled,
    is_fp8_dtype,
)
from ..sharding import (
    global_mesh_resource,
    lax_paral_op,
)


__all__ = [
    "fp8_gemm",
    "gemm",
]


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if tex.get_device_compute_capability() >= 90:
        return 33_554_432
    return 4_194_304


class GemmPrimitive(BasePrimitive):
    """
    cuBlasLt GEMM Primitive w/ support for distributed inputs
    """

    name = "te_gemm"
    impl_static_args = None
    multiple_results = True
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        lhs_aval: ArrayLike,
        lhs_scale_inv_aval: ArrayLike,
        rhs_aval: ArrayLike,
        rhs_scale_inv_aval: ArrayLike,
        bias_aval: ArrayLike,
        out_amax_aval: ArrayLike,
        out_scale_aval: ArrayLike,
        out_dtype: jnp.dtype,
        layout: str,
        do_gelu: bool,
        use_bias: bool,
        grad: bool,
        accumulate: bool,
        use_split_accumulator: bool,
    ):
        """
        cuBlasLt GEMM abstract
        """
        del grad, accumulate, use_split_accumulator

        # Validate input dtypes
        lhs_dtype = dtypes.canonicalize_dtype(lhs_aval.dtype)
        rhs_dtype = dtypes.canonicalize_dtype(rhs_aval.dtype)
        assert lhs_dtype == rhs_dtype, "Mismatched matrix dtypes!"
        is_fp8 = False
        if is_fp8_dtype(lhs_dtype):
            assert (
                lhs_scale_inv_aval.size == 1
                and dtypes.canonicalize_dtype(lhs_scale_inv_aval.dtype) == jnp.float32
            ), "Missing FP8 meta!"
            is_fp8 = True
        if is_fp8_dtype(rhs_dtype):
            assert (
                rhs_scale_inv_aval.size == 1
                and dtypes.canonicalize_dtype(rhs_scale_inv_aval.dtype) == jnp.float32
            ), "Missing FP8 meta!"

        # Validate input layouts
        if is_fp8:
            assert layout == 'NT', "FP8 GEMM custom op only supports 'NT' layout!"
        else:
            assert layout in ['NN', 'NT', 'TN'], "Invalid GEMM layout!"
        lhs_trans = layout[0] == 'T'
        rhs_trans = layout[1] == 'T'
        lhs_outer_idx = -2 if lhs_trans else -1
        lhs_inner_idx = -1 if lhs_trans else -2
        rhs_outer_idx = -1 if rhs_trans else -2
        rhs_inner_idx = -2 if rhs_trans else -1
        assert (
            lhs_aval.shape[lhs_inner_idx] == rhs_aval.shape[rhs_inner_idx]
        ), "Incompatible operand sizes!"
        assert all([
            lhs_batch == rhs_batch for lhs_batch, rhs_batch \
            in zip(lhs_aval.shape[:-2], rhs_aval.shape[:-2])
        ]), "Incompatible batch sizes!"

        # Validate output dtype
        out_dtype = dtypes.canonicalize_dtype(out_dtype)
        if is_fp8_dtype(out_dtype):
            assert (
                is_fp8_dtype(lhs_dtype)
                and is_fp8_dtype(rhs_dtype)
            ), "FP8 GEMM output requires FP8 inputs!"
            assert (
                out_amax_aval.size == out_scale_aval.size == 1
            ), "Invalid/missing output amax and scale!"
            out_amax_updated_type = dtypes.canonicalize_dtype(out_amax_aval.dtype)
            out_scale_updated_type = dtypes.canonicalize_dtype(out_scale_aval.dtype)
            assert (
                out_amax_updated_type == out_scale_updated_type == jnp.float32
            ), "Invalid output amax or scale dtype!"
        else:
            out_dtype = lhs_dtype

        # Infer output size and create abstract arrays
        out_shape = (
            *lhs_aval.shape[:-2],
            lhs_aval.shape[lhs_outer_idx],
            rhs_aval.shape[rhs_outer_idx]
        )
        out_aval = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)
        out_amax_updated_aval = out_amax_aval.update(shape=out_amax_aval.shape, dtype=jnp.float32)
        out_scale_updated_aval = out_scale_aval.update(shape=out_scale_aval.shape, dtype=jnp.float32)

        bias_dtype = jnp.bfloat16 if not use_bias else dtypes.canonicalize_dtype(bias_aval.dtype)
        if use_bias:
            assert (
                bias_aval.size > 0
                and bias_aval.ndim == 1
                and bias_aval.shape[0] == out_shape[-1]
            ), "Incorrect bias shape!"
        else:
            assert bias_aval.size == 0, "Internal TE error!"

        pre_gelu_shape = out_shape if do_gelu else (0, )
        pre_gelu_aval = jax.core.ShapedArray(shape=pre_gelu_shape, dtype=bias_dtype)

        workspace_size = get_cublas_workspace_size_bytes()
        workspace_aval = jax.core.ShapedArray(shape=(workspace_size, ), dtype=jnp.uint8)

        return out_aval, out_amax_updated_aval, out_scale_updated_aval, pre_gelu_aval, \
               workspace_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        cuBlasLt GEMM outer abstract
        """
        out_aval, out_amax_updated_aval, out_scale_updated_aval, pre_gelu_aval, _ = \
            GemmPrimitive.abstract(*args, **kwargs)
        return out_aval, out_amax_updated_aval, out_scale_updated_aval, pre_gelu_aval

    @staticmethod
    def lowering(
        ctx,
        lhs: ArrayLike,
        lhs_scale_inv: ArrayLike,
        rhs: ArrayLike,
        rhs_scale_inv: ArrayLike,
        bias: ArrayLike,
        out_amax: ArrayLike,
        out_scale: ArrayLike,
        out_dtype: jnp.dtype,
        layout: str,
        do_gelu: bool,
        use_bias: bool,
        grad: bool,
        accumulate: bool,
        use_split_accumulator: bool,
    ):
        """
        Fused attention fwd lowering rules
        """
        del do_gelu, use_bias

        lhs_trans = layout[0] == 'T'
        rhs_trans = layout[1] == 'T'
        workspace_size = get_cublas_workspace_size_bytes()

        if is_ffi_enabled():
            name = "te_gemm_ffi"
            return ffi.ffi_lowering(name, operand_output_aliases={5: 1, 6: 2})(
                ctx,
                rhs,
                rhs_scale_inv,
                lhs,
                lhs_scale_inv,
                bias,
                out_amax,
                out_scale,
                lhs_trans=lhs_trans,
                rhs_trans=rhs_trans,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator
            )
        else:
            operands = [
                rhs,
                rhs_scale_inv,
                lhs,
                lhs_scale_inv,
                bias,
                out_amax,
                out_scale,
            ]
            operand_shapes = map(lambda x: x.type.shape, operands)
            out_types = [
                ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
                for output in ctx.avals_out
            ]
            args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

            # LHS:([B], M, K) x RHS:([B], K, N) = OUT:([B], M, N)
            lhs_aval, _, rhs_aval, _, bias_aval, *_ = ctx.avals_in
            lhs_outer_idx = -2 if lhs_trans else -1
            lhs_inner_idx = -1 if lhs_trans else -2
            rhs_outer_idx = -1 if rhs_trans else -2
            batch = reduce(lhs_aval.shape[:-2], operator.mul, 1.) if lhs_aval.ndim > 2 else 1
            m = lhs_aval.shape[lhs_outer_idx]
            n = lhs_aval.shape[lhs_inner_idx]
            k = rhs_aval.shape[rhs_outer_idx]
            operand_dtype = jax_dtype_to_te_dtype(lhs_aval.dtype)
            bias_dtype = jax_dtype_to_te_dtype(bias_aval.dtype)
            opaque = tex.pack_gemm_descriptor(batch, m, n, k, workspace_size, operand_dtype,
                                              jax_dtype_to_te_dtype(out_dtype), bias_dtype,
                                              lhs_trans, rhs_trans, grad,  accumulate,
                                              use_split_accumulator)

            return custom_caller(GemmPrimitive.name, args, opaque, has_side_effect=False)

    @staticmethod
    def impl(
        lhs: ArrayLike,
        lhs_scale_inv: ArrayLike,
        rhs: ArrayLike,
        rhs_scale_inv: ArrayLike,
        bias: ArrayLike,
        out_amax: ArrayLike,
        out_scale: ArrayLike,
        out_dtype: jnp.dtype,
        layout: str,
        do_gelu: bool,
        use_bias: bool,
        grad: bool,
        accumulate: bool,
        use_split_accumulator: bool,
    ):
        assert GemmPrimitive.inner_primitive is not None

        output, out_amax_updated, out_scale_updated, pre_gelu_out, _ = \
            GemmPrimitive.inner_primitive.bind(
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                out_amax,
                out_scale,
                out_dtype,
                layout,
                do_gelu,
                use_bias,
                grad,
                accumulate,
                use_split_accumulator
            )

        return output, out_amax_updated, out_scale_updated, pre_gelu_out

    @staticmethod
    def batcher(batched_args, batch_dims, out_dtype, layout, do_gelu, use_bias, grad, accumulate,
                use_split_accumulator):
        assert GemmPrimitive.outer_primitive is not None
        check_valid_batch_dims(batch_dims)
        _, _, b_bdim, _, amax_bdim, scale_bdim, _ = batch_dims

        out_bdims = b_bdim, amax_bdim, scale_bdim, b_bdim
        return (
            GemmPrimitive.outer_primitive.bind(*batched_args, out_dtype, layout, do_gelu, use_bias,
                                               grad, accumulate, use_split_accumulator),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(out_dtype, layout, do_gelu, use_bias, grad, accumulate,
                                     use_split_accumulator, mesh, arg_infos, result_infos):
        del out_dtype, do_gelu, use_bias, grad, accumulate, use_split_accumulator, result_infos
        lhs_spec = get_padded_spec(arg_infos[0])
        rhs_spec = get_padded_spec(arg_infos[2])
        lhs_trans = layout[0] == 'T'
        rhs_trans = layout[1] == 'T'

        lhs_inner_idx = -1 if lhs_trans else -2
        rhs_inner_idx = -1 if rhs_trans else -2
        rhs_outer_idx = -2 if rhs_trans else -1

        if lhs_spec[lhs_inner_idx] != rhs_spec[rhs_inner_idx]:
            warnings.warn("Forcing the inner dimension of A to match the sharding of inner "
                          + "dimension of B. This can trigger additional communication of A is "
                          + "not partitioned correctly.")
        if rhs_spec[rhs_inner_idx] is not None and rhs_spec[rhs_outer_idx] is not None:
            raise RuntimeError("Both inner and outer dimensions of B cannot be sharded!")

        out_spec = [lhs_spec[rhs_outer_idx], rhs_spec[rhs_outer_idx]]
        if len(lhs_spec) > 2:
            out_spec = lhs_spec[:-2] + out_spec
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        return (out_sharding, fp8_meta_sharding, fp8_meta_sharding, out_sharding)

    @staticmethod
    def partition(out_dtype, layout, do_gelu, use_bias, grad, accumulate, use_split_accumulator,
                  mesh, arg_infos, result_infos):
        del out_dtype, do_gelu, use_bias, grad, accumulate, use_split_accumulator, result_infos
        lhs_spec = get_padded_spec(arg_infos[0])
        rhs_spec = get_padded_spec(arg_infos[2])
        lhs_trans = layout[0] == 'T'
        rhs_trans = layout[1] == 'T'

        # LHS:([B], M, K) x RHS:([B], K, N) = OUT:([B], M, N)
        lhs_outer_idx = -2 if lhs_trans else -1
        lhs_inner_idx = -1 if lhs_trans else -2
        rhs_inner_idx = -1 if rhs_trans else -2
        rhs_outer_idx = -2 if rhs_trans else -1

        if lhs_spec[lhs_inner_idx] != rhs_spec[rhs_inner_idx]:
            warnings.warn("Forcing the inner dimension of A to match the sharding of inner "
                          + "dimension of B. This can trigger additional communication of A is "
                          + "not partitioned correctly.")
        if rhs_spec[rhs_inner_idx] is not None and rhs_spec[rhs_outer_idx] is not None:
            raise RuntimeError("Both inner and outer dimensions of B cannot be sharded!")

        lhs_spec_new = [None, rhs_spec[rhs_inner_idx]]
        if len(lhs_spec) > 2:
            lhs_spec_new = lhs_spec[:-2] + lhs_spec_new
        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_spec_new))
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_spec))
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        out_spec = [lhs_spec[rhs_outer_idx], rhs_spec[rhs_outer_idx]]
        if len(lhs_spec) > 2:
            out_spec = lhs_spec[:-2] + out_spec
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))
        bias_sharding = NamedSharding(mesh, PartitionSpec(out_spec[-1]))

        arg_shardings = (lhs_sharding, fp8_meta_sharding, rhs_sharding, fp8_meta_sharding,
                         bias_sharding, fp8_meta_sharding, fp8_meta_sharding)
        out_shardings = (out_sharding, fp8_meta_sharding, fp8_meta_sharding, out_sharding)

        def impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, out_amax, out_scale, out_dtype,
                 layout, do_gelu, use_bias, grad, accumulate, use_split_accumulator):

            assert GemmPrimitive.inner_primitive is not None

            output, out_amax, out_scale, pre_gelu_out, _ = GemmPrimitive.inner_primitive.bind(
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                out_amax,
                out_scale,
                out_dtype,
                layout,
                do_gelu,
                use_bias,
                grad,
                accumulate,
                use_split_accumulator
            )

            if rhs_spec[rhs_inner_idx] is not None:
                # If the inner dimensions of LHS and RHS are sharded, we all-reduce the GEMM output.
                # If the outer dimension of LHS is also sharded, we reduce-scatter the GEMM output.
                par_op = (
                    jax.lax.psum_scatter
                    if lhs_spec[lhs_outer_idx] is not None
                    else jax.lax.psum
                )
                output = lax_paral_op(output, par_op, global_mesh_resource().tp_resource, mesh)
                if do_gelu:
                    pre_gelu_out = lax_paral_op(pre_gelu_out, par_op,
                                                global_mesh_resource().tp_resource, mesh)

            return output, out_amax, out_scale, pre_gelu_out

        return mesh, impl, out_shardings, arg_shardings


register_primitive(GemmPrimitive)


def fp8_gemm(
    lhs: ArrayLike,
    lhs_scale_inv: ArrayLike,
    rhs: ArrayLike,
    rhs_scale_inv: ArrayLike,
    bias: Optional[ArrayLike] = None,
    out_amax: Optional[ArrayLike] = None,
    out_scale: Optional[ArrayLike] = None,
    out_dtype: Optional[jnp.dtype] = None,
    do_gelu: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> Tuple[ArrayLike, ...]:
    """NT layout GEMM with FP8 inputs"""
    if out_dtype is not None and is_fp8_dtype(out_dtype):
        assert out_amax is not None and out_scale is not None, "Missing output amax and scale!"
    else:
        out_amax = jnp.empty((0, ), dtype=jnp.float32)
        out_scale = jnp.empty((0, ), dtype=jnp.float32)

    use_bias = bias is not None
    if not use_bias:
        bias = jnp.empty((0, ), dtype=jnp.bfloat16)

    out, out_amax, out_scale, pre_gelu_out = GemmPrimitive.outer_primitive.bind(
        lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, out_amax, out_scale, out_dtype, 'NT', do_gelu,
        use_bias, False, accumulate, use_split_accumulator
    )

    outputs = (out, )
    if out_dtype is not None and is_fp8_dtype(out_dtype):
        outputs += (out_amax, out_scale, )
    if do_gelu:
        outputs += (pre_gelu_out, )

    return outputs


def gemm(
    A: ArrayLike,
    B: ArrayLike,
    bias: Optional[ArrayLike] = None,
    layout: str = 'NT',
    do_gelu: bool = False,
    grad: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> Tuple[ArrayLike, ...]:
    """Non-FP8 GEMM"""
    use_bias = bias is not None
    if not use_bias:
        bias = jnp.empty((0, ), dtype=jnp.bfloat16)

    dummy_fp8_meta = jnp.empty((0, ), dtype=jnp.float32)

    D, _, _, pre_gelu_out = GemmPrimitive.outer_primitive.bind(
        A, dummy_fp8_meta, B, dummy_fp8_meta, bias, dummy_fp8_meta, dummy_fp8_meta, A.dtype,
        layout, do_gelu, use_bias, grad, accumulate, use_split_accumulator
    )

    outputs = (D, )
    if do_gelu:
        outputs += (pre_gelu_out, )

    return outputs
