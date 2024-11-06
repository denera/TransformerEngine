# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for cuBlasLt GEMM"""
import warnings
import operator
import math
from typing import Optional, Tuple, Sequence

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
        dimension_numbers: Tuple[Tuple[Tuple[Sequence[int], Sequence[int]], ...], ...],
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
        contracting_dims, batched_dims = dimension_numbers
        (lhs_inner_dim, ), (rhs_inner_dim, ) = contracting_dims
        lhs_bdims, _ = batched_dims
        if is_fp8:
            assert lhs_inner_dim == lhs_aval.ndim - 1, "FP8 GEMM does not support transposed LHS."
            assert rhs_inner_dim == -1, "FP8 GEMM requires transposed RHS."
            rhs_trans = True
        else:
            rhs_trans = (rhs_inner_dim == -1)

        assert (
            lhs_aval.shape[lhs_inner_dim] == rhs_aval.shape[rhs_inner_dim]
        ), "Incompatible operand sizes!"
        assert rhs_aval.ndim == 2, "TE/JAX GEMM does not support batched RHS operand."

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
        batched_shape = [lhs_aval.shape[bdim] for bdim in lhs_bdims]
        projected_size = rhs_aval.shape[0 if rhs_trans else -1]
        out_shape = (*batched_shape, projected_size)
        out_aval = lhs_aval.update(shape=out_shape, dtype=out_dtype)
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
        dimension_numbers: Tuple[Tuple[Sequence[int], Sequence[int]], ...],
        do_gelu: bool,
        use_bias: bool,
        grad: bool,
        accumulate: bool,
        use_split_accumulator: bool,
    ):
        """
        Fused attention fwd lowering rules
        """
        del do_gelu, use_bias, dimension_numbers

        # Batched always reshapes into LHS:([B], M, K) x RHS^T:(N, K) = OUT:([B], M, N)
        lhs_trans = False
        rhs_trans = True

        # Call underlying custom op with flipped LHS/RHS to account for cuBlasLt column-major format
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

            lhs_aval, _, rhs_aval, _, bias_aval, *_ = ctx.avals_in
            m = lhs_aval.shape[0]
            n = rhs_aval.shape[0]
            k = rhs_aval.shape[-1]
            workspace_size = get_cublas_workspace_size_bytes()
            operand_dtype = jax_dtype_to_te_dtype(lhs_aval.dtype)
            bias_dtype = jax_dtype_to_te_dtype(bias_aval.dtype)
            opaque = tex.pack_gemm_descriptor(m, n, k, workspace_size, operand_dtype,
                                              jax_dtype_to_te_dtype(out_dtype), bias_dtype,
                                              lhs_trans, rhs_trans, grad, accumulate,
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
        dimension_numbers: Tuple[Tuple[Sequence[int], Sequence[int]], ...],
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
                dimension_numbers,
                do_gelu,
                use_bias,
                grad,
                accumulate,
                use_split_accumulator
            )

        return output, out_amax_updated, out_scale_updated, pre_gelu_out

    @staticmethod
    def batcher(batched_args, batch_dims, out_dtype, dimension_numbers, do_gelu, use_bias, grad,
                accumulate, use_split_accumulator):
        assert GemmPrimitive.outer_primitive is not None

        lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, out_amax, out_scale = batched_args
        assert rhs.ndim == 2, "TE/JAX GEMM custom op does not support batching RHS operands."

        # Get contracting and batch dimensions out
        contracting_dims, _ = dimension_numbers
        (lhs_inner_dim, ), (rhs_inner_dim, ) = contracting_dims
        lhs_bdims, _, _, _, _, amax_bdims, scale_bdims = batch_dims

        # Put the contracting dimension to the last dimension (if necessary)
        # This means we always execute `nvte_cublas_gemm` with lhs_trans = False and
        # rhs_trans = True
        if lhs_inner_dim != lhs.ndim - 1:
            lhs = jnp.moveaxis(lhs, lhs_inner_dim, -1)
        if rhs_inner_dim != rhs.ndim - 1:
            rhs = jnp.moveaxis(rhs, rhs_inner_dim, -1)

        # Collapse all non-contracting dimensions
        lhs_batch_shape = lhs.shape[:-1]
        lhs = jnp.reshape(lhs, (math.mul(lhs_batch_shape), lhs.shape[-1]))

        outputs = GemmPrimitive.outer_primitive.bind(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias,
                                                     out_amax, out_scale, out_dtype, do_gelu,
                                                     use_bias, grad, accumulate,
                                                     use_split_accumulator)

        # Reshape output to recover original LHS batch shape
        outputs[0] = jnp.reshape(outputs[0], (*lhs_batch_shape, rhs.shape[0]))
        if outputs[3].size > 0:
            outputs[3] = jnp.reshape(outputs[3], outputs[0].shape)
        return outputs, (lhs_bdims, amax_bdims, scale_bdims, lhs_bdims)

    @staticmethod
    def infer_sharding_from_operands(out_dtype, dimension_numbers, do_gelu, use_bias, grad,
                                     accumulate, use_split_accumulator, mesh, arg_infos,
                                     result_infos):
        del out_dtype, do_gelu, use_bias, grad, accumulate, use_split_accumulator, result_infos
        contracting_dims, batched_dims = dimension_numbers
        (lhs_inner_dim, ), (rhs_inner_dim, ) = contracting_dims
        lhs_bdims, _ = batched_dims

        lhs_spec = get_padded_spec(arg_infos[0])
        batch_specs = [lhs_spec[bdim] for bdim in lhs_bdims]
        rhs_spec = get_padded_spec(arg_infos[2])
        projected_spec = rhs_spec[0 if rhs_inner_dim == 1 else 1]

        if lhs_spec[lhs_inner_dim] != rhs_spec[rhs_inner_dim]:
            warnings.warn("Forcing the inner dimension of A to match the sharding of inner "
                          + "dimension of B. This can trigger additional communication of A is "
                          + "not partitioned correctly.")
        if rhs_spec[rhs_inner_dim] is not None and projected_spec is not None:
            raise RuntimeError("Both inner and outer dimensions of B cannot be sharded!")

        out_spec = [*batch_specs, projected_spec]
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        return (out_sharding, fp8_meta_sharding, fp8_meta_sharding, out_sharding)

    @staticmethod
    def partition(out_dtype, dimension_numbers, do_gelu, use_bias, grad, accumulate,
                  use_split_accumulator, mesh, arg_infos, result_infos):
        del out_dtype, do_gelu, use_bias, grad, accumulate, use_split_accumulator, result_infos
        contracting_dims, batched_dims = dimension_numbers
        (lhs_inner_dim, ), (rhs_inner_dim, ) = contracting_dims
        lhs_bdims, _ = batched_dims

        lhs_spec = get_padded_spec(arg_infos[0])
        batch_specs = [lhs_spec[bdim] for bdim in lhs_bdims]
        rhs_spec = get_padded_spec(arg_infos[2])
        projected_spec = rhs_spec[0 if rhs_inner_dim == 1 else 1]

        lhs_spec_new = [None, lhs_spec[lhs_inner_dim]]
        if len(batch_specs) > 1:
            lhs_spec_new = batch_specs[:-1] + lhs_spec_new
        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_spec_new))
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_spec))
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        out_spec = [*batch_specs, projected_spec]
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))
        bias_sharding = NamedSharding(mesh, PartitionSpec(out_spec[-1]))

        arg_shardings = (lhs_sharding, fp8_meta_sharding, rhs_sharding, fp8_meta_sharding,
                         bias_sharding, fp8_meta_sharding, fp8_meta_sharding)
        out_shardings = (out_sharding, fp8_meta_sharding, fp8_meta_sharding, out_sharding)

        def impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, out_amax, out_scale, out_dtype,
                 dimension_numbers, do_gelu, use_bias, grad, accumulate, use_split_accumulator):

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
                dimension_numbers,
                do_gelu,
                use_bias,
                grad,
                accumulate,
                use_split_accumulator
            )

            if rhs_spec[rhs_inner_dim] is not None:
                # If the inner dimensions of LHS and RHS are sharded, we all-reduce the GEMM output.
                # If the outer dimension of LHS is also sharded, we reduce-scatter the GEMM output.
                par_op = (
                    jax.lax.psum_scatter
                    if batch_specs[-1] is not None
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
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = None,
    do_gelu: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> Tuple[ArrayLike, ...]:
    """GEMM with FP8 inputs"""
    if contracting_dims is None:
        contracting_dims = ((1,), (0,))
    lhs_batch_dims = tuple([dim for dim in range(lhs.ndim) if dim not in contracting_dims[0]])
    rhs_batch_dims = tuple([dim for dim in range(rhs.ndim) if dim not in contracting_dims[1]])
    dimension_numbers = (contracting_dims, (lhs_batch_dims, rhs_batch_dims))

    if out_dtype is not None and is_fp8_dtype(out_dtype):
        assert out_amax is not None and out_scale is not None, "Missing output amax and scale!"
    else:
        out_amax = jnp.empty((0, ), dtype=jnp.float32)
        out_scale = jnp.empty((0, ), dtype=jnp.float32)

    use_bias = bias is not None
    if not use_bias:
        bias = jnp.empty((0, ), dtype=jnp.bfloat16)

    out, out_amax, out_scale, pre_gelu_out = GemmPrimitive.outer_primitive.bind(
        lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, out_amax, out_scale, out_dtype,
        dimension_numbers, do_gelu, use_bias, False, accumulate, use_split_accumulator
    )

    outputs = (out, )
    if out_dtype is not None and is_fp8_dtype(out_dtype):
        outputs += (out_amax, out_scale, )
    if do_gelu:
        outputs += (pre_gelu_out, )

    return outputs


def gemm(
    lhs: ArrayLike,
    rhs: ArrayLike,
    bias: Optional[ArrayLike] = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = None,
    do_gelu: bool = False,
    grad: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> Tuple[ArrayLike, ...]:
    """Non-FP8 GEMM"""
    if contracting_dims is None:
        contracting_dims = ((1,), (0,))
    lhs_batch_dims = tuple([dim for dim in range(lhs.ndim) if dim not in contracting_dims[0]])
    rhs_batch_dims = tuple([dim for dim in range(rhs.ndim) if dim not in contracting_dims[1]])
    dimension_numbers = (contracting_dims, (lhs_batch_dims, rhs_batch_dims))

    use_bias = bias is not None
    if not use_bias:
        bias = jnp.empty((0, ), dtype=jnp.bfloat16)

    dummy_fp8_meta = jnp.empty((0, ), dtype=jnp.float32)

    out, _, _, pre_gelu_out = GemmPrimitive.outer_primitive.bind(
        lhs, dummy_fp8_meta, rhs, dummy_fp8_meta, bias, dummy_fp8_meta, dummy_fp8_meta, lhs.dtype,
        dimension_numbers, do_gelu, use_bias, grad, accumulate, use_split_accumulator
    )

    outputs = (out, )
    if do_gelu:
        outputs += (pre_gelu_out, )

    return outputs
