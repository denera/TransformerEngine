# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

from typing import Tuple, Sequence, Union, Dict
from functools import partial, reduce
import operator
import warnings

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding
from jax.ad_checkpoint import checkpoint_name

from transformer_engine_jax import get_device_compute_capability

from .misc import get_padded_spec

from .base import BasePrimitive, register_primitive

from .quantization import quantize_dbias

from ..quantize import (
    ScaledTensor,
    ScalingMode,
    Quantizer,
    QuantizeConfig,
    noop_quantizer_set,
)

from ..sharding import global_mesh_resource


__all__ = ["gemm"]


num_cublas_streams = 4


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if get_device_compute_capability(0) >= 90:
        return 33_554_432
    return 4_194_304


def sanitize_dim(dim, ndim):
    """Convert relative dimension indexing to absolute."""
    assert abs(dim) < ndim, f"Dimension index {dim} is out of range {ndim}."
    if dim < 0:
        return ndim + dim
    return dim


class CollectiveGemm(BasePrimitive):
    """
    Primitive for collective GEMM
    """

    name = "te_collective_gemm_ffi"
    multiple_results = True
    impl_static_args = ()
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_in, *, lhs_trans, rhs_trans,
                 scaling_mode, fuse_bias, fuse_gelu, grad, accumulate, use_split_accumulator):
        del scaling_mode, accumulate, use_split_accumulator

        # Sanity-check contracting dims
        lhs_inner_dim = 0 if lhs_trans else lhs.ndim - 1
        rhs_inner_dim = rhs.ndim - 1 if rhs_trans else 0
        assert lhs.shape[lhs_inner_dim] == rhs.shape[rhs_inner_dim], (
            "Incompatible operands, contracting dimension sizes do not match."
        )
        lhs_trans = lhs_inner_dim != 1
        lhs_outer_dim = 1 if lhs_trans else 0
        rhs_trans = rhs_inner_dim != 0
        rhs_outer_dim = 0 if rhs_trans else 1

        # Sanity check scaling factors
        if scaling_mode != ScalingMode.NO_SCALING:
            assert lhs_scale_inv.size > 0 and rhs_scale_inv.size > 0, (
                "LHS and RHS operand inverse-scaling factors cannot be empty tensors for FP8 GEMM."
            )

        # Determine pure-GEMM output shape (without factoring in any TP scatter)
        out_shape = [lhs.shape[lhs_outer_dim], rhs.shape[rhs_outer_dim]]
        out_dtype = jnp.bfloat16 if scaling_mode != ScalingMode.NO_SCALING else lhs.dtype
        out_aval = jax.core.ShapedArray(out_shape, dtype=out_dtype)

        # Sanity-check bias shape (forward pass) and declare dbias output (backward pass)
        bias_shape = (0, )
        bias_dtype = jnp.bfloat16
        if fuse_bias:
            if not grad:
                assert bias is not None and bias.shape[0] == out_shape[1], (
                    f"Incorrect bias shape, expected {out_shape[1]} but found {bias.shape[0]}."
                )
                bias_shape = bias.shape
                bias_dtype = bias.dtype
            else:
                bias_shape = (out_shape[1], )
        dbias_aval = jax.core.ShapedArray(bias_shape, dtype=bias_dtype)

        # Declare pre-GeLU output (forward pass) and sanity-check pre-GeLU input (backward pass)
        pre_gelu_out_shape = (0, )
        if fuse_gelu:
            if grad:
                assert gelu_in is not None, "Missing pre-GeLU input."
                assert (
                    gelu_in.ndim == 2
                    and all([gelu_in.shape[i] == out_shape[i] for i in range(gelu_in.ndim)])
                ), (
                    "Pre-GELU input has incorrect shape, "
                    f"expected {out_shape} but got {gelu_in.shape}."
                )
            else:
                pre_gelu_out_shape = out_shape.copy()
        pre_gelu_out_aval = jax.core.ShapedArray(pre_gelu_out_shape, dtype=out_dtype)

        # Declare cuBLAS workspace, expanded to the number of cuBLAS compute streams for TP overlap
        workspace_size = get_cublas_workspace_size_bytes()
        workspace_aval = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)

        return out_aval, dbias_aval, pre_gelu_out_aval, workspace_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        (
            out_aval,
            dbias_aval,
            pre_gelu_out_aval,
            _  # discarding cuBLAS workspace
        ) = CollectiveGemm.abstract(*args, **kwargs)
        return out_aval, dbias_aval, pre_gelu_out_aval

    @staticmethod
    def lowering(ctx, lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_in, lhs_trans, rhs_trans,
                 scaling_mode, fuse_bias, fuse_gelu, grad, accumulate, use_split_accumulator):
        return jax.ffi.ffi_lowering(
            CollectiveGemm.name,
            operand_output_aliases={ 4:1, 5:2 },
        )(
            ctx,
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_in,
            lhs_trans=lhs_trans,
            rhs_trans=rhs_trans,
            scaling_mode=int(scaling_mode),
            fuse_bias=fuse_bias,
            fuse_gelu=fuse_gelu,
            grad=grad,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
        )

    @staticmethod
    def impl(*args, lhs_trans, rhs_trans, scaling_mode, fuse_bias, fuse_gelu, grad, accumulate,
             use_split_accumulator):
        assert CollectiveGemm.inner_primitive is not None
        out = CollectiveGemm.inner_primitive.bind(
            *args,
            lhs_trans=lhs_trans,
            rhs_trans=rhs_trans,
            scaling_mode=scaling_mode,
            fuse_bias=fuse_bias,
            fuse_gelu=fuse_gelu,
            grad=grad,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
        )
        return out[:-1]  # out is [out_list, wkspace], only return out_list


    @staticmethod
    def infer_sharding_from_operands(lhs_trans, rhs_trans, scaling_mode, fuse_bias,
                                     fuse_gelu, grad, accumulate, use_split_accumulator, mesh,
                                     arg_infos, result_infos):
        del scaling_mode, accumulate, use_split_accumulator
        del result_infos
        lhs_info, _, rhs_info, *_ = arg_infos
        lhs_spec = get_padded_spec(lhs_info)
        rhs_spec = get_padded_spec(rhs_info)

        lhs_inner_dim = int(not lhs_trans)
        rhs_inner_dim = int(rhs_trans)
        rhs_outer_dim = int(not rhs_trans)

        # Get corrected RHS outer dim sharding (inner dim will be corrected in partitioning later)
        rhs_outer_spec = rhs_spec[rhs_outer_dim]
        if rhs_spec[rhs_inner_dim] is not None and rhs_spec[rhs_outer_dim] is not None:
            if lhs_spec[lhs_inner_dim] == rhs_spec[rhs_inner_dim]:
                rhs_outer_spec = None

        out_shardings = []

        # Final output sharding
        out_spec = (None, rhs_outer_spec)
        out_shardings.append(NamedSharding(mesh, PartitionSpec(*out_spec)))

        # dBias sharding
        dbias_spec = (None, )
        if fuse_bias and grad:
            dbias_spec = (out_spec[1], )
        out_shardings.append(NamedSharding(mesh, PartitionSpec(*dbias_spec)))

        # Pre-GeLU output sharding
        pre_gelu_out_spec = (None, )
        if fuse_gelu and not grad:
            pre_gelu_out_spec = out_spec
        out_shardings.append(NamedSharding(mesh, PartitionSpec(*pre_gelu_out_spec)))

        # cuBLAS workspace sharding
        out_shardings.append(NamedSharding(mesh, PartitionSpec(None)))

        return out_shardings

    @staticmethod
    def partition(lhs_trans, rhs_trans, scaling_mode, fuse_bias, fuse_gelu, grad, accumulate,
                  use_split_accumulator, mesh, arg_infos, result_infos):
        lhs_info, _, rhs_info, _, bias_info, gelu_in_info = arg_infos
        lhs_spec = get_padded_spec(lhs_info)
        rhs_spec = get_padded_spec(rhs_info)
        bias_spec = get_padded_spec(bias_info)
        gelu_in_spec = get_padded_spec(gelu_in_info)

        lhs_inner_dim = int(not lhs_trans)
        lhs_outer_dim = int(lhs_trans)
        rhs_inner_dim = int(rhs_trans)
        rhs_outer_dim = int(not rhs_trans)

        # RHS operand sharding: Do not allow both dimensions to be sharded at the same time.
        # 1. If LHS inner dimension is sharded *and* matches RHS inner dimension, then force
        #    the RHS outer dimension to be un-sharded. This should never trigger for correct inputs.
        # 2. Otherwise, force the RHS inner dimension to be unsharded. This should never trigger
        #    for correct inputs.
        rhs_spec_new = list(rhs_spec).copy()
        if rhs_spec[rhs_inner_dim] is not None and rhs_spec[rhs_outer_dim] is not None:
            if lhs_spec[lhs_inner_dim] == rhs_spec[rhs_inner_dim]:
                warnings.warn(
                    (
                        "Forcing RHS outer dimension to be unsharded. "
                        "This will trigger an extra collective and impact performance."
                    ),
                    RuntimeWarning
                )
                rhs_spec_new[rhs_outer_dim] = None
            else:
                warnings.warn(
                    (
                        "Forcing RHS inner dimension to be unsharded. "
                        "This will trigger an extra collective and impact performance."
                    ),
                    RuntimeWarning
                )
                rhs_spec_new[rhs_inner_dim] = None
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_spec_new))
        rhs_scale_inv_sharding = NamedSharding(mesh, PartitionSpec(None))

        # LHS operand sharding: Do not allow both dimensions to be sharded at the same time.
        # 1. Require outer dimension to be unsharded. This will trigger all-gather
        #    for sequence parallel inputs.
        # 2. Require the inner dimension to match RHS inner dimension sharding.
        #    This should never trigger for correct inputs.
        if lhs_spec[lhs_inner_dim] != rhs_spec_new[rhs_inner_dim]:
            warnings.warn(
                (
                    "Forcing LHS inner dimension sharding to match RHS inner dimension. "
                    "This may trigger an extra collective and impact performance."
                ),
                RuntimeWarning,
            )
        lhs_spec_new = [None, rhs_spec_new[rhs_outer_dim]]
        lhs_sharding = NamedSharding(mesh, PartitionSpec(None, rhs_spec[rhs_inner_dim]))
        lhs_scale_inv_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Set the output spec
        out_spec = (lhs_spec_new[lhs_outer_dim], rhs_spec_new[rhs_outer_dim])

        # Require bias to be sharded to fit the GEMM output.
        # This should never trigger for correct inputs.
        arg_shardings = [lhs_sharding, lhs_scale_inv_sharding, rhs_sharding, rhs_scale_inv_sharding]
        bias_spec_new = (None, )
        if fuse_bias and not grad:
            if bias_spec[0] != out_spec[1]:
                warnings.warn(
                    (
                        "Forcing bias sharding to match GEMM output. "
                        "This may trigger an extra collective and impact performance."
                    ),
                    RuntimeWarning,
                )
            bias_spec_new = (out_spec[1], )
        arg_shardings.append(NamedSharding(mesh, PartitionSpec(*bias_spec_new)))

        # Require gelu_in to match GEMM output.
        # This should never trigger for correct inputs.
        gelu_in_spec_new = (None, )
        if fuse_gelu and grad:
            if any([first != second for first, second in zip(gelu_in_spec, out_spec)]):
                warnings.warn(
                    (
                        "Forcing gelu_in sharding to match GEMM output. "
                        "This may trigger an extra collective and impact performance."
                    ),
                    RuntimeWarning,
                )
            gelu_in_spec_new = out_spec
        arg_shardings.append(NamedSharding(mesh, PartitionSpec(*gelu_in_spec_new)))

        out_shardings = CollectiveGemm.infer_sharding_from_operands(
            lhs_trans, rhs_trans, scaling_mode, fuse_bias, fuse_gelu, grad, accumulate,
            use_split_accumulator, mesh, arg_infos, result_infos
        )

        def sharded_impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_in):
            out, dbias, pre_gelu_out, _ = CollectiveGemm.impl(
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                gelu_in,
                lhs_trans=lhs_trans,
                rhs_trans=rhs_trans,
                scaling_mode=scaling_mode,
                fuse_bias=fuse_bias,
                fuse_gelu=fuse_gelu,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
            )

            if rhs_spec_new[rhs_inner_dim] is not None:
                out = jax.lax.psum(out, global_mesh_resource().tp_resource)
                if fuse_gelu and not grad:
                    pre_gelu_out = jax.lax.psum(pre_gelu_out, global_mesh_resource().tp_resource)

            return out, dbias, pre_gelu_out

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(CollectiveGemm)


class GroupedGemmPrimitive(BasePrimitive):
    """
    Primitive for grouped GEMM
    """

    name = "te_grouped_gemm_ffi"
    multiple_results = True
    impl_static_args = ()
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(*args, num_gemms, scaling_mode, out_dtype, has_bias):
        """
        Args:
            *args: Size num_gemms * 4 or num_gemms * 5 depending on has_bias:
                args[  0         :   num_gemms] are the lhs tensors,
                args[  num_gemms : 2*num_gemms] are the rhs tensors,
                args[2*num_gemms : 3*num_gemms] are the lhs scale_inv tensors,
                args[3*num_gemms : 4*num_gemms] are the rhs scale_inv tensors,
                args[4*num_gemms : 5*num_gemms] are the bias tensors if has_bias is True.
            num_gemms: Number of GEMM operations to perform.
            scaling_mode: Scaling mode for the GEMM operations.
            out_dtype: Data type of the output tensors.
            has_bias: Boolean indicating if bias tensors are provided.

        Returns:
           A tuple of ShapedArray objects of size num_gemms+1:
               ret[0 : num_gemms]: GEMM output tensors,
               ret[num_gemms]:workspace tensor.
        """
        del scaling_mode
        expected_num_args = 5 * num_gemms if has_bias else 4 * num_gemms
        assert (
            len(args) == expected_num_args
        ), f"Expected {expected_num_args} input arguments, but got {len(args)}"
        A_list = args[0:num_gemms]
        B_list = args[num_gemms : 2 * num_gemms]
        # A and B have shapes [1, m, k] and [1, n, k]
        out_list_aval = tuple(
            jax.core.ShapedArray((A.shape[1], B.shape[1]), dtype=out_dtype)
            for A, B in zip(A_list, B_list)
        )
        workspace_size = get_cublas_workspace_size_bytes() * num_cublas_streams
        workspace_aval = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)
        return (*out_list_aval, workspace_aval)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        (out_aval, _) = GroupedGemmPrimitive.abstract(*args, **kwargs)
        return out_aval

    @staticmethod
    def lowering(ctx, *args, num_gemms, scaling_mode, out_dtype, has_bias):
        del out_dtype
        return jax.ffi.ffi_lowering(GroupedGemmPrimitive.name)(
            ctx,
            *args,
            num_gemms=num_gemms,
            scaling_mode=int(scaling_mode),
            has_bias=has_bias,
        )

    @staticmethod
    def impl(*args, num_gemms, scaling_mode, out_dtype, has_bias):
        assert GroupedGemmPrimitive.inner_primitive is not None
        out = GroupedGemmPrimitive.inner_primitive.bind(
            *args,
            num_gemms=num_gemms,
            scaling_mode=scaling_mode.value,
            out_dtype=out_dtype,
            has_bias=has_bias,
        )
        return out[:-1]  # out is [out_list, wkspace], only return out_list


register_primitive(GroupedGemmPrimitive)


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


def _calculate_remaining_shape(shape, contracting_dims):
    return tuple(shape[dim] for dim in range(len(shape)) if dim not in contracting_dims)


def _transpose_contract_dims(ndim, contracting_dims):
    return tuple(ndim - i - 1 for i in contracting_dims)


def _dequantize(x, scale_inv, dq_dtype):
    return x.astype(dq_dtype) * scale_inv.astype(dq_dtype)


# Apply jit to guarantee correctness of FP8 GEMM.
@partial(jax.jit, static_argnums=(2, 3))
def _jax_gemm_tensor_scaling_fp8(lhs, rhs, dim_nums, precision):
    # Need to hard-code the dequantize here instead of calling lhs.dequantize() for pattern matching
    """FP8 GEMM for XLA pattern match"""
    lhs_dq = _dequantize(lhs.data, lhs.scale_inv, lhs.dq_dtype)
    rhs_dq = _dequantize(rhs.data, rhs.scale_inv, rhs.dq_dtype)

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
    if lhs.data_layout == "T":
        lhs_contract = _transpose_contract_dims(lhs_dq.ndim, lhs_contract)
    if rhs.data_layout == "T":
        rhs_contract = _transpose_contract_dims(rhs_dq.ndim, rhs_contract)

    dim_nums = (lhs_contract, rhs_contract), (lhs_batch, rhs_batch)

    return jax.lax.dot_general(
        lhs_dq, rhs_dq, dim_nums, precision=precision, preferred_element_type=lhs.dq_dtype
    )


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

    # Slice out the padding as scaled_matmul does not support padded scales yet
    lhs_scale_3d = jnp.asarray(lhs_scale_3d[:, : lhs_3d.shape[1], : int(lhs_3d.shape[2] / 32)])
    rhs_scale_3d = jnp.asarray(rhs_scale_3d[:, : rhs_3d.shape[1], : int(rhs_3d.shape[2] / 32)])

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
    quantizer_set: Dict["str", Quantizer] = noop_quantizer_set,
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

    if isinstance(lhs, ScaledTensor) and isinstance(rhs, ScaledTensor):
        return _jax_gemm_fp8_impl(lhs, rhs)

    if not isinstance(lhs, ScaledTensor) and not isinstance(rhs, ScaledTensor):
        if quantizer_set != noop_quantizer_set:
            assert type(quantizer_set.x) is type(quantizer_set.kernel)
            (((lhs_contract_dim,), (rhs_contract_dim,)), _) = dim_nums
            lhs_is_rowwise = lhs_contract_dim == lhs.ndim - 1
            rhs_is_rowwise = rhs_contract_dim == rhs.ndim - 1
            # Call JAX quantization so that XLA can do pattern matching (QDQ --> FP8 gemm)
            lhs_q = quantizer_set.x.quantize(
                lhs,
                is_rowwise=lhs_is_rowwise,
                is_colwise=not lhs_is_rowwise,
            )
            rhs_q = quantizer_set.kernel.quantize(
                rhs,
                is_rowwise=rhs_is_rowwise,
                is_colwise=not rhs_is_rowwise,
            )
            return _jax_gemm_fp8_impl(lhs_q, rhs_q)

    if (
        isinstance(lhs, jnp.ndarray)
        and isinstance(rhs, jnp.ndarray)
        and quantizer_set == noop_quantizer_set
    ):
        return jax.lax.dot_general(lhs, rhs, dim_nums, preferred_element_type=lhs.dtype)

    raise NotImplementedError("Not supporting multiplication of ScaledTensor and jnp.array")


def gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    quantizer_set: Dict["str", Quantizer] = noop_quantizer_set,
) -> jnp.ndarray:
    """General matrix multiplication with optional quantization.

    Args:
        lhs: First input matrix.
        rhs: Second input matrix.
        contracting_dims: Tuple of two sequences representing the contracting dimensions.
            The first sequence represents the contracting dimensions of the first matrix,
            and the second sequence represents the contracting dimensions of the second matrix.
        quantizer_set: Set of quantizers for FP8 quantization of the output.
            If None, no quantization is applied and the output has the same dtype as the inputs.

    Returns:
        If quantizer_set is None:
            The matrix multiplication result.
            Shape: (M, N)
            Dtype: Same as input dtype
          If quantizer_set is provided:
            A ScaledTensor containing the quantized matrix multiplication result.
    """

    return _jax_gemm(lhs, rhs, contracting_dims, quantizer_set)


def te_gemm_impl(
    lhs: jnp.ndarray,
    lhs_scale_inv: Union[jnp.ndarray, None],
    rhs: jnp.ndarray,
    rhs_scale_inv: Union[jnp.ndarray, None],
    bias: Union[jnp.ndarray, None],
    pre_gelu_out: Union[jnp.ndarray, None],
    lhs_trans: bool = False,
    rhs_trans: bool = False,
    scaling_mode: ScalingMode = ScalingMode.NO_SCALING,
    fuse_bias: bool = False,
    fuse_gelu: bool = False,
    grad: bool = False,
    accumulate: bool = False,
    use_split_accumulator = False,
) -> Tuple[jnp.ndarray, Union[jnp.ndarray, None], Union[jnp.ndarray[None]]]:
    """TE cuBLAS GEMM"""
    if lhs_scale_inv is None or scaling_mode == ScalingMode.NO_SCALING:
        lhs_scale_inv = jnp.empty(0, dtype=jnp.float32)
    if rhs_scale_inv is None or scaling_mode == ScalingMode.NO_SCALING:
        rhs_scale_inv = jnp.empty(0, dtype=jnp.float32)

    if grad or not fuse_bias:
        bias = jnp.empty(0, dtype=jnp.bfloat16)

    if (grad and not fuse_gelu) or not grad:
        pre_gelu_out = jnp.empty(0, dtype=jnp.bfloat16)

    output, dbias, pre_gelu_out = CollectiveGemm.outer_primitive.bind(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        pre_gelu_out,
        lhs_trans=lhs_trans,
        rhs_trans=rhs_trans,
        scaling_mode=scaling_mode,
        fuse_bias=fuse_bias,
        fuse_gelu=fuse_gelu,
        grad=grad,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    if grad and not fuse_bias:
        dbias = None

    if not grad and not fuse_gelu:
        pre_gelu_out = None

    return output, dbias, pre_gelu_out


def _te_gemm_fwd_rule(lhs, rhs, bias, contracting_dims, quantizer_set, fuse_gelu,
                      accumulate, use_split_accumulator):
    # Detect operand transposes, calculate collapsed 2-dimensional shapes
    lhs_data = lhs.data if isinstance(lhs, ScaledTensor) else lhs
    lhs_inner_dim = sanitize_dim(contracting_dims[0], lhs_data.ndim)
    lhs_trans = lhs_inner_dim != lhs_data.ndim - 1
    lhs_outer_shape = [lhs_data.shape[i] for i in range(lhs_data.ndim) if i != lhs_inner_dim]
    lhs_2d_shape = (
        (lhs_data.shape[lhs_inner_dim], -1)
        if lhs_trans
        else (-1, lhs_data.shape[lhs_inner_dim])
    )

    rhs_data = rhs.data if isinstance(rhs, ScaledTensor) else rhs
    assert rhs_data.ndim == 2, "RHS operand (weight/kernel) must be 2-dimensional."
    rhs_inner_dim = sanitize_dim(contracting_dims[1], rhs_data.ndim)
    rhs_trans = rhs_inner_dim == rhs_data.ndim - 1

    lhs_scale_inv = None
    rhs_scale_inv = None
    scaling_mode = ScalingMode.NO_SCALING
    if quantizer_set != noop_quantizer_set:
        assert type(quantizer_set.x) == type(quantizer_set.kernel)
        if not isinstance(lhs, ScaledTensor):
            lhs_q = quantizer_set.x.quantize(
                lhs, is_rowwise=not lhs_trans, is_colwise=lhs_trans
            )
        else:
            lhs_q = lhs
        lhs_data = lhs_q.data
        lhs_scale_inv = lhs_q.scale_inv
        if not isinstance(rhs, ScaledTensor):
            rhs_q = quantizer_set.kernel.quantize(
                rhs, is_rowwise=not rhs_trans, is_colwise=rhs_trans
            )
        else:
            rhs_q = rhs
        rhs_data = rhs_q.data
        rhs_scale_inv = rhs_q.scale_inv
        scaling_mode = lhs_q.scaling_mode

    # ([B], M, K) x (K, N) = ([B], M, N)
    fuse_bias = bias is not None
    output, _, pre_gelu_out = te_gemm_impl(
        lhs_data.reshape(lhs_2d_shape),
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        None,
        lhs_trans=lhs_trans,
        rhs_trans=rhs_trans,
        scaling_mode=scaling_mode,
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias,
        grad=False,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )
    output = output.reshape((*lhs_outer_shape, -1))

    output = checkpoint_name(output, "output")
    if fuse_gelu:
        pre_gelu_out = checkpoint_name(pre_gelu_out, "pre_gelu_out")

    return output, (
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        pre_gelu_out if fuse_gelu else None,
        lhs_trans,
        rhs_trans,
        fuse_bias,
    )


def _te_gemm_bwd_rule(contracting_dims, quantizer_set, fuse_gelu, accumulate, use_split_accumulator,
                      ctx, dz):
    (
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        pre_gelu_out,
        lhs_trans,
        rhs_trans,
        fuse_bias,
    )= ctx

    # Quantize gradient and compute dbias
    dz_data = dz
    dz_scale_inv = None
    scaling_mode = ScalingMode.NO_SCALING
    bias_grad_q = None
    fuse_bias_dgrad = fuse_bias
    if quantizer_set != noop_quantizer_set:
        if fuse_bias:
            dz_q, bias_grad_q = quantize_dbias(dz, quantizer_set.dgrad)
            fuse_bias_dgrad = False
        else:
            dz_q = quantizer_set.dgrad.quantize(
                dz,
                is_rowwise=lhs_trans,
                is_colwise=not lhs_trans,
            )
        dz_data = dz_q.data
        dz_scale_inv = dz_q.scale_inv
        scaling_mode = dz_q.scaling_mode

    # dLHS: ([B]*M, N) x (K, N)^T = ([B]*M, K)
    lhs_grad, bias_grad_bf16, _ = te_gemm_impl(
        dz_data.reshape((-1, dz_data.shape[-1])),
        dz_scale_inv,
        rhs_data,
        rhs_scale_inv,
        None,
        pre_gelu_out,
        lhs_trans=False,
        rhs_trans=not rhs_trans,
        scaling_mode=scaling_mode,
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias_dgrad,
        grad=True,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator
    )
    lhs_grad = lhs_grad.reshape(lhs_data.shape)

    bias_grad = None
    if fuse_bias:
        if fuse_bias_dgrad:
            bias_grad = bias_grad_q
        else:
            bias_grad = bias_grad_bf16

    # dRHS: ([B]*M, K)^T x ([B]*M, N) = (K, N)
    lhs_inner_dim = sanitize_dim(contracting_dims[0], lhs_data.ndim)
    lhs_2d_shape = (
        (lhs_data.shape[lhs_inner_dim], -1)
        if lhs_trans
        else (-1, lhs_data.shape[lhs_inner_dim])
    )
    rhs_grad, *_ = te_gemm_impl(
        lhs_data.reshape(lhs_2d_shape),
        lhs_scale_inv,
        dz_data.reshape((-1, dz_data.shape[-1])),
        dz_scale_inv,
        None,
        pre_gelu_out,
        lhs_trans=not lhs_trans,
        rhs_trans=False,
        scaling_mode=scaling_mode,
        fuse_gelu=fuse_gelu,
        fuse_bias=False,
        grad=True,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    return lhs_grad, rhs_grad, bias_grad


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def te_gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    bias: Union[jnp.ndarray, None],
    contracting_dims: Tuple[int, int] = (-1, 0),
    quantizer_set: Dict["str", Quantizer] = noop_quantizer_set,
    fuse_gelu: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
):
    """TE cuBLAS GEMM w/ FWD+BWD rules"""
    output, _ = _te_gemm_fwd_rule(lhs, rhs, bias, contracting_dims, quantizer_set, fuse_gelu,
                                  accumulate, use_split_accumulator)
    return output


"""
def swizzled_scale(scales):
    # Swizzle the scale tensor for FP8 GEMM
    assert scales.ndim == 2
    rows, cols = scales.shape
    scales = scales.reshape(rows // 128, 4, 32, cols // 4, 4)
    scales = jnp.transpose(scales, (0, 3, 2, 1, 4))
    scales = scales.reshape(rows, cols)
    return scales


def grouped_gemm(
    lhs_list: List[Union[jnp.ndarray, ScaledTensor]],
    rhs_list: List[Union[jnp.ndarray, ScaledTensor]],
    contracting_dims_list: List[Tuple[Sequence[int], Sequence[int]]],
    bias_list: List[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    # Grouped GEMM for multiple pairs of tensors.
    assert (
        len(lhs_list) == len(rhs_list) == len(contracting_dims_list)
    ), "lhs_list, rhs_list, contracting_dims_list must have the same length"

    num_gemms = len(lhs_list)
    lhs_list_ = []
    rhs_list_ = []
    lhs_sinv_list_ = []
    rhs_sinv_list_ = []
    bias_list_ = []
    for i in range(num_gemms):
        lhs = lhs_list[i]
        rhs = rhs_list[i]
        contracting_dims = contracting_dims_list[i]
        dim_nums = (contracting_dims, ((), ()))
        if isinstance(lhs, ScaledTensor) and isinstance(rhs, ScaledTensor):
            scaling_mode = lhs.scaling_mode
            lhs_shape = lhs.data.shape
            rhs_shape = rhs.data.shape
            out_dtype = lhs.dq_dtype
            # For ScaledTensors and DELAYED_TENSOR_SCALING, need to handle internal data_layout
            if lhs.scaling_mode.is_tensor_scaling():
                assert not (
                    lhs.data.dtype == jnp.float8_e5m2 and rhs.data.dtype == jnp.float8_e5m2
                ), "FP8 GEMM does not support E5M2 * E5M2"
                ((lhs_contract_dim,), (rhs_contract_dim,)) = contracting_dims
                if lhs.data_layout == "T":
                    lhs_contract_dim = (lhs_contract_dim - 1) % lhs.data.ndim
                if rhs.data_layout == "T":
                    rhs_contract_dim = (rhs_contract_dim - 1) % rhs.data.ndim
                dim_nums = ((lhs_contract_dim,), (rhs_contract_dim,)), ((), ())
        else:
            # For jnp.ndarray, only consider contracting_dims, data_layout is always NN
            scaling_mode = ScalingMode.NO_SCALING
            lhs_shape = lhs.shape
            rhs_shape = rhs.shape
            out_dtype = lhs.dtype

        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
        lhs_dn = (lhs_contract, lhs_batch)
        rhs_dn = (rhs_contract, rhs_batch)

        lhs_remain_shape = _calculate_remaining_shape(lhs_shape, lhs_contract)
        rhs_remain_shape = _calculate_remaining_shape(rhs_shape, rhs_contract)

        # Note: do not squeeze() for {lhs, rhs}_3d, it will trigger a D2D memcpy
        if scaling_mode == ScalingMode.NO_SCALING:
            lhs_3d = _shape_normalization(lhs, lhs_dn)
            rhs_3d = _shape_normalization(rhs, rhs_dn)
        elif scaling_mode.is_tensor_scaling():
            lhs_3d = _shape_normalization(lhs.data, lhs_dn, lhs.data_layout == "N")
            rhs_3d = _shape_normalization(rhs.data, rhs_dn, rhs.data_layout == "T")
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            lhs_3d = _shape_normalization(lhs.data, lhs_dn)
            rhs_3d = _shape_normalization(rhs.data, rhs_dn)
            lhs_scale_inv = _shape_normalization(lhs.scale_inv, lhs_dn)
            rhs_scale_inv = _shape_normalization(rhs.scale_inv, rhs_dn)
            # swizzled_scale requires a matrix
            lhs_scale_inv = swizzled_scale(lhs_scale_inv.squeeze())
            rhs_scale_inv = swizzled_scale(rhs_scale_inv.squeeze())
        else:
            raise NotImplementedError("Unsupported ScalingMode: {scaling_mode}")

        # Note: already_transposed doesn't matter for the output shape
        # x.shape = [B, D1, D2]
        # contracting_dims = (2, )    --> output.shape = [1, B * D1, D2]
        # contracting_dims = (0, 1, ) --> output.shape = [1, D2, B * D1]
        # x.shape = [D1, D2]
        # contracting_dims = (1, )    --> output.shape = [1, D1, D2]
        # contracting_dims = (0, )    --> output.shape = [1, D2, D1]
        bm = lhs_remain_shape[0]
        bn = rhs_remain_shape[0]
        kl = lhs_3d.shape[-1]
        kr = rhs_3d.shape[-1]
        assert kl == kr, f"After shape normalization, contracting dim size mismatch: {kl} != {kr}"
        if (bm % 16 != 0) or (bn % 16 != 0) or (kl % 16 != 0):
            print("grouped_gemm input pair {i} has invalid problem shape for lowering: ")
            print(f"m = {bm}, n = {bn}, k = {kl}; ")
            print("cuBLAS requires the problem shapes being multiples of 16")
            assert (bm % 16 == 0) and (bn % 16 == 0) and (kl % 16 == 0)

        lhs_list_.append(lhs_3d)
        rhs_list_.append(rhs_3d)
        if scaling_mode == ScalingMode.NO_SCALING:
            lhs_sinv_list_.append(jnp.ones(1, dtype=jnp.float32))
            rhs_sinv_list_.append(jnp.ones(1, dtype=jnp.float32))
        if scaling_mode.is_tensor_scaling():
            lhs_sinv_list_.append(lhs.scale_inv)
            rhs_sinv_list_.append(rhs.scale_inv)
        if scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            lhs_sinv_list_.append(lhs_scale_inv)
            rhs_sinv_list_.append(rhs_scale_inv)
        if bias_list is not None:
            bias_list_.append(bias_list[i])

    out_list = GroupedGemmPrimitive.outer_primitive.bind(
        *lhs_list_,
        *rhs_list_,
        *lhs_sinv_list_,
        *rhs_sinv_list_,
        *bias_list_,
        num_gemms=num_gemms,
        scaling_mode=scaling_mode,
        out_dtype=out_dtype,
        has_bias=1 if bias_list is not None else 0,
    )

    return out_list
"""
