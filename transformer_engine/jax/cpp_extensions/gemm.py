# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

from typing import Tuple, Sequence, Union, Dict
from functools import partial, reduce
import operator

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding
from jax.ad_checkpoint import checkpoint_name

import transformer_engine_jax as tex

from .misc import get_padded_spec

from .base import BasePrimitive, register_primitive

from .quantization import quantize_dbias

from ..quantize import (
    ScaledTensor,
    ScalingMode,
    Quantizer,
    QuantizerSet,
    QuantizeConfig,
    noop_quantizer_set,
)

from ..sharding import (
    global_mesh_resource,
    get_mesh_axis_size,
)


__all__ = [
    "gemm",
    "te_gemm",
    "get_default_comm_overlap_config",
    "initialize_comm_overlap",
]

min_stream_priority = None
max_stream_priority = None
num_max_comm_overlap_streams = 3
num_cublas_streams = 4


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if tex.get_device_compute_capability(0) >= 90:
        return 33_554_432
    return 4_194_304


def sanitize_dim(dim, ndim):
    """Convert relative dimension indexing to absolute."""
    assert abs(dim) < ndim, f"Dimension index {dim} is out of range {ndim}."
    if dim < 0:
        return ndim + dim
    return dim


class CollectiveGemmPrimitive(BasePrimitive):
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
                 scaling_mode, fuse_bias, fuse_gelu, grad, accumulate, use_split_accumulator,
                 comm_overlap_config, sharded_abstract):
        del scaling_mode, accumulate, use_split_accumulator

        # Sanity check contracting dims
        lhs_inner_dim = lhs.ndim - 2 if lhs_trans else lhs.ndim - 1
        rhs_inner_dim = rhs.ndim - 1 if rhs_trans else rhs.ndim - 2
        assert lhs.shape[lhs_inner_dim] == rhs.shape[rhs_inner_dim], (
            "Incompatible operands, contracting dimension sizes do not match."
        )
        lhs_outer_dim = lhs.ndim - 1 if lhs_trans else lhs.ndim - 2
        rhs_outer_dim = rhs.ndim - 2 if rhs_trans else rhs.ndim - 1

        # Sanity check scaling factors
        if scaling_mode != ScalingMode.NO_SCALING:
            assert lhs_scale_inv.size > 0 and rhs_scale_inv.size > 0, (
                "LHS and RHS operand inverse-scaling factors cannot be empty tensors for FP8 GEMM."
            )

        # Sanity check batched dims
        lhs_batch_size = 1 if lhs.ndim == 2 else reduce(operator.mul, lhs.shape[:-2])
        rhs_batch_size = 1 if rhs.ndim == 2 else reduce(operator.mul, rhs.shape[:-2])
        if lhs.ndim > 2 and rhs.ndim > 2:
            assert lhs_trans, "LHS must be transposed when both operands are batched."
            assert not rhs_trans, "RHS cannot be transposed when both operands are batched."
            assert lhs_batch_size == rhs_batch_size, "Mismatched batch sizes in operands."

        # Determine pure-GEMM output shape (without factoring in any TP scatter)
        out_shape = [lhs.shape[lhs_outer_dim], rhs.shape[rhs_outer_dim]]
        if lhs.ndim > 2 and rhs.ndim == 2:
            out_shape = [*lhs.shape[:-2], *out_shape]
        out_dtype = jnp.bfloat16 if scaling_mode != ScalingMode.NO_SCALING else lhs.dtype
        out_avals = [ jax.core.ShapedArray(out_shape, dtype=out_dtype) ]

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
                bias_shape = (out_shape[-1], )
        out_avals.append(jax.core.ShapedArray(bias_shape, dtype=bias_dtype))

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
        out_avals.append(jax.core.ShapedArray(pre_gelu_out_shape, dtype=out_dtype))

        # Auxiliary output for comm+GEMM overlap
        if comm_overlap_config is not None:
            aux_out_shape = (0, )
            aux_out_dtype = jnp.bfloat16
            if comm_overlap_config["comm_type"] == tex.CommOverlapType.RS:
                # Reduce-scattered GEMM output
                aux_out_dtype = out_dtype
                aux_out_shape = out_shape.copy()
                if sharded_abstract:
                    aux_out_shape[-2] /= comm_overlap_config["tp_size"]
            else:
                # All-gathered GEMM input
                aux_out_dtype = lhs.dtype
                aux_out_shape = list(lhs.shape).copy()
                if sharded_abstract:
                    aux_out_shape[lhs_outer_dim] *= comm_overlap_config["tp_size"]
            out_avals.append(jax.core.ShapedArray(shape=aux_out_shape, dtype=aux_out_dtype))

        # Declare cuBLAS workspace, expanded to the number of cuBLAS compute streams for TP overlap
        workspace_size = get_cublas_workspace_size_bytes()
        if comm_overlap_config is not None:
            workspace_size *= num_max_comm_overlap_streams
        out_avals.append(jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8))

        return out_avals

    @staticmethod
    def outer_abstract(*args, **kwargs):
        out_avals = CollectiveGemmPrimitive.abstract(*args, **kwargs)
        return out_avals[:-1]  # exclude cuBLAS workspace

    @staticmethod
    def lowering(ctx, lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_in, lhs_trans, rhs_trans,
                 scaling_mode, fuse_bias, fuse_gelu, grad, accumulate, use_split_accumulator,
                 comm_overlap_config, sharded_abstract):
        del sharded_abstract
        args = (
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_in
        )
        kwargs = {
            "lhs_trans" : lhs_trans,
            "rhs_trans" : rhs_trans,
            "scaling_mode" : scaling_mode,
            "fuse_bias" : fuse_bias,
            "fuse_gelu" : fuse_gelu,
            "grad" : grad,
            "accumulate" : accumulate,
            "use_split_accumulator" : use_split_accumulator,
        }
        ffi_name = CollectiveGemmPrimitive.name
        if comm_overlap_config is not None:
            kwargs["comm_overlap_id"] = comm_overlap_config["id"]
            kwargs["comm_type"] = comm_overlap_config["comm_type"]
            ffi_name = "te_comm_gemm_overlap_ffi"
        return jax.ffi.ffi_lowering(ffi_name, operand_output_aliases={ 4:1, 5:2 })(
            ctx, *args, **kwargs
        )

    @staticmethod
    def impl(*args, lhs_trans, rhs_trans, scaling_mode, fuse_bias, fuse_gelu, grad, accumulate,
             use_split_accumulator, comm_overlap_config, sharded_abstract):
        assert CollectiveGemmPrimitive.inner_primitive is not None
        outputs = CollectiveGemmPrimitive.inner_primitive.bind(
            *args,
            lhs_trans=lhs_trans,
            rhs_trans=rhs_trans,
            scaling_mode=scaling_mode,
            fuse_bias=fuse_bias,
            fuse_gelu=fuse_gelu,
            grad=grad,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
            comm_overlap_config=comm_overlap_config,
            sharded_abstract=sharded_abstract,
        )
        return outputs[:-1]  # exclude cuBLAS workspace


    @staticmethod
    def infer_sharding_from_operands(lhs_trans, rhs_trans, scaling_mode, fuse_bias,
                                     fuse_gelu, grad, accumulate, use_split_accumulator,
                                     comm_overlap_config, sharded_abstract, mesh, arg_infos,
                                     result_infos):
        del lhs_trans, scaling_mode, accumulate, use_split_accumulator, sharded_abstract
        del result_infos
        lhs_info, _, rhs_info, *_ = arg_infos
        lhs_spec = get_padded_spec(lhs_info)
        rhs_spec = get_padded_spec(rhs_info)
        lhs_outer_dim = lhs_info.ndim - 1 if lhs_trans else lhs_info.ndim - 2
        rhs_outer_dim = rhs_info.ndim - 2 if rhs_trans else rhs_info.ndim - 1

        # Final output sharding
        out_spec = (*lhs_spec[:-2], None, rhs_spec[rhs_outer_dim])
        out_shardings = [ NamedSharding(mesh, PartitionSpec(*out_spec)) ]

        # dBias sharding
        dbias_spec = (None, )
        if fuse_bias and grad:
            dbias_spec = (out_spec[-1], )
        out_shardings.append(NamedSharding(mesh, PartitionSpec(*dbias_spec)))

        # Pre-GeLU output sharding
        pre_gelu_out_spec = (None, )
        if fuse_gelu and not grad:
            pre_gelu_out_spec = out_spec
        out_shardings.append(NamedSharding(mesh, PartitionSpec(*pre_gelu_out_spec)))

        # Reduce-scattered output buffer sharding
        if comm_overlap_config is not None:
            aux_out_spec = (None, )
            if comm_overlap_config["comm_type"] == tex.CommOverlapType.RS:
                aux_out_spec = (*lhs_spec[:-2], global_mesh_resource().tp_resource, None)
            else:
                aux_out_spec = list(lhs_spec).copy()
                aux_out_spec[lhs_outer_dim] = None
            out_shardings.append(NamedSharding(mesh, PartitionSpec(*aux_out_spec)))

        # cuBLAS workspace sharding
        out_shardings.append(NamedSharding(mesh, PartitionSpec(None)))

        return out_shardings

    @staticmethod
    def partition(lhs_trans, rhs_trans, scaling_mode, fuse_bias, fuse_gelu, grad, accumulate,
                  use_split_accumulator, comm_overlap_config, sharded_abstract, mesh, arg_infos,
                  result_infos):
        lhs_info, _, rhs_info, *_ = arg_infos
        lhs_spec = get_padded_spec(lhs_info)
        rhs_spec = get_padded_spec(rhs_info)

        lhs_inner_dim = lhs_info.ndim - 2 if lhs_trans else lhs_info.ndim - 1
        lhs_outer_dim = lhs_info.ndim - 1 if lhs_trans else lhs_info.ndim - 2
        rhs_inner_dim = rhs_info.ndim - 1 if rhs_trans else rhs_info.ndim - 2
        rhs_outer_dim = rhs_info.ndim - 2 if rhs_trans else rhs_info.ndim - 1

        # Do not allow both dimensions to be sharded. If RHS (weights/kernel) is sharded as
        # FSDP+TP, the FSDP axis has to be gathered with a sharding constraint before passing it
        # into this custom op.
        assert not (lhs_spec[lhs_outer_dim] is not None and lhs_spec[lhs_inner_dim] is not None), (
            "LHS operand cannot be sharded in both outer and contracting dimensions."
        )
        assert not (rhs_spec[rhs_outer_dim] is not None and rhs_spec[rhs_inner_dim] is not None), (
            "RHS operand cannot be sharded in both outer and contracting dimensions."
        )
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_spec))
        rhs_scale_inv_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Require the outer dimension of LHS to be unsharded when there is no comm+GEMM overlap.
        # This will trigger an all-gather over the input sequence dimension.
        lhs_spec_new = list(lhs_spec).copy()
        if comm_overlap_config is None:
            lhs_spec_new[lhs_outer_dim] = None
        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_spec_new))
        lhs_scale_inv_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Require bias to be sharded to fit the GEMM output.
        # This should never trigger for correct inputs.
        arg_shardings = [lhs_sharding, lhs_scale_inv_sharding, rhs_sharding, rhs_scale_inv_sharding]
        bias_spec = (None, )
        if fuse_bias:
            bias_spec = (rhs_spec[rhs_outer_dim], )
        arg_shardings.append(NamedSharding(mesh, PartitionSpec(*bias_spec)))

        # Require gelu_in to match GEMM output.
        # This should never trigger for correct inputs.
        gelu_in_spec = (None, )
        if fuse_gelu:
            gelu_in_spec = (*lhs_spec[:-2], lhs_spec_new[lhs_outer_dim], rhs_spec[rhs_outer_dim])
        arg_shardings.append(NamedSharding(mesh, PartitionSpec(*gelu_in_spec)))

        out_shardings = CollectiveGemmPrimitive.infer_sharding_from_operands(
            lhs_trans, rhs_trans, scaling_mode, fuse_bias, fuse_gelu, grad, accumulate,
            use_split_accumulator, comm_overlap_config, sharded_abstract, mesh, arg_infos,
            result_infos
        )

        def sharded_impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_in):
            outputs = CollectiveGemmPrimitive.impl(
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
                comm_overlap_config=comm_overlap_config,
                sharded_abstract=True,
            )

            if comm_overlap_config is None and rhs_spec[rhs_inner_dim] is not None:
                outputs[0] = jax.lax.psum(outputs[0], global_mesh_resource().tp_resource)
                if fuse_gelu and not grad:
                    outputs[2] = jax.lax.psum(outputs[2], global_mesh_resource().tp_resource)

            return outputs[:-1]

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(CollectiveGemmPrimitive)


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


def _dequantize(x, scale_inv, dq_dtype):
    return x.astype(dq_dtype) * scale_inv.astype(dq_dtype)


# Apply jit to guarantee correctness of FP8 GEMM.
@partial(
    jax.jit,
    static_argnums=(
        2,
        3,
        4,
    ),
)
def __jitted_jax_gemm_tensor_scaling_fp8(lhs, rhs, lhs_dn, rhs_dn, precision):
    # Need to hard-code the dequantize here instead of calling lhs.dequantize() for pattern matching
    lhs_dq = _dequantize(lhs.data, lhs.scale_inv, lhs.dq_dtype)
    rhs_dq = _dequantize(rhs.data, rhs.scale_inv, rhs.dq_dtype)

    # Reshape + Transpose
    # [..., M, K] -> [B, M, K]
    # [..., K, M] -> [B, M, K]
    lhs_3d = _shape_normalization(lhs_dq, lhs_dn, lhs.data_layout == "N")
    rhs_3d = _shape_normalization(rhs_dq, rhs_dn, rhs.data_layout == "T")

    dim_nums = (((2,), (2,)), ((0,), (0,)))
    out_3d = jax.lax.dot_general(
        lhs_3d, rhs_3d, dim_nums, precision=precision, preferred_element_type=lhs.dq_dtype
    )
    return out_3d


def _jax_gemm_tensor_scaling_fp8(
    lhs: ScaledTensor, rhs: ScaledTensor, dim_nums: Tuple[Tuple[Sequence[int], Sequence[int]]]
):
    """FP8 GEMM for XLA pattern match"""
    assert rhs.scaling_mode.is_tensor_scaling(), "rhs does not have tensor scaling mode"

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
    if lhs.data_layout == "T":
        lhs_contract = tuple((lhs.data.ndim - 1 - i) % lhs.data.ndim for i in lhs_contract)
    if rhs.data_layout == "T":
        rhs_contract = tuple((rhs.data.ndim - 1 - i) % rhs.data.ndim for i in rhs_contract)

    lhs_dn = (lhs_contract, lhs_batch)
    rhs_dn = (rhs_contract, rhs_batch)

    lhs_remain_shape = _calculate_remaining_shape(lhs.data.shape, lhs_contract)
    rhs_remain_shape = _calculate_remaining_shape(rhs.data.shape, rhs_contract)

    precision = (
        jax.lax.Precision.HIGHEST if QuantizeConfig.FP8_2X_ACC_FPROP else jax.lax.Precision.DEFAULT
    )
    out_3d = __jitted_jax_gemm_tensor_scaling_fp8(lhs, rhs, lhs_dn, rhs_dn, precision)

    # Reshape [B, M, N] -> [..., M, N]
    out = out_3d.reshape(*lhs_remain_shape, *rhs_remain_shape)
    return out


def _jax_gemm_mxfp8_1d(
    lhs: ScaledTensor, rhs: ScaledTensor, dim_nums: Tuple[Tuple[Sequence[int], Sequence[int]]]
):
    """
    JAX GEMM for MXFP8 via scaled_matmul
    """
    assert (
        rhs.scaling_mode == ScalingMode.MXFP8_1D_SCALING
    ), "rhs does not have MXFP8 1D scaling mode"
    from jax._src.cudnn.scaled_matmul_stablehlo import scaled_matmul_wrapper

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
    out_3d = scaled_matmul_wrapper(
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
            return _jax_gemm_tensor_scaling_fp8(lhs, rhs, dim_nums)

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


def _te_gemm_impl(
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
    comm_overlap_config: dict = None,
) -> Tuple[jnp.ndarray, Union[jnp.ndarray, None], Union[jnp.ndarray, None]]:
    if lhs_scale_inv is None or scaling_mode == ScalingMode.NO_SCALING:
        lhs_scale_inv = jnp.empty(0, dtype=jnp.float32)
    if rhs_scale_inv is None or scaling_mode == ScalingMode.NO_SCALING:
        rhs_scale_inv = jnp.empty(0, dtype=jnp.float32)

    if grad or not fuse_bias:
        bias = jnp.empty(0, dtype=jnp.bfloat16)

    if (grad and not fuse_gelu) or not grad:
        pre_gelu_out = jnp.empty(0, dtype=jnp.bfloat16)

    outputs = CollectiveGemmPrimitive.outer_primitive.bind(
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
        comm_overlap_config=comm_overlap_config,
    )

    if grad and not fuse_bias:
        outputs[1] = None  # bias_grad

    if not grad and not fuse_gelu:
        outputs[2] = None  # pre_gelu_out

    return outputs


def _te_gemm_fwd_rule(lhs, rhs, bias, contracting_dims, quantizer_set, fuse_gelu,
                      accumulate, use_split_accumulator, comm_overlap_config):
    # Detect operand transposes, calculate collapsed 2-dimensional shapes
    lhs_data = lhs.data if isinstance(lhs, ScaledTensor) else lhs
    lhs_inner_dim = sanitize_dim(contracting_dims[0], lhs_data.ndim)
    lhs_trans = lhs_inner_dim != lhs_data.ndim - 1

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
    # AG overlap: ([B], M/P, K) --(AG)-> ([B], M, K) x (K, N/P) = ([B], M, N/P)
    # RS overlap: ([B], M, K/P) x (K/P, N) --(RS)-> ([B], M/P, N)
    # AR w/o overlap: ([B], M, K/P) x (K/P, N) --(AR)-> ([B], M, N)
    fuse_bias = bias is not None
    output, *aux_outputs = _te_gemm_impl(
        lhs_data,
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
        comm_overlap_config=comm_overlap_config,
    )
    if comm_overlap_config is not None:
        if comm_overlap_config["comm_type"] == tex.CommOverlapType.RS:
            output = aux_outputs[2]  # reduce-scattered output
        elif comm_overlap_config["save_gathered_lhs"]:
            lhs_data = aux_outputs[2]  # re-use all-gathered LHS in the backward pass

    output = checkpoint_name(output, "output")
    pre_gelu_out = None
    if fuse_gelu:
        pre_gelu_out = checkpoint_name(aux_outputs[1], "pre_gelu_out")

    return output, (
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        pre_gelu_out,
        lhs_trans,
        rhs_trans,
        fuse_bias,
    )


def _te_gemm_bwd_rule(contracting_dims, quantizer_set, fuse_gelu, accumulate, use_split_accumulator,
                      comm_overlap_config, ctx, dz):
    del contracting_dims
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

    # dLHS: ([B], M, N) x (K, N)^T = ([B], M, K)
    # FWD AG overlap: ([B], M, N/P) x (K, N/P)^T --(RS)-> ([B], M/P, K)
    # FWD RS overlap: ([B], M/P, N) --(AG)-> ([B], M, N) x (K/P, N)^T = ([B], M, K/P)
    # FWD AR w/o overlap: ([B], M, N) x (K/P, N)^T = ([B], M, K/P)
    lhs_grad_overlap = None if comm_overlap_config is None else comm_overlap_config["lhs_grad"]
    lhs_grad, *aux_outputs = _te_gemm_impl(
        dz_data,
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
        use_split_accumulator=use_split_accumulator,
        comm_overlap_config=lhs_grad_overlap,
    )
    if lhs_grad_overlap is not None:
        if lhs_grad_overlap["comm_type"] == tex.CommOverlapType.RS:
            lhs_grad = aux_outputs[2]  # reduce-scattered output
        else:
            dz_data = aux_outputs[2]  # all-gathered LHS

    bias_grad = None
    if fuse_bias:
        if fuse_bias_dgrad:
            bias_grad = bias_grad_q
        else:
            bias_grad = aux_outputs[0]

    # dRHS: ([B], M, K)^T x ([B], M, N) = (K, N)
    # FWD AG overlap: ([B], M/P, K)^T --(AG)-> ([B], M, K)^T x ([B], M, N/P) = (K, N/P)
    # FWD RS overlap (using gathered dz from dLHS): ([B], M, K/P)^T x ([B], M, N) = (K/P, N)
    # FWD AR w/o overlap: ([B], M, K/P)^T x ([B], M, N) = (K/P, N)
    rhs_grad_overlap = (
        None
        if comm_overlap_config is None or comm_overlap_config["save_gathered_lhs"]
        else comm_overlap_config["rhs_grad"]
    )
    rhs_grad, *_ = _te_gemm_impl(
        lhs_data,
        lhs_scale_inv,
        dz_data,
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
        comm_overlap_config=rhs_grad_overlap,
    )

    return lhs_grad, rhs_grad, bias_grad


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _te_gemm(lhs, rhs, bias, contracting_dims, quantizer_set, fuse_gelu, accumulate,
             use_split_accumulator, comm_overlap_config):
    output, _ = _te_gemm_fwd_rule(lhs, rhs, bias, contracting_dims, quantizer_set, fuse_gelu,
                                  accumulate, use_split_accumulator, comm_overlap_config)
    return output


_te_gemm.defvjp(_te_gemm_fwd_rule, _te_gemm_bwd_rule)


def te_gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    bias: jnp.ndarray = None,
    contracting_dims: Tuple[int, int] = (-1, 0),
    quantizer_set: QuantizerSet = noop_quantizer_set,
    fuse_gelu: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
    comm_overlap_config: dict = None,
):
    r"""
    Transformer Engine cuBLAS GEMM custom call w/ support for communication overlap.

    Parameters
    ----------
    lhs: Union[jnp.ndarray, ScaledTensor]
        Left-hand side operand. Batched dimension must be leading.
    rhs: Union[jnp.ndarray, ScaledTensor]
        Right-hand side operand. Batch dimension is not supported.
    bias: jnp.ndarray, default = None
        Optional additive bias.
    contracting_dims: Tuple[int, int], default = (-1, 0)
        Inner dimensions in the matrix multiplication. FP8 operands on Hopper are only supported
        with `(0, 0)` contracting dimensions.
    quantizer_set: QuantizerSet, default = noop_quantizer_set
        Set of quantizers for the input (LHS), kernel (RHS) and gradient of the output. If given,
        any `jnp.ndarray` operands will be quantized into `ScaledTensor`s. If the operands are
        already `ScaledTensor`s, the matching quantizer set must also be provided.
    fuse_gelu: bool, default = False
        Fuse the GeLU application on the output into the cuBLAS GEMM kernel prologue.
    accumulate: bool, default = False
        Accumulate the result directly into the output buffer.
    use_split_accumulator: bool, default = False
        Use split accumulator for FP8 GEMM.
    comm_overlap_config: dict, default = None
        Communication overlap options. If the operands are distributed but overlap config is `None`,
        XLA schedules blocking collectives before or after the GEMM custom call.
    """

    return _te_gemm(lhs, rhs, bias, contracting_dims, quantizer_set, fuse_gelu, accumulate,
                    use_split_accumulator, comm_overlap_config)


def get_default_comm_overlap_config(
    method: tex.CommOverlapMethod,
    comm_type: tex.CommOverlapType,
    tp_size: int,
) -> dict:
    """Returns a config dictionary with default options for the given overlap method."""
    if comm_type == tex.CommOverlapType.AG:
        assert method == tex.CommOverlapMethod.RING_EXCHANGE, (
            "All-gather overlap is only supported with the ring-exchange method."
        )

    global min_stream_priority, max_stream_priority
    if min_stream_priority is None or max_stream_priority is None:
        min_stream_priority, max_stream_priority = tex.get_stream_priority_range()

    return {
        "num_splits": tp_size if method == tex.CommOverlapMethod.RING_EXCHANGE else 4,
        "num_max_streams": num_max_comm_overlap_streams,
        "comm_cga_size": 1 if method == tex.CommOverlapMethod.RING_EXCHANGE else 2,
        "comm_priority": max_stream_priority,
        "gemm_priority": min_stream_priority,
        "num_comm_sm": 1 if method == tex.CommOverlapMethod.RING_EXCHANGE else 16,
        "set_sm_margin": not method == tex.CommOverlapMethod.RING_EXCHANGE,
        "use_ce": True,
        "atomic_gemm": False,
        "rs_overlap_first_gemm": False,
        "aggregate_ag": False,
    }


def initialize_comm_overlap(
    buffer_shape: Tuple[int, int],
    buffer_dtype: tex.DType,
    mesh: jax.sharding.Mesh,
    tp_resource: str,
    comm_type: tex.CommOverlapType,
    method: tex.CommOverlapMethod,
    lhs_grad: Union[bool, dict] = True,
    rhs_grad: Union[bool, dict] = True,
    save_gathered_lhs_for_backward: bool = False,
    **kwargs: dict,
) -> dict:
    r"""
    Initializes a comm+GEMM overlap buffer and returns an identifier based on a hash of the
    buffer's shape, data type and the overlap configuration options. Buffer creation is skipped if
    a buffer with the same hashed identifier already exists.

    Parameters
    ----------
    buffer_shape: Tuple[int, int]
        2-dimensional communication buffer shape.
    buffer_dtype: DType
        Transformer Engine data type for the communication buffer.
    mesh: jax.sharding.Mesh
        JAX Mesh with a `tp_resource` axis.
    tp_resource: str,
        Name of the mesh axis used for tensor-parallelism.
    comm_type: tex.CommOverlapType
        Collective communication type to overlap with compute.
    method: tex.CommOverlapMethod
        Implementation method for the communication overlap algorithms.
    lhs_grad: Union[bool, dict], default = True
        Flag for controlling whether this call also allocated the backward-pass buffer for
        communication overlap with the LHS operand gradient. The buffer config options can be
        controlled by passing a dictionary into this option instead of a boolean flag.
    rhs_grad: Union[bool, dict], default = True
        Flag for controlling whether this call also allocated the backward-pass buffer for
        communication overlap with the RHS operand gradient. The buffer config options can be
        controlled by passing a dictionary into this option instead of a boolean flag.
    save_gathered_lhs_for_backward: bool, default = False
        Optional optimization for saving the gathered LHS operand during the all-gather overlap
        in the forward pass, in order to re-use it in the RHS gradient computation in the backward
        pass. This avoids an all-gather in the backward pass at the expense of storing the global
        global LHS operand in the autograd context.
    kwargs: dict, default = {}
        Communication overlap configuration options. Any option not defined here falls back on
        default values set by `get_default_comm_overlap_config()`.
    """
    global num_max_comm_overlap_streams
    num_max_comm_overlap_streams = max(num_max_comm_overlap_streams,
                                       kwargs.get("num_max_streams", num_max_comm_overlap_streams))

    tp_size = get_mesh_axis_size(tp_resource, mesh=mesh)
    config = get_default_comm_overlap_config(method, comm_type, tp_size)
    config.update((k, kwargs[k]) for k in config.keys() & kwargs.keys())
    config["unique_id"] = tex.create_comm_overlap_buffer(
        comm_type, method, buffer_shape, buffer_dtype, tp_size, **config
    )
    config["mesh"] = mesh
    config["tp_resource"] = tp_resource
    config["tp_size"] = tp_size
    config["save_gathered_lhs"] = save_gathered_lhs_for_backward

    config["lhs_grad"] = None
    if lhs_grad:
        lhs_grad_comm_type = (
            tex.CommOverlapType.RS
            if comm_type == tex.CommOverlapType.AG
            else tex.CommOverlapType.AG
        )
        lhs_grad_method = (
            method
            if lhs_grad_comm_type == tex.CommOverlapType.RS
            else tex.CommOverlapMethod.RING_EXCHANGE
        )
        lhs_grad_config = get_default_comm_overlap_config(lhs_grad_comm_type, lhs_grad_method,
                                                          tp_size)
        if isinstance(lhs_grad, dict):
            lhs_grad_config.update(
                (k, lhs_grad[k]) for k in lhs_grad_config.keys() & lhs_grad.keys()
            )
        lhs_grad_config["unique_id"] = tex.create_comm_overlap_buffer(
            lhs_grad_comm_type, lhs_grad_method, buffer_shape, buffer_dtype, tp_size,
            lhs_grad = False, rhs_grad = False, **lhs_grad_config
        )
        lhs_grad_config["mesh"] = mesh
        lhs_grad_config["tp_resource"] = tp_resource
        lhs_grad_config["tp_size"] = tp_size
        lhs_grad_config["save_gathered_lhs"] = False
        lhs_grad_config["lhs_grad"] = None
        lhs_grad_config["rhs_grad"] = None

        config["lhs_grad"] = lhs_grad_config

    config["rhs_grad"] = None
    if (
        rhs_grad
        and not comm_type == tex.CommOverlapType.RS
        and not (comm_type == tex.CommOverlapType.AG and save_gathered_lhs_for_backward)
    ):
        rhs_grad_comm_type = (
            tex.CommOverlapType.RS
            if comm_type == tex.CommOverlapType.AG
            else tex.CommOverlapType.AG
        )
        rhs_grad_method = (
            method
            if rhs_grad_comm_type == tex.CommOverlapType.RS
            else tex.CommOverlapMethod.RING_EXCHANGE
        )
        rhs_grad_config = get_default_comm_overlap_config(rhs_grad_comm_type, rhs_grad_method,
                                                          tp_size)
        if isinstance(rhs_grad, dict):
            rhs_grad_config.update(
                (k, rhs_grad[k]) for k in rhs_grad_config.keys() & rhs_grad.keys()
            )
        rhs_grad_config["unique_id"] = tex.create_comm_overlap_buffer(
            rhs_grad_comm_type, rhs_grad_method, buffer_shape, buffer_dtype, tp_size,
            lhs_grad = False, rhs_grad = False, **rhs_grad_config
        )
        rhs_grad_config["mesh"] = mesh
        rhs_grad_config["tp_resource"] = tp_resource
        rhs_grad_config["tp_size"] = tp_size
        rhs_grad_config["save_gathered_lhs"] = False
        rhs_grad_config["lhs_grad"] = None
        rhs_grad_config["rhs_grad"] = None

        config["rhs_grad"] = rhs_grad_config

    return config


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
