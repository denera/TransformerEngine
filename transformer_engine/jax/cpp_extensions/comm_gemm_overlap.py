
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""
import operator
from dataclasses import dataclass, field
from functools import reduce
from typing import Sequence

import jax
import jax.numpy as jnp

import transformer_engine_jax as tex

from .base import manage_primitives
from .misc import sanitize_dims, jax_dtype_to_te_dtype

from ..sharding import (
    global_mesh_resource,
    get_mesh_axis_size,
    generate_pspec,
    SEQLEN_TP_AXES,
    HIDDEN_TP_AXES,
)


__all__ = [
    "CommOverlapHelper",
    "CommOverlapHelperSet",
]


CUDA_STREAM_PRIORITY_LOWEST = None
CUDA_STREAM_PRIORITY_HIGHEST = None


@dataclass(frozen=True)
class CommOverlapHelper:
    """
    Helper object that carries comm+GEMM overlap configuration, initializes the internal
    communication buffer, and generates lowering arguments and partitioning rules for
    the GemmPrimitive.
    """

    # Core init arguments
    comm_type: tex.CommOverlapType = field(default=tex.CommOverlapType.NONE)
    method: tex.CommOverlapMethod = field(default=tex.CommOverlapMethod.NONE)
    buffer_shape: Sequence[int] = field(default=None)
    buffer_dtype: jnp.dtype = field(default=jnp.bfloat16)
    tp_size: int = field(
        default_factory=lambda: get_mesh_axis_size(global_mesh_resource().tp_resource)
    )

    # Userbuffers bootstrap kwargs
    num_splits: int = field(default=None, kw_only=True)
    num_max_streams: int = field(default=3, kw_only=True)
    comm_cga_size: int = field(default=None, kw_only=True)
    gemm_priority: int = field(default=CUDA_STREAM_PRIORITY_LOWEST, kw_only=True)
    comm_priority: int = field(default=CUDA_STREAM_PRIORITY_HIGHEST, kw_only=True)
    num_comm_sm: int = field(default=None, kw_only=True)
    set_sm_margin: bool = field(default=None, kw_only=True)
    use_ce: bool = field(default=None, kw_only=True)
    atomic_gemm: bool = field(default=False, kw_only=True)
    rs_overlap_first_gemm: bool = field(default=False, kw_only=True)
    aggregate_ag: bool = field(default=False, kw_only=True)

    # Other kwargs not passed to Userbuffers
    tp_resource: str = field(default_factory=lambda: global_mesh_resource().tp_resource)
    logical_tp_axis: str = field(default=HIDDEN_TP_AXES, kw_only=True)
    logical_sp_axis: str = field(default=SEQLEN_TP_AXES, kw_only=True)
    output_all_gathered_lhs: bool = field(default=False, kw_only=True)
    flatten_axis: int = field(default=-1, kw_only=True)

    # Internal attributes
    is_enabled: bool = field(default=False, init=False)
    unique_id: int = field(default=-1, init=False, compare=False)
    sharded_impl: bool = field(default=False, init=False, compare=False)
    gather_dim: int = field(default=-2, init=False, compare=False)
    scatter_dim: int = field(default=-2, init=False, compare=False)

    def __post_init__(self):
        # Update global min/max CUDA stream priority values if not already done
        global CUDA_STREAM_PRIORITY_LOWEST, CUDA_STREAM_PRIORITY_HIGHEST
        if CUDA_STREAM_PRIORITY_LOWEST is None or CUDA_STREAM_PRIORITY_HIGHEST is None:
            (
                CUDA_STREAM_PRIORITY_LOWEST,
                CUDA_STREAM_PRIORITY_HIGHEST,
            ) = tex.get_stream_priority_range()
        if self.gemm_priority is None:
            object.__setattr__(self, "gemm_priority", CUDA_STREAM_PRIORITY_LOWEST)
        if self.comm_priority is None:
            object.__setattr__(self, "comm_priority", CUDA_STREAM_PRIORITY_HIGHEST)

        if self.method != tex.CommOverlapMethod.NONE or self.comm_type != tex.CommOverlapType.NONE:
            assert self.method != tex.CommOverlapMethod.NONE, (
                f"CommOverlapHelper: {self.comm_type} is not a valid collective type for "
                f"{self.method}."
            )
            assert self.comm_type != tex.CommOverlapType.NONE, (
                f"CommOverlapHelper: {self.method} is not a valid overlap method for "
                f"{self.comm_type}."
            )
            assert (
                self.buffer_shape is not None and len(self.buffer_shape) >= 2
            ), f"CommOverlapHelper: {self.buffer_shape} is not a valid buffer shape."
            assert self.tp_resource is not None, (
                "CommOverlapHelper: Communication + GEMM overlap requires a valid TP resource. "
                "This must either be specified via the `tp_resource=` keyword, or "
                "`CommOverlapHelper` needs to be initialized under a "
                "`te.sharding.global_shard_guard()` using a `te.sharding.MeshResource()` with a "
                "valid tensor-parallel mesh axis name."
            )
            assert (
                self.tp_size % 2 == 0
            ), f"CommOverlapHelper: Tensor-parallel axis of {self.tp_size} is not divisible by 2."

            # Comm overlap requires custom TE GEMM
            manage_primitives(enable_names=["GemmPrimitive"])

            if not self.is_bulk() and not self.is_p2p():
                # Pipelined overlap is only for reduce-scatter
                assert not self.is_all_gather(), (
                    f"CommOverlapHelper: {self.method} is not a valid overlap method for "
                    f"{self.comm_type}."
                )

            # Collapse buffer shape to 2D
            if len(self.buffer_shape) > 2:
                if self.flatten_axis < 0:
                    object.__setattr__(
                        self, "flatten_axis", self.flatten_axis + len(self.buffer_shape)
                    )
                object.__setattr__(
                    self,
                    "buffer_shape",
                    (
                        reduce(operator.mul, self.buffer_shape[: self.flatten_axis]),
                        reduce(operator.mul, self.buffer_shape[self.flatten_axis :]),
                    ),
                )

            # Num splits for P2P overlap is always fixed to TP size
            if self.is_p2p():
                object.__setattr__(self, "num_splits", self.tp_size)
            elif self.num_splits is None:
                object.__setattr__(self, "num_splits", self.tp_size)

            # Set conditional defaults for config options not specified at init time
            if self.comm_cga_size is None:
                object.__setattr__(self, "comm_cga_size", 1 if self.is_p2p() else 2)
            if self.num_comm_sm is None:
                object.__setattr__(self, "num_comm_sm", 1 if self.is_p2p() else 16)
            if self.set_sm_margin is None:
                object.__setattr__(self, "set_sm_margin", not self.is_p2p())
            if self.use_ce is None:
                object.__setattr__(self, "use_ce", self.is_p2p())

            # Allocate the communication buffer
            args, kwargs = self.get_bootstrap_args_kwargs()
            object.__setattr__(self, "unique_id", tex.create_comm_overlap_buffer(*args, **kwargs))
            object.__setattr__(self, "is_enabled", True)

    def _set_sharded_impl(self, value):
        assert isinstance(value, bool)
        object.__setattr__(self, "sharded_impl", value)

    def _set_gather_dim(self, value):
        assert isinstance(value, int)
        object.__setattr__(self, "gather_dim", value)

    def _set_scatter_dim(self, value):
        assert isinstance(value, int)
        object.__setattr__(self, "scatter_dim", value)

    def is_bulk(self):
        """Check if this is a bulk overlap."""
        return self.method == tex.CommOverlapMethod.BULK

    def is_p2p(self):
        """Check if this is a peer-to-peer (ring-exchange) overlap."""
        return self.method == tex.CommOverlapMethod.RING_EXCHANGE

    def is_all_gather(self):
        """Check if the overlapped collective is an all-gather."""
        return self.comm_type == tex.CommOverlapType.AG

    def is_reduce_scatter(self):
        """Check if the overlapped collective is a reduce-scatter."""
        return self.comm_type == tex.CommOverlapType.RS

    def has_aux_output(self):
        """Check if the comm+GEMM overlap has an auxiliary output."""
        return self.is_enabled and (
            self.is_bulk() or (self.is_all_gather() and self.output_all_gathered_lhs)
        )

    def get_bootstrap_args_kwargs(self):
        """Generate positional and keyword arguments to bootstrap Userbuffers."""
        args = (
            self.comm_type,
            self.method,
            self.buffer_shape,
            jax_dtype_to_te_dtype(self.buffer_dtype),
            self.tp_size,
        )
        kwargs = {
            "num_splits": self.num_splits,
            "num_max_streams": self.num_max_streams,
            "comm_cga_size": self.comm_cga_size,
            "gemm_priority": self.gemm_priority,
            "comm_priority": self.comm_priority,
            "num_comm_sm": self.num_comm_sm,
            "set_sm_margin": self.set_sm_margin,
            "use_ce": self.use_ce,
            "atomic_gemm": self.atomic_gemm,
            "rs_overlap_first_gemm": self.rs_overlap_first_gemm,
            "aggregate_ag": self.aggregate_ag,
        }
        return args, kwargs

    def get_lowering_kwargs(self):
        """Generate a dictionary of keyword arguments used in GemmPrimitive.lowering()."""
        aux_axis_boundary = -1
        if self.is_enabled and self.sharded_impl:
            if self.is_all_gather():
                assert self.gather_dim >= 0, (
                    "Internal TE error: CommOverlapHelper.gather_dim is not set correctly in "
                    "GemmPrimitive."
                )
                aux_axis_boundary = self.gather_dim + 1
            elif self.is_reduce_scatter():
                assert self.scatter_dim >= 0, (
                    "Internal TE error: CommOverlapHelper.scatter_dim is not set correctly in "
                    "GemmPrimitive."
                )
                aux_axis_boundary = self.scatter_dim + 1

        return {
            "comm_overlap_id": self.unique_id,
            "comm_overlap_method": int(self.method),
            "comm_type": int(self.comm_type),
            "aux_axis_boundary": aux_axis_boundary,
        }

    @staticmethod
    def _split_specs(specs, contracting_dims, batch_dims):
        ndims = len(specs)
        cdims, bdims = map(sanitize_dims, (ndims, ndims), (contracting_dims, batch_dims))

        # Batch specs
        bspecs = tuple(specs[i] for i in bdims)

        # Non-batch leading dimension specs
        lspecs = tuple(specs[i] for i in range(ndims) if i not in cdims + bdims)

        # Non-batch contracting dimension specs
        cspecs = tuple(specs[i] for i in range(ndims) if i in cdims and i not in bdims)

        return bspecs, lspecs, cspecs

    @staticmethod
    def _check_operand_specs(lhs_specs, rhs_specs, contracting_dims, batched_dims):
        lhs_cdims, rhs_cdims = contracting_dims
        lhs_bdims, rhs_bdims = batched_dims
        lhs_ndim, rhs_ndim = map(len, (lhs_specs, rhs_specs))

        (
            (lhs_bspecs, lhs_lspecs, lhs_cspecs),
            (rhs_bspecs, rhs_lspecs, rhs_cspecs),
        ) = map(
            CommOverlapHelper._split_specs,
            (lhs_specs, rhs_specs),
            (lhs_cdims, rhs_cdims),
            (lhs_bdims, rhs_bdims),
        )

        # Batched dimensions must have the same sharding
        if len(lhs_bdims) > 0 and len(rhs_bdims) > 0:
            assert all(
                lhs_bspec == rhs_bspec for lhs_bspec, rhs_bspec in zip(lhs_bspecs, rhs_bspecs)
            ), (
                "cuBLAS GEMM operand batch dimensions must have the same sharding: "
                f"{lhs_specs} @ idx {lhs_bdims} x {rhs_specs} @ idx {rhs_bdims}."
            )

        # Only one each of the non-batched leading dimensions and non-batched contracting
        # dimensions can be sharded
        lhs_ldims, rhs_ldims = map(
            lambda ndim, exclude: tuple(dim for dim in range(ndim) if dim not in exclude),
            (lhs_ndim, rhs_ndim),
            (lhs_bdims + lhs_cdims, rhs_bdims + rhs_cdims),
        )
        (lhs_lspec_not_none, rhs_lspec_not_none, lhs_cspec_not_none, rhs_cspec_not_none) = map(
            lambda specs: tuple(spec for spec in specs if spec is not None),
            (lhs_lspecs, rhs_lspecs, lhs_cspecs, rhs_cspecs),
        )
        assert len(lhs_lspec_not_none) <= 1 and len(rhs_lspec_not_none) <= 1, (
            "cuBLAS GEMM operands can have only one sharded non-batched leading dimension: "
            f"{lhs_specs} @ idx {lhs_ldims} x {rhs_specs} @ idx {rhs_ldims}."
        )
        assert len(lhs_cspec_not_none) <= 1 and len(rhs_cspec_not_none) <= 1, (
            "cuBLAS GEMM operands can have only one sharded non-batched contracting dimension: "
            f"{lhs_specs} @ idx {lhs_cdims} x {rhs_specs} @ idx {rhs_cdims}."
        )

        # Extract single leading and contracting dimension specs
        (lhs_lspec, rhs_lspec, lhs_cspec, rhs_cspec) = map(
            lambda specs: None if len(specs) == 0 else specs[0],
            (lhs_lspec_not_none, rhs_lspec_not_none, lhs_cspec_not_none, rhs_cspec_not_none),
        )
        return (lhs_lspec, lhs_cspec), (rhs_lspec, rhs_cspec)

    def _get_no_overlap_rules(
        self,
        lhs_specs,
        rhs_specs,
        aux_in_specs,
        contracting_dims,
        batched_dims,
        sequence_parallel_output,
        sequence_dim,
    ):
        lhs_ndim, rhs_ndim = map(len, (lhs_specs, rhs_specs))
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs_ndim, rhs_ndim), contracting_dims)
        lhs_bdims, rhs_bdims = map(sanitize_dims, (lhs_ndim, rhs_ndim), batched_dims)

        (_, lhs_cspec), (_, rhs_cspec) = self._check_operand_specs(
            lhs_specs, rhs_specs, (lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims)
        )

        # Partitioning rules:
        # ([B], M, K1) x ([B], N, K2)^T = ([B], M, N)
        # 1. K1 == K2 != None
        #   - Require non-batched non-contracting dims of both LHS and RHS to be unsharded.
        #   - If `sequence_parallel_output=True`, then reduce-scatter the output.
        #   - Otherwise, all-reduce the output.
        # 2. Otherwise
        #   - Require contracting dimensions of both LHS and RHS to be unsharded.
        #   - Require non-batched non-contracting dims of LHS to be unsharded.
        reduce_output = rhs_cspec is not None and lhs_cspec == rhs_cspec
        reduce_spec = scatter_dim = None
        if reduce_output:
            reduce_spec = rhs_cspec
            if sequence_parallel_output:
                # If the sequence dimension is not specified, assume it to be the first
                # non-batched non-contracting dimension of the LHS operand.
                lhs_ldims = tuple(i for i in range(lhs_ndim) if i not in lhs_bdims + lhs_cdims)
                scatter_dim = sequence_dim if sequence_dim is not None else lhs_ldims[0]

        # Always require the non-batched non-contracting dims of LHS to be unsharded
        # NOTE: This will all-gather sequence-parallel inputs and preserve tensor-parallel params.
        lhs_specs = tuple(
            lhs_specs[i] if i in set(lhs_bdims + lhs_cdims) else None for i in range(lhs_ndim)
        )
        if reduce_output:
            # When reducing GEMM output, require non-batched non-contracting dims of the RHS
            # operand to be unsharded (i.e. FSDP)
            rhs_specs = tuple(
                None if i not in set(rhs_bdims + rhs_cdims) else rhs_specs[i]
                for i in range(rhs_ndim)
            )
        else:
            # Otherwise, require contracting dims of both operands to be unsharded
            lhs_specs = tuple(None if i in lhs_cdims else lhs_specs[i] for i in range(lhs_ndim))
            rhs_specs = tuple(None if i in rhs_cdims else rhs_specs[i] for i in range(rhs_ndim))

        # Combine modified LHS and RHS specs into the output
        lhs_non_contracting_specs, rhs_non_contracting_specs = map(
            lambda specs, cdims: tuple(specs[i] for i in range(len(specs)) if i not in cdims),
            (lhs_specs, rhs_specs),
            (lhs_cdims, rhs_cdims),
        )
        out_specs = [*lhs_non_contracting_specs, *rhs_non_contracting_specs]

        # Bias and Pre-GeLU sharding is based on GEMM output before any scatter
        bias_specs = tuple(list(out_specs[len(lhs_non_contracting_specs) :]).copy())
        gelu_specs = tuple(list(out_specs).copy())

        # Set output scatter dim to the tensor-parallel spec
        if sequence_parallel_output:
            out_specs[scatter_dim] = reduce_spec

        return (
            (lhs_specs, rhs_specs, bias_specs, gelu_specs, aux_in_specs),
            (out_specs, bias_specs, gelu_specs, (None, )),
            (reduce_spec, scatter_dim),
        )

    def _get_bulk_overlap_rules(
        self,
        lhs_specs,
        rhs_specs,
        aux_in_specs,
        contracting_dims,
        batched_dims,
        sequence_parallel_output,
        sequence_dim,
    ):
        assert self.tp_resource in aux_in_specs, (
            "CommOverlapHelper: Auxiliary input for bulk all-gather overlap is not sharded "
            f"over the tensor-parallel mesh resource '{self.tp_resource}' in any dimension."
        )

        aux_out_specs = (None,)
        bulk_comm_dim = aux_in_specs.index(self.tp_resource)
        aux_in_specs_batch = aux_in_specs[:bulk_comm_dim]
        aux_in_specs_tail = aux_in_specs[bulk_comm_dim + 1 :]
        if self.is_all_gather():
            assert all(spec is None for spec in aux_in_specs_tail), (
                "CommOverlapHelper: Trailing dimensions of the auxiliary input for bulk all-gather "
                "overlap cannot be sharded."
            )
            self._set_gather_dim(bulk_comm_dim)
            aux_out_specs = (
                *aux_in_specs_batch,
                None,  # all-gathered dimension
                *aux_in_specs_tail,
            )
        else:
            assert all(spec is None for spec in aux_in_specs[bulk_comm_dim:]), (
                "CommOverlapHelper: Non-batch dimensions of the auxiliary input for bulk "
                "reduce-scatter overlap cannot be sharded."
            )
            self._set_scatter_dim(bulk_comm_dim)
            aux_out_specs = (
                *aux_in_specs_batch,
                self.tp_resource,
                *aux_in_specs_tail,
            )

        # GEMM is independent of communication so specs are as if there is no overlap
        operand_specs, output_specs, xla_reduce_info = self._get_no_overlap_rules(
            lhs_specs,
            rhs_specs,
            aux_in_specs,
            contracting_dims,
            batched_dims,
            sequence_parallel_output,
            sequence_dim,
        )

        return (
            operand_specs,
            (*output_specs[:-1], aux_out_specs),
            xla_reduce_info,
        )

    def _get_all_gather_rules(
        self,
        lhs_specs,
        rhs_specs,
        aux_in_specs,
        contracting_dims,
        batched_dims,
        sequence_parallel_output,
        sequence_dim,
    ):
        del sequence_parallel_output, sequence_dim

        lhs_ndim, rhs_ndim = map(len, (lhs_specs, rhs_specs))
        lhs_cdims, rhs_cdims, lhs_bdims, rhs_bdims = map(
            sanitize_dims, 2 * [lhs_ndim, rhs_ndim], contracting_dims + batched_dims
        )

        (lhs_lspec, _), _ = self._check_operand_specs(
            lhs_specs, rhs_specs, (lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims)
        )
        assert lhs_lspec == self.tp_resource, (
            "CommOverlapHelper: Non-batch leading dimension of the LHS operand for AG->GEMM "
            f"overlap must be sharded over the tensor-parallel mesh resource '{self.tp_resource}', "
            f"but got {lhs_lspec} sharding instead."
        )

        # AG->GEMM overlap: Require non-batched contracting dimensions to be unsharded (e.g. FSDP)
        # LHS: (B, M, None)
        # RHS: (None, N)
        # OUT: (B, M, None) --(AG)-> (B, None, None) x (None, N) = (B, None, N)
        lhs_specs = tuple(
            None if i in lhs_cdims and i not in lhs_bdims else lhs_specs[i] for i in range(lhs_ndim)
        )
        rhs_specs = tuple(
            None if i in rhs_cdims and i not in rhs_bdims else rhs_specs[i] for i in range(rhs_ndim)
        )

        # GEMM output spec keeps LHS batch spec and RHS non-contracting specs, but is None
        # in the non-batched leading dimensions.
        lhs_non_cspecs_gathered = list(
            lhs_specs[i] if i in lhs_bdims else None for i in range(lhs_ndim) if i not in lhs_cdims
        )
        rhs_non_cspecs = tuple(rhs_specs[i] for i in range(rhs_ndim) if i not in rhs_cdims)
        out_specs = (*lhs_non_cspecs_gathered, *rhs_non_cspecs)
        self._set_gather_dim(lhs_specs.index(lhs_lspec))

        # Bias and Pre-GeLU sharding is based on GEMM output
        bias_specs = out_specs[len(lhs_non_cspecs_gathered) :]
        gelu_specs = out_specs

        # Auxiliary input/output specs depend on bulk vs. non-bulk overlap
        aux_out_specs = (None,)
        if self.output_all_gathered_lhs:
            # Auxiliary output is the same as the LHS spec, except the gathered dimension unsharded
            aux_out_specs = list(lhs_specs).copy()
            aux_out_specs[self.gather_dim] = None

        return (
            (lhs_specs, rhs_specs, bias_specs, gelu_specs, aux_in_specs),
            (out_specs, bias_specs, gelu_specs, aux_out_specs),
            (None, None),
        )

    def _get_reduce_scatter_rules(
        self,
        lhs_specs,
        rhs_specs,
        aux_in_specs,
        contracting_dims,
        batched_dims,
        sequence_parallel_output,
        sequence_dim,
    ):
        del sequence_parallel_output, sequence_dim

        lhs_ndim, rhs_ndim = map(len, (lhs_specs, rhs_specs))
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs_ndim, rhs_ndim), contracting_dims)
        lhs_bdims, rhs_bdims = map(sanitize_dims, (lhs_ndim, rhs_ndim), batched_dims)

        (_, lhs_cspec), (_, rhs_cspec) = self._check_operand_specs(
            lhs_specs, rhs_specs, (lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims)
        )
        assert lhs_cspec == rhs_cspec == self.tp_resource, (
            "CommOverlapHelper: Non-batched contracting dimensions of LHS and RHS operands for "
            "GEMM->RS overlap must be sharded over the tensor-parallel resource "
            f"{self.tp_resource}, but got LHS:{lhs_cspec} and RHS:{rhs_cspec} sharding instead."
        )

        # GEMM->RS overlap: Require non-contracting non-batch dimensions to be unsharded (e.g. FSDP)
        # LHS: (B, None, K)
        # RHS: (K, None)
        # OUT: (B, None, K) x (K, None) = (B, None, None) --(UB-RS)-> (B, M, None)
        lhs_specs = tuple(
            None if i not in lhs_bdims + lhs_cdims else lhs_specs[i] for i in range(lhs_ndim)
        )
        rhs_specs = tuple(
            None if i not in rhs_bdims + rhs_cdims else rhs_specs[i] for i in range(rhs_ndim)
        )

        # GEMM output is the internal communication buffer, but we will use the XLA output buffer
        # as the final reduce-scattered output so we shard it accordingly here.
        lhs_bspecs = tuple(
            lhs_specs[i] for i in range(lhs_ndim) if i in lhs_bdims and i not in lhs_cdims
        )
        lhs_lspecs = tuple(lhs_specs[i] for i in range(lhs_ndim) if i not in lhs_bdims + lhs_cdims)
        rhs_non_cspecs = tuple(rhs_specs[i] for i in range(rhs_ndim) if i not in rhs_cdims)
        out_specs = (
            *lhs_bspecs,
            self.tp_resource,
            *[None for _ in range(len(lhs_lspecs) - 1)],
            *rhs_non_cspecs,
        )
        self._set_scatter_dim(out_specs.index(self.tp_resource))

        # Bias and Pre-GeLU sharding is based on GEMM output
        bias_specs = out_specs[len(lhs_bspecs) + len(lhs_lspecs) :]
        gelu_specs = out_specs

        return (
            (lhs_specs, rhs_specs, bias_specs, gelu_specs, aux_in_specs),
            (out_specs, bias_specs, gelu_specs, (None,)),
            (None, None),
        )

    def get_partitioning_rules(
        self,
        lhs_specs,
        rhs_specs,
        aux_in_specs,
        contracting_dims,
        batched_dims,
        sequence_parallel_output,
        sequence_dim,
    ):
        """
        Correct operand specs to partititions suitable for the GemmPrimitive, and infer the
        partition specs of the outputs.
        """
        if self.is_bulk():
            return self._get_bulk_overlap_rules(
                lhs_specs,
                rhs_specs,
                aux_in_specs,
                contracting_dims,
                batched_dims,
                sequence_parallel_output,
                sequence_dim,
            )

        impl_map = {
            tex.CommOverlapType.NONE: self._get_no_overlap_rules,
            tex.CommOverlapType.AG: self._get_all_gather_rules,
            tex.CommOverlapType.RS: self._get_reduce_scatter_rules,
        }
        return impl_map[self.comm_type](
            lhs_specs,
            rhs_specs,
            aux_in_specs,
            contracting_dims,
            batched_dims,
            sequence_parallel_output,
            sequence_dim,
        )

    def get_output_spec(
        self,
        lhs_specs,
        rhs_specs,
        contracting_dims,
        batched_dims,
        sequence_parallel_output,
        sequence_dim,
        from_logical_axes=False,
        with_flax_rules=True,
    ):
        """
        Infer a PartitionSpec for the GEMM output based on the LHS and RHS operand specs. This can
        optionally generate the output spec from operand logical axes as well.
        """
        if from_logical_axes:
            lhs_specs = generate_pspec(lhs_specs, with_flax_rules=with_flax_rules, padded=True)
            rhs_specs = generate_pspec(rhs_specs, with_flax_rules=with_flax_rules, padded=True)

        (_, (out_specs, *_), _) = self.get_partitioning_rules(
            lhs_specs,
            rhs_specs,
            (None, ),
            contracting_dims,
            batched_dims,
            sequence_parallel_output,
            sequence_dim,
        )
        return jax.sharding.PartitionSpec(*out_specs)



@dataclass(frozen=True)
class CommOverlapHelperSet:
    """
    A set of CommOverlapHelper objects that provide complementary comm+GEMM overlap configurations
    for FPROP, DGRAD and WGRAD GEMMs in FWD/BWD passes through Dense-layers.
    """

    fprop: CommOverlapHelper = field(default=None)
    dgrad: CommOverlapHelper = field(default=None)
    wgrad: CommOverlapHelper = field(default=None)

    def _sanity_check(self):
        # Require any argument that exists to be a `CommOverlapHelper` instance
        for overlap, name in zip((self.fprop, self.dgrad, self.wgrad), ("fprop", "dgrad", "wgrad")):
            if overlap is not None:
                assert isinstance(overlap, CommOverlapHelper), (
                    f"CommOverlapHelperSet: Expected `{name}` to be a {CommOverlapHelper} but got "
                    f"{type(overlap)} instead."
                )

        # If FPROP overlap is not defined or not enabled, require DGRAD and WGRAD to also not be
        # be defined or not enabled
        if self.fprop is None or not self.fprop.is_enabled:
            assert (self.dgrad is None or not self.dgrad.is_enabled) and (
                self.wgrad is None or not self.wgrad.is_enabled
            ), (
                "CommOverlapHelperSet: Cannot do communication overlap for DGRAD and/or WGRAD when "
                "there is no communication overlap for FPROP."
            )
            return

        assert (
            not self.fprop.is_bulk()
        ), "CommOverlapHelperSet: Cannot overlap bulk collectives with FPROP."

        if self.fprop.is_all_gather():
            if self.dgrad is not None and self.dgrad.is_enabled:
                if self.dgrad.is_bulk() and self.dgrad.is_all_gather():
                    assert not self.fprop.output_all_gathered_lhs, (
                        "CommOverlapHelperSet: AG->GEMM FPROP does not support BULK-AG overlap for "
                        "DGRAD when the all-gathered LHS is already saved in the forward pass."
                    )
                    assert (
                        self.wgrad is not None
                        and self.wgrad.is_enabled
                        and self.wgrad.is_bulk()
                        and self.wgrad.is_reduce_scatter()
                    ), (
                        "CommOverlapHelperSet: AG->GEMM FPROP with BULK-AG overlap for DGRAD "
                        "requires BULK-RS overlap for WGRAD."
                    )

                elif not self.dgrad.is_bulk() and self.dgrad.is_reduce_scatter():
                    assert self.wgrad is None or not self.wgrad.is_enabled, (
                        "CommOverlapHelperSet: AG->GEMM FPROP with GEMM->RS DGRAD does not support "
                        "communication overlap for WGRAD."
                    )

                else:
                    raise AssertionError(
                        "CommOverlapHelperSet: AG->GEMM FPROP requires communication overlap for "
                        "DGRAD to be either BULK-AG or GEMM->RS."
                    )
            else:
                assert self.wgrad is None or not self.wgrad.is_enabled, (
                    "CommOverlapHelperSet: AG->GEMM FPROP with no communication overlap for DGRAD"
                    "does not support communication overlap for WGRAD."
                )

        elif self.fprop.is_reduce_scatter():
            if self.dgrad is not None and self.dgrad.is_enabled:
                assert not self.dgrad.is_bulk() and self.dgrad.is_all_gather(), (
                    "CommOverlapHelperSet: GEMM->RS FPROP requires communication overlap for DGRAD "
                    "to be AG->GEMM."
                )

            assert self.wgrad is None or not self.wgrad.is_enabled, (
                "CommOverlapHelperSet: GEMM->RS FPROP does not support communication overlap "
                "for WGRAD."
            )

        else:
            raise RuntimeError(
                "CommOverlapHelperSet: Internal TE error, unrecognized collective type "
                f"{self.fprop.comm_type} in communication overlap for FPROP."
            )

    def __post_init__(self):
        self._sanity_check()

        if self.fprop is None:
            object.__setattr__(self, "fprop", CommOverlapHelper())

        if not self.fprop.is_enabled:
            object.__setattr__(self, "dgrad", CommOverlapHelper())
            object.__setattr__(self, "wgrad", CommOverlapHelper())

        # Column-parallel layers: QKV projection and MLP FFN1
        #   FPROP with AG->GEMM:
        #     LHS:(B, M, None)--(AG)->(B, None, None) x RHS:(None, N) = OUT:(B, None, N)
        #   DGRAD w/ BULK-AG for LHS:
        #     GRAD:(B, None, N) x RHS:(None, N)^T = DGRAD:(B, None, None)
        #     LHS:(B, M, None)--(BULK-AG)->(B, None, None)
        #   WGRAD w/ BULK-RS for DGRAD:
        #     LHS:(B, None, None)^T x GRAD:(B, None, N) = WGRAD:(None, N)
        #     DGRAD:(B, None, None)--(BULK-RS)->(B, M, None)
        #
        # Row-parallel layers: Post-attention projection and MLP FFN2
        #   FPROP with GEMM->RS:
        #     LHS:(B, None, K) x RHS:(K, None) = (B, None, None)--(RS)->(B, M, None)
        #   DGRAD with AG->GEMM (all-gathered GRAD saved for WGRAD):
        #     GRAD:(B, M, None)--(AG)->(B, None, None) x RHS:(K, None)^T = (B, None, K)
        #   WGRAD with NO OVERLAP:
        #     LHS:(B, None, K)^T x GRAD:(B, None, None) = (K, None)
        if self.dgrad is None:
            dgrad_overlap = None

            if self.fprop.is_all_gather():
                if self.fprop.output_all_gathered_lhs:
                    # FPROP AG->GEMM w/ saved global LHS and DGRAD GEMM->RS
                    dgrad_overlap = CommOverlapHelper(
                        method=tex.CommOverlapMethod.RING_EXCHANGE,
                        comm_type=tex.CommOverlapType.RS,
                        buffer_shape=self.fprop.buffer_shape,
                        buffer_dtype=self.fprop.buffer_dtype,
                        tp_size=self.fprop.tp_size,
                        logical_tp_axis=self.fprop.logical_tp_axis,
                        logical_sp_axis=self.fprop.logical_sp_axis,
                    )
                else:
                    # FPROP AG->GEMM and DGRAD BULK-AG for LHS
                    dgrad_overlap = CommOverlapHelper(
                        method=tex.CommOverlapMethod.BULK,
                        comm_type=tex.CommOverlapType.AG,
                        buffer_shape=self.fprop.buffer_shape,
                        buffer_dtype=self.fprop.buffer_dtype,
                        tp_size=self.fprop.tp_size,
                        logical_tp_axis=self.fprop.logical_tp_axis,
                        logical_sp_axis=self.fprop.logical_sp_axis,
                    )

            elif self.fprop.is_reduce_scatter():
                # FPROP GEMM->RS and DGRAD AG->GEMM
                dgrad_overlap = CommOverlapHelper(
                    method=tex.CommOverlapMethod.RING_EXCHANGE,
                    comm_type=tex.CommOverlapType.AG,
                    buffer_shape=self.fprop.buffer_shape,
                    buffer_dtype=self.fprop.buffer_dtype,
                    tp_size=self.fprop.tp_size,
                    logical_tp_axis=self.fprop.logical_tp_axis,
                    logical_sp_axis=self.fprop.logical_sp_axis,
                    output_all_gathered_lhs=False,
                )

            else:
                dgrad_overlap = CommOverlapHelper()

            object.__setattr__(self, "dgrad", dgrad_overlap)

        if self.wgrad is None:
            wgrad_overlap = self.wgrad

            if (
                self.fprop.is_all_gather()
                and not self.fprop.output_all_gathered_lhs
                and self.dgrad.is_enabled
                and self.dgrad.is_bulk()
                and self.dgrad.is_all_gather()
            ):
                # FPROP AG->GEMM, DGRAD BULK-AG for LHS and WGRAD BULK-RS for DGRAD
                wgrad_overlap = CommOverlapHelper(
                    method=tex.CommOverlapMethod.BULK,
                    comm_type=tex.CommOverlapType.RS,
                    buffer_shape=self.fprop.buffer_shape,
                    buffer_dtype=self.fprop.buffer_dtype,
                    tp_size=self.fprop.tp_size,
                    logical_tp_axis=self.fprop.logical_tp_axis,
                    logical_sp_axis=self.fprop.logical_sp_axis,
                )

            else:
                wgrad_overlap = CommOverlapHelper()

            object.__setattr__(self, "wgrad", wgrad_overlap)
