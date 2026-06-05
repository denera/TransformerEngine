# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for grouped FP8 block-scaling quantization."""

import math
from typing import Iterable, List

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch import Float8BlockQuantizer, GroupedLinear, autocast
import transformer_engine.pytorch.module.grouped_linear as grouped_linear_module
import transformer_engine_torch as tex


fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)


def _make_quantizer(block_scaling_dim: int, rowwise: bool, columnwise: bool):
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=block_scaling_dim,
    )


def _assert_optional_equal(name: str, got, ref) -> None:
    if ref is None:
        assert got is None, name
        return
    assert got is not None, name
    assert got.shape == ref.shape, name
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def _valid_scale_shape(shape, quantizer, columnwise: bool):
    rows = math.prod(shape[:-1])
    cols = shape[-1] if shape else 1
    block_len = quantizer.block_len
    row_blocks = (rows + block_len - 1) // block_len
    col_blocks = (cols + block_len - 1) // block_len

    if quantizer.block_scaling_dim == 2:
        return (col_blocks, row_blocks) if columnwise else (row_blocks, col_blocks)
    return (row_blocks, cols) if columnwise else (col_blocks, rows)


def _assert_scale_equal(name: str, got, ref, shape, quantizer, columnwise: bool) -> None:
    if ref is None:
        assert got is None, name
        return
    assert got is not None, name
    assert got.shape == ref.shape, name
    valid_rows, valid_cols = _valid_scale_shape(shape, quantizer, columnwise)
    torch.testing.assert_close(
        got[:valid_rows, :valid_cols],
        ref[:valid_rows, :valid_cols],
        rtol=0,
        atol=0,
    )


def _assert_blockwise_tensor_equal(got, ref) -> None:
    _assert_optional_equal("rowwise_data", got._rowwise_data, ref._rowwise_data)
    _assert_optional_equal("columnwise_data", got._columnwise_data, ref._columnwise_data)
    shape = tuple(got.size())
    quantizer = got._quantizer
    _assert_scale_equal(
        "rowwise_scale_inv",
        got._rowwise_scale_inv,
        ref._rowwise_scale_inv,
        shape,
        quantizer,
        False,
    )
    _assert_scale_equal(
        "columnwise_scale_inv",
        got._columnwise_scale_inv,
        ref._columnwise_scale_inv,
        shape,
        quantizer,
        True,
    )


def _manual_quantize_one(part: torch.Tensor, quantizer):
    if (
        quantizer.block_scaling_dim == 2
        and quantizer.columnwise_usage
        and not quantizer.rowwise_usage
    ):
        # The non-grouped 2D blockwise kernel computes rowwise output while
        # optionally producing columnwise output. Use that path as the loop
        # reference and drop the rowwise buffers before comparison.
        ref_quantizer = Float8BlockQuantizer(
            fp8_dtype=quantizer.dtype,
            rowwise=True,
            columnwise=True,
            force_pow_2_scales=quantizer.force_pow_2_scales,
            amax_epsilon=quantizer.amax_epsilon,
            block_scaling_dim=quantizer.block_scaling_dim,
        )
        ref = tex.quantize(part.contiguous(), ref_quantizer)
        ref.update_usage(rowwise_usage=False, columnwise_usage=True)
        return ref
    return tex.quantize(part.contiguous(), quantizer)


def _manual_quantize(parts: Iterable[torch.Tensor], quantizer) -> List[object]:
    return [_manual_quantize_one(part, quantizer) for part in parts]


def _run_grouped_linear(module, inp, m_splits, fp8_recipe):
    module.zero_grad(set_to_none=True)
    inp = inp.detach().clone().requires_grad_(True)
    with autocast(enabled=True, recipe=fp8_recipe):
        out = module(inp, m_splits)
    loss = out.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    weight_grads = [
        getattr(module, f"weight{i}").grad.detach().clone()
        for i in range(module.num_gemms)
    ]
    return out.detach(), inp.grad.detach(), weight_grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not fp8_block_scaling_available,
    reason=reason_for_no_fp8_block_scaling,
)
@pytest.mark.parametrize("block_scaling_dim", [1, 2])
@pytest.mark.parametrize("rowwise,columnwise", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("splits", [[128, 256, 384, 512], [129, 0, 255, 303]])
def test_group_quantize_fp8_blockwise_matches_manual_loop(
    block_scaling_dim: int,
    rowwise: bool,
    columnwise: bool,
    splits: List[int],
) -> None:
    """Direct grouped quantize matches independent FP8 blockwise quantization."""

    torch.manual_seed(1234)
    cols = 300
    inp = torch.randn(sum(splits), cols, dtype=torch.bfloat16, device="cuda")
    first_dims = torch.tensor(splits, dtype=torch.int64, device="cuda")
    quantizer = _make_quantizer(block_scaling_dim, rowwise, columnwise)

    grouped = tex.group_quantize(inp, quantizer, len(splits), first_dims)
    got_parts = grouped.split_into_quantized_tensors()
    ref_parts = _manual_quantize(torch.split(inp, splits), quantizer)

    if rowwise:
        assert grouped.scale_inv_offsets is not None
    else:
        assert grouped.scale_inv is None
    if columnwise:
        assert grouped.columnwise_scale_inv_offsets is not None
    else:
        assert grouped.columnwise_scale_inv is None

    for got, ref in zip(got_parts, ref_parts):
        _assert_blockwise_tensor_equal(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not fp8_block_scaling_available,
    reason=reason_for_no_fp8_block_scaling,
)
@pytest.mark.parametrize("block_scaling_dim", [1, 2])
def test_group_quantize_fp8_blockwise_aligned_jagged_direct_matches_manual_loop(
    block_scaling_dim: int,
) -> None:
    """Aligned jagged direct grouped quantize matches independent FP8 blockwise quantize."""

    torch.manual_seed(1357)
    splits = [512, 1024, 1536, 2048]
    cols = 512
    inp = torch.randn(sum(splits), cols, dtype=torch.bfloat16, device="cuda")
    first_dims = torch.tensor(splits, dtype=torch.int64, device="cuda")
    quantizer = _make_quantizer(block_scaling_dim, rowwise=True, columnwise=True)

    assert first_dims.device.type == "cuda"
    assert cols % quantizer.block_len == 0
    assert sum(splits) % quantizer.block_len == 0
    assert all(split % quantizer.block_len == 0 for split in splits if split)

    grouped = tex.group_quantize(inp, quantizer, len(splits), first_dims)

    assert grouped.first_dims is not None
    assert grouped._fp8_row_block_offsets is not None
    assert grouped._fp8_row_block_offsets.shape == (
        len(splits) + 1,
        sum(splits) // quantizer.block_len,
    )
    assert grouped._fp8_rowwise_scale_inv_offsets is not None
    assert grouped._fp8_columnwise_scale_inv_offsets is not None
    assert grouped._fp8_rowwise_scale_inv_offsets.shape == (len(splits) + 1,)
    assert grouped._fp8_columnwise_scale_inv_offsets.shape == (len(splits) + 1,)
    assert grouped.scale_inv_offsets is not None
    assert grouped.columnwise_scale_inv_offsets is not None

    got_parts = grouped.split_into_quantized_tensors()
    ref_parts = _manual_quantize(torch.split(inp, splits), quantizer)

    for got, ref in zip(got_parts, ref_parts):
        _assert_blockwise_tensor_equal(got, ref)
        _assert_optional_equal("rowwise_scale_inv", got._rowwise_scale_inv, ref._rowwise_scale_inv)
        _assert_optional_equal(
            "columnwise_scale_inv",
            got._columnwise_scale_inv,
            ref._columnwise_scale_inv,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not fp8_block_scaling_available,
    reason=reason_for_no_fp8_block_scaling,
)
@pytest.mark.parametrize("block_scaling_dim", [1, 2])
def test_group_quantize_fp8_blockwise_uniform_no_first_dims_splits_correctly(
    block_scaling_dim: int,
) -> None:
    """Uniform direct grouped quantize exposes per-member blockwise tensor views."""

    torch.manual_seed(4321)
    num_tensors = 4
    rows_per_tensor = 129
    cols = 300
    inp = torch.randn(num_tensors * rows_per_tensor, cols, dtype=torch.bfloat16, device="cuda")
    quantizer = _make_quantizer(block_scaling_dim, rowwise=True, columnwise=True)

    grouped = tex.group_quantize(inp, quantizer, num_tensors)
    got_parts = grouped.split_into_quantized_tensors()
    ref_parts = _manual_quantize(torch.split(inp, rows_per_tensor), quantizer)

    assert grouped.first_dims is None
    assert grouped.tensor_shapes == [[rows_per_tensor, cols]] * num_tensors
    assert grouped.scale_inv_offsets is not None
    assert grouped.columnwise_scale_inv_offsets is not None

    for got, ref in zip(got_parts, ref_parts):
        assert tuple(got.shape) == (rows_per_tensor, cols)
        assert got._rowwise_scale_inv.shape == quantizer.get_scale_shape(got.shape, False)
        assert got._columnwise_scale_inv.shape == quantizer.get_scale_shape(got.shape, True)
        _assert_blockwise_tensor_equal(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not fp8_block_scaling_available,
    reason=reason_for_no_fp8_block_scaling,
)
@pytest.mark.parametrize("block_scaling_dim", [1, 2])
@pytest.mark.parametrize("rowwise,columnwise", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize(
    "splits",
    [[64, 128, 0, 192, 320], [128, 128, 128, 128], [128, 256, 384, 512]],
)
def test_split_quantize_fp8_blockwise_uses_grouped_layout(
    block_scaling_dim: int,
    rowwise: bool,
    columnwise: bool,
    splits: List[int],
) -> None:
    """Compatible split quantizers return the same views as the manual loop."""

    torch.manual_seed(5678)
    cols = 512
    inp = torch.randn(sum(splits), cols, dtype=torch.bfloat16, device="cuda")
    quantizers = [_make_quantizer(block_scaling_dim, rowwise, columnwise) for _ in splits]

    got_parts = tex.split_quantize(inp, splits, quantizers)
    ref_parts = _manual_quantize(torch.split(inp, splits), quantizers[0])

    for got, ref in zip(got_parts, ref_parts):
        _assert_blockwise_tensor_equal(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not fp8_block_scaling_available,
    reason=reason_for_no_fp8_block_scaling,
)
def test_split_quantize_fp8_blockwise_incompatible_quantizers_fall_back() -> None:
    """Incompatible FP8 blockwise split quantizers still match independent quantize calls."""

    torch.manual_seed(8765)
    splits = [128, 256, 128]
    cols = 512
    inp = torch.randn(sum(splits), cols, dtype=torch.bfloat16, device="cuda")
    quantizers = [
        _make_quantizer(1, rowwise=True, columnwise=True),
        _make_quantizer(2, rowwise=True, columnwise=True),
        _make_quantizer(1, rowwise=True, columnwise=False),
    ]

    got_parts = tex.split_quantize(inp, splits, quantizers)
    ref_parts = [
        _manual_quantize_one(part, quantizer)
        for part, quantizer in zip(torch.split(inp, splits), quantizers)
    ]

    for got, ref in zip(got_parts, ref_parts):
        _assert_blockwise_tensor_equal(got, ref)


def test_grouped_linear_fp8_block_weight_helper_fallbacks() -> None:
    """The grouped-weight helper declines cache, workspace, skip, debug, and mismatch cases."""

    weights = tuple(torch.randn(128, 128, dtype=torch.bfloat16) for _ in range(2))
    quantizers = [_make_quantizer(2, rowwise=True, columnwise=True) for _ in weights]

    def call(test_quantizers=None, **overrides):
        kwargs = {
            "debug": False,
            "cache_weight": False,
            "weight_workspaces": [None] * len(weights),
            "skip_fp8_weight_update": None,
        }
        kwargs.update(overrides)
        return grouped_linear_module._try_group_quantize_fp8_block_weights(
            weights,
            quantizers if test_quantizers is None else test_quantizers,
            **kwargs,
        )

    assert call(cache_weight=True) is None
    assert call(weight_workspaces=[object(), None]) is None
    assert call(skip_fp8_weight_update=torch.zeros((), dtype=torch.int32)) is None
    assert call(debug=True) is None
    assert call(
        test_quantizers=[
            _make_quantizer(2, rowwise=True, columnwise=True),
            _make_quantizer(1, rowwise=True, columnwise=True),
        ]
    ) is None
    assert call(test_quantizers=[quantizers[0], None]) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not fp8_block_scaling_available,
    reason=reason_for_no_fp8_block_scaling,
)
def test_grouped_linear_float8_block_scaling_grouped_weight_forward_backward(
    monkeypatch,
) -> None:
    """GroupedLinear reaches grouped weight quantize and matches the per-weight loop."""

    torch.manual_seed(2468)
    num_gemms = 4
    in_features = 256
    out_features = 256
    m_splits = [128] * num_gemms
    fp8_recipe = recipe.Float8BlockScaling()

    fast = GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        fuse_wgrad_accumulation=False,
        device="cuda",
    )
    ref = GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        fuse_wgrad_accumulation=False,
        device="cuda",
    )
    with torch.no_grad():
        for i in range(num_gemms):
            getattr(ref, f"weight{i}").copy_(getattr(fast, f"weight{i}"))

    inp = torch.randn(sum(m_splits), in_features, dtype=torch.bfloat16, device="cuda")
    with monkeypatch.context() as patch:
        patch.setattr(
            grouped_linear_module,
            "_try_group_quantize_fp8_block_weights",
            lambda *args, **kwargs: None,
        )
        ref_out, ref_inp_grad, ref_weight_grads = _run_grouped_linear(
            ref, inp, m_splits, fp8_recipe
        )

    group_quantize_calls = []
    split_quantize_calls = []
    original_group_quantize = grouped_linear_module.tex.group_quantize
    original_split_quantize = grouped_linear_module.tex.split_quantize

    def tracked_group_quantize(tensor, quantizer, num_tensors, first_dims=None, output=None):
        if isinstance(quantizer, Float8BlockQuantizer):
            group_quantize_calls.append(
                {
                    "shape": tuple(tensor.shape),
                    "num_tensors": num_tensors,
                    "first_dims": first_dims,
                    "block_scaling_dim": quantizer.block_scaling_dim,
                }
            )
        return original_group_quantize(tensor, quantizer, num_tensors, first_dims, output)

    def tracked_split_quantize(tensor, splits, quantizers, *args, **kwargs):
        split_quantize_calls.append(
            [getattr(quantizer, "block_scaling_dim", None) for quantizer in quantizers]
        )
        return original_split_quantize(tensor, splits, quantizers, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(grouped_linear_module.tex, "group_quantize", tracked_group_quantize)
        patch.setattr(grouped_linear_module.tex, "split_quantize", tracked_split_quantize)
        fast_out, fast_inp_grad, fast_weight_grads = _run_grouped_linear(
            fast, inp, m_splits, fp8_recipe
        )

    weight_calls = [
        call
        for call in group_quantize_calls
        if call["shape"] == (num_gemms * out_features, in_features)
    ]
    assert weight_calls
    assert weight_calls[0]["num_tensors"] == num_gemms
    assert weight_calls[0]["first_dims"] is None
    assert weight_calls[0]["block_scaling_dim"] == 2
    assert any(call == [1] * num_gemms for call in split_quantize_calls)

    torch.testing.assert_close(fast_out, ref_out, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(fast_inp_grad, ref_inp_grad, rtol=1e-3, atol=1e-2)
    for fast_grad, ref_grad in zip(fast_weight_grads, ref_weight_grads):
        torch.testing.assert_close(fast_grad, ref_grad, rtol=1e-3, atol=1e-2)
