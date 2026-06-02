# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for grouped FP8 block-scaling quantization."""

from typing import Iterable, List

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import Float8BlockQuantizer
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


def _assert_blockwise_tensor_equal(got, ref) -> None:
    _assert_optional_equal("rowwise_data", got._rowwise_data, ref._rowwise_data)
    _assert_optional_equal("columnwise_data", got._columnwise_data, ref._columnwise_data)
    _assert_optional_equal("rowwise_scale_inv", got._rowwise_scale_inv, ref._rowwise_scale_inv)
    _assert_optional_equal(
        "columnwise_scale_inv",
        got._columnwise_scale_inv,
        ref._columnwise_scale_inv,
    )


def _manual_quantize(parts: Iterable[torch.Tensor], quantizer) -> List[object]:
    return [tex.quantize(part.contiguous(), quantizer) for part in parts]


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
@pytest.mark.parametrize("rowwise,columnwise", [(True, False), (False, True), (True, True)])
def test_split_quantize_fp8_blockwise_uses_grouped_layout(
    block_scaling_dim: int,
    rowwise: bool,
    columnwise: bool,
) -> None:
    """Compatible split quantizers return the same views as the manual loop."""

    torch.manual_seed(5678)
    splits = [64, 128, 0, 192, 320]
    cols = 512
    inp = torch.randn(sum(splits), cols, dtype=torch.bfloat16, device="cuda")
    quantizers = [_make_quantizer(block_scaling_dim, rowwise, columnwise) for _ in splits]

    got_parts = tex.split_quantize(inp, splits, quantizers)
    ref_parts = _manual_quantize(torch.split(inp, splits), quantizers[0])

    for got, ref in zip(got_parts, ref_parts):
        _assert_blockwise_tensor_equal(got, ref)
