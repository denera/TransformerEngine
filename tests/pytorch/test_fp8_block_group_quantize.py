# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped FP8 2D block-scaling quantize tests."""

import pytest
import torch

from transformer_engine.common import recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch import Float8BlockQuantizer
import transformer_engine_torch as tex


fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)


def _make_quantizer(rowwise: bool, columnwise: bool) -> Float8BlockQuantizer:
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=2,
    )


def _reference_loop(
    x: torch.Tensor,
    split_sections: list[int],
    rowwise: bool,
    columnwise: bool,
) -> list:
    refs = []
    for chunk in torch.split(x, split_sections):
        refs.append(_make_quantizer(rowwise, columnwise)(chunk.contiguous()))
    return refs


def _assert_tensor_or_none_equal(got, ref) -> None:
    if ref is None:
        assert got is None
        return
    assert got is not None
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    if ref.numel() > 0:
        torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)


def _assert_blockwise_outputs_equal(got, ref) -> None:
    _assert_tensor_or_none_equal(got._rowwise_data, ref._rowwise_data)
    _assert_tensor_or_none_equal(got._columnwise_data, ref._columnwise_data)
    _assert_tensor_or_none_equal(got._rowwise_scale_inv, ref._rowwise_scale_inv)
    _assert_tensor_or_none_equal(got._columnwise_scale_inv, ref._columnwise_scale_inv)


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
@pytest.mark.parametrize("rowwise,columnwise", [(True, False), (False, True), (True, True)])
def test_group_quantize_fp8_block_scaling_2d_matches_loop(rowwise, columnwise) -> None:
    split_sections = [129, 0, 255, 303]
    k_dim = 300
    torch.manual_seed(123)
    x = torch.randn((sum(split_sections), k_dim), dtype=torch.bfloat16, device="cuda")

    refs = _reference_loop(x, split_sections, rowwise, columnwise)
    first_dims = torch.tensor(split_sections, dtype=torch.int64, device="cuda")
    grouped = tex.group_quantize(
        x,
        _make_quantizer(rowwise, columnwise),
        len(split_sections),
        first_dims,
    )
    members = grouped.split_into_quantized_tensors()

    assert len(members) == len(refs)
    for got, ref in zip(members, refs):
        _assert_blockwise_outputs_equal(got, ref)


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
@pytest.mark.parametrize("split_sections", [[128, 128, 128, 128], [64, 193, 0, 319]])
def test_split_quantize_fp8_block_scaling_2d_matches_loop(split_sections) -> None:
    k_dim = 272
    torch.manual_seed(456)
    x = torch.randn((sum(split_sections), k_dim), dtype=torch.float16, device="cuda")
    rowwise = True
    columnwise = True

    refs = _reference_loop(x, split_sections, rowwise, columnwise)
    quantizers = [_make_quantizer(rowwise, columnwise) for _ in split_sections]
    outputs = tex.split_quantize(x, split_sections, quantizers)

    assert len(outputs) == len(refs)
    for got, ref in zip(outputs, refs):
        _assert_blockwise_outputs_equal(got, ref)


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_group_quantize_fp8_block_scaling_1d_stays_unsupported() -> None:
    x = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
    first_dims = torch.tensor([128, 128], dtype=torch.int64, device="cuda")
    quantizer = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=1,
    )

    with pytest.raises(RuntimeError, match="FP8 2D block"):
        tex.group_quantize(x, quantizer, 2, first_dims)


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_grouped_linear_fp8_block_scaling_uses_grouped_weight_quantize(monkeypatch) -> None:
    num_gemms = 4
    hidden_size = 256
    m_splits = [128, 128, 128, 128]
    grouped_weight_calls = []
    original_group_quantize = tex.group_quantize

    def wrapped_group_quantize(tensor, quantizer, num_tensors, first_dims):
        if (
            isinstance(quantizer, Float8BlockQuantizer)
            and quantizer.block_scaling_dim == 2
            and first_dims is None
        ):
            grouped_weight_calls.append((tuple(tensor.shape), num_tensors))
        return original_group_quantize(tensor, quantizer, num_tensors, first_dims)

    monkeypatch.setattr(tex, "group_quantize", wrapped_group_quantize)

    module = te.GroupedLinear(
        num_gemms,
        hidden_size,
        hidden_size,
        bias=False,
        params_dtype=torch.float16,
    ).cuda()
    x = torch.randn(
        (sum(m_splits), hidden_size),
        dtype=torch.float16,
        device="cuda",
        requires_grad=True,
    )

    with te.autocast(enabled=True, recipe=recipe.Float8BlockScaling()):
        out = module(x, m_splits)
    out.sum().backward()

    assert out.shape == (sum(m_splits), hidden_size)
    assert x.grad is not None
    assert any(
        shape == (num_gemms * hidden_size, hidden_size) and call_num_gemms == num_gemms
        for shape, call_num_gemms in grouped_weight_calls
    )
