# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for grouped FP8 2D block-scaling quantization."""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch import Float8BlockQuantizer, GroupedLinear, autocast
import transformer_engine_torch as tex


fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)


def _make_quantizer(rowwise: bool = True, columnwise: bool = True) -> Float8BlockQuantizer:
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=2,
    )


def _ceildiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _valid_scale_tiles(scale: torch.Tensor, shape: torch.Size, columnwise: bool) -> torch.Tensor:
    row_tiles = _ceildiv(shape[0], 128)
    col_tiles = _ceildiv(shape[1], 128)
    if columnwise:
        return scale[:col_tiles, :row_tiles]
    return scale[:row_tiles, :col_tiles]


def _assert_same_quantized(actual, expected) -> None:
    for name in ("_rowwise_data", "_columnwise_data"):
        actual_tensor = getattr(actual, name, None)
        expected_tensor = getattr(expected, name, None)
        if actual_tensor is None or expected_tensor is None:
            assert actual_tensor is None and expected_tensor is None
            continue
        assert actual_tensor.shape == expected_tensor.shape
        assert torch.equal(actual_tensor, expected_tensor), name

    for name, columnwise in (
        ("_rowwise_scale_inv", False),
        ("_columnwise_scale_inv", True),
    ):
        actual_tensor = getattr(actual, name, None)
        expected_tensor = getattr(expected, name, None)
        if actual_tensor is None or expected_tensor is None:
            assert actual_tensor is None and expected_tensor is None
            continue
        assert actual_tensor.shape == expected_tensor.shape
        assert torch.equal(
            _valid_scale_tiles(actual_tensor, actual.shape, columnwise),
            _valid_scale_tiles(expected_tensor, expected.shape, columnwise),
        ), name


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
@pytest.mark.parametrize("rowwise,columnwise", [(True, False), (False, True), (True, True)])
def test_group_quantize_fp8_block_scaling_2d_matches_loop(rowwise: bool, columnwise: bool) -> None:
    splits = [129, 0, 255, 303]
    cols = 300
    x = torch.randn(sum(splits), cols, device="cuda", dtype=torch.bfloat16)
    first_dims = torch.tensor(splits, device="cuda", dtype=torch.int64)
    quantizer = _make_quantizer(rowwise=rowwise, columnwise=columnwise)

    grouped = tex.group_quantize(x, quantizer, len(splits), first_dims)
    actual_parts = grouped.split_into_quantized_tensors()

    for actual, split_x in zip(actual_parts, torch.split(x, splits)):
        if split_x.numel() == 0:
            assert actual.numel() == 0
            continue
        expected = tex.quantize(split_x, quantizer)
        _assert_same_quantized(actual, expected)


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_split_quantize_fp8_block_scaling_2d_matches_unfused_loop() -> None:
    splits = [17, 128, 129, 1, 256]
    cols = 129
    x = torch.randn(sum(splits), cols, device="cuda", dtype=torch.float16)
    quantizers = [_make_quantizer(rowwise=True, columnwise=True) for _ in splits]

    actual_parts = tex.split_quantize(x, splits, quantizers)
    expected_parts = [
        tex.quantize(split_x, quantizer)
        for split_x, quantizer in zip(torch.split(x, splits), quantizers)
    ]

    for actual, expected in zip(actual_parts, expected_parts):
        _assert_same_quantized(actual, expected)


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_split_quantize_fp8_block_scaling_1d_stays_supported_by_fallback() -> None:
    splits = [128, 128]
    x = torch.randn(sum(splits), 128, device="cuda", dtype=torch.bfloat16)
    quantizers = [
        Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            force_pow_2_scales=True,
            amax_epsilon=0.0,
            block_scaling_dim=1,
        )
        for _ in splits
    ]

    outputs = tex.split_quantize(x, splits, quantizers)
    assert len(outputs) == len(splits)
    assert [tuple(out.shape) for out in outputs] == [(split, x.shape[1]) for split in splits]


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_grouped_linear_uses_grouped_fp8_block_scaling_2d_weight_path(monkeypatch) -> None:
    num_groups = 4
    in_features = 128
    out_features = 256
    split_sizes = [64, 96, 32, 128]
    x = torch.randn(sum(split_sizes), in_features, device="cuda", dtype=torch.bfloat16)
    fp8_recipe = recipe.Float8BlockScaling()

    seen_grouped_block_scaling_2d = {"value": False}
    orig_group_quantize = tex.group_quantize

    def wrapped_group_quantize(tensor, quantizer, num_tensors, first_dims):
        if (
            isinstance(quantizer, Float8BlockQuantizer)
            and quantizer.block_scaling_dim == 2
            and first_dims is None
            and num_tensors == num_groups
        ):
            seen_grouped_block_scaling_2d["value"] = True
        return orig_group_quantize(tensor, quantizer, num_tensors, first_dims)

    monkeypatch.setattr(tex, "group_quantize", wrapped_group_quantize)

    module = GroupedLinear(
        num_groups,
        in_features,
        out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        single_grouped_weight=True,
    ).cuda()

    with autocast(enabled=True, recipe=fp8_recipe):
        out = module(x, split_sizes)

    assert out.shape == (sum(split_sizes), out_features)
    assert seen_grouped_block_scaling_2d["value"]
