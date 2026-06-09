# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped FP8 block-scaling quantization tests."""

import math
from typing import List

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import Float8BlockQuantizer


fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)


def _make_quantizer(block_scaling_dim: int, rowwise: bool, columnwise: bool) -> Float8BlockQuantizer:
    return Float8BlockQuantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=block_scaling_dim,
    )


def _valid_scale_mask(
    quantizer: Float8BlockQuantizer,
    rows: int,
    cols: int,
    *,
    columnwise: bool,
) -> torch.Tensor:
    scale_shape = quantizer.get_scale_shape((rows, cols), columnwise)
    mask = torch.zeros(scale_shape, dtype=torch.bool, device="cuda")
    if quantizer.block_scaling_dim == 1:
        if columnwise:
            mask[:, :cols] = True
        else:
            mask[:, :rows] = True
    else:
        if columnwise:
            mask[:, : math.ceil(rows / quantizer.block_len)] = True
        else:
            mask[:, : math.ceil(cols / quantizer.block_len)] = True
    return mask


def _assert_scale_matches(
    got: torch.Tensor,
    expected: torch.Tensor,
    quantizer: Float8BlockQuantizer,
    rows: int,
    cols: int,
    *,
    columnwise: bool,
) -> None:
    assert got.shape == expected.shape
    if got.numel() == 0:
        return
    mask = _valid_scale_mask(quantizer, rows, cols, columnwise=columnwise)
    torch.testing.assert_close(got[mask], expected[mask], atol=0.0, rtol=0.0)


def _make_inputs(rows_per_tensor: List[int], cols: int, dtype: torch.dtype) -> List[torch.Tensor]:
    torch.manual_seed(1234)
    tensors = []
    for i, rows in enumerate(rows_per_tensor):
        tensor = torch.randn((rows, cols), dtype=dtype, device="cuda")
        tensors.append(tensor * float(i + 1))
    return tensors


def _check_group_quantize_case(
    *,
    block_scaling_dim: int,
    rows_per_tensor: List[int],
    cols: int,
    dtype: torch.dtype,
    rowwise: bool,
    columnwise: bool,
) -> None:
    quantizer = _make_quantizer(block_scaling_dim, rowwise, columnwise)
    tensors = _make_inputs(rows_per_tensor, cols, dtype)
    grouped_input = torch.cat(tensors, dim=0) if tensors else torch.empty((0, cols), device="cuda")
    first_dims = torch.tensor(rows_per_tensor, dtype=torch.int64, device="cuda")

    grouped = tex.group_quantize(grouped_input, quantizer, len(rows_per_tensor), first_dims)
    split_grouped = grouped.split_into_quantized_tensors()

    assert len(split_grouped) == len(tensors)
    assert grouped.scale_inv_offsets is None or len(grouped.scale_inv_offsets) == len(tensors) + 1
    assert (
        grouped.columnwise_scale_inv_offsets is None
        or len(grouped.columnwise_scale_inv_offsets) == len(tensors) + 1
    )

    data_offset = 0
    rowwise_scale_offset = 0
    columnwise_scale_offset = 0
    for tensor, output in zip(tensors, split_grouped):
        rows = tensor.shape[0]
        expected = quantizer(tensor)
        numel = rows * cols

        if rowwise:
            torch.testing.assert_close(
                output._rowwise_data.view(dtype=torch.uint8),
                expected._rowwise_data.view(dtype=torch.uint8),
                atol=0.0,
                rtol=0.0,
            )
            assert output._rowwise_data.data_ptr() == grouped.rowwise_data.data_ptr() + data_offset
            _assert_scale_matches(
                output._rowwise_scale_inv,
                expected._rowwise_scale_inv,
                quantizer,
                rows,
                cols,
                columnwise=False,
            )
            assert (
                output._rowwise_scale_inv.data_ptr()
                == grouped.scale_inv.data_ptr() + rowwise_scale_offset * 4
            )
            rowwise_scale_offset += output._rowwise_scale_inv.numel()

        if columnwise:
            torch.testing.assert_close(
                output._columnwise_data.view(dtype=torch.uint8),
                expected._columnwise_data.view(dtype=torch.uint8),
                atol=0.0,
                rtol=0.0,
            )
            assert (
                output._columnwise_data.data_ptr()
                == grouped.columnwise_data.data_ptr() + data_offset
            )
            _assert_scale_matches(
                output._columnwise_scale_inv,
                expected._columnwise_scale_inv,
                quantizer,
                rows,
                cols,
                columnwise=True,
            )
            assert (
                output._columnwise_scale_inv.data_ptr()
                == grouped.columnwise_scale_inv.data_ptr() + columnwise_scale_offset * 4
            )
            columnwise_scale_offset += output._columnwise_scale_inv.numel()

        data_offset += numel


@pytest.mark.skipif(
    not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling
)
@pytest.mark.parametrize(
    "block_scaling_dim,rows_per_tensor,cols,dtype,rowwise,columnwise",
    [
        (1, [128], 128, torch.bfloat16, True, False),
        (1, [127, 128], 272, torch.float16, False, True),
        (1, [1, 129, 256, 7], 256, torch.bfloat16, True, True),
        (2, [128] * 8, 128, torch.float16, True, False),
        (2, [64, 512, 127, 129, 256, 3, 1, 128], 272, torch.bfloat16, False, True),
        (
            2,
            [1, 2, 17, 64, 127, 128, 129, 255, 256, 257, 3, 5, 8, 13, 21, 34],
            512,
            torch.bfloat16,
            True,
            True,
        ),
    ],
)
def test_group_quantize_matches_manual_loop(
    block_scaling_dim: int,
    rows_per_tensor: List[int],
    cols: int,
    dtype: torch.dtype,
    rowwise: bool,
    columnwise: bool,
) -> None:
    _check_group_quantize_case(
        block_scaling_dim=block_scaling_dim,
        rows_per_tensor=rows_per_tensor,
        cols=cols,
        dtype=dtype,
        rowwise=rowwise,
        columnwise=columnwise,
    )


@pytest.mark.skipif(
    not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling
)
def test_split_quantize_uses_grouped_fp8_block_path() -> None:
    rows_per_tensor = [127, 128, 129, 64]
    cols = 256
    tensors = _make_inputs(rows_per_tensor, cols, torch.bfloat16)
    grouped_input = torch.cat(tensors, dim=0)
    quantizers = [_make_quantizer(2, True, True) for _ in rows_per_tensor]

    outputs = tex.split_quantize(grouped_input, rows_per_tensor, quantizers)

    assert len(outputs) == len(tensors)
    for tensor, output, quantizer in zip(tensors, outputs, quantizers):
        expected = quantizer(tensor)
        torch.testing.assert_close(
            output._rowwise_data.view(dtype=torch.uint8),
            expected._rowwise_data.view(dtype=torch.uint8),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output._columnwise_data.view(dtype=torch.uint8),
            expected._columnwise_data.view(dtype=torch.uint8),
            atol=0.0,
            rtol=0.0,
        )
        _assert_scale_matches(
            output._rowwise_scale_inv,
            expected._rowwise_scale_inv,
            quantizer,
            tensor.shape[0],
            cols,
            columnwise=False,
        )
        _assert_scale_matches(
            output._columnwise_scale_inv,
            expected._columnwise_scale_inv,
            quantizer,
            tensor.shape[0],
            cols,
            columnwise=True,
        )
