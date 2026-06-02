# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped FP8 2D block-scaling quantize tests."""

import pytest
import torch

from transformer_engine.common import recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch import Float8BlockQuantizer
from transformer_engine.pytorch.cpu_offload import get_cpu_offload_context
import transformer_engine.pytorch.module.grouped_linear as grouped_linear_mod
import transformer_engine_torch as tex


fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)


def _make_quantizer(
    rowwise: bool,
    columnwise: bool,
    fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
) -> Float8BlockQuantizer:
    return Float8BlockQuantizer(
        fp8_dtype=fp8_dtype,
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
    fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
) -> list:
    refs = []
    for chunk in torch.split(x, split_sections):
        refs.append(_make_quantizer(rowwise, columnwise, fp8_dtype)(chunk.contiguous()))
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
@pytest.mark.parametrize(
    "split_sections,k_dim,input_dtype,fp8_dtype",
    [
        ([129], 129, torch.float32, tex.DType.kFloat8E4M3),
        ([255, 303], 272, torch.float16, tex.DType.kFloat8E5M2),
        ([129, 0, 255, 303], 300, torch.bfloat16, tex.DType.kFloat8E4M3),
        ([64, 128, 0, 129, 255, 303, 1, 512], 129, torch.float32, tex.DType.kFloat8E5M2),
    ],
)
def test_group_quantize_fp8_block_scaling_2d_matches_loop(
    rowwise, columnwise, split_sections, k_dim, input_dtype, fp8_dtype
) -> None:
    torch.manual_seed(123)
    x = torch.randn((sum(split_sections), k_dim), dtype=input_dtype, device="cuda")

    refs = _reference_loop(x, split_sections, rowwise, columnwise, fp8_dtype)
    first_dims = torch.tensor(split_sections, dtype=torch.int64, device="cuda")
    grouped = tex.group_quantize(
        x,
        _make_quantizer(rowwise, columnwise, fp8_dtype),
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
def test_group_quantize_fp8_block_scaling_2d_rejects_non_contiguous_input() -> None:
    split_sections = [129, 255]
    base = torch.randn((300, sum(split_sections)), dtype=torch.bfloat16, device="cuda")
    x = base.transpose(0, 1)
    assert not x.is_contiguous()
    first_dims = torch.tensor(split_sections, dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="contiguous 2D tensor"):
        tex.group_quantize(x, _make_quantizer(True, True), len(split_sections), first_dims)


def _make_grouped_linear_module(num_gemms: int, hidden_size: int) -> te.GroupedLinear:
    return te.GroupedLinear(
        num_gemms,
        hidden_size,
        hidden_size,
        bias=False,
        params_dtype=torch.float16,
    ).cuda()


def _copy_grouped_linear_weights(
    dst: te.GroupedLinear,
    src: te.GroupedLinear,
    num_gemms: int,
) -> None:
    with torch.no_grad():
        for i in range(num_gemms):
            getattr(dst, f"weight{i}").copy_(getattr(src, f"weight{i}"))


def _run_grouped_linear(
    module: te.GroupedLinear,
    x: torch.Tensor,
    m_splits: list[int],
    grad: torch.Tensor,
    num_gemms: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    with te.autocast(enabled=True, recipe=recipe.Float8BlockScaling()):
        out = module(x, m_splits)
    out.backward(grad)
    weight_grads = [getattr(module, f"weight{i}").grad.detach().clone() for i in range(num_gemms)]
    return out.detach(), x.grad.detach().clone(), weight_grads


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_grouped_linear_fp8_block_scaling_matches_per_weight_fallback(monkeypatch) -> None:
    num_gemms = 4
    hidden_size = 256
    m_splits = [128, 128, 128, 128]
    torch.manual_seed(789)

    candidate = _make_grouped_linear_module(num_gemms, hidden_size)
    reference = _make_grouped_linear_module(num_gemms, hidden_size)
    _copy_grouped_linear_weights(reference, candidate, num_gemms)

    x = torch.randn(
        (sum(m_splits), hidden_size),
        dtype=torch.float16,
        device="cuda",
        requires_grad=True,
    )
    x_ref = x.detach().clone().requires_grad_(True)
    grad = torch.randn((sum(m_splits), hidden_size), dtype=torch.float16, device="cuda")

    grouped_weight_calls = 0
    original_try_group_quantize = grouped_linear_mod._try_group_quantize_fp8_block_weights

    def wrapped_try_group_quantize(*args, **kwargs):
        nonlocal grouped_weight_calls
        result = original_try_group_quantize(*args, **kwargs)
        if result is not None:
            grouped_weight_calls += 1
        return result

    monkeypatch.setattr(
        grouped_linear_mod, "_try_group_quantize_fp8_block_weights", wrapped_try_group_quantize
    )
    candidate_out, candidate_x_grad, candidate_weight_grads = _run_grouped_linear(
        candidate, x, m_splits, grad, num_gemms
    )
    assert grouped_weight_calls > 0

    monkeypatch.setattr(
        grouped_linear_mod,
        "_try_group_quantize_fp8_block_weights",
        lambda *args, **kwargs: None,
    )
    reference_out, reference_x_grad, reference_weight_grads = _run_grouped_linear(
        reference, x_ref, m_splits, grad, num_gemms
    )

    torch.testing.assert_close(candidate_out, reference_out, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(candidate_x_grad, reference_x_grad, atol=1e-3, rtol=1e-3)
    for candidate_grad, reference_grad in zip(candidate_weight_grads, reference_weight_grads):
        torch.testing.assert_close(candidate_grad, reference_grad, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_grouped_linear_cpu_offload_skips_grouped_weight_quantize(monkeypatch) -> None:
    num_gemms = 4
    hidden_size = 256
    m_splits = [128, 128, 128, 128]
    grouped_weight_attempts = []
    original_try_group_quantize = grouped_linear_mod._try_group_quantize_fp8_block_weights

    def wrapped_try_group_quantize(*args, **kwargs):
        grouped_weight_attempts.append(True)
        return original_try_group_quantize(*args, **kwargs)

    monkeypatch.setattr(
        grouped_linear_mod, "_try_group_quantize_fp8_block_weights", wrapped_try_group_quantize
    )

    module = _make_grouped_linear_module(num_gemms, hidden_size)
    x = torch.randn(
        (sum(m_splits), hidden_size),
        dtype=torch.float16,
        device="cuda",
        requires_grad=True,
    )

    offload_context, sync_function = get_cpu_offload_context(
        enabled=True,
        num_layers=1,
        model_layers=2,
    )
    with offload_context, te.autocast(enabled=True, recipe=recipe.Float8BlockScaling()):
        out = module(x, m_splits)
    out = sync_function(out)
    out.sum().backward()

    assert out.shape == (sum(m_splits), hidden_size)
    assert x.grad is not None
    assert grouped_weight_attempts == []
