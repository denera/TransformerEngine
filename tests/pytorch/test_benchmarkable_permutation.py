# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Benchmarkable PyTorch MoE permutation example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.benchmarkable import BenchmarkCase, BenchmarkSkip

try:
    import pytest
except ImportError:
    pytest = None


@dataclass
class _PermutationState:
    torch: Any
    te_permute: Any
    tokens: Any
    indices: Any
    num_out_tokens: int
    actual_bytes: int


def _require_backend():
    try:
        import torch
    except ImportError as exc:
        raise BenchmarkSkip("PyTorch is not installed.") from exc

    if not torch.cuda.is_available():
        raise BenchmarkSkip("CUDA is not available for the PyTorch permutation case.")

    try:
        import transformer_engine.pytorch  # noqa: F401
        from transformer_engine.pytorch import moe_permute
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise BenchmarkSkip("Transformer Engine PyTorch extension is not available.") from exc

    return torch, moe_permute


def _pytorch_permute_index_map(tokens, indices, num_out_tokens):
    if indices.dim() == 1:
        top_k = 1
    else:
        top_k = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch_argsort(flatten_indices)
    permuted_tokens = tokens.index_select(0, sorted_indices[:num_out_tokens] // top_k)
    return permuted_tokens, sorted_indices


def torch_argsort(tensor):
    return tensor.argsort(stable=True)


def _setup_case(params):
    torch, te_permute = _require_backend()
    dtype = getattr(torch, params["dtype"])
    generator = torch.Generator(device="cuda")
    generator.manual_seed(params["seed"])

    tokens = torch.rand(
        (params["num_tokens"], params["hidden_size"]),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    indices = torch.stack(
        [
            torch.randperm(params["num_experts"], device="cuda", generator=generator)[
                : params["top_k"]
            ]
            for _ in range(params["num_tokens"])
        ]
    ).to(torch.int32)
    num_out_tokens = params["num_out_tokens"] or params["num_tokens"] * params["top_k"]
    actual_bytes = (
        tokens.numel() * tokens.element_size()
        + indices.numel() * indices.element_size()
        + num_out_tokens * params["hidden_size"] * tokens.element_size()
    )
    return _PermutationState(torch, te_permute, tokens, indices, num_out_tokens, actual_bytes)


def _reference(state):
    return _pytorch_permute_index_map(state.tokens, state.indices, state.num_out_tokens)[0]


def _evaluate(state):
    return state.te_permute(
        state.tokens,
        state.indices,
        state.num_out_tokens,
        map_type="index",
    )[0]


def _check(state):
    expected = _reference(state)
    actual = _evaluate(state)
    state.torch.testing.assert_close(actual, expected)


def _bandwidth_gbps(state, median_ms):
    if median_ms <= 0:
        return 0.0
    return state.actual_bytes / (median_ms / 1000.0) / 1.0e9


def _make_case(params):
    return BenchmarkCase(
        case_id=(
            "pytorch.moe_permute.index."
            f"tokens{params['num_tokens']}.hidden{params['hidden_size']}."
            f"experts{params['num_experts']}.topk{params['top_k']}"
        ),
        framework="pytorch",
        component="moe",
        operation="moe_permute_index_map_forward",
        params=params,
        setup=lambda: _setup_case(params),
        reference=_reference,
        evaluate=_evaluate,
        check=_check,
        tags=("example", "moe", "permutation"),
        metrics={"bandwidth_GBps_actual_bytes": _bandwidth_gbps},
        unit_test="tests/pytorch/test_benchmarkable_permutation.py::test_benchmarkable_permutation",
        regression_threshold={"relative": 0.05, "absolute_ms": 0.01},
    )


BENCHMARKABLE_CASES = [
    _make_case(
        {
            "num_tokens": 128,
            "hidden_size": 128,
            "num_experts": 4,
            "top_k": 1,
            "num_out_tokens": None,
            "dtype": "float16",
            "map_type": "index",
            "seed": 1234,
        }
    ),
    _make_case(
        {
            "num_tokens": 1024,
            "hidden_size": 256,
            "num_experts": 8,
            "top_k": 1,
            "num_out_tokens": None,
            "dtype": "float16",
            "map_type": "index",
            "seed": 1234,
        }
    ),
]


def iter_benchmark_cases():
    return list(BENCHMARKABLE_CASES)


if pytest is not None:

    @pytest.mark.benchmarkable
    @pytest.mark.parametrize("case", BENCHMARKABLE_CASES, ids=lambda case: case.case_id)
    def test_benchmarkable_permutation(case):
        try:
            case.run_check()
        except BenchmarkSkip as exc:
            pytest.skip(str(exc))
