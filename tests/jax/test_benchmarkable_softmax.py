# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Benchmarkable JAX softmax example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.benchmarkable import BenchmarkCase, BenchmarkSkip

try:
    import pytest
except ImportError:
    pytest = None


@dataclass
class _SoftmaxState:
    jnp: Any
    nn: Any
    lax: Any
    softmax: Any
    logits: Any
    mask: Any
    scale_factor: float
    fusion_type: Any
    actual_bytes: int
    dtype_name: str


def _require_backend(params):
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax, nn
    except ImportError as exc:
        raise BenchmarkSkip("JAX is not installed.") from exc

    try:
        from transformer_engine.jax.cpp_extensions import is_softmax_kernel_available
        from transformer_engine.jax.cpp_extensions.attention import AttnSoftmaxType
        from transformer_engine.jax.softmax import SoftmaxFusionType, softmax
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise BenchmarkSkip("Transformer Engine JAX extension is not available.") from exc

    dtype = getattr(jnp, params["dtype"])
    fusion_type = getattr(SoftmaxFusionType, params["softmax_fusion_type"])
    if not is_softmax_kernel_available(
        fusion_type,
        AttnSoftmaxType.VANILLA_SOFTMAX,
        params["batch_size"],
        params["num_heads"],
        params["max_seqlen_q"],
        params["max_seqlen_kv"],
        dtype,
    ):
        raise BenchmarkSkip("TE fused softmax kernel is not available for this configuration.")

    return jax, jnp, lax, nn, softmax, fusion_type


def _setup_case(params):
    jax, jnp, lax, nn, softmax, fusion_type = _require_backend(params)
    dtype = getattr(jnp, params["dtype"])
    key = jax.random.PRNGKey(params["seed"])
    logits_key, mask_key = jax.random.split(key, 2)
    logits_shape = (
        params["batch_size"],
        params["num_heads"],
        params["max_seqlen_q"],
        params["max_seqlen_kv"],
    )
    mask_shape = (
        params["batch_size"],
        1,
        params["max_seqlen_q"],
        params["max_seqlen_kv"],
    )
    logits = jax.random.uniform(logits_key, logits_shape, dtype, -1.0, 1.0)
    if params["softmax_fusion_type"] == "SCALED":
        mask = None
        mask_bytes = 0
    else:
        mask = jax.random.bernoulli(mask_key, shape=mask_shape).astype(jnp.uint8)
        mask_bytes = mask.size
    actual_bytes = logits.size * logits.dtype.itemsize * 2 + mask_bytes
    return _SoftmaxState(
        jnp=jnp,
        nn=nn,
        lax=lax,
        softmax=softmax,
        logits=logits,
        mask=mask,
        scale_factor=params["scale_factor"],
        fusion_type=fusion_type,
        actual_bytes=actual_bytes,
        dtype_name=params["dtype"],
    )


def _reference(state):
    logits = state.logits
    if state.mask is not None:
        logits = logits + state.lax.select(
            state.mask > 0,
            state.jnp.full(state.mask.shape, -1e10).astype(logits.dtype),
            state.jnp.full(state.mask.shape, 0.0).astype(logits.dtype),
        )
    return state.nn.softmax(logits * state.scale_factor)


def _evaluate(state):
    return state.softmax(state.logits, state.mask, state.scale_factor, state.fusion_type)


def _check(state):
    import numpy as np

    expected = _reference(state)
    actual = _evaluate(state)
    expected.block_until_ready()
    actual.block_until_ready()
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2.0e-2,
        atol=1.0e-3 if state.dtype_name == "float16" else 2.0e-2,
    )


def _bandwidth_gbps(state, median_ms):
    if median_ms <= 0:
        return 0.0
    return state.actual_bytes / (median_ms / 1000.0) / 1.0e9


def _make_case(params):
    return BenchmarkCase(
        case_id=(
            "jax.softmax.forward."
            f"b{params['batch_size']}.h{params['num_heads']}."
            f"q{params['max_seqlen_q']}.kv{params['max_seqlen_kv']}."
            f"{params['softmax_fusion_type'].lower()}.{params['dtype']}"
        ),
        framework="jax",
        component="softmax",
        operation="fused_softmax_forward",
        params=params,
        setup=lambda: _setup_case(params),
        reference=_reference,
        evaluate=_evaluate,
        check=_check,
        tags=("example", "softmax"),
        metrics={"bandwidth_GBps_actual_bytes": _bandwidth_gbps},
        unit_test="tests/jax/test_benchmarkable_softmax.py::test_benchmarkable_softmax",
        regression_threshold={"relative": 0.05, "absolute_ms": 0.01},
    )


BENCHMARKABLE_CASES = [
    _make_case(
        {
            "batch_size": 8,
            "num_heads": 16,
            "max_seqlen_q": 16,
            "max_seqlen_kv": 16,
            "scale_factor": 0.125,
            "softmax_fusion_type": "SCALED",
            "dtype": "float16",
            "seed": 0,
        }
    ),
    _make_case(
        {
            "batch_size": 8,
            "num_heads": 16,
            "max_seqlen_q": 512,
            "max_seqlen_kv": 512,
            "scale_factor": 0.125,
            "softmax_fusion_type": "SCALED",
            "dtype": "float16",
            "seed": 0,
        }
    ),
]


def iter_benchmark_cases():
    return list(BENCHMARKABLE_CASES)


if pytest is not None:

    @pytest.mark.benchmarkable
    @pytest.mark.parametrize("case", BENCHMARKABLE_CASES, ids=lambda case: case.case_id)
    def test_benchmarkable_softmax(case):
        try:
            case.run_check()
        except BenchmarkSkip as exc:
            pytest.skip(str(exc))
