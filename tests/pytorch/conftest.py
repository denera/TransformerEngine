# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""PyTorch test configuration."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "benchmarkable: mark tests that expose benchmarks.benchmarkable BenchmarkCase objects.",
    )
