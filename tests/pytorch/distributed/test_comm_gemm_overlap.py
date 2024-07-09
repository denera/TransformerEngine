# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import os
import subprocess
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

RNG_SEED: int = 1234
NUM_PROCS: int = torch.cuda.device_count()
SEQ_LENGTH: int = 2024
BATCH_SIZE: int = 2
NUM_HEADS: int = 64
HEAD_DIM: int = 128

TEST_PATH = Path(__file__).parent.resolve() / "run_gemm_with_overlap.py"
TEST_CMD_BASE = [
    "torchrun",
    f"--nproc-per-node={min(torch.cuda.device_count(), 4)}",
    str(TEST_PATH),
    "-s",
    str(SEQ_LENGTH),
    "-b",
    str(BATCH_SIZE),
    "-n",
    str(NUM_HEADS),
    "-d",
    str(HEAD_DIM),
    "--seed",
    str(RNG_SEED),
    "--check-numerics",
    "--warmup-iters",
    str(0),
    "--timing-iters",
    str(1),
    "--verbose",
]

# Sync GPU launch order to the host CPU
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# Fall back on CUDA IPC if the platform does not support CUDA multicast
if not tex.comm_overlap_supports_multicast():
    os.environ["UB_SKIPMC"] = "1"


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Comm+GEMM overlap requires at least 2 GPUs.")
@pytest.mark.parametrize(
    "fp8,p2p,comm_type,aggregate,atomic,bulk",
    [
        # FP8, P2P, Type, Aggregate, Atomic
        (False, True, "AG", False, False, False),
        (False, True, "AG", True, False, False),
        (True, True, "AG", False, False, False),
        (True, True, "AG", True, False), False,
        (False, False, "RS", False, False, False),
        (False, True, "RS", False, False, False),
        (True, False, "RS", False, False, False),
        (True, True, "RS", False, False, False),
        # (True, False, "RS", False, True, False),
        (True, True, "RS", False, True, False),
        (False, False, "AG", False, False, True),
        (False, False, "RS", False, False, True)
    ],
    ids=[
        " AG + SPLIT GEMM | BF16 | RING-EXCHANGE",
        " AG + SPLIT GEMM | BF16 | 2X AGGREGATED RING-EXCHANGE",
        " AG + SPLIT GEMM | FP8  | RING-EXCHANGE",
        " AG + SPLIT GEMM | FP8  | 2X AGGREGATED RING-EXCHANGE",
        " SPLIT GEMM + RS | BF16 | PIPELINE",
        " SPLIT GEMM + RS | BF16 | RING-EXCHANGE",
        " SPLIT GEMM + RS | FP8  | PIPELINE",
        " SPLIT GEMM + RS | FP8  | RING-EXCHANGE",
        # "ATOMIC GEMM + RS | FP8  | PIPELINE",
        "ATOMIC GEMM + RS | FP8  | RING-EXCHANGE",
        "  BULK AG + GEMM | BF16 | PIPELINE",
        "  GEMM + BULK RS | BF16 | PIPELINE"
    ],
)
def test_overlap_algos(fp8, p2p, comm_type, aggregate, atomic, bulk):
    """
    Test comm+GEMM overlap algorithms with direct calls to
    te.cpp_extensions.gemm or te.cpp_extensions.fp8_gemm
    """
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    test_cmd = TEST_CMD_BASE + ["--comm-type", comm_type]
    if bulk:
        test_cmd.append("--bulk-overlap")
    else:
        if fp8:
            test_cmd.append("--fp8")
        if p2p:
            test_cmd.append("--p2p")
        if aggregate:
            test_cmd.append("--aggregate")
        if atomic:
            test_cmd.append("--atomic")
    assert not bool(subprocess.call(test_cmd, env=os.environ))
