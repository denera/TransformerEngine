# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import os
import subprocess
import copy
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

RNG_SEED: int = 1234
SEQ_LENGTH: int = 2024
BATCH_SIZE: int = 2
NUM_HEADS: int = 64
HEAD_DIM: int = 128

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = min(torch.cuda.device_count(), 4)
TORCHRUN_CMD = [
    "torchrun",
    f"--nproc_per_node={NUM_PROCS}"
]

# Force GPU kernels to launch in the order they're executed by the host CPU
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# Fall back on CUDA IPC if the platform does not support CUDA multicast
if not tex.device_supports_multicast():
    os.environ["UB_SKIPMC"] = "1"


@pytest.mark.skipif(NUM_PROCS < 2, reason="Comm+GEMM overlap requires at least 2 GPUs.")
@pytest.mark.parametrize(
    "fp8,p2p,comm_type,aggregate,atomic,bulk,backend",
    [
        # FP8, P2P, Type, Aggregate, Atomic, bulk, backend
        (False, True, "AG", False, False, False, 'user_buffers'),
        (False, True, "AG", True, False, False, 'user_buffers'),
        (True, True, "AG", False, False, False, 'user_buffers'),
        (True, True, "AG", True, False, False, 'user_buffers'),
        (False, False, "RS", False, False, False, 'user_buffers'),
        (False, True, "RS", False, False, False, 'user_buffers'),
        (True, False, "RS", False, False, False, 'user_buffers'),
        (True, True, "RS", False, False, False, 'user_buffers'),
        # (True, False, "RS", False, True, False, 'user_buffers'),
        (True, True, "RS", False, True, False, 'user_buffers'),
        (False, False, "AG", False, False, True, 'user_buffers'),
        (False, False, "RS", False, False, True, 'user_buffers'),

        (False, True, "AG", False, False, False, 'nvshmem'),
        (False, True, "RS", False, False, False, 'nvshmem'),
        (True, True, "AG", False, False, False, 'nvshmem'),
        (True, True, "RS", False, False, False, 'nvshmem'),
    ],
    ids=[
        " UB      | AG + SPLIT GEMM | BF16 | RING-EXCHANGE ",
        " UB      | AG + SPLIT GEMM | BF16 | 2X AGGREGATED RING-EXCHANGE ",
        " UB      | AG + SPLIT GEMM | FP8  | RING-EXCHANGE ",
        " UB      | AG + SPLIT GEMM | FP8  | 2X AGGREGATED RING-EXCHANGE ",
        " UB      | SPLIT GEMM + RS | BF16 | PIPELINE ",
        " UB      | SPLIT GEMM + RS | BF16 | RING-EXCHANGE ",
        " UB      | SPLIT GEMM + RS | FP8  | PIPELINE ",
        " UB      | SPLIT GEMM + RS | FP8  | RING-EXCHANGE ",
        # " UB      | ATOMIC GEMM + RS | FP8  | PIPELINE ",
        " UB      | ATOMIC GEMM + RS | FP8  | RING-EXCHANGE ",
        " UB      |   BULK AG + GEMM | BF16 | PIPELINE ",
        " UB      |   GEMM + BULK RS | BF16 | PIPELINE ",

        " NVSHMEM | AG + SPLIT GEMM | BF16 | RING-EXCHANGE ",
        " NVSHMEM | RS + SPLIT GEMM | BF16 | RING-EXCHANGE ",
        " NVSHMEM | AG + SPLIT GEMM | FP8  | RING-EXCHANGE ",
        " NVSHMEM | RS + SPLIT GEMM | FP8  | RING-EXCHANGE ",
    ],
)
def test_gemm_with_overlap(fp8, p2p, comm_type, aggregate, atomic, bulk, backend):
    """
    Test comm+GEMM overlap algorithms with direct calls to
    te.cpp_extensions.gemm or te.cpp_extensions.fp8_gemm
    """
    test_path = TEST_ROOT / "run_gemm_with_overlap.py"
    test_cmd = TORCHRUN_CMD + [ str(test_path) ] + [
        f"--seq-length={SEQ_LENGTH}",
        f"--batch-size={BATCH_SIZE}",
        f"--num-heads={NUM_HEADS}",
        f"--head-dim={HEAD_DIM}",
        f"--seed={RNG_SEED}",
        "--check-numerics",
        "--warmup-iters=0",
        "--timing-iters=1",
        "--verbose",
        f"--comm-type={comm_type}",
        f"--backend={backend}",
    ]

    if bulk:
        test_cmd.append("--bulk-overlap")
    else:
        if fp8:
            if not fp8_available:
                pytest.skip(reason_for_no_fp8)
            test_cmd.append("--fp8")
        if p2p:
            test_cmd.append("--p2p")
        if aggregate:
            test_cmd.append("--aggregate")
        if atomic:
            if torch.cuda.get_device_properties(0).major < 9:
                pytest.skip("Atomic GEMM requires device compute capability 9.0 or higher.")
            test_cmd.append("--atomic")

    assert not bool(subprocess.call(test_cmd, env=os.environ))


# @pytest.mark.skipif(NUM_PROCS < 2, reason="Comm+GEMM overlap requires at least 2 GPUs.")
# def test_transformer_layer_with_overlap():
#     """Test TransformerLayer with comm+GEMM overlap enabled in all layers."""
#     test_path = TEST_ROOT / "run_transformer_layer_with_overlap.py"
#     test_cmd = TORCHRUN_CMD + [ str(test_path) ] + [
#         f"--seq-length={SEQ_LENGTH}",
#         f"--batch-size={BATCH_SIZE}",
#         f"--num-heads={NUM_HEADS}",
#         f"--head-dim={HEAD_DIM}",
#         "--no-grad"
#     ]

#     assert not bool(subprocess.call(test_cmd, env=os.environ))
