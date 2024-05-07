#!/bin/bash

set -euo pipefail

mpiexec \
-np 2 \
-x PATH \
-x MASTER_ADDR=$(hostname) \
-x MASTER_PORT=40587 \
-x CUDA_DEVICE_MAX_CONNECTIONS=1 \
-x NVTE_BIAS_GELU_NVFUSION=0 \
-x UB_SKIPMC=1 \
-bind-to none \
-map-by slot \
-mca pml ob1 \
-mca btl ^openib \
python test_ln_mlp.py --comm-overlap
