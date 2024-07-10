#!/bin/bash

##########################################################################
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
##########################################################################

export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_DISABLE_NCCL=1 

# mpirun --allow-run-as-root -np 4 ./build/tests/ag_gemm -m 16 -n 16 -k 16 --csv 
for N in 32 64 128 256 512 1024 2048 4096 8192
do
    # mpirun --allow-run-as-root -np 4 ./build/tests/ag_gemm -m ${N} -n ${N} -k ${N} --csv | grep ">>>>"
    nsys profile --force-overwrite=true -o out_${N} mpirun --allow-run-as-root -np 4 ./build/tests/ag_gemm -m ${N} -n ${N} -k ${N}
done