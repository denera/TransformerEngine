#!/bin/bash

set -eux
set -o pipefail

export NVSHMEM_DISABLE_CUDA_VMM=1 
export NVSHMEM_DISABLE_NCCL=1 
export NVSHMEM_REMOTE_TRANSPORT=none 
#export LD_LIBRARY_PATH=/workdir/TransformerEngine/:/workdir/libnvshmem_2.11.0-5+cuda12.0_x86_64/lib:$LD_LIBRARY_PATH 
export LD_LIBRARY_PATH=/workdir/TransformerEngine/:/workdir/libnvshmem_3.0.12-0+cuda12.3_x86_64/lib:$LD_LIBRARY_PATH 

# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=0 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
# 
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
# NVTE_NVSHMEM_WAIT=1 torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem

torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend nvshmem
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend nvshmem

torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend user_buffers 
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend user_buffers 
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend user_buffers 
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend user_buffers 
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 64 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend user_buffers 
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 96 -d 128 --p2p --comm-type ag --warmup-iters 5 --timing-iters 10 --backend user_buffers 
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend user_buffers 
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 2048 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend user_buffers 
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend user_buffers 
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 4096 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend user_buffers 
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 64 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend user_buffers 
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s 8192 -n 96 -d 128 --p2p --comm-type rs --warmup-iters 5 --timing-iters 10 --backend user_buffers 


