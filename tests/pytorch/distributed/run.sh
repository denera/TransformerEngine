#!/bin/bash

set -eux
set -o pipefail

export CUDA_MODULE_LOADING=EAGER
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_DISABLE_NCCL=1
export NVSHMEM_REMOTE_TRANSPORT=none
export LD_LIBRARY_PATH=/workdir/:/workdir/libnvshmem_3.0.12-0+cuda12.3_x86_64/lib:$LD_LIBRARY_PATH
ngpus=4

for s in 2048 4096 8192
do
	for n in 64 96
	do 
		for backend in nvshmem user_buffers
		do

			echo "skipmc 0 b 2 s ${s} n ${n} d 128 p2p ag bf16 split ${backend}"
			torchrun --nproc-per-node=${ngpus} tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s ${s} -n ${n} -d 128 --p2p --comm-type ag                --warmup-iters 5 --timing-iters 10 --backend ${backend}
			
			echo "skipmc 0 b 2 s ${s} n ${n} d 128 p2p rs bf16 split ${backend}"
			torchrun --nproc-per-node=${ngpus} tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s ${s} -n ${n} -d 128 --p2p --comm-type rs                --warmup-iters 5 --timing-iters 10 --backend ${backend}
			
			echo "skipmc 0 b 2 s ${s} n ${n} d 128 p2p ag fp8 split ${backend}"
			torchrun --nproc-per-node=${ngpus} tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s ${s} -n ${n} -d 128 --p2p --comm-type ag --fp8          --warmup-iters 5 --timing-iters 10 --backend ${backend}

			echo "skipmc 0 b 2 s ${s} n ${n} d 128 p2p rs fp8 split ${backend}"
			torchrun --nproc-per-node=${ngpus} tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s ${s} -n ${n} -d 128 --p2p --comm-type rs --fp8          --warmup-iters 5 --timing-iters 10 --backend ${backend}
			
			echo "skipmc 0 b 2 s ${s} n ${n} d 128 p2p rs fp8 atomic ${backend}"
			torchrun --nproc-per-node=${ngpus} tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s ${s} -n ${n} -d 128 --p2p --comm-type rs --fp8 --atomic --warmup-iters 5 --timing-iters 10 --backend ${backend}
		
			if [[ "$backend" == "user_buffers" ]]

				echo "skipmc 1 b 2 s ${s} n ${n} d 128 p2p ag bf16 split ${backend}"
				UB_SKIPMC=1 torchrun --nproc-per-node=${ngpus} tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s ${s} -n ${n} -d 128 --p2p --comm-type ag                --warmup-iters 5 --timing-iters 10 --backend ${backend}
				
				echo "skipmc 1 b 2 s ${s} n ${n} d 128 p2p rs bf16 split ${backend}"
				UB_SKIPMC=1 torchrun --nproc-per-node=${ngpus} tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s ${s} -n ${n} -d 128 --p2p --comm-type rs                --warmup-iters 5 --timing-iters 10 --backend ${backend}
				
				echo "skipmc 1 b 2 s ${s} n ${n} d 128 p2p ag fp8 split ${backend}"
				UB_SKIPMC=1 torchrun --nproc-per-node=${ngpus} tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s ${s} -n ${n} -d 128 --p2p --comm-type ag --fp8          --warmup-iters 5 --timing-iters 10 --backend ${backend}

				echo "skipmc 1 b 2 s ${s} n ${n} d 128 p2p rs fp8 split ${backend}"
				UB_SKIPMC=1 torchrun --nproc-per-node=${ngpus} tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s ${s} -n ${n} -d 128 --p2p --comm-type rs --fp8          --warmup-iters 5 --timing-iters 10 --backend ${backend}
				
				echo "skipmc 1 b 2 s ${s} n ${n} d 128 p2p rs fp8 atomic ${backend}"
				UB_SKIPMC=1 torchrun --nproc-per-node=${ngpus} tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics -b 2 -s ${s} -n ${n} -d 128 --p2p --comm-type rs --fp8 --atomic --warmup-iters 5 --timing-iters 10 --backend ${backend}

			fi
		done
	done
done

