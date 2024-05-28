# Requirements
- MPI (`MPI_HOME`)
- NVSHMEM (`NVSHMEM_HOME`)
- NCCL (`NCCL_HOME`)
- CUDA Toolkit (`CUDACXX=...`) for NVCC and cuBLAS

# Build
```
mkdir build
cd build
CXX=g++-11 MPI_HOME=/usr/ CUDACXX=/usr/local/cuda/bin/nvcc cmake -DCMAKE_CUDA_ARCHITECTURES=80-real -DNVSHMEM_HOME=/home/scratch.lcambier_ent/nvshmem-2.11.0/ -DNCCL_HOME=/home/scratch.lcambier_ent/nccl_2.21.5-1+cuda12.4_x86_64/ -DCMAKE_CUDA_FLAGS="-Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Werror" ..
make -j
```

# Run

## AG+GEMM
```
$ LD_LIBRARY_PATH=/home/scratch.lcambier_ent/nvshmem-2.11.0/lib:/usr/lib/x86_64-linux-gnu/openmpi/lib/:$LD_LIBRARY_PATH mpirun -np 2 -tag-output -x NVSHMEM_REMOTE_TRANSPORT=none -x CUDA_DEVICE_MAX_CONNECTIONS=1 -x NVSHMEM_BOOTSTRAP=MPI -x NVSHMEM_DISABLE_NCCL=1 ./tester -m 1024 -n 1024 -k 1024
[1,0]<stdout>:MPI Hello from 0/2
[1,1]<stdout>:MPI Hello from 1/2
[1,0]<stdout>:AG+GEMM:
[1,0]<stdout>:num_ranks 2
[1,0]<stdout>:m 1024
[1,0]<stdout>:n 1024
[1,0]<stdout>:k 1024
[1,0]<stdout>:cycles 10
[1,0]<stdout>:skip 5
[1,0]<stdout>:Performance:
[1,0]<stdout>:NVSHMEM (max) 0.080384 ms
[1,0]<stdout>:NVSHMEM (average) 0.080333 ms
[1,0]<stdout>:NCCL (max) 0.107008 ms
[1,0]<stdout>:NCCL (average) 0.107008 ms
[1,0]<stdout>:PASSED
```

## GEMM+RS
```
$ LD_LIBRARY_PATH=/home/scratch.lcambier_ent/nvshmem-2.11.0/lib:/usr/lib/x86_64-linux-gnu/openmpi/lib/:$LD_LIBRARY_PATH mpirun -np 4 -tag-output -x NVSHMEM_REMOTE_TRANSPORT=none -x CUDA_DEVICE_MAX_CONNECTIONS=1 -x NVSHMEM_BOOTSTRAP=MPI -x NVSHMEM_DISABLE_NCCL=1 ./gemm_rs -m 1024 -n 1024 -k 1024
[1,3]<stdout>:MPI Hello from 3/4
[1,0]<stdout>:MPI Hello from 0/4
[1,1]<stdout>:MPI Hello from 1/4
[1,2]<stdout>:MPI Hello from 2/4
[1,0]<stdout>:GEMM+RS:
[1,0]<stdout>:num_ranks 4
[1,0]<stdout>:m 1024
[1,0]<stdout>:n 1024
[1,0]<stdout>:k 1024
[1,0]<stdout>:cycles 10
[1,0]<stdout>:skip 5
[1,0]<stdout>:Performance:
[1,0]<stdout>:NVSHMEM (max) 0.138138 ms
[1,0]<stdout>:NVSHMEM (average) 0.137779 ms
[1,0]<stdout>:NCCL (max) 0.091750 ms
[1,0]<stdout>:NCCL (average) 0.091674 ms
[1,0]<stdout>:PASSED
```