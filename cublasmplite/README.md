# Requirements
- MPI (`MPI_HOME`)
- NVSHMEM (`NVSHMEM_HOME`)
- NCCL (`NCCL_HOME`)
- CUDA Toolkit (`CUDACXX=...`) for NVCC and cuBLAS

# Build
```
mkdir build
cd build
CXX=g++-11 MPI_HOME=/usr/ cmake -DCMAKE_CUDA_ARCHITECTURES=80-real -DNVSHMEM_HOME=/home/scratch.lcambier_ent/nvshmem-2.11.0/ -DNCCL_HOME=/home/scratch.lcambier_ent/nccl_2.21.5-1+cuda12.4_x86_64/ -DCMAKE_CUDA_FLAGS="-Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Werror" -DCMAKE_INSTALL_PREFIX=$(pwd)/../install ..
VERBOSE=1 make -j8 install
```

# Run

## AG+GEMM
```
$ mpirun -np 2 -tag-output -x NVSHMEM_REMOTE_TRANSPORT=none -x NVSHMEM_BOOTSTRAP=MPI -x NVSHMEM_DISABLE_NCCL=1 ./tests/ag_gemm -m 32 -n 64 -k 128
[1,1]<stdout>:MPI Hello from 1/2
[1,0]<stdout>:MPI Hello from 0/2
[1,0]<stdout>:AG+GEMM:
[1,0]<stdout>:num_ranks 2
[1,0]<stdout>:m 32
[1,0]<stdout>:n 64
[1,0]<stdout>:k 128
[1,0]<stdout>:cycles 10
[1,0]<stdout>:skip 5
[1,0]<stdout>:Performance:
[1,0]<stdout>:NVSHMEM (max) 0.047718 ms
[1,0]<stdout>:NVSHMEM (average) 0.046848 ms
[1,0]<stdout>:NCCL (max) 0.026214 ms
[1,0]<stdout>:NCCL (average) 0.026163 ms
[1,0]<stdout>:PASSED
```

## GEMM+RS
```
$ mpirun -np 2 -tag-output -x NVSHMEM_REMOTE_TRANSPORT=none -x NVSHMEM_BOOTSTRAP=MPI -x NVSHMEM_DISABLE_NCCL=1 ./tests/gemm_rs -m 32 -n 64 -k 128
[1,0]<stdout>:MPI Hello from 0/2
[1,1]<stdout>:MPI Hello from 1/2
[1,0]<stdout>:GEMM+RS:
[1,0]<stdout>:num_ranks 2
[1,0]<stdout>:m 32
[1,0]<stdout>:n 64
[1,0]<stdout>:k 128
[1,0]<stdout>:cycles 10
[1,0]<stdout>:skip 5
[1,0]<stdout>:Performance:
[1,0]<stdout>:NVSHMEM (max) 0.051712 ms
[1,0]<stdout>:NVSHMEM (average) 0.051610 ms
[1,0]<stdout>:NCCL (max) 0.021914 ms
[1,0]<stdout>:NCCL (average) 0.021914 ms
[1,0]<stdout>:PASSED
```