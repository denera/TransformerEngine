# Quickstart - standalone AG+GEMM or GEMM+RS tests

## Requirements
- MPI (`MPI_HOME`)
- NVSHMEM (`NVSHMEM_HOME`)
- NCCL (`NCCL_HOME`)
- CUDA Toolkit (`CUDACXX=...`) for NVCC and cuBLAS

## Build
```
mkdir cublasmplite/build
cd cublasmplite/build
CXX=g++-11 MPI_HOME=/usr/ cmake -DCMAKE_CUDA_ARCHITECTURES=80-real -DNVSHMEM_HOME=/home/scratch.lcambier_ent/nvshmem-2.11.0/ -DNCCL_HOME=/home/scratch.lcambier_ent/nccl_2.21.5-1+cuda12.4_x86_64/ ..
make -j8
```

## Run

### AG+GEMM
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

### GEMM+RS
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

# Quickstart - inside of TransformerEngine

## Requirements
- MPI (`MPI_HOME`)
- NVSHMEM (`NVSHMEM_HOME`)
- NCCL (`NCCL_HOME`)
- CUDA Toolkit (`CUDACXX=...`) for NVCC and cuBLAS

MPI and NCCL are only used by the tests.

## Git clone, copy NVSHMEM + start docker

Clone repo
```
git clone -b lcambier/ub_nvshmem --recurse-submodules ssh://git@gitlab-master.nvidia.com:12051/lcambier/TransformerEngine.git TransformerEngine
cd TransformerEngine
```

Install NVSHMEM
```
wget https://developer.download.nvidia.com/compute/redist/nvshmem/2.11.0/builds/cuda12/txz/agnostic/x64/libnvshmem_2.11.0-5+cuda12.0_x86_64.txz
tar -xvf libnvshmem_2.11.0-5+cuda12.0_x86_64.txz
```
NVSHMEM is now installed in `libnvshmem_2.11.0-5+cuda12.0_x86_64`.

Start docker
```
docker run -it -v $(pwd):/workdir --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all gitlab-master.nvidia.com:5005/dl/dgx/pytorch:master-py3-devel bash -i
```

## Build cuBLASMplite

`cuBLASMplite` (naming is hard) is a small library that encapsultes NVSHMEM ops and a little more.

```
mkdir -p /workdir/cublasmplite/build
cd /workdir/cublasmplite/build
NVSHMEM_HOME=/workdir/libnvshmem_2.11.0-5+cuda12.0_x86_64 cmake -DCMAKE_INSTALL_PREFIX=/workdir/cublasmplite/install  -DCMAKE_CUDA_ARCHITECTURES=90-real ..
make install -j8
```
cuBLASMplite is now installed in `/workdir/cublasmplite/install/`.

## Build TE + cuBLASMplite

```
cd /workdir
CPATH=/workdir/cublasmplite/install/include:$CPATH CUBLASMPLITE_HOME=/workdir/cublasmplite/install/ NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/opt/hpcx/ompi/ pip install --verbose -e .[test]
```

## Run tests

With UB
```
root@64642578d1c9:/workdir# UB_SKIPMC=1 LD_LIBRARY_PATH=/workdir/libnvshmem_2.11.0-5+cuda12.0_x86_64/lib:/workdir/cublasmplite/install/lib/:$LD_LIBRARY_PATH torchrun --nproc-per-node=4 examples/pytorch/comm_gemm_overlap/test_gemm.py --check-numerics --p2p --comm-type ag
W0529 16:36:48.919000 140329987028096 torch/distributed/run.py:778]
W0529 16:36:48.919000 140329987028096 torch/distributed/run.py:778] *****************************************
W0529 16:36:48.919000 140329987028096 torch/distributed/run.py:778] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0529 16:36:48.919000 140329987028096 torch/distributed/run.py:778] *****************************************
Rank 1/4, hello from test_gemm.py
Rank 3/4, hello from test_gemm.py
Rank 0/4, hello from test_gemm.py
Rank 2/4, hello from test_gemm.py
UB_TIMEOUT is set to 110 sec, 89100000000 cycles, freq: 810000khz
MC NOT initialized and used
!!! [UB] communicator initialized
!!! [UB] registered buffer 1
[rank:2] input: [1024, 8192]  | kernel_1: [8192, 8192] | output: [4096, 8192]
[rank:0] input: [1024, 8192]  | kernel_1: [8192, 8192] | output: [4096, 8192]
[rank:3] input: [1024, 8192]  | kernel_1: [8192, 8192] | output: [4096, 8192]
[rank:1] input: [1024, 8192]  | kernel_1: [8192, 8192] | output: [4096, 8192]
[GLOBAL] inp_g: [4096, 8192]  | ker_g: [8192, 32768] | out_g: [4096, 32768] | ref_g: [4096, 32768]
PASSED
```

With NVSHMEM
```
root@64642578d1c9:/workdir# NVTE_NVSHMEM=1 NVSHMEM_DISABLE_NCCL=1 NVSHMEM_REMOTE_TRANSPORT=none LD_LIBRARY_PATH=/workdir/libnvshmem_2.11.0-5+cuda12.0_x86_64/lib:/workdir/cublasmplite/install/lib/:$LD_LIBRARY_PATH torchrun --nproc-per-node=4 examples/pytorch/comm_gemm_overlap/test_gemm.py --check-numerics --p2p --comm-type ag
W0529 16:37:12.250000 140499262506112 torch/distributed/run.py:778]
W0529 16:37:12.250000 140499262506112 torch/distributed/run.py:778] *****************************************
W0529 16:37:12.250000 140499262506112 torch/distributed/run.py:778] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0529 16:37:12.250000 140499262506112 torch/distributed/run.py:778] *****************************************
Rank 1/4, hello from test_gemm.py
Rank 3/4, hello from test_gemm.py
Rank 2/4, hello from test_gemm.py
Rank 0/4, hello from test_gemm.py
UID bootstrap network already initialized using:  eth0:172.18.0.11<0>

UID bootstrap network already initialized using:  eth0:172.18.0.11<0>

UID bootstrap network already initialized using:  eth0:172.18.0.11<0>

UID bootstrap network already initialized using:  eth0:172.18.0.11<0>
UID bootstrap network already initialized using:  eth0:172.18.0.11<0>


[rank:1] input: [1024, 8192]  | kernel_1: [8192, 8192] | output: [4096, 8192]
[rank:2] input: [1024, 8192]  | kernel_1: [8192, 8192] | output: [4096, 8192]
[rank:3] input: [1024, 8192]  | kernel_1: [8192, 8192] | output: [4096, 8192]
[rank:0] input: [1024, 8192]  | kernel_1: [8192, 8192] | output: [4096, 8192]
[GLOBAL] inp_g: [4096, 8192]  | ker_g: [8192, 32768] | out_g: [4096, 32768] | ref_g: [4096, 32768]
PASSED
```