# Transformer Engine

NVIDIA library for accelerating Transformer models with FP8/MXFP8/NVFP4 precision on NVIDIA GPUs.
Supports PyTorch and JAX backends. C++ common library with Python bindings via pybind11/XLA FFI.

## Build

```bash
# Build all frameworks (auto-detected)
NVTE_USE_CCACHE=1 pip install --verbose --no-build-isolation --editable .

# Build only PyTorch
NVTE_FRAMEWORK=pytorch NVTE_USE_CCACHE=1 pip install --verbose --no-build-isolation --editable .

# Build only JAX
NVTE_FRAMEWORK=jax NVTE_USE_CCACHE=1 pip install --verbose --no-build-isolation --editable .

# Control parallelism
NVTE_BUILD_MAX_JOBS=8 NVTE_BUILD_THREADS_PER_JOB=4 NVTE_USE_CCACHE=1 \
  pip install --verbose --no-build-isolation --editable .
```

When ccache is enabled, set `CCACHE_DIR=/tmp/${USER}_nvte_ccache` and `CCACHE_MAXSIZE=10G` if
desired.

## Clean

```bash
# Remove build artifacts (build/, *.so, *.egg-info)
rm -rf build/ build_tools/build/ *.egg-info *.so transformer_engine/*.so

# Remove Python/pytest caches
find . -name __pycache__ -type d -exec rm -rf {} + && rm -rf .pytest_cache

# Remove pip installations (local user site)
rm -rf "$(python3 -m site --user-site)"/*transformer*engine*

# Remove pip installations (system site)
rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')"/*transformer*engine*
```

## Testing

```bash
# PyTorch unit tests
TE_PATH=$(pwd) bash qa/L0_pytorch_unittest/test.sh

# JAX unit tests (non-distributed)
TE_PATH=$(pwd) bash qa/L0_jax_unittest/test.sh

# C++ unit tests (builds with cmake first)
TE_PATH=$(pwd) bash qa/L0_cppunittest/test.sh

# Run individual tests directly
python3 -m pytest tests/pytorch/test_sanity.py
python3 -m pytest -c tests/jax/pytest.ini tests/jax/ -k 'not distributed'
```

All `qa/L0_*/test.sh` scripts default `TE_PATH=/opt/transformerengine`; always set it to `$(pwd)` locally.
JAX tests need `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.

## Linting

```bash
TE_PATH=$(pwd) bash qa/L0_pytorch_lint/test.sh   # C++ (cpplint) + Python (pylint), PyTorch
TE_PATH=$(pwd) bash qa/L0_jax_lint/test.sh       # C++ + Python, JAX
CPP_ONLY=1    TE_PATH=$(pwd) bash qa/L0_pytorch_lint/test.sh
PYTHON_ONLY=1 TE_PATH=$(pwd) bash qa/L0_jax_lint/test.sh
```

Pinned versions: `cpplint==1.6.0`, `pylint==3.3.1`.

## Architecture

```
transformer_engine/
  common/               # C++ CUDA kernels, framework-agnostic → libtransformer_engine.so
    include/            # Public C++ API headers
    comm_gemm/          # GEMM + communication overlap kernels (cuBLASMp backend)
    comm_gemm_overlap/  # GEMM + communication overlap kernels (Userbuffers backend)
    fused_attn/         # Fused attention backends
    gemm/               # GEMM kernels
    normalization/      # LayerNorm/RMSNorm
  pytorch/              # PyTorch Python API + pybind11 C++ extensions
    ops/                # Composable ops (ops/fused/ for autograd-fused paths)
    module/             # nn.Module wrappers (TransformerLayer, Linear, etc.)
    tensor/             # Custom tensor types (Float8Tensor, NVFp4Tensor, etc.)
    cpp_extensions/     # PyTorch C++ extensions
    csrc/               # C++ pybind11 bindings
  jax/                  # JAX Python API + XLA custom calls
    cpp_extensions/     # JAX primitive definitions (BasePrimitive subclasses)
    csrc/               # C++ pybind11 bindings (XLA FFI handlers)
    flax/               # Flax module wrappers
tests/
  pytorch/              # PyTorch pytest suite
  jax/                  # JAX pytest suite
  cpp/                  # C++ googletest suite (cmake in tests/cpp/build/)
  cpp_distributed/      # Distributed C++ tests (NCCL-based)
qa/                     # CI test scripts: L0=unit, L1=distributed, L2=extended
examples/               # Runnable examples (jax/, pytorch/)
```

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `NVTE_FRAMEWORK` | Frameworks to build: `pytorch`, `jax`, or `pytorch,jax` |
| `NVTE_CMAKE_BUILD_DIR` | Reuse CMake build dir for incremental builds |
| `NVTE_USE_CCACHE=1` | Enable ccache (set `CCACHE_DIR` and `CCACHE_MAXSIZE` as needed) |
| `NVTE_BUILD_DEBUG=1` | Debug build with `-g`, no optimizations |
| `NVTE_FUSED_ATTN=0` | Disable fused attention (useful for test isolation) |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | Force deterministic algorithms |
| `NVTE_TORCH_COMPILE=0` | Disable torch.compile |
| `NVTE_UB_WITH_MPI=1` | Enable MPI userbuffers (requires `MPI_HOME`) |
| `NVTE_WITH_CUBLASMP=1` | Enable the cuBLASMp backend for communication + GEMM overlap (requires `CUBLAMP_HOME`) |
| `NVTE_WITH_NVSHMEM=1` | Enable NVSHMEM for communication (requires `NVSHMEM_HOME`) |

Full list: `docs/envvars.rst`

## Gotchas

- **Import order for C++ bindings**: The low-level C++ extension modules (`transformer_engine_jax`,
  `transformer_engine_torch`) depend on `libtransformer_engine.so` which is loaded by the parent
  package. Always `import transformer_engine.jax` or `import transformer_engine.pytorch` before
  importing `transformer_engine_jax` or `transformer_engine_torch` directly. The parent import
  loads the common shared library that the extension module links against.
- **Commits must be signed-off**: always use `git commit -s` (DCO required).
- **C++ style**: Google C++ Style Guide enforced by cpplint.
- **License header required** on all new files; exempt files listed in `qa/L0_license/config.json`.
- **JAX XLA FFI**: New ops need a `BasePrimitive` subclass in `jax/cpp_extensions/` and a C++
  handler registered in `jax/csrc/extensions/pybind.cpp`.
- **Framework auto-detection**: build detects installed frameworks; use `NVTE_FRAMEWORK` to override
  when both are installed but you only want to build one.
