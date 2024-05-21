# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess
from subprocess import CalledProcessError
import sys
import sysconfig
import platform
from typing import List, Optional, Tuple, Union

import setuptools

from te_version import te_version

# Project directory root
root_path: Path = Path(__file__).resolve().parent

@lru_cache(maxsize=1)
def with_debug_build() -> bool:
    """Whether to build with a debug configuration"""
    for arg in sys.argv:
        if arg == "--debug":
            sys.argv.remove(arg)
            return True
    if int(os.getenv("NVTE_BUILD_DEBUG", "0")):
        return True
    return False

# Call once in global scope since this function manipulates the
# command-line arguments. Future calls will use a cached value.
with_debug_build()

def found_cmake() -> bool:
    """"Check if valid CMake is available

    CMake 3.18 or newer is required.

    """

    # Check if CMake is available
    try:
        _cmake_bin = cmake_bin()
    except FileNotFoundError:
        return False

    # Query CMake for version info
    output = subprocess.run(
        [_cmake_bin, "--version"],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )
    match = re.search(r"version\s*([\d.]+)", output.stdout)
    version = match.group(1).split('.')
    version = tuple(int(v) for v in version)
    return version >= (3, 18)

def cmake_bin() -> Path:
    """Get CMake executable

    Throws FileNotFoundError if not found.

    """

    # Search in CMake Python package
    _cmake_bin: Optional[Path] = None
    try:
        import cmake
    except ImportError:
        pass
    else:
        cmake_dir = Path(cmake.__file__).resolve().parent
        _cmake_bin = cmake_dir / "data" / "bin" / "cmake"
        if not _cmake_bin.is_file():
            _cmake_bin = None

    # Search in path
    if _cmake_bin is None:
        _cmake_bin = shutil.which("cmake")
        if _cmake_bin is not None:
            _cmake_bin = Path(_cmake_bin).resolve()

    # Return executable if found
    if _cmake_bin is None:
        raise FileNotFoundError("Could not find CMake executable")
    return _cmake_bin

def found_ninja() -> bool:
    """"Check if Ninja is available"""
    return shutil.which("ninja") is not None

def found_pybind11() -> bool:
    """"Check if pybind11 is available"""

    # Check if Python package is installed
    try:
        import pybind11
    except ImportError:
        pass
    else:
        return True

    # Check if CMake can find pybind11
    if not found_cmake():
        return False
    try:
        subprocess.run(
            [
                str(cmake_bin()),
                "--find-package",
                "-DMODE=EXIST",
                "-DNAME=pybind11",
                "-DCOMPILER_ID=CXX",
                "-DLANGUAGE=CXX",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (CalledProcessError, OSError):
        pass
    else:
        return True
    return False

@lru_cache(maxsize=1)
def lib_extension():
    """Get the library file extension for the operating system"""
    system = platform.system()
    if system == "Linux":
        lib_ext = "so"
    elif system == "Darwin":
        lib_ext = "dylib"
    elif system == "Windows":
        lib_ext = "dll"
    else:
        raise RuntimeError(f"Unsupported operating system ({system})")
    return lib_ext

@lru_cache
def cuda_path() -> Tuple[str, str]:
    """CUDA root path and NVCC binary path as a tuple.

    Throws FileNotFoundError if NVCC is not found."""
    # Try finding NVCC
    nvcc_bin: Optional[Path] = None
    if nvcc_bin is None and os.getenv("CUDA_HOME"):
        # Check in CUDA_HOME
        cuda_home = Path(os.getenv("CUDA_HOME"))
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if nvcc_bin is None:
        # Check if nvcc is in path
        nvcc_bin = shutil.which("nvcc")
        if nvcc_bin is not None:
            cuda_home = Path(nvcc_bin.strip("/bin/nvcc"))
            nvcc_bin = Path(nvcc_bin)
    if nvcc_bin is None:
        # Last-ditch guess in /usr/local/cuda
        cuda_home = Path("/usr/local/cuda")
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if not nvcc_bin.is_file():
        raise FileNotFoundError(f"Could not find NVCC at {nvcc_bin}")

    return cuda_home, nvcc_bin

def cuda_version() -> Tuple[int, ...]:
    """CUDA Toolkit version as a (major, minor) tuple."""
    # Query NVCC for version info
    _, nvcc_bin = cuda_path()
    output = subprocess.run(
        [nvcc_bin, "-V"],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )
    match = re.search(r"release\s*([\d.]+)", output.stdout)
    version = match.group(1).split('.')
    return tuple(int(v) for v in version)

@lru_cache
def cudnn_path() -> Tuple[str, str]:
    """cuDNN include and library paths.

    Throws FileNotFoundError if libcudnn.so is not found."""
    assert os.path.exists(root_path / "3rdparty" / "cudnn-frontend"), (
        "Could not find cuDNN frontend API. Try running 'git submodule update --init --recursive' "
        "within the Transformer Engine source.")
    try:
        shutil.copy2(root_path / "3rdparty" / "cudnn-frontend" / "cmake" / "cuDNN.cmake",
                     root_path / "FindCUDNN.cmake")
        find_cudnn = subprocess.run(
            [
                str(cmake_bin()),
                "--find-package",
                f"-DCMAKE_MODULE_PATH={str(root_path)}",
                "-DCMAKE_FIND_DEBUG_MODE=ON",
                "-DMODE=EXIST",
                "-DCOMPILER_ID=CXX",
                "-DLANGUAGE=CXX",
                "-DNAME=CUDNN",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        os.remove(root_path / "FindCUDNN.cmake")
    except subprocess.CalledProcessError as e:
        raise FileNotFoundError("Could not find a cuDNN installation.") from e

    cudnn_include = None
    cudnn_link = None
    lib_ext = lib_extension()
    for line in find_cudnn.stderr.splitlines():
        if "cudnn.h" in line:
            cudnn_include = Path(line.lstrip()).parent
        elif "libcudnn." + lib_ext in line:
            cudnn_link = Path(line.lstrip()).parent
        if cudnn_include is not None and cudnn_link is not None:
            break
    if cudnn_include is None or cudnn_link is None:
        raise FileNotFoundError("Could not find a cuDNN installation.")

    return Path(cudnn_include), Path(cudnn_link)

@lru_cache(maxsize=1)
def with_userbuffers() -> bool:
    """Check if userbuffers support is enabled"""
    if int(os.getenv("NVTE_WITH_USERBUFFERS", "0")):
        assert os.getenv("MPI_HOME"), \
            "MPI_HOME must be set if NVTE_WITH_USERBUFFERS=1"
        return True
    return False

@lru_cache(maxsize=1)
def frameworks() -> List[str]:
    """DL frameworks to build support for"""
    _frameworks: List[str] = []
    supported_frameworks = ["pytorch", "jax", "paddle"]

    # Check environment variable
    if os.getenv("NVTE_FRAMEWORK"):
        _frameworks.extend(os.getenv("NVTE_FRAMEWORK").split(","))

    # Check command-line arguments
    for arg in sys.argv.copy():
        if arg.startswith("--framework="):
            _frameworks.extend(arg.replace("--framework=", "").split(","))
            sys.argv.remove(arg)

    # Detect installed frameworks if not explicitly specified
    if not _frameworks:
        try:
            import torch
        except ImportError:
            pass
        else:
            _frameworks.append("pytorch")
        try:
            import jax
        except ImportError:
            pass
        else:
            _frameworks.append("jax")
        try:
            import paddle
        except ImportError:
            pass
        else:
            _frameworks.append("paddle")

    # Special framework names
    if "all" in _frameworks:
        _frameworks = supported_frameworks.copy()
    if "none" in _frameworks:
        _frameworks = []

    # Check that frameworks are valid
    _frameworks = [framework.lower() for framework in _frameworks]
    for framework in _frameworks:
        if framework not in supported_frameworks:
            raise ValueError(
                f"Transformer Engine does not support framework={framework}"
            )

    return _frameworks

# Call once in global scope since this function manipulates the
# command-line arguments. Future calls will use a cached value.
frameworks()

def install_and_import(package):
    """Install a package via pip (if not already installed) and import into globals."""
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    finally:
        globals()[package] = importlib.import_module(package)

def setup_requirements() -> Tuple[List[str], List[str], List[str]]:
    """Setup Python dependencies

    Returns dependencies for build, runtime, and testing.

    """

    # Common requirements
    setup_reqs: List[str] = []
    install_reqs: List[str] = [
        "pydantic",
        "importlib-metadata>=1.0; python_version<'3.8'",
        "packaging",
    ]
    test_reqs: List[str] = ["pytest"]

    def add_unique(l: List[str], vals: Union[str, List[str]]) -> None:
        """Add entry to list if not already included"""
        if isinstance(vals, str):
            vals = [vals]
        for val in vals:
            if val not in l:
                l.append(val)

    # Requirements that may be installed outside of Python
    if not found_cmake():
        add_unique(setup_reqs, "cmake>=3.18")
    if not found_ninja():
        add_unique(setup_reqs, "ninja")

    # Framework-specific requirements
    if "pytorch" in frameworks():
        add_unique(install_reqs, ["torch", "flash-attn>=2.0.6,<=2.5.8,!=2.0.9,!=2.1.0"])
        add_unique(test_reqs, ["numpy", "onnxruntime", "torchvision"])
    if "jax" in frameworks():
        add_unique(install_reqs, ["jax", "flax>=0.7.1"])
        add_unique(test_reqs, ["numpy", "praxis"])
    if "paddle" in frameworks():
        add_unique(install_reqs, "paddlepaddle-gpu")
        add_unique(test_reqs, "numpy")

    return setup_reqs, install_reqs, test_reqs

# PyTorch extension modules require special handling
if "pytorch" in frameworks():
    from torch.utils.cpp_extension import BuildExtension
elif "paddle" in frameworks():
    from paddle.utils.cpp_extension import BuildExtension
else:
    install_and_import('pybind11')
    from pybind11.setup_helpers import build_ext as BuildExtension


class CMakeBuildExtension(BuildExtension):
    """Setuptools command with support for CMake extension modules"""

    def run(self) -> None:

        # Build CMake extensions
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                print(f"Building CMake extension {ext.name}")
                # Set up incremental builds for CMake extensions
                setup_dir = Path(__file__).resolve().parent
                build_dir = setup_dir / "build" / "cmake"
                build_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
                package_path = Path(self.get_ext_fullpath(ext.name))
                install_dir = package_path.resolve().parent
                ext._build_cmake(
                    build_dir=build_dir,
                    install_dir=install_dir,
                )

        # Paddle requires linker search path for libtransformer_engine.so
        paddle_ext = None
        if "paddle" in frameworks():
            for ext in self.extensions:
                if "paddle" in ext.name:
                    ext.library_dirs.append(self.build_lib)
                    paddle_ext = ext
                    break

        # Build non-CMake extensions as usual
        all_extensions = self.extensions
        self.extensions = [
            ext for ext in self.extensions
            if not isinstance(ext, CMakeExtension)
        ]
        super().run()
        self.extensions = all_extensions

        # Manually write stub file for Paddle extension
        if paddle_ext is not None:

            # Load libtransformer_engine.so to avoid linker errors
            for path in Path(self.build_lib).iterdir():
                if path.name.startswith("libtransformer_engine."):
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

            # Figure out stub file path
            module_name = paddle_ext.name
            assert module_name.endswith("_pd_"), \
                "Expected Paddle extension module to end with '_pd_'"
            stub_name = module_name[:-4]  # remove '_pd_'
            stub_path = os.path.join(self.build_lib, stub_name + ".py")

            # Figure out library name
            # Note: This library doesn't actually exist. Paddle
            # internally reinserts the '_pd_' suffix.
            so_path = self.get_ext_fullpath(module_name)
            _, so_ext = os.path.splitext(so_path)
            lib_name = stub_name + so_ext

            # Write stub file
            print(f"Writing Paddle stub for {lib_name} into file {stub_path}")
            from paddle.utils.cpp_extension.extension_utils import custom_write_stub
            custom_write_stub(lib_name, stub_path)

    def build_extensions(self):
        if "pytorch" not in frameworks() and "paddle" not in frameworks():
            # Ensure at least an empty list of flags for 'cxx' and 'nvcc' when
            # extra_compile_args is a dict.
            for extension in self.extensions:
                if isinstance(extension.extra_compile_args, dict):
                    for ext in ['cxx', 'nvcc']:
                        if ext not in extension.extra_compile_args:
                            extension.extra_compile_args[ext] = []

            # BuildExtension from PyTorch and Paddle already handle NVCC builds for .cu files,
            # but we need to customize it manually for the TE/JAX Pybind11Extension.
            # Add .cu to list of supported extensions.
            self.compiler.src_extensions += [ '.cu', '.cuh' ]

            # Store reference to default compiler_so and _compile methods
            # to dispatch non-CUDA compiles.
            default_compiler_so = self.compiler.compiler_so
            default_compile_fn = self.compiler._compile

            # Define new _compile_fn that uses NVCC_BIN and extra_compile_args['nvcc']
            # for .cu files, while preserving existing compiler with extra_compile_args['cxx']
            # for everything else.
            _, nvcc_bin = cuda_path()
            def _compile_fn(obj, src, ext, cc_args, extra_postargs, pp_opts):
                try:
                    if os.path.splitext(src)[1] in [ '.cu', '.cuh' ]:
                        self.compiler.set_executable('compiler_so', str(nvcc_bin))
                        if isinstance(extra_postargs, dict):
                            postargs = extra_postargs['nvcc']
                        else:
                            postargs = extra_postargs
                        postargs += ['--compiler-options', "'-fPIC'"]
                    else:
                        if isinstance(extra_postargs, dict):
                            postargs = extra_postargs['cxx']
                        else:
                            postargs = extra_postargs
                    default_compile_fn(obj, src, ext, cc_args, postargs, pp_opts)
                finally:
                    self.compiler.compiler_so = default_compiler_so

            # Set the redefined _compile_fn back into the compiler.
            self.compiler._compile = _compile_fn

        BuildExtension.build_extensions(self)


class CMakeExtension(setuptools.Extension):
    """CMake extension module"""

    def __init__(
            self,
            name: str,
            cmake_path: Path,
            cmake_flags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name, sources=[])  # No work for base class
        self.cmake_path: Path = cmake_path
        self.cmake_flags: List[str] = [] if cmake_flags is None else cmake_flags

    def _build_cmake(self, build_dir: Path, install_dir: Path) -> None:

        # Make sure paths are str
        _cmake_bin = str(cmake_bin())
        cmake_path = str(self.cmake_path)
        build_dir = str(build_dir)
        install_dir = str(install_dir)

        # CMake configure command
        build_type = "Debug" if with_debug_build() else "Release"
        configure_command = [
            _cmake_bin,
            "-S",
            cmake_path,
            "-B",
            build_dir,
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_INCLUDE_DIR={sysconfig.get_path('include')}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        configure_command += self.cmake_flags
        if found_ninja():
            configure_command.append("-GNinja")

        import pybind11
        pybind11_dir = Path(pybind11.__file__).resolve().parent
        pybind11_dir = pybind11_dir / "share" / "cmake" / "pybind11"
        configure_command.append(f"-Dpybind11_DIR={pybind11_dir}")

        # CMake build and install commands
        build_command = [_cmake_bin, "--build", build_dir]
        install_command = [_cmake_bin, "--install", build_dir]

        # Run CMake commands
        for command in [configure_command, build_command, install_command]:
            print(f"Running command {' '.join(command)}")
            try:
                subprocess.run(command, cwd=build_dir, check=True)
            except (CalledProcessError, OSError) as e:
                raise RuntimeError(f"Error when running CMake: {e}")


GLIBCXX_USECXX11_ABI = None

def setup_common_extension() -> CMakeExtension:
    """Setup CMake extension for common library

    Also builds JAX or userbuffers support if needed.

    """
    cmake_flags = []
    if with_userbuffers():
        cmake_flags.append("-DNVTE_WITH_USERBUFFERS=ON")
        # FindPythonInterp and FindPythonLibs are deprecated in newer CMake versions,
        # but PyBind11 still tries to use them unless we set PYBIND11_FINDPYTHON=ON.
        cmake_flags.append("-DPYBIND11_FINDPYTHON=ON")

    # If we need to build TE/PyTorch extensions later, we need to compile core TE library with
    # the same C++ ABI version as PyTorch.
    if "pytorch" in frameworks():
        import torch
        global GLIBCXX_USECXX11_ABI
        GLIBCXX_USECXX11_ABI = torch.compiled_with_cxx11_abi()
        cmake_flags.append(f"-DUSE_CXX11_ABI={int(GLIBCXX_USECXX11_ABI)}")

    return CMakeExtension(
        name="transformer_engine",
        cmake_path=root_path / "transformer_engine",
        cmake_flags=cmake_flags,
    )

def _all_files_in_dir(path):
    return list(path.iterdir())

def setup_pytorch_extension() -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files
    src_dir = root_path / "transformer_engine" / "pytorch" / "csrc"
    extensions_dir = src_dir / "extensions"
    sources = [
        src_dir / "common.cu",
        src_dir / "ts_fp8_op.cpp",
    ] + \
    _all_files_in_dir(extensions_dir)

    # Header files
    include_dirs = [
        root_path / "transformer_engine" / "common" / "include",
        root_path / "transformer_engine" / "pytorch" / "csrc",
        root_path / "transformer_engine",
        root_path / "3rdparty" / "cudnn-frontend" / "include",
    ]

    # Compiler flags
    cxx_flags = ["-O3"]
    nvcc_flags = [
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]

    # Version-dependent CUDA options
    try:
        version = cuda_version()
    except FileNotFoundError:
        print("Could not determine CUDA Toolkit version")
    else:
        if version >= (11, 2):
            nvcc_flags.extend(["--threads", "4"])
        if version >= (11, 0):
            nvcc_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
        if version >= (11, 8):
            nvcc_flags.extend(["-gencode", "arch=compute_90,code=sm_90"])

    # Link core TE library
    lib_kwargs = { 'libraries' : [ 'transformer_engine' ] }
    if lib_extension() == 'dll':
        # Windows can load from the same folder as the extension library but needs full DLL name.
        lib_kwargs['libraries'] = [ 'libtransformer_engine.dll' ]
    else:
        # Linux and maxOS support RPATH set to the extension library's $ORIGIN path.
        # This ensures we can dynamically load libtransformer_engine.so from the same path
        # that `transformer_engine-extensions.X-Y-Z.so` gets installed into by `pip`.
        lib_kwargs['extra_link_args'] = [ '-Wl,-rpath,$ORIGIN' ]

    # Add missing PyBind flags
    import pybind11
    include_dirs.append(pybind11.get_include())
    if lib_extension() == "dll":
        cxx_flags += [ "/EHsc", "/bigobj" ]
    else:
        cxx_flags += [ "-fvisibility=hidden" ]

    # userbuffers support
    if with_userbuffers():
        if os.getenv("MPI_HOME"):
            mpi_home = Path(os.getenv("MPI_HOME"))
            include_dirs.append(mpi_home / "include")
        cxx_flags.append("-DNVTE_WITH_USERBUFFERS")
        nvcc_flags.append("-DNVTE_WITH_USERBUFFERS")
        if lib_extension() == 'dll':
            lib_kwargs['libraries'].append('transformer_engine_userbuffers.dll')
        else:
            lib_kwargs['libraries'].append('transformer_engine_userbuffers')

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CUDAExtension
    return CUDAExtension(
        name="transformer_engine_extensions",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        **lib_kwargs
    )

def setup_jax_extension() -> setuptools.Extension:
    """Setup PyBind11 extension for JAX support"""
    # Source files
    src_dir = root_path / "transformer_engine" / "jax" / "csrc"
    sources = [
        src_dir / "extensions.cpp",
        src_dir / "modules.cpp",
        src_dir / "utils.cu",
    ]

    # Header files
    cuda_home, _ = cuda_path()
    cudnn_include, cudnn_link = cudnn_path()
    include_dirs = [
        cuda_home / "include",
        cudnn_include,
        root_path / "transformer_engine" / "common" / "include",
        root_path / "transformer_engine" / "jax" / "csrc",
        root_path / "transformer_engine",
    ]

    # Compile flags
    cxx_flags = [ "-O3" ]
    nvcc_flags = [
        "-O3",
        "--forward-unknown-opts",
    ]
    if GLIBCXX_USECXX11_ABI is not None:
        # Compile JAX extensions with same ABI as core library
        flag = f"-D_GLIBCXX_USE_CXX11_ABI={int(GLIBCXX_USECXX11_ABI)}"
        cxx_flags.append(flag)
        nvcc_flags.append(flag)

    # Linked libraries
    libraries = [
        'cudart',
        'cublas',
        'cublasLt',
        'cudnn',
    ]
    lib_kwargs = {}
    if lib_extension() == 'dll':
        # Windows needs explicit file paths for each library.
        cuda_lib_dir = cuda_home / 'lib' / 'x64'
        libraries = [ cuda_lib_dir / f'lib{name}.dll' for name in libraries ]

        # Core TE library is in the same folder as the extension library so it doesn't
        # need an absolute path, just the full file name.
        lib_kwargs['libraries'] = [ str(path) for path in libraries ] + \
                                  [ 'libtransformer_engine.dll' ]
    else:
        # Set link and runtime paths for CUDA libraries.
        cuda_lib_dir = cuda_home / 'lib64'
        if (not cuda_lib_dir.exists() and (cuda_home / 'lib').exists()):
            cuda_lib_dir = cuda_home / 'lib'
        lib_kwargs['libraries'] = libraries
        lib_kwargs['library_dirs'] = [
            str(cuda_lib_dir),
            str(cudnn_link),
        ]
        lib_kwargs['extra_link_args'] = [
            f"-Wl,-rpath,{path}" for path in lib_kwargs['library_dirs']
        ]
        # Add core TE library with RPATH same as the extension library's $ORIGIN dir.
        # This ensures we can dynamically load libtransformer_engine.so from the same path
        # that `transformer_engine-extensions.X-Y-Z.so` gets installed into by `pip`.
        lib_kwargs['libraries'] += [ 'transformer_engine' ]
        lib_kwargs['library_dirs'] += [ '.' ]
        lib_kwargs['extra_link_args'] += [ '-Wl,-rpath,$ORIGIN' ]

    # Define TE/JAX as a Pybind11Extension
    from pybind11.setup_helpers import Pybind11Extension

    class Pybind11CUDAExtension(Pybind11Extension):
        """Modified Pybind11Extension to allow combined CXX + NVCC compile flags."""

        def _add_cflags(self, flags: List[str]) -> None:
            if isinstance(self.extra_compile_args, dict):
                cxx_flags = self.extra_compile_args.pop('cxx', [])
                cxx_flags += flags
                self.extra_compile_args['cxx'] = cxx_flags
            else:
                self.extra_compile_args[:0] = flags

    return Pybind11CUDAExtension(
        "transformer_engine_jax",
        sources=[str(path) for path in sources],
        include_dirs=[str(path) for path in include_dirs],
        language=["c++"],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags
        },
        **lib_kwargs,
    )

def setup_paddle_extension() -> setuptools.Extension:
    """Setup CUDA extension for Paddle support"""

    # Source files
    src_dir = root_path / "transformer_engine" / "paddle" / "csrc"
    sources = [
        src_dir / "extensions.cu",
        src_dir / "common.cpp",
        src_dir / "custom_ops.cu",
    ]

    # Header files
    include_dirs = [
        root_path / "transformer_engine" / "common" / "include",
        root_path / "transformer_engine" / "paddle" / "csrc",
        root_path / "transformer_engine",
    ]

    # Compiler flags
    cxx_flags = ["-O3"]
    nvcc_flags = [
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]
    if GLIBCXX_USECXX11_ABI is not None:
        # Compile PaddlePaddle extensions with same ABI as core library
        flag = f"-D_GLIBCXX_USE_CXX11_ABI={int(GLIBCXX_USECXX11_ABI)}"
        cxx_flags.append(flag)
        nvcc_flags.append(flag)

    # Version-dependent CUDA options
    try:
        version = cuda_version()
    except FileNotFoundError:
        print("Could not determine CUDA Toolkit version")
    else:
        if version >= (11, 2):
            nvcc_flags.extend(["--threads", "4"])
        if version >= (11, 0):
            nvcc_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
        if version >= (11, 8):
            nvcc_flags.extend(["-gencode", "arch=compute_90,code=sm_90"])

    # Link core TE library
    lib_kwargs = { 'libraries' : [ 'transformer_engine' ] }
    if lib_extension() == 'dll':
        # Windows can load from the same folder as the extension library but needs full DLL name.
        lib_kwargs['libraries'] = [ 'libtransformer_engine.dll' ]
    else:
        # Linux and maxOS support RPATH set to the extension library's $ORIGIN path.
        # This ensures we can dynamically load libtransformer_engine.so from the same path
        # that `transformer_engine-extensions.X-Y-Z.so` gets installed into by `pip`.
        lib_kwargs['extra_link_args'] = [ '-Wl,-rpath,$ORIGIN' ]

    # Construct Paddle CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from paddle.utils.cpp_extension import CUDAExtension
    ext = CUDAExtension(
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        **lib_kwargs
    )
    ext.name = "transformer_engine_paddle_pd_"
    return ext

def main():

    # Submodules to install
    packages = setuptools.find_packages(
        include=["transformer_engine", "transformer_engine.*"],
    )

    # Dependencies
    setup_requires, install_requires, test_requires = setup_requirements()

    # Extensions
    ext_modules = [setup_common_extension()]
    if "pytorch" in frameworks():
        ext_modules.append(setup_pytorch_extension())

    if "jax" in frameworks():
        ext_modules.append(setup_jax_extension())

    if "paddle" in frameworks():
        ext_modules.append(setup_paddle_extension())

    # Configure package
    setuptools.setup(
        name="transformer_engine",
        version=te_version(),
        packages=packages,
        description="Transformer acceleration library",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension},
        setup_requires=setup_requires,
        install_requires=install_requires,
        extras_require={"test": test_requires},
        license_files=("LICENSE",),
    )


if __name__ == "__main__":
    main()
