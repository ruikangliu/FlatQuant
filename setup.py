from setuptools import setup
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import pathlib
setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent

def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

def get_cuda_arch_flags():
    return [
        '-gencode', 'arch=compute_75,code=sm_75',  # Turing
        '-gencode', 'arch=compute_80,code=sm_80',  # Ampere
        '-gencode', 'arch=compute_86,code=sm_86',  # Ampere
    ]
    
def third_party_cmake(extra_pip_flags=None):
    import subprocess, sys, shutil
    
    cmake = shutil.which('cmake')
    if cmake is None:
            raise RuntimeError('Cannot find CMake executable.')

    retcode = subprocess.call([
        cmake, 
        "-DCMAKE_CUDA_ARCHITECTURES=75;80;86",
        HERE
    ])
    if retcode != 0:
        sys.stderr.write("Error: CMake configuration failed.\n")
        sys.exit(1)

    # install fast hadamard transform
    hadamard_dir = os.path.join(HERE, 'third-party/fast-hadamard-transform')
    pip = shutil.which('pip')
    
    # Build pip command with base flags
    pip_cmd = [pip, 'install', '-e', hadamard_dir]
    
    # Add extra flags if provided
    if extra_pip_flags:
        pip_cmd.extend(extra_pip_flags)
    
    retcode = subprocess.call(pip_cmd)

def get_build_args():
    """Get pip build arguments from BUILD_ARGS environment variable"""
    build_args = os.environ.get('BUILD_ARGS', '')
    if build_args:
        return build_args.split()
    return []

def get_kernels():
    extra_kernels = os.environ.get('BUILD_KERNELS', '')
    default_kernels = [
        'deploy/kernels/bindings.cpp',
        'deploy/kernels/gemm.cu',
        'deploy/kernels/quant.cu',
        'deploy/kernels/flashinfer.cu',
    ]
    if extra_kernels:
        return extra_kernels.split() + default_kernels
    else:
        return default_kernels

if __name__ == '__main__':
    # Get build args from environment variable
    extra_pip_flags = get_build_args()
    
    # Call third_party_cmake with extra flags
    third_party_cmake(extra_pip_flags if extra_pip_flags else None)
    
    remove_unwanted_pytorch_nvcc_flags()
    setup(
        name='flatquant',
        packages=['flatquant', 'deploy'],
        ext_modules=[
            CUDAExtension(
                name='deploy._CUDA',
                sources=get_kernels(),
                include_dirs=[
                    os.path.join(setup_dir, 'deploy/kernels/include'),
                    os.path.join(setup_dir, 'third-party/cutlass/include'),
                    os.path.join(setup_dir, 'third-party/cutlass/tools/util/include')
                ],
                extra_compile_args={
                    'cxx': [],
                    'nvcc': get_cuda_arch_flags(),
                }
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
