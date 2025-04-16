import torch
import os
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from setuptools import setup

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

if torch.cuda.is_available():
    print("CUDA is available. Building CUDA extension...")
    ext_modules = [
        CUDAExtension('cuda_knn', [
            os.path.join(current_dir, 'ext.cpp'),
            os.path.join(current_dir, 'knn.cu'),
            os.path.join(current_dir, 'knn_cpu.cpp'),
        ], extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']})
    ]
else:
    print("CUDA is not available. Building CPU-only extension...")
    ext_modules = [
        CppExtension('cuda_knn', [
            os.path.join(current_dir, 'ext.cpp'),
            os.path.join(current_dir, 'knn_cpu.cpp'),
        ], extra_compile_args={'cxx': ['-O3']})
    ]

setup(
    name='cuda_knn',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
) 