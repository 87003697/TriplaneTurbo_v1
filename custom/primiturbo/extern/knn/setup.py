import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from setuptools import setup

if torch.cuda.is_available():
    print("CUDA is available. Building CUDA extension...")
    ext_modules = [
        CUDAExtension('cuda_knn', [
            'ext.cpp',
            'knn.cu',
            'knn_cpu.cpp',
        ], extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']})
    ]
else:
    print("CUDA is not available. Building CPU-only extension...")
    ext_modules = [
        CppExtension('cuda_knn', [
            'ext.cpp',
            'knn_cpu.cpp',
        ], extra_compile_args={'cxx': ['-O3']})
    ]

setup(
    name='cuda_knn',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
) 