import torch
import os
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from setuptools import setup

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

if torch.cuda.is_available():
    print("CUDA is available. Building CUDA extension...")
    ext_modules = [
        CUDAExtension(
            name='cuda_kdon',
            sources=[
                os.path.join(current_dir, 'ext.cpp'),
                os.path.join(current_dir, 'kdon.cu'),
                os.path.join(current_dir, 'kdon_cpu.cpp'),
            ],
            extra_compile_args={'cxx': ['-O3', '-std=c++17'],
                                'nvcc': ['-O3', '-std=c++17']}
        )
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
    name='cuda_kdon',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='K-Dense Opacity Neighbor (KDON) PyTorch extension',
    long_description='CUDA and CPU implementation for finding K-Dense Opacity Neighbors using Mahalanobis distance for density and opacity.',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
) 