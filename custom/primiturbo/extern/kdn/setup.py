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
            name='cuda_kdn',
            sources=[
                os.path.join(current_dir, 'ext.cpp'),
                os.path.join(current_dir, 'kdn.cu'),
                os.path.join(current_dir, 'kdn_cpu.cpp'),
            ],
            extra_compile_args={'cxx': ['-O3'],
                                'nvcc': ['-O3',
                                         # Add any necessary CUDA compute capabilities if needed
                                         # Example: '-gencode=arch=compute_70,code=sm_70'
                                         ]}
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
    name='cuda_kdn',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
) 