from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this setup.py
# _src_path = os.path.dirname(os.path.abspath(__file__))
# No longer needed as sources are relative to setup.py itself

setup(
    name='frequency_encoding_cuda_ext', # Valid package name for install
    version='0.0.1', # Add a version
    ext_modules=[
        CUDAExtension(
            name='_frequency_encoding_cuda_ext', # Internal name for Python import
            sources=[
                'frequency_encoding_bindings.cpp', # Updated name
                'frequency_encoding_kernel.cu',   # Updated name
            ],
            # Add extra compile args if needed
            # extra_compile_args={'cxx': ['-g'],
            #                     'nvcc': ['-O3', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }) 