#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="center_depth_rasterization",
    packages=['center_depth_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="center_depth_rasterization._C",
            sources=[
            "ext.cpp",
            "rasterize_points.cu"
            ],
            extra_compile_args={
                "nvcc": [
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                    "-arch=sm_89",
                    "--expt-relaxed-constexpr"
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
