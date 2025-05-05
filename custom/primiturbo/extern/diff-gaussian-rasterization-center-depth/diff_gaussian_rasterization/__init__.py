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

# Simplified __init__.py for center depth rasterization only

from typing import NamedTuple
import torch.nn as nn
import torch

try:
    from . import _C
    # Directly expose the only function we expect to be compiled
    # Pass required args directly, matching the C++ signature
    def rasterize_gaussians_center_depth(
        means3D: torch.Tensor,
        viewmatrix: torch.Tensor,
        projmatrix: torch.Tensor,
        tanfovx: float,
        tanfovy: float,
        image_height: int,
        image_width: int,
        scale_modifier: float = 1.0, # Provide default
        kernel_size: float = 0.0, # Provide default
        prefiltered: bool = False,
        debug: bool = False):
        """Rasterizes Gaussian centers to produce opacity and depth maps.

        Args:
            means3D (Tensor): Gaussian centers (N, 3).
            viewmatrix (Tensor): Camera view matrix (4, 4).
            projmatrix (Tensor): Camera projection matrix (4, 4).
            tanfovx (float): Tangent of half FoV_X.
            tanfovy (float): Tangent of half FoV_Y.
            image_height (int): Image height.
            image_width (int): Image width.
            scale_modifier (float): Scale modifier (controls size, default 1.0).
            kernel_size (float): Kernel size for smoothing (default 0.0).
            prefiltered (bool): Whether the inputs are prefiltered (default False).
            debug (bool): Debug flag (default False).

        Returns:
            Tuple[Tensor, Tensor]:
                - opacity_map (Tensor): (H, W) tensor, 1 where a center is rendered, 0 otherwise.
                - depth_map (Tensor): (H, W) tensor with the depth of the nearest center, infinity otherwise.
        """
        return _C.rasterize_gaussians_center_depth(
            means3D,
            viewmatrix,
            projmatrix,
            tanfovx,
            tanfovy,
            image_height,
            image_width,
            scale_modifier,
            kernel_size,
            prefiltered,
            debug
        )

except ImportError as e:
    # Keep the original error message style
    print("Could not import the current diff_gaussian_rasterization C++ extension (_C).")
    print(f"Import error: {e}")
    print("Please make sure the extension in the current directory (...) is compiled (e.g., run 'pip install -e .' in this directory).")
    # Define a dummy function to avoid crashing scripts that might try to import it conditionally
    def rasterize_gaussians_center_depth(*args, **kwargs):
        raise ImportError("diff_gaussian_rasterization._C failed to import.")

# Remove all code related to the original rasterizer:
# - cpu_deep_copy_tuple
# - rasterize_gaussians (the autograd function wrapper)
# - _RasterizeGaussians (the autograd.Function class)
# - GaussianRasterizationSettings (NamedTuple)
# - GaussianRasterizer (nn.Module class)
# - integrate function (if it existed here)

