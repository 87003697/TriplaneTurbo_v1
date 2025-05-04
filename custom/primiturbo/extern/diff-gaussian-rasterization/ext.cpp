/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA, "Rasterize Gaussians (CUDA)",
        py::arg("background"), 
        py::arg("means3D"),
        py::arg("colors"),
        py::arg("opacity"),
        py::arg("scales"),
        py::arg("rotations"),
        py::arg("scale_modifier"),
        py::arg("cov3D_precomp"),
        py::arg("viewmatrix"),
        py::arg("projmatrix"),
        py::arg("tan_fovx"), 
        py::arg("tan_fovy"),
        py::arg("kernel_size"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("sh"),
        py::arg("degree"),
        py::arg("campos"),
        py::arg("prefiltered"),
        py::arg("require_coord"),
        py::arg("require_depth"),
        py::arg("require_center") = false,
        py::arg("debug") = false
  );
  m.def("integrate_gaussians_to_points", &IntegrateGaussiansToPointsCUDA, "Integrate Gaussians To Points (CUDA)");
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA, "Backward pass for Rasterize Gaussians (CUDA)");
  m.def("mark_visible", &markVisible, "Mark visible Gaussians (CUDA)");

  m.def("rasterize_gaussians_center_depth", &RasterizeGaussiansCenterDepthCUDA,
        "Rasterize Gaussian centers to get depth and opacity maps (non-differentiable CUDA)",
        py::arg("means3D"),
        py::arg("viewmatrix"),
        py::arg("projmatrix"),
        py::arg("tan_fovx"),
        py::arg("tan_fovy"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("debug") = false
  );
}