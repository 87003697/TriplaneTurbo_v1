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

py::tuple RasterizeGaussiansCenterDepthPython(
	const torch::Tensor& means3D,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const float scale_modifier,
	const float kernel_size,
	const bool prefiltered,
	const bool debug)
{
	auto result_tuple = RasterizeGaussiansCenterDepthCUDA(means3D, viewmatrix, projmatrix, tan_fovx, tan_fovy, image_height, image_width, scale_modifier, kernel_size, prefiltered, debug);
	return py::make_tuple(std::get<0>(result_tuple), std::get<1>(result_tuple));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rasterize_gaussians_center_depth", &RasterizeGaussiansCenterDepthPython);
}