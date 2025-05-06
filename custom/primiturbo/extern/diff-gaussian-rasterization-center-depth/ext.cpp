#include <torch/extension.h> // Include PyTorch C++ headers
#include <tuple> // Add include for std::tuple
#include "rasterize_points.h"

// Modify function signature to return py::tuple
py::tuple RasterizeGaussiansCenterDepthPython(
	const torch::Tensor& means3D,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& mvp_matrix_T,
	const torch::Tensor& w2c_matrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const float near_plane,
	const float far_plane,
	const float scale_modifier,
	const float kernel_size,
	const bool prefiltered,
	const bool debug)
{
	// Call the CUDA function with correct arg order
	auto result_tuple = RasterizeGaussiansCenterDepthCUDA(means3D, viewmatrix, mvp_matrix_T, w2c_matrix, tan_fovx, tan_fovy, image_height, image_width, near_plane, far_plane, scale_modifier, kernel_size, prefiltered, debug);

	// Unpack the TWO tensors and return as py::tuple
	return py::make_tuple(std::get<0>(result_tuple), std::get<1>(result_tuple));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rasterize_gaussians_center_depth", &RasterizeGaussiansCenterDepthPython,
			"Rasterize Gaussians to get center depth and opacity",
			py::arg("means3D"),
			py::arg("viewmatrix"),
			py::arg("mvp_matrix_T"),
			py::arg("w2c_matrix"),
			py::arg("tan_fovx"),
			py::arg("tan_fovy"),
			py::arg("image_height"),
			py::arg("image_width"),
			py::arg("near_plane"),
			py::arg("far_plane"),
			py::arg("scale_modifier"),
			py::arg("kernel_size"),
			py::arg("prefiltered"),
			py::arg("debug"));
} 