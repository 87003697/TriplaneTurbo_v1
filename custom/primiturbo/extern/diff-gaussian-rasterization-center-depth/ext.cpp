#include <torch/extension.h> // Include PyTorch C++ headers
#include <tuple> // Add include for std::tuple
#include "rasterize_points.h"

// Modify function signature to return py::tuple
py::tuple RasterizeGaussiansCenterDepthPython(
	const torch::Tensor& means3D,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& mvp_matrix_T,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const float scale_modifier,
	const float kernel_size,
	const bool prefiltered,
	const bool debug)
{
	// Call the CUDA function which now returns TWO tensors
	auto result_tuple = RasterizeGaussiansCenterDepthCUDA(means3D, viewmatrix, mvp_matrix_T, tan_fovx, tan_fovy, image_height, image_width, scale_modifier, kernel_size, prefiltered, debug);

	// Unpack the TWO tensors and return as py::tuple
	return py::make_tuple(std::get<0>(result_tuple), std::get<1>(result_tuple));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rasterize_gaussians_center_depth", &RasterizeGaussiansCenterDepthPython);
} 