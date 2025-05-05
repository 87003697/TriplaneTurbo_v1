#include <torch/extension.h>
#include <tuple>
#include <functional>

// Define the new interface function for center depth/opacity rasterization
// Update to return two tensors
std::tuple<torch::Tensor, torch::Tensor>
RasterizeGaussiansCenterDepthCUDA(
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
	const bool debug
); 