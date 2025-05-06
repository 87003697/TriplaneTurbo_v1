#include <torch/extension.h>
#include <tuple>
#include <functional>

// Function declaration returning 2 Tensors
std::tuple<torch::Tensor, torch::Tensor>
RasterizeGaussiansCenterDepthCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& viewmatrix, // W2C.T
	const torch::Tensor& mvp_matrix_T,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const float near_plane,
	const float far_plane,
	const float scale_modifier,
	const float kernel_size,
	const bool prefiltered,
	const bool debug
); 