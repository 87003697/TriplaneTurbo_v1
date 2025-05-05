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

#include "rasterize_points.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/config.h"
#include <torch/extension.h>
#include <functional>
#include <limits>
#include <ATen/ATen.h>
#include <iostream>

// <<< Define the new simple test kernel >>>
/*
__global__ void testKernel(const float* means, float* output, int P)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    // Simple operation: output = means.x + 1.0
    if (means != nullptr && output != nullptr) {
        output[idx] = means[idx * 3 + 0] + 1.0f;
    }
    // Optional: Add printf for debugging within kernel
    // if (idx == 0) { printf("testKernel: means[0]=%f, output[0]=%f\n", means[0], output[idx]); }
}
*/

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

// Modify function signature to return only two tensors
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
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
) {

	// Create output tensors internally
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(means3D.device());
	auto out_opacity = torch::zeros({image_height, image_width}, options);
	auto out_depth = torch::full({image_height, image_width}, std::numeric_limits<float>::infinity(), options);
	// <<< Create the new debug output tensor >>>
	/*
	int P = means3D.size(0);
	auto debug_output = torch::zeros({P}, options);
	*/

	if (means3D.numel() == 0)
	{
		// <<< Return 3 tensors even if input is empty >>>
        // Return 2 tensors instead
		// return std::make_tuple(out_opacity, out_depth, debug_output);
        return std::make_tuple(out_opacity, out_depth);
	}

	// Get data pointers
	const float* means_ptr = means3D.data_ptr<float>();
	const float* view_ptr = viewmatrix.data_ptr<float>(); // Keep for potential future steps
	const float* proj_ptr = projmatrix.data_ptr<float>(); // Keep for potential future steps
	float* opacity_ptr = out_opacity.data_ptr<float>();
	float* depth_ptr = out_depth.data_ptr<float>();
	// <<< Get pointer for the debug output tensor >>>
	/*
	float* debug_output_ptr = debug_output.data_ptr<float>();
	*/

	// Setup temporary buffer tensors using resizeFunctional (Not strictly needed for this step, but keep)
	torch::Device device(means3D.device());
	torch::TensorOptions byte_opts(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, byte_opts.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, byte_opts.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, byte_opts.device(device));
	auto geomFunc = resizeFunctional(geomBuffer);
	auto binningFunc = resizeFunctional(binningBuffer);
	auto imgFunc = resizeFunctional(imgBuffer);

	// UNCOMMENT the call to CudaRasterizer::Rasterizer::forward
	// /*

    // Calculate cam_pos from viewmatrix (inverse)
    torch::Tensor viewmatrix_inv = torch::inverse(viewmatrix.cpu()).to(device); // CPU inverse might be more stable
    const float* cam_pos_ptr = viewmatrix_inv.slice(1, 3, 4).slice(0, 0, 3).contiguous().data_ptr<float>();
    // Alternative using campos from Python if available & passed:
    // const float* cam_pos_ptr = campos.data_ptr<float>(); // Assuming campos is passed to this function

	CudaRasterizer::Rasterizer::forward(
		geomFunc,       // std::function<char*(size_t N)>
        binningFunc,
        imgFunc,
        means3D.size(0), // P
        3, // D
        16, // M
        // torch::tensor({}), // background (const float*) <-- WRONG TYPE
        nullptr,          // background (const float*) - Pass nullptr
        image_width,    // W
        image_height,   // H
        means_ptr,      // means3D (const float*)
        nullptr, // shs (const float*) - Pass nullptr
        nullptr, // colors_precomp (const float*) - Pass nullptr
        nullptr, // opacities (const float*) - Pass nullptr
        nullptr, // scales (const float*) - Pass nullptr
        scale_modifier,
        nullptr, // rotations (const float*) - Pass nullptr
        nullptr, // cov3D_precomp (const float*) - Pass nullptr
		view_ptr,         // viewmatrix (const float*)
		proj_ptr,         // projmatrix (const float*)
		// campos.contiguous(), // cam_pos (const float*) <-- Needs source
        cam_pos_ptr,      // cam_pos (const float*)
        tan_fovx,
        tan_fovy,
        kernel_size,      // kernel_size
        prefiltered,
        // torch::tensor({}), // radii (const int*) <-- Pass nullptr
        nullptr,          // radii (const int*)
        debug,
        opacity_ptr,    // out_opacity_ptr (use opacity output)
		depth_ptr       // out_depth_ptr
	);
	// */

    // >>> ADD CUDA Synchronization <<<
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        // Maybe print an error message here, but avoid complex iostream inside .cu if possible
        // Consider returning an error code or a special tensor value if sync fails.
        printf("CUDA sync error after forward: %s\n", cudaGetErrorString(sync_err));
        // Optionally, handle the error e.g., by returning tensors filled with error indicators
    }

	// <<< Return the THREE tensors >>>
    // Return TWO tensors
	// return std::make_tuple(out_opacity, out_depth, debug_output);
    return std::make_tuple(out_opacity, out_depth);
}