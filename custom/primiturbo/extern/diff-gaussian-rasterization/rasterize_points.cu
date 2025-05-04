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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/forward.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

// ****** Move RasterizeGaussiansCenterDepthCUDA IMPLEMENTATION here ******
std::tuple<torch::Tensor, torch::Tensor>
RasterizeGaussiansCenterDepthCUDA(
	const torch::Tensor& means3D,    // Gaussian centers (P, 3)
	const torch::Tensor& viewmatrix, // Camera view matrix (4, 4)
	const torch::Tensor& projmatrix, // Camera projection matrix (4, 4)
	const float tan_fovx,            // Tangent of half FoV in x
	const float tan_fovy,            // Tangent of half FoV in y
    const int image_height,          // Image height H
    const int image_width,           // Image width W
	const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto float_opts = means3D.options().dtype(torch::kFloat32);
	auto byte_opts = torch::TensorOptions().dtype(torch::kByte).device(means3D.device());

	// Create output tensors for opacity (byte) and depth (float)
	torch::Tensor out_opacity = torch::zeros({image_height, image_width}, byte_opts);
	// Depth map initialized to 0.0f (required output for invalid pixels).
	// CUDA kernel will handle this initial 0 value during atomicMin.
	torch::Tensor out_depth = torch::zeros({image_height, image_width}, float_opts);

	if (means3D.is_cuda())
	{
		// CHECK_CONTIGUOUS(means3D); // Temporarily commented out
		// CHECK_CONTIGUOUS(viewmatrix); // Temporarily commented out
		// CHECK_CONTIGUOUS(projmatrix); // Temporarily commented out
		// CHECK_CONTIGUOUS(out_opacity); // Temporarily commented out
		// CHECK_CONTIGUOUS(out_depth); // Temporarily commented out

		// --- REMOVE KERNEL LAUNCH ---
		/*
		// Setup CUDA launch parameters
		const int BLOCK_SIZE_X = 256; // Common block size
		dim3 blockDim(BLOCK_SIZE_X); 
		dim3 gridDim((P + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);

		// Launch the kernel using the fully qualified name
		FORWARD::compute_center_depth_kernel<<<gridDim, blockDim>>>(
			P, H, W,
			means3D.data_ptr<float>(),
			viewmatrix.data_ptr<float>(),
			projmatrix.data_ptr<float>(),
			tan_fovx, tan_fovy,
			out_opacity.data_ptr<unsigned char>(),
			out_depth.data_ptr<float>()
		);
		*/

		// --- ADD METHOD CALL --- 
		// Requires #include "cuda_rasterizer/rasterizer.h"
		CudaRasterizer::Rasterizer::compute_center_depth(
			P,                          // const int P
			H,                          // const int H
			W,                          // const int W
			means3D.contiguous().data_ptr<float>(),  // const float* means3D (Ensure contiguous)
			viewmatrix.contiguous().data_ptr<float>(),// const float* viewmatrix (Ensure contiguous)
			projmatrix.contiguous().data_ptr<float>(),// const float* projmatrix (Ensure contiguous)
			tan_fovx,                   // const float tan_fovx
			tan_fovy,                   // const float tan_fovy
			out_opacity.contiguous().data_ptr<unsigned char>(), // unsigned char* out_opacity (Ensure contiguous)
			out_depth.contiguous().data_ptr<float>(), // float* out_depth (Ensure contiguous)
			debug                       // bool debug
		);

		// Check for CUDA errors (after the call)
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "CUDA kernel launch error in RasterizeGaussiansCenterDepthCUDA: " 
					  << cudaGetErrorString(err) << std::endl;
			// Handle error appropriately, e.g., throw exception or return error code
		}
	}
	else
	{
		TORCH_CHECK(false, "CPU Rasterization for center depth not supported");
	}

	return std::make_tuple(out_opacity, out_depth);
}
// ****** END RasterizeGaussiansCenterDepthCUDA IMPLEMENTATION ******

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const float kernel_size,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool require_coord,
	const bool require_depth,
	const bool require_center,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_mdepth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_coord = torch::full({3, H, W}, 0.0, float_opts);
  torch::Tensor out_mcoord = torch::full({3, H, W}, 0.0, float_opts);
  torch::Tensor out_alpha = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_normal = torch::full({3, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		kernel_size,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_coord.contiguous().data<float>(),
		out_mcoord.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		out_mdepth.contiguous().data<float>(),
		out_alpha.contiguous().data<float>(),
		out_normal.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		require_coord,
		require_depth,
		debug);
  }

  torch::Tensor center_opacity;
  torch::Tensor center_depth;

  if (require_center)
  {
    std::tie(center_opacity, center_depth) = RasterizeGaussiansCenterDepthCUDA(
			means3D,
			viewmatrix,
			projmatrix,
			tan_fovx,
			tan_fovy,
			image_height,
			image_width,
			debug
	  );
  }
  else
  {
	  auto options_byte = torch::TensorOptions().dtype(torch::kByte).device(means3D.device());
	  auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(means3D.device());
	  center_opacity = torch::empty({0}, options_byte);
	  center_depth = torch::empty({0}, options_float);
  }

  return std::make_tuple(rendered, out_color, out_coord, out_mcoord, out_alpha, out_normal, out_depth, out_mdepth, radii, geomBuffer, binningBuffer, imgBuffer, center_opacity, center_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const float kernel_size,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_coord,
	const torch::Tensor& dL_dout_mcoord,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& dL_dout_mdepth,
	const torch::Tensor& dL_dout_alpha,
	const torch::Tensor& dL_dout_normal,
	const torch::Tensor& normalmap,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& alphas,
	const bool require_coord,
	const bool require_depth,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dview_points = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dts = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcamera_planes = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dray_planes = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_dnormals = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  alphas.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  kernel_size,
	  radii.contiguous().data<int>(),
	  normalmap.contiguous().data<float>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_coord.contiguous().data<float>(),
	  dL_dout_mcoord.contiguous().data<float>(),
	  dL_dout_depth.contiguous().data<float>(),
	  dL_dout_mdepth.contiguous().data<float>(),
	  dL_dout_alpha.contiguous().data<float>(),
	  dL_dout_normal.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dview_points.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dts.contiguous().data<float>(),
	  dL_dcamera_planes.contiguous().data<float>(),
	  dL_dray_planes.contiguous().data<float>(),
	  dL_dnormals.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  require_coord,
	  require_depth,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
IntegrateGaussiansToPointsCUDA(
	const torch::Tensor& background,
	const torch::Tensor& points3D,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& view2gaussian_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const float kernel_size,
	const torch::Tensor& subpixel_offset,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  if (points3D.ndimension() != 2 || points3D.size(1) != 3) {
    AT_ERROR("points3D must have dimensions (num_points, 3)");
  }

  const int PN = points3D.size(0);
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({9, H, W}, 0.0, float_opts);
  torch::Tensor accum_alpha = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_alpha_integrated = torch::full({PN}, 1.0, float_opts);
  torch::Tensor out_color_integrated = torch::full({PN, 3}, 0.0, float_opts);
  torch::Tensor out_coordinate2d = torch::full({PN, 2}, 0.0, float_opts);
  torch::Tensor out_sdf = torch::full({PN}, -1000.0, float_opts);
  torch::Tensor invraycov = torch::full({P, 6}, 0.0, float_opts);
  torch::Tensor condition = torch::full({PN}, 0.0, means3D.options().dtype(torch::kBool));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  torch::Tensor pointBuffer = torch::empty({0}, options.device(device));
  torch::Tensor point_binningBuffer = torch::empty({0}, options.device(device));
  
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> pointFunc = resizeFunctional(pointBuffer);
  std::function<char*(size_t)> point_binningFunc = resizeFunctional(point_binningBuffer);
  
//   if (DEBUG_INTEGRATE && PRINT_INTEGRATE_INFO){
// 		printf("IntegrateGaussiansToPointsCUDA\n");
// 		printf("P: %d\n", P);
// 		printf("PN: %d\n", PN);
//   }
  
  int rendered = 0;
  if(P != 0 && PN != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::integrate(
	    geomFunc,
		binningFunc,
		imgFunc,
		pointFunc,
		point_binningFunc,
	    PN, P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		points3D.contiguous().data<float>(),
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		view2gaussian_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		kernel_size,
		subpixel_offset.contiguous().data<float>(),
		prefiltered,
		out_color.contiguous().data<float>(),
		accum_alpha.contiguous().data<float>(),
		invraycov.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		out_alpha_integrated.contiguous().data<float>(),
		out_color_integrated.contiguous().data<float>(),
		out_coordinate2d.contiguous().data<float>(),
		out_sdf.contiguous().data<float>(),
		condition.contiguous().data<bool>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_alpha_integrated, out_color_integrated, out_coordinate2d, out_sdf, radii, geomBuffer, binningBuffer, imgBuffer);
}