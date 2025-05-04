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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct GeometryState;
struct RasterizationState;
struct BinningState;
struct ImageState;

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float tan_fovx, float tan_fovy,
		const float focal_x, float focal_y,
		const float kernel_size,
		int* radii,
		float2* points_xy_image,
		float3* view_points,
		float* depths,
		float* camera_planes,
		float2* ray_planes,
		float* ts,
		float3* normals,
		float* cov3Ds,
		float* rgb,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool integrate,
		float* invraycov3Ds,
		bool* conditions);

	// Rasterize one Gaussian per thread. Keep track of contributions from all Gaussians
	// and accumulate them per pixel using atomics. Code runs ONCE per Gaussian.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* background,
		const float2* means2D,
		const float* colors,
		const float* opacities,
		const float* depths,
		const float* mdepths,
		const float* alphas,
		const float2* covs2D,
		const float3* coords,
		const float3* mcoords,
		const float4* normals,
		float* final_T,
		uint32_t* n_contrib,
		const float* M,
		float* out_color,
		float* out_coord,
		float* out_mcoord,
		float* out_depth,
		float* out_mdepth,
		float* out_alpha,
		float* out_normal,
		bool require_coord,
		bool require_depth,
		bool debug);


	// Function to compute center depth and opacity map -- MOVED TO rasterizer_impl.cu
	/*
	__global__ void compute_center_depth_kernel(
		const uint32_t N, // Number of gaussians
		const float* __restrict__ points, // Input: Gaussian centers (X, Y, Z)
		const float* __restrict__ viewmatrix, // Input: View matrix (4x4)
		const float* __restrict__ projmatrix, // Input: Projection matrix (4x4)
		const float focal_x, // Input: Camera focal length X
		const float focal_y, // Input: Camera focal length Y
		const float tan_fovx, // Input: Tangent of half FOV X
		const float tan_fovy, // Input: Tangent of half FOV Y
		const int W, // Input: Image width
		const int H, // Input: Image height
		float* __restrict__ out_depth, // Output: Center depth map (initialized to 0)
		unsigned char* __restrict__ out_opacity // Output: Center opacity map (initialized to 0)
	);
	*/

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* view_points,
		const float2* means2D,
		const float* colors,
		const float* ts,
		const float* camera_planes,
		const float2* ray_planes,
		const float3* normals,
		const float4* conic_opacity,
		const float focal_x, float focal_y,
		float* out_alpha,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_coord,
		float* out_mcoord,
		float* out_normal,
		float* out_depth,
		float* out_mdepth,
		float* accum_coord,
		float* accum_depth,
		float* normal_length,
		bool require_coord,
		bool require_depth);

	//follow code is adopted from GOF for marching tetrahedra https://github.com/autonomousvision/gaussian-opacity-fields
	// Perform initial steps for each Point prior to integration.
	void preprocess_points(int PN, int D, int M,
		const float* points3D,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		float2* points2D,
		float* depths,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	
	// Main rasterization method.
	void integrate(
		const dim3 grid, dim3 block,
		const uint2* gaussian_ranges,
		const uint2* point_ranges,
		const uint32_t* gaussian_list,
		const uint32_t* point_list,
		int W, int H,
		float focal_x, float focal_y,
		const float2* subpixel_offset,
		const float2* points2D,
		const float2* gaussians2D,
		const float* features,
		const float* depths_plane,
		const float2* ray_planes,
		const float* cov3Ds,
		const float* viewmatrix,
		const float3* points3D,
		const float3* gaussians3D,
		const float3* scales,
		const float* invraycov3Ds,
		const float* point_depths,
		const float* gaussian_depths,
		const float4* conic_opacity,
		const bool* condition,
		float* final_T,
		uint32_t* n_contrib,
		// float* center_depth,
		// float4* center_alphas,
		const float* bg_color,
		float* out_color,
		float* out_alpha_integrated,
		float* out_color_integrated,
		float* out_coordinate2d,
		float* out_sdf);

}

#endif