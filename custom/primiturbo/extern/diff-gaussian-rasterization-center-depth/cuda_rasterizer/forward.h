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
#include <cooperative_groups.h>

namespace FORWARD
{
	// Struct to store gradient information (if needed later for backward pass, keep for now)
	struct Gradients
	{
		float3 *grad_means2D, *grad_view_points;
		float *grad_opacities;
		float4 *grad_conics;
		float3 *grad_colors;
        float3 *grad_normals;
        float *grad_normal_lengths;
        float *grad_depths;
        float3 *grad_coords;
	};

	// Perform initial steps for each Gaussian prior to rasterization (Simplified for Center Depth)
	void preprocess(int P, int D, int M, const float* means3D, const float* viewmatrix, const float* projmatrix, const float tan_fovx, const float tan_fovy, const int W, int H, float3* points_xy_image, float* depths, float* out_opacity, float* out_depth, bool debug);

	// Perform main rasterization step (Simplified for Center Depth)
	void render(dim3 grid, dim3 block, int P, int W, int H, const float2* points_xy_image, const float* depths, float* out_opacity, float* out_depth);

	// >>>>> Dummy Kernel Declaration REMOVED From Here <<<<<
	/*
	__global__ void dummyRenderCUDA(
		int W, int H,
		float* __restrict__ out_depth
	);
	*/

	// void preprocess_diff(int P, int D, int M, const float* means3D, const glm::vec3* scales, const float scale_modifier, const glm::vec4* rotations, const float* opacities,
	// 	const float* shs, bool* clamped, const float* cov3D_precomp, const float* colors_precomp, const float* viewmatrix, const float* projmatrix, const glm::vec3* campos,
	// 	const int W, int H, const float tanfovx, float tanfovy, const float focal_x, float focal_y, int* radii, float2* means2D, float* cov3Ds, float4* conic_opacity,
	// 	float* rgb, const dim3 grid, uint32_t* tiles_touched, bool prefiltered, float* means3d_camera, bool* condition);

	void preprocess_points(int PN, int D, int M, const float* points3D, const float* viewmatrix, const float* projmatrix, const glm::vec3* cam_pos, const int W, int H, 
						 const float focal_x, float focal_y, const float tan_fovx, float tan_fovy, float2* points2D, float* depths, const dim3 grid, uint32_t* tiles_touched, bool prefiltered);

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