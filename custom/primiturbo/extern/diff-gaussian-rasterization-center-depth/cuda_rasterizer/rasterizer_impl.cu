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

#include "rasterizer_impl.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"
#include "rasterizer.h"

// >>>>> Dummy Kernel Declaration for Debugging (REMOVED) <<<<<
/*
__global__ void dummyRenderCUDA(
	int W, int H,
	float* __restrict__ out_depth
);
*/

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}


__global__ void createWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	const uint32_t* tiles_touched,
	uint64_t* points_keys_unsorted,
	uint32_t* points_values_unsorted,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Points
	if (tiles_touched[idx] > 0)
	{
		// Find this Point's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		
		// determine the tile that the point is in
		const float2 p = points_xy[idx];
		int x = min(grid.x - 1, max((int)0, (int)(p.x / BLOCK_X)));
		int y = min(grid.y - 1, max((int)0, (int)(p.y / BLOCK_Y)));
	
		uint64_t key = y * grid.x + x;
		key <<= 32;
		key |= *((uint32_t*)&depths[idx]);
		points_keys_unsorted[off] = key;
		points_values_unsorted[off] = idx;
	}
}


// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.camera_planes, P * 6, 128);
	obtain(chunk, geom.ray_planes, P, 128);
	obtain(chunk, geom.ts, P, 128);
	obtain(chunk, geom.normals, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.view_points, P * 3, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::PointState CudaRasterizer::PointState::fromChunk(char*& chunk, size_t P)
{
	PointState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.points2D, P, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	// obtain(chunk, img.accum_alpha, N * 4, 128);
	obtain(chunk, img.n_contrib, N * 2, 128);
	obtain(chunk, img.ranges, N, 128);
	obtain(chunk, img.point_ranges, N, 128);
	obtain(chunk, img.accum_coord, N * 3, 128);
	obtain(chunk, img.accum_depth, N, 128);
	obtain(chunk, img.normal_length, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Main forward pass function - Ensure signature matches rasterizer.h
void CudaRasterizer::Rasterizer::forward(
	std::function<char*(size_t N)> geomFunc,
    std::function<char*(size_t N)> binningFunc,
    std::function<char*(size_t N)> imgFunc,
	const int P, int D, int M, 
	const float* background,
	const int W, int H,
	const float* means3D,
	const float* shs, 
	const float* colors_precomp, 
	const float* opacities, 
	const float* scales, 
	const float scale_modifier, 
	const float* rotations, 
	const float* cov3D_precomp, 
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos, 
	const float tan_fovx, const float tan_fovy,
	const float kernel_size, 
	const bool prefiltered, 
	const int* radii, 
	const bool debug,
	float* out_opacity_ptr,
	float* out_depth_ptr
)
{
    // >>> ADD Printf at the beginning of the function <<<
    printf("[Rasterizer Impl Debug] Entering CudaRasterizer::Rasterizer::forward (P=%d)\n", P);

	// P = number of gaussians
	if (P == 0) {
        // >>> ADD Printf for early exit <<<
        printf("[Rasterizer Impl Debug] Exiting early because P == 0\n");
        return;
    }

	// Directly create tensors for intermediate buffers
	auto options_float3 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA); // Assuming CUDA
	// Need device from input tensors, e.g., means3D (but means3D is float*, not Tensor here)
	// We need access to one of the input Tensors to get the device, or assume CUDA:0
	// Let's assume the device of out_depth_ptr (though it's just a pointer)
	// SAFER: Need to pass a tensor reference (like means3D) into this function to get device.
	// FOR NOW: Assume CUDA device where the output pointers reside (risky assumption)
	// Hacky way to get device from pointer - THIS IS NOT ROBUST, NEED INPUT TENSOR
	// torch::Device device = torch::kCUDA; // Fallback, needs improvement
	// Let's try getting device from out_depth_ptr - still requires a tensor object.
	// We absolutely need an input tensor reference in the function signature.
	// --- TEMPORARY WORKAROUND: Assume CUDA:0 --- (NEEDS FIX LATER by changing signature)
	torch::Device device(torch::kCUDA, 0); // <<< VERY LIKELY NEEDS FIXING
	auto options_f3 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
	auto points_xy_image_tensor = torch::empty({P, 3}, options_f3); // Shape P, 3 for float3? No, float3 is struct. {P} size needed.
	auto depths_tensor = torch::empty({P}, options_f3);
	float3* points_xy_image_ptr = reinterpret_cast<float3*>(points_xy_image_tensor.data_ptr<float>());
	float* depths_ptr = depths_tensor.data_ptr<float>();

	// --- Call the simplified FORWARD::preprocess --- 
	// UNCOMMENT the call

    // >>> ADD Printf right before calling FORWARD::preprocess <<<
    printf("[Rasterizer Impl Debug] About to call FORWARD::preprocess...\n");
    // Check if pointers are valid just before calling
    printf("[Rasterizer Impl Debug]   points_xy_image ptr: %p\n", points_xy_image_ptr);
    printf("[Rasterizer Impl Debug]   depths ptr: %p\n", depths_ptr);
    printf("[Rasterizer Impl Debug]   out_opacity_ptr: %p\n", out_opacity_ptr);
    printf("[Rasterizer Impl Debug]   out_depth_ptr: %p\n", out_depth_ptr);

	// Ensure the call is NOT commented out
	FORWARD::preprocess(
		P,
        D, M, // Make sure D and M are correctly passed or defined
		means3D,
		viewmatrix,
		projmatrix,
        tan_fovx, tan_fovy, // Make sure these are correctly passed
		W,
		H,
		points_xy_image_ptr, // float3*
		depths_ptr,         // float*
		out_opacity_ptr,    // float*
		out_depth_ptr,      // float*
		debug               // bool
	);

	// --- Binning and Sorting Removed --- 

	// --- Rendering Bypassed --- 
}

// Backward pass (commented out)
/*
void CudaRasterizer::Rasterizer::backward(...)
{
	...
}
*/