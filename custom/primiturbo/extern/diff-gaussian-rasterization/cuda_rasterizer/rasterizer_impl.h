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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size = 0;
		float* depths = nullptr;
		float* camera_planes = nullptr; // float[6]*
		float2* ray_planes = nullptr;    // float* or float2*?
		float* ts = nullptr;
		float3* normals = nullptr;       // float* or float3*?
		bool* clamped = nullptr;        // bool[3]*
		int* internal_radii = nullptr;
		float2* means2D = nullptr;
		float* view_points = nullptr;   // float3*
		float* cov3D = nullptr;         // float[6]*
		float4* conic_opacity = nullptr; // float4*
		float* rgb = nullptr;           // float3*
		uint32_t* tiles_touched = nullptr;
		char* scanning_space = nullptr;
		uint32_t* point_offsets = nullptr;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct PointState
	{
		size_t scan_size;
		float* depths;
		float2* points2D;
		uint32_t* tiles_touched;
		char* scanning_space;
		uint32_t* point_offsets;

		static PointState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint32_t* n_contrib = nullptr;
		uint2* ranges = nullptr;
		uint2* point_ranges = nullptr;
		float* accum_coord = nullptr;
		float* accum_depth = nullptr;
		float* normal_length = nullptr;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint32_t* point_list;
		uint32_t* point_list_unsorted;
		uint64_t* point_list_keys;
		uint64_t* point_list_keys_unsorted;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};