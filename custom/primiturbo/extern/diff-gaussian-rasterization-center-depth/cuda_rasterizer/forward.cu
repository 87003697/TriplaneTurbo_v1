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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <ATen/cuda/Atomic.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include "config.h"
#include <stdio.h> // For printf

namespace cg = cooperative_groups;

#define BLOCK_MAX_SPLATS 512 // Define missing macro

// >>> ADD Helper function for float atomicMin <<<
__device__ __forceinline__ float atomicMinFloat(float *addr, float value) {
    float old = *addr;
    // Loop until atomic CAS succeeds
    while (value < old) {
        float assumed = old;
        // Read the value at addr, compare it with assumed, and write value if they match
        old = atomicCAS((unsigned int*)addr, __float_as_uint(assumed), __float_as_uint(value));
        // Need to cast float* to unsigned int* for atomicCAS and reinterpret bits
        old = __uint_as_float(old); // Interpret the returned bits as float
        // If another thread wrote a smaller value already, old will be updated and the loop continues
        // If the value was successfully swapped, old == assumed, loop might continue if value < old still holds (unlikely but possible with concurrent writes)
        // A robust loop continues as long as the new value is still smaller than the current memory value
    }
    return old; // Return the value present before the potential update by this thread
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
template<bool INTE = false>
__device__ bool computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, float kernel_size, const float* cov3D, const float* viewmatrix, 
							float* cov2D, float* camera_plane, float3* output_normal, float2* ray_plane, float& coef, float* invraycov3Ds = nullptr)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	float txtz = t.x / t.z;
	float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	txtz = t.x / t.z;
	tytz = t.y / t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// output[0] = { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
	cov2D[0] = float(cov[0][0] + kernel_size);
	cov2D[1] = float(cov[0][1]);
	cov2D[2] = float(cov[1][1] + kernel_size);
	const float det_0 = max(1e-6, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
	const float det_1 = max(1e-6, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);
	coef = sqrt(det_0 / (det_1+1e-6) + 1e-6);
	if (det_0 <= 1e-6 || det_1 <= 1e-6){
		coef = 0.0f;
	}

	// glm::mat3 testm = glm::mat3{
	// 	1,2,3,
	// 	4,5,6,
	// 	7,8,9,
	// };
	// glm::vec3 testv = {1,1,1};
	// glm::vec3 resultm = testm * testv;
	// printf("%f %f %f\n", resultm[0], resultm[1],resultm[2]); 12.000000 15.000000 18.000000

	glm::mat3 Vrk_eigen_vector;
	glm::vec3 Vrk_eigen_value;
	int D = glm_modification::findEigenvaluesSymReal(Vrk,Vrk_eigen_value,Vrk_eigen_vector);

	unsigned int min_id = Vrk_eigen_value[0]>Vrk_eigen_value[1]? (Vrk_eigen_value[1]>Vrk_eigen_value[2]?2:1):(Vrk_eigen_value[0]>Vrk_eigen_value[2]?2:0);

	glm::mat3 Vrk_inv;
	bool well_conditioned = Vrk_eigen_value[min_id]>0.00000001;
	glm::vec3 eigenvector_min;
	if(well_conditioned)
	{
		glm::mat3 diag = glm::mat3( 1/Vrk_eigen_value[0], 0, 0,
									0, 1/Vrk_eigen_value[1], 0,
									0, 0, 1/Vrk_eigen_value[2] );
		Vrk_inv = Vrk_eigen_vector * diag * glm::transpose(Vrk_eigen_vector);
	}
	else
	{
		eigenvector_min = Vrk_eigen_vector[min_id];
		Vrk_inv = glm::outerProduct(eigenvector_min,eigenvector_min);
	}
	
	glm::mat3 cov_cam_inv = glm::transpose(W) * Vrk_inv * W;
	glm::vec3 uvh = {txtz, tytz, 1};
	glm::vec3 uvh_m = cov_cam_inv * uvh;
	glm::vec3 uvh_mn = glm::normalize(uvh_m);

	if(isnan(uvh_mn.x)|| D==0)
	{
		for(int ch = 0; ch < 6; ch++)
			camera_plane[ch] = 0;
		*output_normal = {0,0,0};
		*ray_plane = {0,0};
	}
	else
	{
		float u2 = txtz * txtz;
		float v2 = tytz * tytz;
		float uv = txtz * tytz;

		float l = sqrt(t.x*t.x+t.y*t.y+t.z*t.z);
		glm::mat3 nJ = glm::mat3(
			1 / t.z, 0.0f, -(t.x) / (t.z * t.z),
			0.0f, 1 / t.z, -(t.y) / (t.z * t.z),
			t.x/l, t.y/l, t.z/l);

		glm::mat3 nJ_inv = glm::mat3(
			v2 + 1,	-uv, 		0,
			-uv,	u2 + 1,		0,
			-txtz,	-tytz,		0
		);

		if constexpr (INTE)
		{
			glm::mat3 inv_cov_ray;
			if(well_conditioned)
			{
				float ltz = u2+v2+1;
				glm::mat3 nJ_inv_full = t.z/(u2+v2+1) * \
										glm::mat3(
											v2 + 1,	-uv, 		txtz/l*ltz,
											-uv,	u2 + 1,		tytz/l*ltz,
											-txtz,	-tytz,		1/l*ltz);
				glm::mat3 T2 = W * glm::transpose(nJ_inv_full);
				inv_cov_ray = glm::transpose(T2) * Vrk_inv * T2;
			}
			else
			{
				glm::mat3 T2 = W * nJ;
				glm::mat3 cov_ray = glm::transpose(T2) * Vrk_inv * T2;
				glm::mat3 cov_eigen_vector;
				glm::vec3 cov_eigen_value;
				glm_modification::findEigenvaluesSymReal(cov_ray,cov_eigen_value,cov_eigen_vector);
				unsigned int min_id = cov_eigen_value[0]>cov_eigen_value[1]? (cov_eigen_value[1]>cov_eigen_value[2]?2:1):(cov_eigen_value[0]>cov_eigen_value[2]?2:0);
				float lambda1 = cov_eigen_value[(min_id+1)%3];
				float lambda2 = cov_eigen_value[(min_id+2)%3];
				float lambda3 = cov_eigen_value[min_id];
				glm::mat3 new_cov_eigen_vector = glm::mat3();
				new_cov_eigen_vector[0] = cov_eigen_vector[(min_id+1)%3];
				new_cov_eigen_vector[1] = cov_eigen_vector[(min_id+2)%3];
				new_cov_eigen_vector[2] = cov_eigen_vector[min_id];
				glm::vec3 r3 = glm::vec3(new_cov_eigen_vector[0][2],new_cov_eigen_vector[1][2],new_cov_eigen_vector[2][2]);

				glm::mat3 cov2d = glm::mat3(
					1/lambda1,0,-r3[0]/r3[2]/lambda1,
					0,1/lambda2,-r3[1]/r3[2]/lambda2,
					-r3[0]/r3[2]/lambda1,-r3[1]/r3[2]/lambda2,0
				);
				glm::mat3 inv_cov_ray = new_cov_eigen_vector * cov2d * glm::transpose(new_cov_eigen_vector);
			}
			glm::mat3 scale = glm::mat3(1/focal_x,0,0,
										0, 1/focal_y,0,
										0,0,1);
			inv_cov_ray = scale * inv_cov_ray * scale;
			invraycov3Ds[0] = inv_cov_ray[0][0];
			invraycov3Ds[1] = inv_cov_ray[0][1];
			invraycov3Ds[2] = inv_cov_ray[0][2];
			invraycov3Ds[3] = inv_cov_ray[1][1];
			invraycov3Ds[4] = inv_cov_ray[1][2];
			invraycov3Ds[5] = inv_cov_ray[2][2];
		}


		float vbn = glm::dot(uvh_mn, uvh);
		float factor_normal = l / (u2+v2+1);
		glm::vec3 plane = nJ_inv * (uvh_mn/max(vbn,0.0000001f));
		float nl = u2+v2+1;
		glm::vec2 camera_plane_x = {(-(v2 + 1)*t.z+plane[0]*t.x)/nl/focal_x, (uv*t.z+plane[1]*t.x)/nl/focal_y};
		glm::vec2 camera_plane_y = {(uv*t.z+plane[0]*t.y)/nl/focal_x, (-(u2 + 1)*t.z+plane[1]*t.y)/nl/focal_y};
		glm::vec2 camera_plane_z = {(t.x+plane[0]*t.z)/nl/focal_x, (t.y+plane[1]*t.z)/nl/focal_y};

		*ray_plane = {plane[0]*l/nl/focal_x, plane[1]*l/nl/focal_y};

		camera_plane[0] = camera_plane_x.x;
		camera_plane[1] = camera_plane_x.y;
		camera_plane[2] = camera_plane_y.x;
		camera_plane[3] = camera_plane_y.y;
		camera_plane[4] = camera_plane_z.x;
		camera_plane[5] = camera_plane_z.y;


		glm::vec3 ray_normal_vector = {-plane[0]*factor_normal, -plane[1]*factor_normal, -1};
		glm::vec3 cam_normal_vector = nJ * ray_normal_vector;
		glm::vec3 normal_vector = glm::normalize(cam_normal_vector);

		*output_normal = {normal_vector.x, normal_vector.y, normal_vector.z};

	}
	return well_conditioned;
}


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

namespace {

// Kernel called by FORWARD::preprocess
__global__ void preprocessCUDA(
	int P,
	const float* means3D,
	const float* viewmatrix,
	const float* projmatrix,
	const float tan_fovx, const float tan_fovy,
	const int W, const int H,
	float3* points_xy_image,
	float* depths,
	float* out_opacity,
	float* out_depth,
	bool debug)
{
	auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P) return;

	/* REMOVED old debug block */

	float3 p_orig = {means3D[idx*3+0], means3D[idx*3+1], means3D[idx*3+2]};

	// Transform point by view matrix
	float4 p_view = transformPoint4x4(p_orig, viewmatrix);

	// Check if point is behind camera
	bool valid = (p_view.z > 0.0f);

	// Project
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float3 p_proj = {(p_hom.x / p_hom.w + 1.f) * W / 2.f, (p_hom.y / p_hom.w + 1.f) * H / 2.f, 0.0f};

    // REMOVE Step 2/3 Debug printf block
    /*
    if (idx == 0 && blockIdx.x == 0)
    {
        printf("[Step 2 Debug] idx=0: p_view.z = %f, p_proj.x = %f, p_proj.y = %f\n", p_view.z, p_proj.x, p_proj.y);

        // >>> TRY STEP 3 Simplified: Write directly to out_depth[0] <<<
        printf("[Step 3 Simple Debug] idx=0: Attempting to write -99.0 to out_depth[0]...\n");
        if (out_depth != nullptr) { // Basic null check
             out_depth[0] = -99.0f;
             printf("[Step 3 Simple Debug] idx=0: Write attempted. Value at out_depth[0] after write: %f\n", out_depth[0]);
        } else {
             printf("[Step 3 Simple Debug] idx=0: out_depth pointer is NULL!\n");
        }
    }
    */

	// Store intermediate depth (optional, keep for now)
	depths[idx] = p_view.z;

	// Store intermediate output pixel coordinates (optional, keep for now)
	points_xy_image[idx] = p_proj;

	// --- Step 4: Atomic Min Depth Write ---
	if (valid) {
        // Calculate integer pixel coordinates
        int px = static_cast<int>(roundf(p_proj.x - 0.5f));
        int py = static_cast<int>(roundf(p_proj.y - 0.5f));

        // Check if pixel is within bounds
        if (px >= 0 && px < W && py >= 0 && py < H)
        {
            // Calculate linear pixel index
            int pix_id = py * W + px;

            // Use atomicMin to write the minimum depth
            // Replace standard atomicMin with float version
            // atomicMin(&out_depth[pix_id], p_view.z);
            // atomicMinFloat(&out_depth[pix_id], p_view.z);

            // >>> REVERT TO Non-atomic write for debugging <<<
            out_depth[pix_id] = p_view.z;

            // Set opacity to 1.0 for the pixel (non-atomic, last write wins - acceptable for simple opacity)
            out_opacity[pix_id] = 1.0f;
        }
	} else {
		// Mark invalid points (e.g., behind camera) in intermediate buffers
		points_xy_image[idx].x = -1.f;
		depths[idx] = std::numeric_limits<float>::infinity();
        // Note: No write to final out_depth/out_opacity for invalid points
	}
}

// Kernel called by FORWARD::render
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	int P,
	int W, int H,
	const float2* __restrict__ points_xy_image, 
	const float* __restrict__ depths,         
	float* __restrict__ out_opacity,          
	float* __restrict__ out_depth            
)
{
	auto block = cg::this_thread_block();

	// Find the range of pixels that this thread block needs to process.
	int blockId = blockIdx.x;
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	int2 pix_min = { blockId % horizontal_blocks * BLOCK_X, blockId / horizontal_blocks * BLOCK_Y };

	// Iterate through pixels in the tile.
	for (int i = block.thread_rank(); i < BLOCK_X * BLOCK_Y; i += block.size())
	{
		int px = pix_min.x + i % BLOCK_X;
		int py = pix_min.y + i / BLOCK_X;

		bool inside = (px < W && py < H);
		if (!inside) continue;

		const int pix_id = px + py * W;

		// >>> DEBUG: Check initial depth value <<<
		// if (block.thread_rank() == 0 && blockIdx.x == 0 && i == 0) { // Print only for the first thread of the first block
		// 	printf("Initial out_depth[%d] = %f\n", pix_id, out_depth[pix_id]);
		// }

		// Iterate through ALL Gaussians (less efficient)
		for (int idx = 0; idx < P; idx++)
		{
			const float2 gaussian_pos = points_xy_image[idx];
			// >>> DEBUG: Remove distance check <<<
			// float dx = gaussian_pos.x - (float(px) + 0.5f);
			// float dy = gaussian_pos.y - (float(py) + 0.5f);
			// if (dx * dx + dy * dy < 0.25f) { // Check distance < 0.5
			// Check if gaussian projection is valid (already checked in preprocess)
			if (gaussian_pos.x >= 0) { // Simplified check: was preprocess successful?
				float depth_value = depths[idx];

				if (isfinite(depth_value)) { // Check if depth is not inf/nan
					 // gpuAtomicMin(&out_depth[pix_id], depth_value);
					 // --- DEBUG: Use Non-Atomic Write ---
					 out_depth[pix_id] = depth_value; 
					 // Write opacity = 1 if depth is valid and center is close
					 out_opacity[pix_id] = 1.0f; 
				}
			}

		}
		// n_contrib calculation removed as it's not part of the simplified signature
	}
}

// Kernel called by FORWARD::preprocess_points
template<int C>
__global__ void preprocessPointsCUDA(int P, int D, int M,
	const float* points3D,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	float2* points2D,
	float* depths,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, points3D, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { points3D[3 * idx], points3D[3 * idx + 1], points3D[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	float2 point_image = {focal_x * p_view.x / (p_view.z + 0.0000001f) + W/2., focal_y * p_view.y / (p_view.z + 0.0000001f) + H/2.};

	// If the point is outside the image, quit.
	if (point_image.x < 0 || point_image.x >= W || point_image.y < 0 || point_image.y >= H)
		return;

	// Store some useful helper data for the next steps.
	depths[idx] = sqrt(p_view.x*p_view.x+p_view.y*p_view.y+p_view.z*p_view.z);
	points2D[idx] = point_image;
	tiles_touched[idx] = 1;
}

} // <<< END of Anonymous Namespace >>>


// --- FUNCTIONS WITHIN FORWARD NAMESPACE --- 
namespace FORWARD {

// Function definition for FORWARD::preprocess
void preprocess(
	int P, int D, int M, 
	const float* means3D,
	const float* viewmatrix,
	const float* projmatrix,
	const float tan_fovx, const float tan_fovy,
	const int W, const int H,
	float3* points_xy_image,
	float* depths,
	float* out_opacity,
	float* out_depth,
	bool debug)
{
	// --- BODY RESTORED --- 
	dim3 P_block(BLOCK_SIZE, 1, 1);
	dim3 P_grid((P + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
	preprocessCUDA <<< P_grid, P_block, 0, 0 >>> ( 
		P,
		means3D,
		viewmatrix,
		projmatrix,
		tan_fovx, tan_fovy,
		W, H, 
		points_xy_image,
		depths,
		out_opacity, 
		out_depth,   
		debug);
}


// Function definition for FORWARD::render
void render(
	dim3 grid, dim3 block,
	int P, int W, int H,
	const float2* points_xy_image,
	const float* depths,
	float* out_opacity,
	float* out_depth
)
{
	if (W == 0 || H == 0) return;
	renderCUDA<<<grid, block>>> (
		P,
		W, H,
		points_xy_image,
		depths,
		out_opacity,
		out_depth
	);
}

// Function definition for FORWARD::preprocess_points
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
		bool prefiltered)
{
	preprocessPointsCUDA<NUM_CHANNELS> << <(PN + 255) / 256, 256 >> > (
		PN, D, M,
		points3D,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		points2D,
		depths,
		grid,
		tiles_touched,
		prefiltered
		);
}

} // <<< END of namespace FORWARD >>>

// --- GOF integrate function (Keep global) ---
