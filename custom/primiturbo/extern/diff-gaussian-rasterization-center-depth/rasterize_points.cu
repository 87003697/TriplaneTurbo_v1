#include <torch/extension.h>
#include <limits>
#include <ATen/ATen.h>
#include <iostream>
#include <tuple>
#include <functional>
#include <vector_types.h> // For float2, float3, float4
#include <c10/cuda/CUDAException.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

// <<< Add helper function needed by the new kernel >>>
namespace {
    __device__ inline float4 transformPoint4x4(const float3& p, const float* matrix) {
        float4 result;
        result.x = p.x * matrix[0] + p.y * matrix[4] + p.z * matrix[8] + matrix[12];
        result.y = p.x * matrix[1] + p.y * matrix[5] + p.z * matrix[9] + matrix[13];
        result.z = p.x * matrix[2] + p.y * matrix[6] + p.z * matrix[10] + matrix[14];
        result.w = p.x * matrix[3] + p.y * matrix[7] + p.z * matrix[11] + matrix[15];
        return result;
    }
}

// <<< Kernel 1: Project Points >>>
__global__ void projectPointsKernel(
    int P,
    const float* means3D,
    const float* viewmatrix, // W2C.T
    const float* mvp_matrix_T, // MVP Transposed
    const int W, const int H,
    int*   out_pixel_indices, // Output: P ints
    float* out_view_depths    // Output: P floats
)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    // Default invalid values
    out_pixel_indices[idx] = -1;
    out_view_depths[idx] = std::numeric_limits<float>::infinity();

    float3 p_orig = {means3D[idx*3+0], means3D[idx*3+1], means3D[idx*3+2]};

    // World to View Space 
    float4 p_view_h = transformPoint4x4(p_orig, viewmatrix);
    float p_view_z = p_view_h.z;

    // World to Clip Space 
    float4 p_clip_h = transformPoint4x4(p_orig, mvp_matrix_T);

    // Clip to NDC 
    float w = p_clip_h.w;
    if (abs(w) < 1e-8) return; 
    float3 ndc = make_float3(p_clip_h.x / w, p_clip_h.y / w, p_clip_h.z / w);

    // NDC to Screen Coords
    float screen_x = (ndc.x + 1.f) * W * 0.5f;
    float screen_y = (ndc.y + 1.f) * H * 0.5f;

    // Screen Coords to Pixel Coords
    int px = static_cast<int>(roundf(screen_x - 0.5f));
    int py = static_cast<int>(roundf(screen_y - 0.5f));

    // Bounds and Validity Check
    if (px >= 0 && px < W && py >= 0 && py < H && isfinite(p_view_z)) {
        out_pixel_indices[idx] = py * W + px;
        out_view_depths[idx] = p_view_z;
    }
}

// <<< Kernel 2: Select Depth after Sorting >>>
__global__ void selectDepthKernel(
    int P, // Number of valid points after projection
    const int* sorted_pixel_indices,
    const float* sorted_view_depths,
    float* out_depth // Final HxW depth map
)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    int pix_id = sorted_pixel_indices[idx];
    if (pix_id < 0) return; // Should have been filtered before sorting ideally, but check again

    // Check if this is the first occurrence of this pixel index in the sorted list
    bool is_first = (idx == 0) || (pix_id != sorted_pixel_indices[idx - 1]);

    if (is_first) {
        // Because we sorted by depth descending first, then stable sort by pix_id,
        // the first time we see a pix_id, its corresponding depth is the maximum (closest).
        out_depth[pix_id] = sorted_view_depths[idx];
    }
}

// Helper function to create a resize lambda
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		// Check if the tensor's current size (in bytes) is sufficient
		if (t.numel() * t.element_size() < N) {
			// If not, resize the tensor.
			// Note: This assumes N is the desired size in *bytes*.
			// We need to calculate the number of elements based on the tensor's dtype.
			size_t num_elements = (N + t.element_size() - 1) / t.element_size(); // Calculate elements needed
			t.resize_({(long long)num_elements}); // Resize to the required number of elements
		}
		// Return the raw data pointer as char*
		return reinterpret_cast<char*>(t.data_ptr());
	};
	return lambda; // Return the created lambda function object
}

// <<< Modify RasterizeGaussiansCenterDepthCUDA for GPU Sorting >>>
std::tuple<torch::Tensor, torch::Tensor> // Return Opacity, Final Depth Map
RasterizeGaussiansCenterDepthCUDA(
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
    const auto options_float = means3D.options().dtype(torch::kFloat32);
    const auto options_int = means3D.options().dtype(torch::kInt32);
    const int P = means3D.size(0);
    const int W = image_width;
    const int H = image_height;
    const auto device = means3D.device();

    auto out_opacity = torch::zeros({H, W}, options_float);
    auto out_depth = torch::full({H, W}, std::numeric_limits<float>::infinity(), options_float);

    // Temporary buffers for Kernel 1 output
    auto pixel_indices_tensor = torch::empty({(long long)P}, options_int); 
    auto view_depths_tensor = torch::empty({(long long)P}, options_float);

	if (P == 0) {
        return std::make_tuple(out_opacity, out_depth);
	}

    // Get pointers for Kernel 1
    const float* means_ptr = means3D.contiguous().data_ptr<float>();
    const float* view_ptr = viewmatrix.contiguous().data_ptr<float>();
    const float* mvp_T_ptr = mvp_matrix_T.contiguous().data_ptr<float>();
    int*   pixel_indices_ptr = pixel_indices_tensor.data_ptr<int>();
    float* view_depths_ptr = view_depths_tensor.data_ptr<float>();

    // Launch Kernel 1: Project Points
    const int threads = 128; 
    const dim3 blocks_proj((P + threads - 1) / threads);
    projectPointsKernel<<<blocks_proj, threads>>>(
        P,
        means_ptr,
        view_ptr, 
        mvp_T_ptr, 
        W, H,
        pixel_indices_ptr,
        view_depths_ptr
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // <<< GPU Sorting using Thrust >>>
    // Note: Ideally filter invalid points (-1 indices) *before* sorting for efficiency.
    // However, for simplicity now, we sort everything and handle -1 in selectDepthKernel.
    // We could use thrust::copy_if and then sort the smaller valid arrays.
    
    // Wrap raw pointers with device_ptr
    thrust::device_ptr<int>   thrust_pix_idx_ptr(pixel_indices_ptr);
    thrust::device_ptr<float> thrust_view_depth_ptr(view_depths_ptr);
    
    // Create a sequence of indices [0, 1, ..., P-1] to track original order if needed (or just sort values directly)
    // We need to sort pixel_indices and view_depths together.
    // Strategy: Sort depths descending, then stable sort by pixel index ascending.
    
    try {
        // 1. Sort primarily by depth descending.
        //    We sort the *values* (depths) and apply the same permutation to the *keys* (pixel indices).
        thrust::sort_by_key(thrust::cuda::par, 
                              thrust_view_depth_ptr, thrust_view_depth_ptr + P, // Sort keys (depths) using >
                              thrust_pix_idx_ptr, // Apply permutation to values (pixel indices)
                              thrust::greater<float>()); // Descending order for depths
        C10_CUDA_KERNEL_LAUNCH_CHECK(); // Check for errors after Thrust call

        // 2. Stable sort primarily by pixel index ascending.
        //    We sort the *keys* (pixel indices) and apply the same permutation to the *values* (depths).
        //    Stable sort preserves the descending depth order for ties in pixel index.
        thrust::stable_sort_by_key(thrust::cuda::par,
                                   thrust_pix_idx_ptr, thrust_pix_idx_ptr + P, // Sort keys (pixel indices) using <
                                   thrust_view_depth_ptr); // Apply permutation to values (depths)
        C10_CUDA_KERNEL_LAUNCH_CHECK(); 
    } catch (const thrust::system_error &e) {
        throw std::runtime_error(std::string("Thrust error during sorting: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Unknown error during Thrust sorting.");
    }
    // <<< End Sorting >>>

    // Get pointer for final output depth map
    float* out_depth_ptr = out_depth.data_ptr<float>();

    // Launch Kernel 2: Select Depth
    // Note: We launch P threads, matching the size of the sorted arrays.
    const dim3 blocks_select((P + threads - 1) / threads);
    selectDepthKernel<<<blocks_select, threads>>>(
        P,
        pixel_indices_ptr, // Pass pointer to sorted indices
        view_depths_ptr,   // Pass pointer to sorted depths
        out_depth_ptr      // Pass pointer to final output map
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Return final opacity and depth maps
    return std::make_tuple(out_opacity, out_depth);
} 