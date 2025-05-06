#include <torch/extension.h>
#include <limits>
#include <ATen/ATen.h>
#include <iostream>
#include <tuple>
#include <functional>
#include <vector_types.h> // For float2, float3, float4
#include <c10/cuda/CUDAException.h>

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

// <<< Kernel for Step 4-DEBUG 2.0 >>>
__global__ void centerPointDepthKernel_Debug(
    int P,
    const float* means3D,
    const float* viewmatrix, // W2C.T
    const float* mvp_matrix_T, // MVP Transposed
    const int W, const int H,
    float* out_depth, // Keep for pointer check if needed
    float* debug_info_ptr // Output: P x 8 (idx, p_view_z, ndc.x, ndc.y, screen_x, screen_y, px, py)
)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    float3 p_orig = {means3D[idx*3+0], means3D[idx*3+1], means3D[idx*3+2]};

    // 1. World to View Space
    float4 p_view_h = transformPoint4x4(p_orig, viewmatrix);
    float p_view_z = p_view_h.z;

    // 2. World to Clip Space (using TRANSPOSED MVP)
    float4 p_clip_h = transformPoint4x4(p_orig, mvp_matrix_T);

    // 3. Clip to NDC 
    float w = p_clip_h.w;
    float3 ndc = make_float3(-10.f, -10.f, -10.f); // Default invalid
    if (abs(w) > 1e-8) { 
        ndc.x = p_clip_h.x / w;
        ndc.y = p_clip_h.y / w;
        ndc.z = p_clip_h.z / w;
    }

    // 4. NDC to Screen Coords
    float screen_x = (ndc.x + 1.f) * W * 0.5f;
    float screen_y = (ndc.y + 1.f) * H * 0.5f;

    // 5. Screen Coords to Pixel Coords
    int px = static_cast<int>(roundf(screen_x - 0.5f));
    int py = static_cast<int>(roundf(screen_y - 0.5f));

    // --- Write debug info --- 
    if (debug_info_ptr != nullptr) {
        int offset = idx * 8;
        debug_info_ptr[offset + 0] = (float)idx;
        debug_info_ptr[offset + 1] = p_view_z;
        debug_info_ptr[offset + 2] = ndc.x;
        debug_info_ptr[offset + 3] = ndc.y;
        debug_info_ptr[offset + 4] = screen_x;
        debug_info_ptr[offset + 5] = screen_y;
        debug_info_ptr[offset + 6] = (float)px;
        debug_info_ptr[offset + 7] = (float)py;
    }
    
    // <<< REMOVE atomicMinFloat call >>>
    
}

// <<< Corrected atomicMinFloat implementation >>>
__device__ inline void atomicMinFloatCorrected(float* addr, float value)
{
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old_uint = *addr_as_uint;
    float old_float = __uint_as_float(old_uint);

    // Loop while the new value is smaller than the current value in memory
    while (value < old_float)
    {
        unsigned int new_uint = __float_as_uint(value);
        // Try to swap if the value hasn't changed since we last read it
        unsigned int returned_uint = atomicCAS(addr_as_uint, old_uint, new_uint);

        // If the swap was successful, we're done
        if (returned_uint == old_uint)
            return;

        // If swap failed, update our "old" values and retry the loop
        old_uint = returned_uint;
        old_float = __uint_as_float(old_uint);
    }
}

// <<< Corrected atomicMaxFloat implementation >>>
__device__ inline void atomicMaxFloatCorrected(float* addr, float value)
{
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old_uint = *addr_as_uint;
    float old_float = __uint_as_float(old_uint);

    while (value > old_float)
    {
        unsigned int new_uint = __float_as_uint(value);
        unsigned int returned_uint = atomicCAS(addr_as_uint, old_uint, new_uint);
        if (returned_uint == old_uint)
            return;
        old_uint = returned_uint;
        old_float = __uint_as_float(old_uint);
    }
}

// <<< Final Kernel using atomicMaxFloatCorrected >>>
__global__ void centerPointDepthKernel(
    int P,
    const float* means3D,
    const float* viewmatrix, // W2C.T
    const float* mvp_matrix_T, // MVP Transposed
    const int W, const int H,
    float* out_depth
)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    float3 p_orig = {means3D[idx*3+0], means3D[idx*3+1], means3D[idx*3+2]};

    // 1. World to View Space
    float4 p_view_h = transformPoint4x4(p_orig, viewmatrix);
    float p_view_z = p_view_h.z;

    // 2. World to Clip Space (using TRANSPOSED MVP)
    float4 p_clip_h = transformPoint4x4(p_orig, mvp_matrix_T);

    // 3. Clip to NDC 
    float w = p_clip_h.w;
    float3 ndc = make_float3(-10.f, -10.f, -10.f); // Default invalid
    if (abs(w) > 1e-8) { 
        ndc.x = p_clip_h.x / w;
        ndc.y = p_clip_h.y / w;
        ndc.z = p_clip_h.z / w;
    }

    // 4. NDC to Screen Coords
    float screen_x = (ndc.x + 1.f) * W * 0.5f;
    float screen_y = (ndc.y + 1.f) * H * 0.5f;

    // 5. Screen Coords to Pixel Coords
    int px = static_cast<int>(roundf(screen_x - 0.5f));
    int py = static_cast<int>(roundf(screen_y - 0.5f));

    // Bounds and validity checks
    if (px >= 0 && px < W && py >= 0 && py < H) {
        if (isfinite(p_view_z)) { 
            int pix_id = py * W + px;
            // <<< REMOVE atomicMaxFloat, direct write instead >>>
            // atomicMaxFloatCorrected(&out_depth[pix_id], p_view_z);
            out_depth[pix_id] = p_view_z; // WARNING: Race condition!
        }
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

// <<< RasterizeGaussiansCenterDepthCUDA launching the final kernel >>>
std::tuple<torch::Tensor, torch::Tensor>
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
    const auto options = means3D.options(); // <<< Back to float options >>>
    const int P = means3D.size(0);
    const int W = image_width;
    const int H = image_height;
    const auto device = means3D.device();

    auto out_opacity = torch::zeros({H, W}, options);
    // <<< out_depth back to float >>>
	auto out_depth = torch::full({H, W}, std::numeric_limits<float>::infinity(), options);

	if (P == 0) {
        return std::make_tuple(out_opacity, out_depth);
	}

    // Get pointers
    const float* means_ptr = means3D.contiguous().data_ptr<float>();
    const float* view_ptr = viewmatrix.contiguous().data_ptr<float>();
    const float* mvp_T_ptr = mvp_matrix_T.contiguous().data_ptr<float>();
	float* opacity_ptr = out_opacity.data_ptr<float>(); // Although unused by kernel, keep consistent?
	float* depth_ptr = out_depth.data_ptr<float>();

    // Launch the final kernel (centerPointDepthKernel)
    const int threads = 128; 
    const dim3 blocks((P + threads - 1) / threads);
    centerPointDepthKernel<<<blocks, threads>>>(
        P,
        means_ptr,
        view_ptr, 
        mvp_T_ptr, 
        W, H,
        depth_ptr // Correct 7 arguments
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Return TWO tensors
    return std::make_tuple(out_opacity, out_depth);
} 