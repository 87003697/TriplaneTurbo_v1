#include <torch/extension.h>
#include <limits>
#include <ATen/ATen.h>
#include <iostream>
#include <tuple>
#include <functional>
#include <vector_types.h> // For float2, float3, float4

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

// <<< Define the new Step 2 verification kernel >>>
__global__ void preprocessStep2Kernel(
    int P,
    const float* means3D,
    const float* viewmatrix,
    const float* projmatrix,
    const int W, const int H,
    float* intermediate_depths, // Output buffer 1
    float2* intermediate_xy,      // Output buffer 2
    float* out_depth
)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    float3 p_orig = {means3D[idx*3+0], means3D[idx*3+1], means3D[idx*3+2]};

    // Transform point by view matrix
    float4 p_view_h = transformPoint4x4(p_orig, viewmatrix);
    float p_view_z = p_view_h.z; // View space depth

    // Project to screen space
    float4 p_proj_h = transformPoint4x4(p_orig, projmatrix);
    float w = (abs(p_proj_h.w) > 1e-5) ? p_proj_h.w : 1e-5;
    float3 p_proj = {(p_proj_h.x / w + 1.f) * W / 2.f, (p_proj_h.y / w + 1.f) * H / 2.f, w};

    if (idx == 0 && blockIdx.x == 0) {
        printf("[Step 3.1 Debug] idx=0: out_depth pointer = %p\n", out_depth);
        if (out_depth != nullptr) {
             printf("[Step 3.1 Debug] idx=0: Initial out_depth[0] = %f\n", out_depth[0]);
             out_depth[0] = -99.0f;
             printf("[Step 3.2 Debug] idx=0: Attempted write. After write, out_depth[0] = %f\n", out_depth[0]);
        } else {
             printf("[Step 3.1 Debug] idx=0: out_depth pointer is NULL!\n");
        }
        printf("[Step 2 Debug] idx=0: p_view.z = %f, p_proj.x = %f, p_proj.y = %f\n", p_view_z, p_proj.x, p_proj.y);
    }

    // Check validity (example: point in front of camera)
    bool valid = (p_view_z < -0.01f);

    if(valid)
    {
        // Write to intermediate buffers
        intermediate_depths[idx] = p_view_z;
        intermediate_xy[idx] = make_float2(p_proj.x, p_proj.y);
    }
    else
    {
        intermediate_depths[idx] = std::numeric_limits<float>::infinity();
        intermediate_xy[idx] = make_float2(-1.f, -1.f); // Mark as invalid
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

// RasterizeGaussiansCenterDepthCUDA - Modified for Step 2 direct kernel launch
std::tuple<torch::Tensor, torch::Tensor>
RasterizeGaussiansCenterDepthCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const float scale_modifier, // Unused now
    const float kernel_size,    // Unused now
    const bool prefiltered,    // Unused now
    const bool debug            // Unused now, but kept in signature
)
{
    const auto options = means3D.options();
    const int P = means3D.size(0);
    const int W = image_width;
    const int H = image_height;
    const auto device = means3D.device();

    // Create final output tensors (will remain mostly unchanged)
    auto out_opacity = torch::zeros({H, W}, options);
	auto out_depth = torch::full({H, W}, std::numeric_limits<float>::infinity(), options);

	if (P == 0) {
        return std::make_tuple(out_opacity, out_depth);
	}

    // Get pointers to inputs
    const float* means_ptr = means3D.contiguous().data_ptr<float>();
    const float* view_ptr = viewmatrix.contiguous().data_ptr<float>();
    const float* proj_ptr = projmatrix.contiguous().data_ptr<float>();

    // <<< Create and get pointers for intermediate buffers >>>
    auto intermediate_depths_tensor = torch::empty({(long long)P}, options);
    auto intermediate_xy_tensor = torch::empty({(long long)P, 2}, options); // float2
    float* intermediate_depths_ptr = intermediate_depths_tensor.data_ptr<float>();
    float2* intermediate_xy_ptr = reinterpret_cast<float2*>(intermediate_xy_tensor.data_ptr<float>());

    // <<< Launch the new Step 2 Kernel >>>
    const int threads = 128; 
    const dim3 blocks((P + threads - 1) / threads);
    preprocessStep2Kernel<<<blocks, threads>>>(
        P,
        means_ptr,
        view_ptr,
        proj_ptr,
        W, H,
        intermediate_depths_ptr,
        intermediate_xy_ptr,
        out_depth.data_ptr<float>()
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after preprocessStep2Kernel launch: %s\\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); 
    err = cudaGetLastError(); // Check error again after sync
    if (err != cudaSuccess) printf("CUDA Error after kernel sync: %s\\n", cudaGetErrorString(err));

    // <<< Return the (unmodified) final output tensors >>>
    return std::make_tuple(out_opacity, out_depth);
} 