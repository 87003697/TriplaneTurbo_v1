#include <torch/extension.h>
#include <limits>
#include <ATen/ATen.h>
#include <iostream>
#include <tuple>
#include <functional>
#include <vector_types.h> // For float2, float3, float4

// <<< Add atomicMinFloat helper function >>>
__device__ inline float atomicMinFloat(float* addr, float value) {
    float old = *addr;
    while (value < old) {
        unsigned int old_int = __float_as_uint(old);
        unsigned int assumed_int = old_int; 
        unsigned int returned_int = atomicCAS((unsigned int*)addr, assumed_int, __float_as_uint(value));
        if (returned_int == assumed_int) {
            break; 
        }
        old = __uint_as_float(returned_int); 
    }
    return old;
}

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
    float* intermediate_depths, // Still passed, but not written to in this step
    float2* intermediate_xy,      // Still passed, but not written to in this step
    float* out_depth
)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    // Calculate p_view_z and p_proj (as before)
    float3 p_orig = {means3D[idx*3+0], means3D[idx*3+1], means3D[idx*3+2]};
    float4 p_view_h = transformPoint4x4(p_orig, viewmatrix);
    float p_view_z = p_view_h.z;
    float4 p_proj_h = transformPoint4x4(p_orig, projmatrix);
    float w = (abs(p_proj_h.w) > 1e-5) ? p_proj_h.w : 1e-5;
    float3 p_proj = {(p_proj_h.x / w + 1.f) * W / 2.f, (p_proj_h.y / w + 1.f) * H / 2.f, w};

    // Calculate pixel coordinates for ALL threads
    int px = static_cast<int>(roundf(p_proj.x - 0.5f));
    int py = static_cast<int>(roundf(p_proj.y - 0.5f));

    // Bounds and validity checks for ALL threads
    if (px >= 0 && px < W && py >= 0 && py < H) {
        if (p_view_z < -0.01f && isfinite(p_view_z)) {
            int pix_id = py * W + px;
            // --- Perform atomicMinFloat for ALL valid threads --- 
            atomicMinFloat(&out_depth[pix_id], p_view_z);
        }
    }

    // Intermediate writes remain commented out
    /*
    bool valid = (p_view_z < -0.01f);
    if(valid)
    {
        intermediate_depths[idx] = p_view_z;
        intermediate_xy[idx] = make_float2(p_proj.x, p_proj.y);
    }
    else
    {
        intermediate_depths[idx] = std::numeric_limits<float>::infinity();
        intermediate_xy[idx] = make_float2(-1.f, -1.f); 
    }
    */
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