#include <torch/extension.h>
#include <limits>
#include <ATen/ATen.h>
#include <iostream>
#include <tuple>
#include <functional>
#include <vector_types.h>
#include <c10/cuda/CUDAException.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

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

__global__ void projectPointsKernel(
    int P,
    const float* means3D,
    const float* viewmatrix,   // W2C.T
    const float* mvp_matrix_T, // (P@W2C).T
    const float* w2c_matrix,   // W2C (Row-Major)
    const int W, const int H,
    const float near_plane,
    const float far_plane,
    int*   out_pixel_indices,
    float* out_view_depths
)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    out_pixel_indices[idx] = -1;
    out_view_depths[idx] = std::numeric_limits<float>::infinity();

    float3 p_orig = {means3D[idx*3+0], means3D[idx*3+1], means3D[idx*3+2]};

    float p_view_z_correct = p_orig.x * w2c_matrix[8] + p_orig.y * w2c_matrix[9] + p_orig.z * w2c_matrix[10] + w2c_matrix[11];

    if (p_view_z_correct < -far_plane || p_view_z_correct > -near_plane) {
         return;
    }

    float4 p_clip_h = transformPoint4x4(p_orig, mvp_matrix_T);
    float w = p_clip_h.w;
    if (abs(w) < 1e-8) return;
    float3 ndc = make_float3(p_clip_h.x / w, p_clip_h.y / w, p_clip_h.z / w);

    if (abs(ndc.x) > 1.0f || abs(ndc.y) > 1.0f) {
        return;
    }

    float screen_x = (ndc.x + 1.f) * W * 0.5f;
    float screen_y = (ndc.y + 1.f) * H * 0.5f;
    int px = static_cast<int>(roundf(screen_x - 0.5f));
    int py = static_cast<int>(roundf(screen_y - 0.5f));

    if (px >= 0 && px < W && py >= 0 && py < H && isfinite(p_view_z_correct)) {
        out_pixel_indices[idx] = py * W + px;
        out_view_depths[idx] = -p_view_z_correct;
    }
}

__global__ void selectDepthKernel(
    int P,
    const int* sorted_pixel_indices,
    const float* sorted_view_depths,
    float* out_depth,
    float* out_opacity
)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    int pix_id = sorted_pixel_indices[idx];
    if (pix_id < 0) return;
    bool is_first = (idx == 0) || (pix_id != sorted_pixel_indices[idx - 1]);
    if (is_first) {
        out_depth[pix_id] = sorted_view_depths[idx];
        out_opacity[pix_id] = 1.0f;
    }
}

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		if (t.numel() * t.element_size() < N) {
			size_t num_elements = (N + t.element_size() - 1) / t.element_size();
			t.resize_({(long long)num_elements});
		}
		return reinterpret_cast<char*>(t.data_ptr());
	};
	return lambda;
}

std::tuple<torch::Tensor, torch::Tensor>
RasterizeGaussiansCenterDepthCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& viewmatrix, // W2C.T
    const torch::Tensor& mvp_matrix_T,
    const torch::Tensor& w2c_matrix, // W2C
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const float near_plane,
    const float far_plane,
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

    auto pixel_indices_tensor = torch::empty({(long long)P}, options_int);
    auto view_depths_tensor = torch::empty({(long long)P}, options_float);

	if (P == 0) {
        return std::make_tuple(out_opacity, out_depth);
	}

    const float* means_ptr = means3D.contiguous().data_ptr<float>();
    const float* view_ptr = viewmatrix.contiguous().data_ptr<float>();
    const float* mvp_T_ptr = mvp_matrix_T.contiguous().data_ptr<float>();
    const float* w2c_ptr = w2c_matrix.contiguous().data_ptr<float>();
    int*   pixel_indices_ptr = pixel_indices_tensor.data_ptr<int>();
    float* view_depths_ptr = view_depths_tensor.data_ptr<float>();

    const int threads = 128;
    const dim3 blocks_proj((P + threads - 1) / threads);

    projectPointsKernel<<<blocks_proj, threads>>>(
        P,
        means_ptr,
        view_ptr,
        mvp_T_ptr,
        w2c_ptr,
        W, H,
        near_plane, far_plane,
        pixel_indices_ptr,
        view_depths_ptr
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    thrust::device_ptr<int>   thrust_pix_idx_ptr(pixel_indices_ptr);
    thrust::device_ptr<float> thrust_view_depth_ptr(view_depths_ptr);

    try {
        thrust::sort_by_key(thrust::cuda::par,
                              thrust_view_depth_ptr, thrust_view_depth_ptr + P,
                              thrust_pix_idx_ptr,
                              thrust::less<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        thrust::stable_sort_by_key(thrust::cuda::par,
                                   thrust_pix_idx_ptr, thrust_pix_idx_ptr + P,
                                   thrust_view_depth_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } catch (const thrust::system_error &e) {
        throw std::runtime_error(std::string("Thrust error during sorting: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Unknown error during Thrust sorting.");
    }

    float* out_depth_ptr = out_depth.data_ptr<float>();
    float* out_opacity_ptr = out_opacity.data_ptr<float>();
    int*   sorted_pixel_indices_ptr = pixel_indices_ptr;
    float* sorted_view_depths_ptr = view_depths_ptr;
    const dim3 blocks_select((P + threads - 1) / threads);
    selectDepthKernel<<<blocks_select, threads>>>(
        P,
        sorted_pixel_indices_ptr,
        sorted_view_depths_ptr,
        out_depth_ptr,
        out_opacity_ptr
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(out_opacity, out_depth);
} 