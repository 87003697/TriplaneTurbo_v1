#include "kdon.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAGuard.h>
#include <cstdio>
#include <algorithm> // for std::max

// CUDA错误检查宏
#define CUDA_CHECK_ERROR(val) { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
}

// 常量
constexpr int THREADS_PER_BLOCK = 128;

// Helper function to compute Mahalanobis distance squared
// Modified to work with float internally for precision, even if inputs are half
template <typename scalar_t> // Input type
__device__ __forceinline__ float compute_mahalanobis_sq_float(
    const scalar_t* __restrict__ query_point,
    const scalar_t* __restrict__ ref_point,
    const scalar_t* __restrict__ inv_cov_ptr,
    int dim) 
{
    if (dim != 3) {
        // Consider using __trap() or asserting if only dim=3 is supported
        // For now, return a large value to indicate error
        return INFINITY;
    }

    float diff[3];
    float q[3], r[3];
    float inv_cov[9];

    // Load and convert inputs to float
    #pragma unroll
    for(int i=0; i<3; ++i) q[i] = static_cast<float>(query_point[i]);
    #pragma unroll
    for(int i=0; i<3; ++i) r[i] = static_cast<float>(ref_point[i]);
    #pragma unroll
    for(int i=0; i<9; ++i) inv_cov[i] = static_cast<float>(inv_cov_ptr[i]);

    diff[0] = q[0] - r[0];
    diff[1] = q[1] - r[1];
    diff[2] = q[2] - r[2];

    // temp = inv_cov * diff (in float)
    float temp[3];
    temp[0] = inv_cov[0] * diff[0] + inv_cov[1] * diff[1] + inv_cov[2] * diff[2];
    temp[1] = inv_cov[3] * diff[0] + inv_cov[4] * diff[1] + inv_cov[5] * diff[2];
    temp[2] = inv_cov[6] * diff[0] + inv_cov[7] * diff[1] + inv_cov[8] * diff[2];

    // result = dot_product(diff, temp) (in float)
    float result = diff[0] * temp[0] + diff[1] * temp[1] + diff[2] * temp[2];

    return result; // Return float
}

// --- Unified KDON Kernel using Shared Memory --- 
template <typename scalar_t> // scalar_t is the INPUT/OUTPUT type (e.g., half or float)
__global__ void kdon_unified_shared_mem_kernel(
    const scalar_t* __restrict__ query_points,
    const scalar_t* __restrict__ reference_points,
    const scalar_t* __restrict__ reference_inv_covariances, 
    const scalar_t* __restrict__ reference_opacities, // Added opacities
    const int64_t* __restrict__ query_lengths,
    const int64_t* __restrict__ reference_lengths,
    scalar_t* __restrict__ distances, // Output: Mahalanobis squared distances (in scalar_t)
    int64_t* __restrict__ indices,   // Output indices
    int batch_size,
    int num_query_points,
    int num_reference_points,
    int dim,
    int k
) {
    // 获取当前线程的全局索引
    const int thread_idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算当前处理的批次和查询点索引
    const int batch_idx = thread_idx_global / num_query_points;
    const int query_idx = thread_idx_global % num_query_points;
    
    // 检查批次索引是否越界
    if (batch_idx >= batch_size) return;
    
    // 获取当前批次有效的查询点和参考点数量
    const int64_t actual_query_length = query_lengths[batch_idx];
    const int64_t actual_ref_length = reference_lengths[batch_idx];

    // 检查查询点索引是否越界
    if (query_idx >= actual_query_length) {
         // 对于无效查询点，写入特殊值 (INFINITY and -1)
         #pragma unroll
         for (int i = 0; i < k; ++i) {
            distances[thread_idx_global * k + i] = INFINITY;
            indices[thread_idx_global * k + i] = -1;
         }
        return;
    }
    
    // 计算实际有效的 k 值
    const int valid_k = min((int)k, (int)actual_ref_length);
    
    // 如果没有有效的参考点
    if (actual_ref_length <= 0) {
        #pragma unroll
        for (int i = 0; i < k; i++) {
            distances[thread_idx_global * k + i] = INFINITY;
            indices[thread_idx_global * k + i] = -1;
        }
        return;
    }
    
    // 计算当前查询点在全局内存中的偏移量
    const size_t query_offset = (size_t)batch_idx * num_query_points * dim + (size_t)query_idx * dim;
    const scalar_t* q_ptr = query_points + query_offset;
    
    // Base pointer for inverse covariances for the current batch
    const size_t inv_cov_batch_offset = (size_t)batch_idx * num_reference_points * (dim * dim);
    const scalar_t* inv_cov_batch_ptr = reference_inv_covariances + inv_cov_batch_offset;
    const size_t opacity_batch_offset = (size_t)batch_idx * num_reference_points; // Opacity is (B, M, 1)
    const scalar_t* opacity_batch_ptr = reference_opacities + opacity_batch_offset;

    // --- Shared Memory Setup --- 
    extern __shared__ char shared_mem[];
    // Store scores and distances as FLOAT internally
    const int size_per_thread_score = k * sizeof(float); // Use float for score storage
    const int size_per_thread_dist = k * sizeof(float);  // Keep float for internal distance
    const int size_per_thread_idx = k * sizeof(int64_t);
    const int size_per_thread_total = size_per_thread_score + size_per_thread_dist + size_per_thread_idx;
    
    char* thread_shared_mem_start = shared_mem + threadIdx.x * size_per_thread_total;
    float* sh_scores = (float*)thread_shared_mem_start; // Store float scores
    float* sh_dists = (float*)(thread_shared_mem_start + size_per_thread_score); // Store float distances
    int64_t* sh_idxs = (int64_t*)(thread_shared_mem_start + size_per_thread_score + size_per_thread_dist);

    // Initialize shared memory 
    #pragma unroll
    for (int i = 0; i < k; i++) {
        sh_scores[i] = -INFINITY; // Use float negative infinity
        sh_dists[i] = INFINITY;   // Largest possible float distance
        sh_idxs[i] = -1;
    }
    
    // Iterate through all valid reference points
    for (int r = 0; r < actual_ref_length; r++) {
        size_t ref_offset = (size_t)batch_idx * num_reference_points * dim + (size_t)r * dim;
        const scalar_t* r_ptr = reference_points + ref_offset;
        const scalar_t* inv_cov_ptr = inv_cov_batch_ptr + (size_t)r * (dim * dim);
        // Load opacity and convert to float first
        const float opacity = static_cast<float>(opacity_batch_ptr[r]); 
        
        // Calculate Mahalanobis distance squared using float internally
        float dist_sq_float = compute_mahalanobis_sq_float<scalar_t>(q_ptr, r_ptr, inv_cov_ptr, dim);
        
        // Calculate density using float - Using faster intrinsic __expf
        float density_float = __expf(-0.5f * dist_sq_float); // Use __expf 
        
        // Calculate score using float
        float score_float = density_float * opacity;
        
        // --- Maintain Top-K based on MAX score (using FLOAT comparisons) --- 
        if (score_float > sh_scores[k - 1]) { 
            int insert_pos = k - 1;
            // Use float comparison
            while (insert_pos > 0 && score_float > sh_scores[insert_pos - 1]) { 
                // Shift elements down
                sh_scores[insert_pos] = sh_scores[insert_pos - 1];
                sh_dists[insert_pos] = sh_dists[insert_pos - 1]; 
                sh_idxs[insert_pos] = sh_idxs[insert_pos - 1];
                insert_pos--;
            }
            // Insert the new score (float), distance (float), and index
            sh_scores[insert_pos] = score_float; // Store float score
            sh_dists[insert_pos] = dist_sq_float; // Store corresponding float Mahalanobis distance
            sh_idxs[insert_pos] = r;
        }
    }

    __syncthreads(); // Wait for all threads to finish calculations
    
    // Write back the Mahalanobis distances and indices corresponding to Top-K scores
    const size_t output_base_idx = thread_idx_global * k;
    #pragma unroll
    for (int i = 0; i < valid_k; i++) {
        // Convert internal float distance back to output type (scalar_t)
        distances[output_base_idx + i] = static_cast<scalar_t>(sh_dists[i]); 
        indices[output_base_idx + i] = sh_idxs[i];
    }
    
    // Fill remaining slots if k > valid_k
    #pragma unroll
    for (int i = valid_k; i < k; i++) {
        distances[output_base_idx + i] = static_cast<scalar_t>(INFINITY); // Use static_cast for infinity too
        indices[output_base_idx + i] = -1;
    }
}

// --- Kernel Launcher --- 
template <typename scalar_t>
void kdon_cuda_impl(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& reference_inv_covariances,
    const torch::Tensor& reference_opacities, // Added
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    torch::Tensor& distances, // Mahalanobis distances (output type scalar_t)
    torch::Tensor& indices,
    int k
) {
    // 获取维度信息
    const int batch_size = query_points.size(0);
    const int num_query_points = query_points.size(1);
    const int num_reference_points = reference_points.size(1);
    const int dim = query_points.size(2);
    
    // 计算所需的总线程数
    long long total_threads_ll = (long long)batch_size * num_query_points;
    // 检查是否超过整型限制
    if (total_threads_ll > std::numeric_limits<int>::max()) {
        char error_msg[200];
        snprintf(error_msg, sizeof(error_msg), "Error: Total number of threads required (%lld) exceeds integer limits.", total_threads_ll);
        throw std::runtime_error(error_msg);
    }
    const int total_threads = (int)total_threads_ll;
    
    // 计算所需的线程块数量
    const int blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // 检查线程块数量是否超过设备限制
    int max_grid_dim_x;
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&max_grid_dim_x, cudaDevAttrMaxGridDimX, query_points.device().index()));
    if (blocks > max_grid_dim_x) {
         char error_msg[200];
         snprintf(error_msg, sizeof(error_msg), "Error: Required number of blocks (%d) exceeds device limit (%d).", blocks, max_grid_dim_x);
         throw std::runtime_error(error_msg);
    }

    // Calculate shared memory size (using FLOAT for score and dist)
    const int size_per_thread = k * (sizeof(float) + sizeof(float) + sizeof(int64_t)); // Back to float score
    const int shared_memory_size = THREADS_PER_BLOCK * size_per_thread;
    
    // 检查总共享内存是否超过块限制
    int max_shared_mem_per_block;
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, query_points.device().index()));
    if (shared_memory_size > max_shared_mem_per_block) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "Error: k=%d requires %d bytes shared memory per block (for KDON), exceeds device limit (%d). Reduce k.", 
              k, shared_memory_size, max_shared_mem_per_block);
        throw std::runtime_error(error_msg);
    }
    
    // 根据 norm 类型启动统一的共享内存核函数
    if (dim != 3) {
         throw std::runtime_error("kdon_cuda_impl currently only supports dim=3");
    }
    
    // Launch the KDON kernel 
    kdon_unified_shared_mem_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK, shared_memory_size>>>(
        query_points.data_ptr<scalar_t>(),
        reference_points.data_ptr<scalar_t>(),
        reference_inv_covariances.data_ptr<scalar_t>(), 
        reference_opacities.data_ptr<scalar_t>(), // Pass opacities
        query_lengths.data_ptr<int64_t>(),
        reference_lengths.data_ptr<int64_t>(),
        distances.data_ptr<scalar_t>(),
        indices.data_ptr<int64_t>(),
        batch_size,
        num_query_points,
        num_reference_points,
        dim,
        k
    );
    
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// Exported C++ interface function
std::tuple<at::Tensor, at::Tensor> KDenseOpacityNeighborCuda(
    const at::Tensor& query_points,
    const at::Tensor& reference_points,
    const at::Tensor& reference_inv_covariances,
    const at::Tensor& reference_opacities, // Added
    const at::Tensor& query_lengths,
    const at::Tensor& reference_lengths,
    int K,
    int version 
) {
    // 输入张量检查 (device, shape, dtype)
    TORCH_CHECK(query_points.is_cuda(), "query_points must be a CUDA tensor");
    TORCH_CHECK(reference_points.is_cuda(), "reference_points must be a CUDA tensor");
    TORCH_CHECK(query_lengths.is_cuda(), "query_lengths must be a CUDA tensor");
    TORCH_CHECK(reference_lengths.is_cuda(), "reference_lengths must be a CUDA tensor");
    
    TORCH_CHECK(query_points.dim() == 3, "query_points must be 3D (B, N, D)");
    TORCH_CHECK(reference_points.dim() == 3, "reference_points must be 3D (B, M, D)");
    TORCH_CHECK(query_lengths.dim() == 1, "query_lengths must be 1D (B)");
    TORCH_CHECK(reference_lengths.dim() == 1, "reference_lengths must be 1D (B)");

    TORCH_CHECK(query_points.size(0) == reference_points.size(0), "Batch size mismatch between query and reference points");
    TORCH_CHECK(query_points.size(0) == query_lengths.size(0), "Batch size mismatch between query points and query lengths");
    TORCH_CHECK(query_points.size(0) == reference_lengths.size(0), "Batch size mismatch between query points and reference lengths");
    TORCH_CHECK(query_points.size(2) == reference_points.size(2), "Dimension mismatch between query and reference points");

    TORCH_CHECK(query_points.scalar_type() == reference_points.scalar_type(), "Scalar type mismatch between query and reference points");
    TORCH_CHECK(query_lengths.scalar_type() == torch::kInt64, "query_lengths must be torch.int64");
    TORCH_CHECK(reference_lengths.scalar_type() == torch::kInt64, "reference_lengths must be torch.int64");

    TORCH_CHECK(K > 0, "k must be positive");

    // 获取维度信息
    const int batch_size = query_points.size(0);
    const int num_query_points = query_points.size(1);
    
    // 设置当前的CUDA设备
    at::cuda::CUDAGuard device_guard(query_points.device());
    
    // 为结果分配内存（在CUDA上）
    auto options_dist = query_points.options();
    auto options_idx = query_points.options().dtype(torch::kInt64);

    auto distances = torch::empty({batch_size, num_query_points, K}, options_dist);
    auto indices = torch::empty({batch_size, num_query_points, K}, options_idx);
    
    // 根据数据类型调用相应的CUDA实现
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_points.scalar_type(), "KDenseOpacityNeighborCuda", ([&] {
        kdon_cuda_impl<scalar_t>(
            query_points,
            reference_points,
            reference_inv_covariances,
            reference_opacities, // Pass opacities
            query_lengths,
            reference_lengths,
            distances,
            indices,
            K
        );
    }));
    
    return std::make_tuple(distances, indices);
} 