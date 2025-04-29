#include "kdn.h"
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
constexpr int THREADS_PER_BLOCK = 256;

// 辅助函数，用于计算马氏距离平方 (device code)
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_mahalanobis_sq(
    const scalar_t* query_point,
    const scalar_t* ref_point,
    const scalar_t* inv_cov_ptr, // Pointer to the 3x3 inverse covariance matrix for ref_point
    int dim) {
    
    // Assuming dim = 3 for explicit calculation in device code
    // A compile-time check or template specialization might be better for non-3D
    scalar_t diff[3];
    diff[0] = query_point[0] - ref_point[0];
    diff[1] = query_point[1] - ref_point[1];
    diff[2] = query_point[2] - ref_point[2];

    // temp = inv_cov * diff
    scalar_t temp[3];
    temp[0] = inv_cov_ptr[0] * diff[0] + inv_cov_ptr[1] * diff[1] + inv_cov_ptr[2] * diff[2];
    temp[1] = inv_cov_ptr[3] * diff[0] + inv_cov_ptr[4] * diff[1] + inv_cov_ptr[5] * diff[2];
    temp[2] = inv_cov_ptr[6] * diff[0] + inv_cov_ptr[7] * diff[1] + inv_cov_ptr[8] * diff[2];

    // result = dot_product(diff, temp)
    scalar_t result = diff[0] * temp[0] + diff[1] * temp[1] + diff[2] * temp[2];

    // Clamp result to be non-negative (though theoretically should be)
    // return max(static_cast<scalar_t>(0.0), result); // Removed clamping
    return result;
}

// --- Unified KDN Kernel using Shared Memory --- 
template <typename scalar_t>
__global__ void kdn_unified_shared_mem_kernel(
    const scalar_t* __restrict__ query_points,
    const scalar_t* __restrict__ reference_points,
    const scalar_t* __restrict__ reference_inv_covariances, // Added inv_cov input
    const int64_t* __restrict__ query_lengths,
    const int64_t* __restrict__ reference_lengths,
    scalar_t* __restrict__ distances, // Output distances (Mahalanobis squared)
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

    // 使用动态共享内存为每个线程存储 Top-K
    extern __shared__ char shared_mem[];
    // More explicit pointer calculation within the shared memory block
    const int size_per_thread_dist = k * sizeof(scalar_t);
    const int size_per_thread_idx = k * sizeof(int64_t);
    const int size_per_thread_total = size_per_thread_dist + size_per_thread_idx;
    
    char* thread_shared_mem_start = shared_mem + threadIdx.x * size_per_thread_total;
    scalar_t* sh_dists = (scalar_t*)thread_shared_mem_start;
    int64_t* sh_idxs = (int64_t*)(thread_shared_mem_start + size_per_thread_dist);

    // 初始化共享内存中的 Top-K (距离为无穷大，索引为-1)
    #pragma unroll
    for (int i = 0; i < k; i++) {
        sh_dists[i] = INFINITY;
        sh_idxs[i] = -1;
    }
    
    // 遍历所有有效的参考点，计算距离并维护 Top-K
    for (int r = 0; r < actual_ref_length; r++) {
        // 计算参考点偏移量
        size_t ref_offset = (size_t)batch_idx * num_reference_points * dim + (size_t)r * dim;
        const scalar_t* r_ptr = reference_points + ref_offset;

        // Calculate pointer to the inv_cov for this specific reference point r
        // Each inv_cov is dim*dim elements (e.g., 3*3=9 for dim=3)
        const scalar_t* inv_cov_ptr = inv_cov_batch_ptr + (size_t)r * (dim * dim);
        
        // Calculate Mahalanobis distance squared
        scalar_t dist = compute_mahalanobis_sq<scalar_t>(
            q_ptr, 
            r_ptr,
            inv_cov_ptr,
            dim
        );
        
        // 如果当前距离小于已存储的最大距离，则进行插入排序
        if (dist < sh_dists[k - 1]) {
            int insert_pos = k - 1;
            // 找到正确的插入位置
            while (insert_pos > 0 && dist < sh_dists[insert_pos - 1]) {
                sh_dists[insert_pos] = sh_dists[insert_pos - 1];
                sh_idxs[insert_pos] = sh_idxs[insert_pos - 1];
                insert_pos--;
            }
            // 插入新的距离和索引
            sh_dists[insert_pos] = dist;
            sh_idxs[insert_pos] = r;
        }
    }

    // --- Add synchronization before writing back --- 
    __syncthreads();
    
    // 将最终的 Top-K 结果从共享内存写回全局内存
    #pragma unroll
    for (int i = 0; i < valid_k; i++) {
        distances[thread_idx_global * k + i] = sh_dists[i];
        indices[thread_idx_global * k + i] = sh_idxs[i];
    }
    
    // 对于 k > valid_k 的情况，填充无效值
    #pragma unroll
    for (int i = valid_k; i < k; i++) {
        distances[thread_idx_global * k + i] = INFINITY;
        indices[thread_idx_global * k + i] = -1;
    }
}

// --- Kernel Launcher --- 
template <typename scalar_t>
void kdn_cuda_impl(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& reference_inv_covariances, // Added inv_cov
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    torch::Tensor& distances,
    torch::Tensor& indices,
    int k
    // Removed norm parameter
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

    // --- Always use the Shared Memory Kernel --- 
    // 计算每个线程所需的共享内存大小
    const int size_per_thread = k * (sizeof(scalar_t) + sizeof(int64_t));
    const int shared_mem_size = size_per_thread * THREADS_PER_BLOCK;
    
    // 检查总共享内存是否超过块限制
    int max_shared_mem_per_block;
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, query_points.device().index()));
    if (shared_mem_size > max_shared_mem_per_block) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "Error: k=%d requires %d bytes shared memory per block, exceeds device limit (%d). Reduce k or increase shared memory.", 
              k, shared_mem_size, max_shared_mem_per_block);
        throw std::runtime_error(error_msg);
    }
    
    // 根据 norm 类型启动统一的共享内存核函数
    if (dim != 3) {
         throw std::runtime_error("kdn_cuda_impl currently only supports dim=3");
    }
    
    // Launch the KDN kernel (no norm dispatch needed)
    kdn_unified_shared_mem_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
        query_points.data_ptr<scalar_t>(),
        reference_points.data_ptr<scalar_t>(),
        reference_inv_covariances.data_ptr<scalar_t>(), // Pass inv_cov pointer
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
    
    // 检查核函数启动和执行中的 CUDA 错误
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// Exported C++ interface function
std::tuple<at::Tensor, at::Tensor> KDenseNeighborCuda(
    const at::Tensor& query_points,
    const at::Tensor& reference_points,
    const at::Tensor& reference_inv_covariances, // Added inv_cov
    const at::Tensor& query_lengths,
    const at::Tensor& reference_lengths,
    int K,
    int version // currently unused
    // Removed norm parameter
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
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_points.scalar_type(), "KDenseNeighborCuda", ([&] {
        kdn_cuda_impl<scalar_t>(
            query_points,
            reference_points,
            reference_inv_covariances, // Pass inv_cov
            query_lengths,
            reference_lengths,
            distances,
            indices,
            K
        );
    }));
    
    return std::make_tuple(distances, indices);
} 