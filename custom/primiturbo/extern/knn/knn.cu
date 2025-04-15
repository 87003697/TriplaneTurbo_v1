#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAGuard.h>
#include <cstdio>

// CUDA错误检查宏
#define CUDA_CHECK_ERROR(val) { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(-1); \
    } \
}

// 常量
constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_SHARED_MEM_SIZE = 49152;  // 48KB

// 辅助函数，用于计算一批距离
template <typename scalar_t, int NORM>
__device__ __forceinline__ scalar_t compute_distance(
    const scalar_t* query_point,
    const scalar_t* ref_point,
    int dim) {
    scalar_t result = 0;
    for (int d = 0; d < dim; d++) {
        scalar_t diff = query_point[d] - ref_point[d];
        if (NORM == 1) {
            result += abs(diff);
        } else {
            result += diff * diff;
        }
    }
    return result;
}

// 计算所有查询点与参考点之间的距离
template <typename scalar_t, int NORM>
__global__ void compute_distances_kernel(
    const scalar_t* __restrict__ query_points,   // [B, N, D]
    const scalar_t* __restrict__ reference_points, // [B, M, D]
    const int64_t* __restrict__ query_lengths,     // [B]
    const int64_t* __restrict__ reference_lengths, // [B]
    scalar_t* __restrict__ distances,              // [B, N, M]
    int batch_size,
    int num_query_points,
    int num_reference_points,
    int dim) {
    
    const int b = blockIdx.z;
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= batch_size) return;
    
    const int64_t actual_query_length = query_lengths[b];
    const int64_t actual_ref_length = reference_lengths[b];
    
    if (q_idx >= actual_query_length || r_idx >= actual_ref_length) return;
    
    // 计算对应点的偏移量
    const int query_offset = b * num_query_points * dim + q_idx * dim;
    const int ref_offset = b * num_reference_points * dim + r_idx * dim;
    
    // 计算距离
    scalar_t dist = compute_distance<scalar_t, NORM>(
        &query_points[query_offset],
        &reference_points[ref_offset],
        dim
    );
    
    // 存储计算结果
    distances[b * num_query_points * num_reference_points + q_idx * num_reference_points + r_idx] = dist;
}

// CUDA 核函数：计算查询点与参考点的距离（L2范数）
template <typename scalar_t>
__global__ void knn_l2_kernel(
    const scalar_t* __restrict__ query_points,
    const scalar_t* __restrict__ reference_points,
    const int64_t* __restrict__ query_lengths,
    const int64_t* __restrict__ reference_lengths,
    scalar_t* __restrict__ distances,
    int64_t* __restrict__ indices,
    int batch_size,
    int num_query_points,
    int num_reference_points,
    int dim,
    int k
) {
    // 获取当前线程的索引
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算当前处理的批次和查询点
    int batch_idx = thread_idx / num_query_points;
    int query_idx = thread_idx % num_query_points;
    
    // 如果超出有效范围，退出
    if (batch_idx >= batch_size) return;
    
    // 检查当前查询点是否在有效范围内
    if (query_idx >= query_lengths[batch_idx]) {
        // 超出有效范围，将结果设为0
        for (int i = 0; i < k; i++) {
            distances[batch_idx * num_query_points * k + query_idx * k + i] = 0;
            indices[batch_idx * num_query_points * k + query_idx * k + i] = 0;
        }
        return;
    }
    
    // 获取当前批次中参考点的有效数量
    int valid_ref_length = reference_lengths[batch_idx];
    
    // 计算最大k值，不能超过参考点数量
    int valid_k = min(k, valid_ref_length);
    
    // 如果没有参考点，退出
    if (valid_ref_length <= 0) {
        for (int i = 0; i < k; i++) {
            distances[batch_idx * num_query_points * k + query_idx * k + i] = INFINITY;
            indices[batch_idx * num_query_points * k + query_idx * k + i] = -1;
        }
        return;
    }
    
    // 获取当前查询点的索引
    int query_offset = batch_idx * num_query_points * dim + query_idx * dim;
    
    // 创建临时数组保存所有距离和索引
    // 使用动态共享内存优化
    extern __shared__ char shared_mem[];
    scalar_t* dists = (scalar_t*)shared_mem;
    int64_t* idxs = (int64_t*)(dists + blockDim.x * k);
    
    // 初始化距离为最大值，索引为-1
    for (int i = 0; i < k; i++) {
        dists[threadIdx.x * k + i] = INFINITY;
        idxs[threadIdx.x * k + i] = -1;
    }
    
    // 计算当前查询点与所有参考点的距离
    for (int r = 0; r < valid_ref_length; r++) {
        int ref_offset = batch_idx * num_reference_points * dim + r * dim;
        
        // 计算L2距离
        scalar_t dist = 0;
        for (int d = 0; d < dim; d++) {
            scalar_t diff = query_points[query_offset + d] - reference_points[ref_offset + d];
            dist += diff * diff;
        }
        
        // 将当前参考点插入到k个最近邻中
        // 使用插入排序以保持有序状态
        if (dist < dists[threadIdx.x * k + k-1]) {
            // 找到适合插入的位置
            int insert_idx = k - 1;
            while (insert_idx > 0 && dist < dists[threadIdx.x * k + insert_idx - 1]) {
                insert_idx--;
            }
            
            // 移动元素腾出空间
            for (int j = k - 1; j > insert_idx; j--) {
                dists[threadIdx.x * k + j] = dists[threadIdx.x * k + j - 1];
                idxs[threadIdx.x * k + j] = idxs[threadIdx.x * k + j - 1];
            }
            
            // 插入新的距离和索引
            dists[threadIdx.x * k + insert_idx] = dist;
            idxs[threadIdx.x * k + insert_idx] = r;
        }
    }
    
    // 将结果拷贝到全局内存
    for (int i = 0; i < valid_k; i++) {
        distances[batch_idx * num_query_points * k + query_idx * k + i] = dists[threadIdx.x * k + i];
        indices[batch_idx * num_query_points * k + query_idx * k + i] = idxs[threadIdx.x * k + i];
    }
    
    // 填充剩余位置（如果k > valid_k）
    for (int i = valid_k; i < k; i++) {
        distances[batch_idx * num_query_points * k + query_idx * k + i] = INFINITY;
        indices[batch_idx * num_query_points * k + query_idx * k + i] = -1;
    }
}

// CUDA 核函数：计算查询点与参考点的距离（L1范数）
template <typename scalar_t>
__global__ void knn_l1_kernel(
    const scalar_t* __restrict__ query_points,
    const scalar_t* __restrict__ reference_points,
    const int64_t* __restrict__ query_lengths,
    const int64_t* __restrict__ reference_lengths,
    scalar_t* __restrict__ distances,
    int64_t* __restrict__ indices,
    int batch_size,
    int num_query_points,
    int num_reference_points,
    int dim,
    int k
) {
    // 获取当前线程的索引
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算当前处理的批次和查询点
    int batch_idx = thread_idx / num_query_points;
    int query_idx = thread_idx % num_query_points;
    
    // 如果超出有效范围，退出
    if (batch_idx >= batch_size) return;
    
    // 检查当前查询点是否在有效范围内
    if (query_idx >= query_lengths[batch_idx]) {
        // 超出有效范围，将结果设为0
        for (int i = 0; i < k; i++) {
            distances[batch_idx * num_query_points * k + query_idx * k + i] = 0;
            indices[batch_idx * num_query_points * k + query_idx * k + i] = 0;
        }
        return;
    }
    
    // 获取当前批次中参考点的有效数量
    int valid_ref_length = reference_lengths[batch_idx];
    
    // 计算最大k值，不能超过参考点数量
    int valid_k = min(k, valid_ref_length);
    
    // 如果没有参考点，退出
    if (valid_ref_length <= 0) {
        for (int i = 0; i < k; i++) {
            distances[batch_idx * num_query_points * k + query_idx * k + i] = INFINITY;
            indices[batch_idx * num_query_points * k + query_idx * k + i] = -1;
        }
        return;
    }
    
    // 获取当前查询点的索引
    int query_offset = batch_idx * num_query_points * dim + query_idx * dim;
    
    // 创建临时数组保存所有距离和索引
    // 使用动态共享内存优化
    extern __shared__ char shared_mem[];
    scalar_t* dists = (scalar_t*)shared_mem;
    int64_t* idxs = (int64_t*)(dists + blockDim.x * k);
    
    // 初始化距离为最大值，索引为-1
    for (int i = 0; i < k; i++) {
        dists[threadIdx.x * k + i] = INFINITY;
        idxs[threadIdx.x * k + i] = -1;
    }
    
    // 计算当前查询点与所有参考点的距离
    for (int r = 0; r < valid_ref_length; r++) {
        int ref_offset = batch_idx * num_reference_points * dim + r * dim;
        
        // 计算L1距离
        scalar_t dist = 0;
        for (int d = 0; d < dim; d++) {
            dist += abs(query_points[query_offset + d] - reference_points[ref_offset + d]);
        }
        
        // 将当前参考点插入到k个最近邻中
        // 使用插入排序以保持有序状态
        if (dist < dists[threadIdx.x * k + k-1]) {
            // 找到适合插入的位置
            int insert_idx = k - 1;
            while (insert_idx > 0 && dist < dists[threadIdx.x * k + insert_idx - 1]) {
                insert_idx--;
            }
            
            // 移动元素腾出空间
            for (int j = k - 1; j > insert_idx; j--) {
                dists[threadIdx.x * k + j] = dists[threadIdx.x * k + j - 1];
                idxs[threadIdx.x * k + j] = idxs[threadIdx.x * k + j - 1];
            }
            
            // 插入新的距离和索引
            dists[threadIdx.x * k + insert_idx] = dist;
            idxs[threadIdx.x * k + insert_idx] = r;
        }
    }
    
    // 将结果拷贝到全局内存
    for (int i = 0; i < valid_k; i++) {
        distances[batch_idx * num_query_points * k + query_idx * k + i] = dists[threadIdx.x * k + i];
        indices[batch_idx * num_query_points * k + query_idx * k + i] = idxs[threadIdx.x * k + i];
    }
    
    // 填充剩余位置（如果k > valid_k）
    for (int i = valid_k; i < k; i++) {
        distances[batch_idx * num_query_points * k + query_idx * k + i] = INFINITY;
        indices[batch_idx * num_query_points * k + query_idx * k + i] = -1;
    }
}

// 针对大规模数据的优化批处理核函数
template <typename scalar_t, int NORM>
__global__ void knn_batch_kernel(
    const scalar_t* __restrict__ query_points,
    const scalar_t* __restrict__ reference_points,
    const int64_t* __restrict__ query_lengths,
    const int64_t* __restrict__ reference_lengths,
    scalar_t* __restrict__ distances,
    int64_t* __restrict__ indices,
    int batch_size,
    int num_query_points,
    int num_reference_points,
    int dim,
    int k
) {
    // 获取当前线程的索引
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算当前处理的批次和查询点
    const int batch_idx = thread_idx / num_query_points;
    const int query_idx = thread_idx % num_query_points;
    
    // 如果超出有效范围，退出
    if (batch_idx >= batch_size) return;
    
    // 检查当前查询点是否在有效范围内
    if (query_idx >= query_lengths[batch_idx]) {
        // 超出有效范围，将结果设为0
        for (int i = 0; i < k; i++) {
            distances[batch_idx * num_query_points * k + query_idx * k + i] = 0;
            indices[batch_idx * num_query_points * k + query_idx * k + i] = 0;
        }
        return;
    }
    
    // 获取当前批次中参考点的有效数量
    const int valid_ref_length = reference_lengths[batch_idx];
    
    // 计算最大k值，不能超过参考点数量
    const int valid_k = min(k, valid_ref_length);
    
    // 如果没有参考点，退出
    if (valid_ref_length <= 0) {
        for (int i = 0; i < k; i++) {
            distances[batch_idx * num_query_points * k + query_idx * k + i] = INFINITY;
            indices[batch_idx * num_query_points * k + query_idx * k + i] = -1;
        }
        return;
    }
    
    // 获取当前查询点的索引
    const int query_offset = batch_idx * num_query_points * dim + query_idx * dim;
    
    // 使用寄存器存储k个最近邻
    // 注意：k不能太大，否则会导致寄存器溢出
    scalar_t top_dists[32];  // 最多支持k=32
    int64_t top_idxs[32];
    
    // 检查k是否超出数组大小
    if (k > 32) {
        if (thread_idx == 0) {
            printf("警告：k值(%d)超出寄存器数组大小(32)，结果可能不正确\n", k);
        }
        return;
    }
    
    // 初始化
    for (int i = 0; i < k; i++) {
        top_dists[i] = INFINITY;
        top_idxs[i] = -1;
    }
    
    // 计算当前查询点与所有参考点的距离
    for (int r = 0; r < valid_ref_length; r++) {
        const int ref_offset = batch_idx * num_reference_points * dim + r * dim;
        
        // 计算距离
        scalar_t dist = 0;
        
        if (NORM == 1) {  // L1范数
            for (int d = 0; d < dim; d++) {
                dist += abs(query_points[query_offset + d] - reference_points[ref_offset + d]);
            }
        } else {  // L2范数
            for (int d = 0; d < dim; d++) {
                const scalar_t diff = query_points[query_offset + d] - reference_points[ref_offset + d];
                dist += diff * diff;
            }
        }
        
        // 将当前参考点插入到k个最近邻中
        if (dist < top_dists[k-1]) {
            // 找到适合插入的位置
            int insert_idx = k - 1;
            while (insert_idx > 0 && dist < top_dists[insert_idx - 1]) {
                insert_idx--;
            }
            
            // 移动元素腾出空间
            for (int j = k - 1; j > insert_idx; j--) {
                top_dists[j] = top_dists[j - 1];
                top_idxs[j] = top_idxs[j - 1];
            }
            
            // 插入新的距离和索引
            top_dists[insert_idx] = dist;
            top_idxs[insert_idx] = r;
        }
    }
    
    // 将结果拷贝到全局内存
    for (int i = 0; i < valid_k; i++) {
        distances[batch_idx * num_query_points * k + query_idx * k + i] = top_dists[i];
        indices[batch_idx * num_query_points * k + query_idx * k + i] = top_idxs[i];
    }
    
    // 填充剩余位置（如果k > valid_k）
    for (int i = valid_k; i < k; i++) {
        distances[batch_idx * num_query_points * k + query_idx * k + i] = INFINITY;
        indices[batch_idx * num_query_points * k + query_idx * k + i] = -1;
    }
}

// 启动CUDA核函数，包含内存管理和启动配置
template <typename scalar_t>
void knn_cuda_impl(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    torch::Tensor& distances,
    torch::Tensor& indices,
    int k,
    int norm
) {
    // 获取维度信息
    int batch_size = query_points.size(0);
    int num_query_points = query_points.size(1);
    int num_reference_points = reference_points.size(1);
    int dim = query_points.size(2);
    
    // 计算CUDA网格和块大小
    // 每个线程处理一个查询点
    int total_threads = batch_size * num_query_points;
    int blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // 检查k值是否在支持范围内
    if (k > 32) {
        // 使用共享内存版本
        const int shared_mem_items_per_thread = k * 2;  // k distances and k indices per thread
        const int items_size = sizeof(scalar_t) > sizeof(int64_t) ? sizeof(scalar_t) : sizeof(int64_t);
        const int shared_mem_size = shared_mem_items_per_thread * THREADS_PER_BLOCK * items_size;
        
        // 检查共享内存大小
        if (shared_mem_size > MAX_SHARED_MEM_SIZE) {
            printf("警告: 需要的共享内存(%d)超出设备限制(%d)，结果可能不正确\n", 
                  shared_mem_size, MAX_SHARED_MEM_SIZE);
        }
        
        // 使用L1或L2范数的核函数
        if (norm == 1) {
            knn_l1_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                query_points.data_ptr<scalar_t>(),
                reference_points.data_ptr<scalar_t>(),
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
        } else {
            knn_l2_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                query_points.data_ptr<scalar_t>(),
                reference_points.data_ptr<scalar_t>(),
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
        }
    } else {
        // 使用寄存器优化版本，k <= 32
        if (norm == 1) {
            knn_batch_kernel<scalar_t, 1><<<blocks, THREADS_PER_BLOCK>>>(
                query_points.data_ptr<scalar_t>(),
                reference_points.data_ptr<scalar_t>(),
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
        } else {
            knn_batch_kernel<scalar_t, 2><<<blocks, THREADS_PER_BLOCK>>>(
                query_points.data_ptr<scalar_t>(),
                reference_points.data_ptr<scalar_t>(),
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
        }
    }
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__);
    } else {
        // 同步设备以确保所有操作完成
        cudaDeviceSynchronize();
    }
}

// 计算KNN的CUDA入口函数
std::tuple<torch::Tensor, torch::Tensor> KNearestNeighborCuda(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    int k,
    int norm
) {
    // 确保输入张量在CUDA上
    TORCH_CHECK(query_points.is_cuda(), "query_points must be a CUDA tensor");
    TORCH_CHECK(reference_points.is_cuda(), "reference_points must be a CUDA tensor");
    TORCH_CHECK(query_lengths.is_cuda(), "query_lengths must be a CUDA tensor");
    TORCH_CHECK(reference_lengths.is_cuda(), "reference_lengths must be a CUDA tensor");
    
    // 确保输入张量形状正确
    TORCH_CHECK(query_points.dim() == 3, "query_points must be 3D");
    TORCH_CHECK(reference_points.dim() == 3, "reference_points must be 3D");
    TORCH_CHECK(query_lengths.dim() == 1, "query_lengths must be 1D");
    TORCH_CHECK(reference_lengths.dim() == 1, "reference_lengths must be 1D");
    
    // 获取维度信息
    int batch_size = query_points.size(0);
    int num_query_points = query_points.size(1);
    
    // 设置当前的CUDA设备
    at::cuda::CUDAGuard device_guard(query_points.device());
    
    // 为结果分配内存（在CUDA上）
    auto distances = torch::empty({batch_size, num_query_points, k}, 
                                 query_points.options());
    auto indices = torch::empty({batch_size, num_query_points, k}, 
                               query_points.options().dtype(torch::kInt64));
    
    // 根据数据类型调用相应的实现
    AT_DISPATCH_FLOATING_TYPES(query_points.scalar_type(), "KNearestNeighborCuda", ([&] {
        knn_cuda_impl<scalar_t>(
            query_points,
            reference_points,
            query_lengths,
            reference_lengths,
            distances,
            indices,
            k,
            norm
        );
    }));
    
    return std::make_tuple(distances, indices);
} 