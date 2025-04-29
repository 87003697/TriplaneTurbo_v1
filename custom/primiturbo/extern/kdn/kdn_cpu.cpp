#include "kdn.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <omp.h>

// 计算点与点之间的马氏距离平方
template <typename scalar_t>
scalar_t compute_mahalanobis_sq(
    const scalar_t* query_point,
    const scalar_t* ref_point,
    const scalar_t* inv_cov_ptr, // Pointer to the 3x3 inverse covariance matrix for ref_point
    int dim) {
    
    // Assuming dim = 3 for simplicity in explicit calculation
    if (dim != 3) {
        // Fallback or error for non-3D cases (not implemented here)
        // For now, return a large value or throw an error
        // Let's throw an error for clarity, though a real implementation might need a general solution
        throw std::runtime_error("compute_mahalanobis_sq currently only supports dim=3");
        // return std::numeric_limits<scalar_t>::max(); 
    }

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

    // Ensure non-negative result, although Mahalanobis squared should be >= 0 if inv_cov is positive semi-definite
    // return std::max(static_cast<scalar_t>(0.0), result); // Removed clamping to allow comparison
    return result;
}

// 距离和索引对，用于排序
template <typename scalar_t>
struct DistIndexPair {
    scalar_t distance;
    int64_t index;
    
    bool operator<(const DistIndexPair& other) const {
        return distance < other.distance;
    }
};

// KDN CPU实现模板
template <typename scalar_t>
void kdn_cpu_impl(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& reference_inv_covariances,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    torch::Tensor& distances,
    torch::Tensor& indices,
    int k) {
    
    // 获取维度信息
    int batch_size = query_points.size(0);
    int num_query_points = query_points.size(1);
    int num_reference_points = reference_points.size(1);
    int dim = query_points.size(2);
    
    if (dim != 3) {
         throw std::runtime_error("kdn_cpu_impl currently only supports dim=3 due to compute_mahalanobis_sq");
    }
    
    // 获取数据指针
    const scalar_t* query_ptr = query_points.data_ptr<scalar_t>();
    const scalar_t* ref_ptr = reference_points.data_ptr<scalar_t>();
    const scalar_t* inv_cov_base_ptr = reference_inv_covariances.data_ptr<scalar_t>();
    const int64_t* query_lengths_ptr = query_lengths.data_ptr<int64_t>();
    const int64_t* ref_lengths_ptr = reference_lengths.data_ptr<int64_t>();
    scalar_t* distances_ptr = distances.data_ptr<scalar_t>();
    int64_t* indices_ptr = indices.data_ptr<int64_t>();
    
    // 使用OpenMP并行化批次和查询点的处理
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int q = 0; q < num_query_points; q++) {
            // 跳过超出有效长度的查询点
            if (q >= query_lengths_ptr[b]) {
                // 填充超出有效长度的结果为0
                for (int i = 0; i < k; i++) {
                    int result_idx = (b * num_query_points + q) * k + i;
                    distances_ptr[result_idx] = 0;
                    indices_ptr[result_idx] = 0;
                }
                continue;
            }
            
            // 计算最大k值，不能超过实际参考点数量
            int valid_k = std::min(k, static_cast<int>(ref_lengths_ptr[b]));
            
            // 获取当前查询点的数据指针
            const scalar_t* q_ptr = query_ptr + (b * num_query_points + q) * dim;
            int64_t current_ref_length = ref_lengths_ptr[b];
            
            // 计算与所有参考点的距离
            std::vector<DistIndexPair<scalar_t>> dist_idx_pairs;
            dist_idx_pairs.reserve(current_ref_length);
            
            for (int r = 0; r < current_ref_length; r++) {
                const scalar_t* r_ptr = ref_ptr + (b * num_reference_points + r) * dim;
                const scalar_t* inv_cov_ptr = inv_cov_base_ptr + (b * num_reference_points + r) * (dim * dim);
                scalar_t dist = compute_mahalanobis_sq<scalar_t>(q_ptr, r_ptr, inv_cov_ptr, dim);
                dist_idx_pairs.push_back({dist, r});
            }
            
            // 部分排序，只找前k个最近的
            if (valid_k < dist_idx_pairs.size()) {
                std::partial_sort(dist_idx_pairs.begin(), 
                                 dist_idx_pairs.begin() + valid_k,
                                 dist_idx_pairs.end());
            } else {
                std::sort(dist_idx_pairs.begin(), dist_idx_pairs.end());
            }
            
            // 存储结果
            for (int i = 0; i < valid_k; i++) {
                int result_idx = (b * num_query_points + q) * k + i;
                distances_ptr[result_idx] = dist_idx_pairs[i].distance;
                indices_ptr[result_idx] = dist_idx_pairs[i].index;
            }
            
            // 如果k大于实际参考点数量，填充剩余位置
            for (int i = valid_k; i < k; i++) {
                int result_idx = (b * num_query_points + q) * k + i;
                distances_ptr[result_idx] = std::numeric_limits<scalar_t>::max();
                indices_ptr[result_idx] = -1;
            }
        }
    }
}

// 针对大规模点云的优化KDN CPU实现
template <typename scalar_t>
void kdn_cpu_optimized_impl(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& reference_inv_covariances,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    torch::Tensor& distances,
    torch::Tensor& indices,
    int k) {
    
    // 获取维度信息
    int batch_size = query_points.size(0);
    int num_query_points = query_points.size(1);
    int num_reference_points = reference_points.size(1);
    int dim = query_points.size(2);
    
    if (dim != 3) {
         throw std::runtime_error("kdn_cpu_optimized_impl currently only supports dim=3 due to compute_mahalanobis_sq");
    }
    
    const scalar_t* inv_cov_base_ptr = reference_inv_covariances.data_ptr<scalar_t>();
    
    // 获取数据指针
    const scalar_t* query_ptr = query_points.data_ptr<scalar_t>();
    const scalar_t* ref_ptr = reference_points.data_ptr<scalar_t>();
    const int64_t* query_lengths_ptr = query_lengths.data_ptr<int64_t>();
    const int64_t* ref_lengths_ptr = reference_lengths.data_ptr<int64_t>();
    scalar_t* distances_ptr = distances.data_ptr<scalar_t>();
    int64_t* indices_ptr = indices.data_ptr<int64_t>();
    
    // 使用OpenMP并行化批次的处理
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        int valid_query_len = query_lengths_ptr[b];
        int valid_ref_len = ref_lengths_ptr[b];
        int valid_k = std::min(k, valid_ref_len);
        
        // 为每个OpenMP线程预分配内存，避免频繁分配
        std::vector<std::vector<DistIndexPair<scalar_t>>> all_dist_idx_pairs(valid_query_len);
        
        // 预分配内存
        for (int q = 0; q < valid_query_len; q++) {
            all_dist_idx_pairs[q].reserve(valid_ref_len);
        }
        
        // 计算所有距离
        for (int q = 0; q < valid_query_len; q++) {
            const scalar_t* q_ptr = query_ptr + (b * num_query_points + q) * dim;
            auto& dist_idx_pairs = all_dist_idx_pairs[q];
            
            for (int r = 0; r < valid_ref_len; r++) {
                const scalar_t* r_ptr = ref_ptr + (b * num_reference_points + r) * dim;
                const scalar_t* inv_cov_ptr = inv_cov_base_ptr + (b * num_reference_points + r) * (dim * dim);
                scalar_t dist = compute_mahalanobis_sq<scalar_t>(q_ptr, r_ptr, inv_cov_ptr, dim);
                dist_idx_pairs.push_back({dist, r});
            }
        }
        
        // 找出每个查询点的k个最近邻
        #pragma omp parallel for
        for (int q = 0; q < valid_query_len; q++) {
            auto& dist_idx_pairs = all_dist_idx_pairs[q];
            
            // 部分排序，只找前k个最近的
            if (valid_k < dist_idx_pairs.size()) {
                std::partial_sort(dist_idx_pairs.begin(), 
                                 dist_idx_pairs.begin() + valid_k,
                                 dist_idx_pairs.end());
            } else {
                std::sort(dist_idx_pairs.begin(), dist_idx_pairs.end());
            }
            
            // 存储结果
            for (int i = 0; i < valid_k; i++) {
                int result_idx = (b * num_query_points + q) * k + i;
                distances_ptr[result_idx] = dist_idx_pairs[i].distance;
                indices_ptr[result_idx] = dist_idx_pairs[i].index;
            }
            
            // 填充剩余位置
            for (int i = valid_k; i < k; i++) {
                int result_idx = (b * num_query_points + q) * k + i;
                distances_ptr[result_idx] = std::numeric_limits<scalar_t>::max();
                indices_ptr[result_idx] = -1;
            }
        }
        
        // 填充超出有效查询长度的结果
        for (int q = valid_query_len; q < num_query_points; q++) {
            for (int i = 0; i < k; i++) {
                int result_idx = (b * num_query_points + q) * k + i;
                distances_ptr[result_idx] = 0;
                indices_ptr[result_idx] = 0;
            }
        }
    }
}

// 根据数据点数量自动选择最佳的实现
template <typename scalar_t>
void kdn_cpu_dispatch(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& reference_inv_covariances,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    torch::Tensor& distances,
    torch::Tensor& indices,
    int k) {
    
    // 获取最大的查询点数和参考点数
    int64_t max_query_len = query_lengths.max().item<int64_t>();
    int64_t max_ref_len = reference_lengths.max().item<int64_t>();
    
    // 如果点数小，使用简单实现
    if (max_query_len * max_ref_len < 1000000) {
        kdn_cpu_impl<scalar_t>(
            query_points, reference_points, reference_inv_covariances,
            query_lengths, reference_lengths,
            distances, indices, k);
    } else {
        // 大规模点云使用优化实现
        kdn_cpu_optimized_impl<scalar_t>(
            query_points, reference_points, reference_inv_covariances,
            query_lengths, reference_lengths,
            distances, indices, k);
    }
}

// 导出的C++接口函数
std::tuple<torch::Tensor, torch::Tensor> KDenseNeighborCpu(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& reference_inv_covariances,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    int k) {
    
    // 获取维度和设备信息
    int batch_size = query_points.size(0);
    int num_query_points = query_points.size(1);
    int dim = query_points.size(2);

    // Check inv_cov dimensions
    TORCH_CHECK(reference_inv_covariances.dim() == 4, "reference_inv_covariances must be 4D");
    TORCH_CHECK(reference_inv_covariances.size(0) == batch_size, "Batch sizes of points and inv_covariances do not match");
    TORCH_CHECK(reference_inv_covariances.size(1) == reference_points.size(1), "Num reference points mismatch between points and inv_covariances");
    TORCH_CHECK(reference_inv_covariances.size(2) == dim, "Inv_covariance dimension mismatch (dim 1)");
    TORCH_CHECK(reference_inv_covariances.size(3) == dim, "Inv_covariance dimension mismatch (dim 2)");
    TORCH_CHECK(reference_inv_covariances.device() == query_points.device(), "Inv_covariances must be on the same device as query_points");
    TORCH_CHECK(reference_inv_covariances.scalar_type() == query_points.scalar_type(), "Inv_covariances must have the same scalar type as query_points");

    // 创建输出张量
    auto options = query_points.options();
    auto long_options = options.dtype(torch::kInt64);
    torch::Tensor distances = torch::empty({batch_size, num_query_points, k}, options);
    torch::Tensor indices = torch::empty({batch_size, num_query_points, k}, long_options);
    
    // 根据数据类型分发
    AT_DISPATCH_FLOATING_TYPES(query_points.scalar_type(), "KDenseNeighborCpu", ([&] {
        kdn_cpu_dispatch<scalar_t>(
            query_points, 
            reference_points, 
            reference_inv_covariances,
            query_lengths, 
            reference_lengths, 
            distances, 
            indices, 
            k
        );
    }));
    
    return std::make_tuple(distances, indices);
} 