#include <torch/extension.h>
#include <ATen/ATen.h>
#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <omp.h>

// 计算点与点之间的距离
template <typename scalar_t, int NORM>
scalar_t compute_distance(
    const scalar_t* query_point,
    const scalar_t* ref_point,
    int dim) {
    
    scalar_t result = 0;
    for (int d = 0; d < dim; d++) {
        scalar_t diff = query_point[d] - ref_point[d];
        if (NORM == 1) {
            result += std::abs(diff);
        } else {
            result += diff * diff;
        }
    }
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

// KNN CPU实现模板
template <typename scalar_t, int NORM>
void knn_cpu_impl(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
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
    
    // 获取数据指针
    const scalar_t* query_ptr = query_points.data_ptr<scalar_t>();
    const scalar_t* ref_ptr = reference_points.data_ptr<scalar_t>();
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
            
            // 计算与所有参考点的距离
            std::vector<DistIndexPair<scalar_t>> dist_idx_pairs;
            dist_idx_pairs.reserve(ref_lengths_ptr[b]);
            
            for (int r = 0; r < ref_lengths_ptr[b]; r++) {
                const scalar_t* r_ptr = ref_ptr + (b * num_reference_points + r) * dim;
                scalar_t dist = compute_distance<scalar_t, NORM>(q_ptr, r_ptr, dim);
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

// 针对大规模点云的优化KNN CPU实现
template <typename scalar_t, int NORM>
void knn_cpu_optimized_impl(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
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
                scalar_t dist = compute_distance<scalar_t, NORM>(q_ptr, r_ptr, dim);
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
template <typename scalar_t, int NORM>
void knn_cpu_dispatch(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
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
        knn_cpu_impl<scalar_t, NORM>(
            query_points, reference_points,
            query_lengths, reference_lengths,
            distances, indices, k);
    } else {
        // 大规模点云使用优化实现
        knn_cpu_optimized_impl<scalar_t, NORM>(
            query_points, reference_points,
            query_lengths, reference_lengths,
            distances, indices, k);
    }
}

// 导出的C++接口函数
std::tuple<torch::Tensor, torch::Tensor> KNearestNeighborCpu(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    int k,
    int norm) {
    
    // 获取维度信息
    int batch_size = query_points.size(0);
    int num_query_points = query_points.size(1);
    
    // 为结果分配内存
    auto distances = torch::empty({batch_size, num_query_points, k}, 
                                 query_points.options());
    auto indices = torch::empty({batch_size, num_query_points, k}, 
                               query_points.options().dtype(torch::kInt64));
    
    // 根据指定的范数调用适当的实现
    if (norm == 1) {
        AT_DISPATCH_FLOATING_TYPES(query_points.scalar_type(), "KNearestNeighborCpu_L1", ([&] {
            knn_cpu_dispatch<scalar_t, 1>(
                query_points,
                reference_points,
                query_lengths,
                reference_lengths,
                distances,
                indices,
                k
            );
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES(query_points.scalar_type(), "KNearestNeighborCpu_L2", ([&] {
            knn_cpu_dispatch<scalar_t, 2>(
                query_points,
                reference_points,
                query_lengths,
                reference_lengths,
                distances,
                indices,
                k
            );
        }));
    }
    
    return std::make_tuple(distances, indices);
} 