#include "kdon.h"
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

// Score/Distance/Index struct for sorting KDON results
template <typename scalar_t>
struct ScoreDistIndexPair {
    scalar_t score;    // Score = density * opacity (used for sorting)
    scalar_t distance; // Corresponding Mahalanobis squared distance (for output)
    int64_t index;     // Index of the reference point

    // Operator for sorting in DESCENDING order by score (to find largest scores)
    bool operator<(const ScoreDistIndexPair& other) const {
        // We want largest score first, so `a < b` should be true if a.score > b.score
        // However, std::sort/partial_sort find the smallest elements based on `<`.
        // To get largest K: sort descending or use a max-heap structure.
        // Let's use a custom comparator for clarity with std::sort/partial_sort.
        return score > other.score; // Use > for descending sort
    }
};

// KDON CPU implementation template (Simple version)
template <typename scalar_t>
void kdon_cpu_impl(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& reference_inv_covariances,
    const torch::Tensor& reference_opacities,
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
         throw std::runtime_error("kdon_cpu_impl currently only supports dim=3 due to compute_mahalanobis_sq");
    }
    
    // 获取数据指针
    const scalar_t* query_ptr = query_points.data_ptr<scalar_t>();
    const scalar_t* ref_ptr = reference_points.data_ptr<scalar_t>();
    const scalar_t* inv_cov_base_ptr = reference_inv_covariances.data_ptr<scalar_t>();
    const int64_t* query_lengths_ptr = query_lengths.data_ptr<int64_t>();
    const int64_t* ref_lengths_ptr = reference_lengths.data_ptr<int64_t>();
    scalar_t* distances_ptr = distances.data_ptr<scalar_t>();
    int64_t* indices_ptr = indices.data_ptr<int64_t>();
    
    const scalar_t* opacities_base_ptr = reference_opacities.data_ptr<scalar_t>(); // Get opacity pointer

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int q = 0; q < num_query_points; q++) {
            // Handle invalid query points
            if (q >= query_lengths_ptr[b]) {
                for (int i = 0; i < k; i++) {
                    int result_idx = (b * num_query_points + q) * k + i;
                    distances_ptr[result_idx] = std::numeric_limits<scalar_t>::infinity(); // Padding value
                    indices_ptr[result_idx] = -1;
                }
                continue;
            }
            
            int valid_k = std::min(k, static_cast<int>(ref_lengths_ptr[b]));
            const scalar_t* q_ptr = query_ptr + (b * num_query_points + q) * dim;
            int64_t current_ref_length = ref_lengths_ptr[b];

            if (current_ref_length == 0) { // Handle case with no reference points
                 for (int i = 0; i < k; i++) {
                     int result_idx = (b * num_query_points + q) * k + i;
                     distances_ptr[result_idx] = std::numeric_limits<scalar_t>::infinity();
                     indices_ptr[result_idx] = -1;
                 }
                 continue;
            }
            
            // Calculate scores for all reference points
            std::vector<ScoreDistIndexPair<scalar_t>> score_dist_idx_pairs;
            score_dist_idx_pairs.reserve(current_ref_length);
            
            for (int r = 0; r < current_ref_length; r++) {
                const scalar_t* r_ptr = ref_ptr + (b * num_reference_points + r) * dim;
                const scalar_t* inv_cov_ptr = inv_cov_base_ptr + (b * num_reference_points + r) * (dim * dim);
                const scalar_t opacity = opacities_base_ptr[b * num_reference_points + r]; // Get opacity
                
                scalar_t dist_sq = compute_mahalanobis_sq<scalar_t>(q_ptr, r_ptr, inv_cov_ptr, dim);
                scalar_t density = std::exp(-0.5 * dist_sq); // Use std::exp
                scalar_t score = density * opacity;
                
                score_dist_idx_pairs.push_back({score, dist_sq, static_cast<int64_t>(r)}); // Store score, dist_sq, index
            }
            
            // Find the K elements with the largest scores using partial_sort
            // std::partial_sort sorts the range [first, middle) based on operator<.
            // Since our operator< sorts descending by score, the K largest scores will be in the beginning.
            if (valid_k > 0) { // Only sort if we need at least one neighbor
                if (valid_k < score_dist_idx_pairs.size()) {
                    std::partial_sort(score_dist_idx_pairs.begin(), 
                                     score_dist_idx_pairs.begin() + valid_k,
                                     score_dist_idx_pairs.end()); // operator< sorts descending by score
                } else {
                    // If k >= M, sort everything
                    std::sort(score_dist_idx_pairs.begin(), score_dist_idx_pairs.end()); // operator< sorts descending by score
                }
            }
            
            // Store results (Mahalanobis distance and index of top K scores)
            for (int i = 0; i < valid_k; i++) {
                int result_idx = (b * num_query_points + q) * k + i;
                distances_ptr[result_idx] = score_dist_idx_pairs[i].distance; // Store Mahalanobis dist sq
                indices_ptr[result_idx] = score_dist_idx_pairs[i].index;    // Store index
            }
            
            // Fill remaining slots if k > valid_k
            for (int i = valid_k; i < k; i++) {
                int result_idx = (b * num_query_points + q) * k + i;
                distances_ptr[result_idx] = std::numeric_limits<scalar_t>::infinity();
                indices_ptr[result_idx] = -1;
            }
        }
    }
}

// KDON CPU implementation template (Optimized version)
template <typename scalar_t>
void kdon_cpu_optimized_impl(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& reference_inv_covariances,
    const torch::Tensor& reference_opacities,
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
         throw std::runtime_error("kdon_cpu_optimized_impl currently only supports dim=3 due to compute_mahalanobis_sq");
    }
    
    const scalar_t* inv_cov_base_ptr = reference_inv_covariances.data_ptr<scalar_t>();
    const scalar_t* opacities_base_ptr = reference_opacities.data_ptr<scalar_t>(); // Get opacity pointer
    
    // 获取数据指针
    const scalar_t* query_ptr = query_points.data_ptr<scalar_t>();
    const scalar_t* ref_ptr = reference_points.data_ptr<scalar_t>();
    const int64_t* query_lengths_ptr = query_lengths.data_ptr<int64_t>();
    const int64_t* ref_lengths_ptr = reference_lengths.data_ptr<int64_t>();
    scalar_t* distances_ptr = distances.data_ptr<scalar_t>();
    int64_t* indices_ptr = indices.data_ptr<int64_t>();
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        int valid_query_len = query_lengths_ptr[b];
        int valid_ref_len = ref_lengths_ptr[b];
        int valid_k = std::min(k, valid_ref_len);
        
        // Preallocate per-thread storage (or per-batch if outer loop is parallel)
        std::vector<std::vector<ScoreDistIndexPair<scalar_t>>> all_score_dist_idx_pairs(valid_query_len);
        for (int q = 0; q < valid_query_len; q++) {
            all_score_dist_idx_pairs[q].reserve(valid_ref_len);
        }
        
        // Calculate all scores
        // This part could potentially be parallelized further if needed
        for (int q = 0; q < valid_query_len; q++) {
            const scalar_t* q_ptr = query_ptr + (b * num_query_points + q) * dim;
            auto& current_pairs = all_score_dist_idx_pairs[q];
            
            for (int r = 0; r < valid_ref_len; r++) {
                const scalar_t* r_ptr = ref_ptr + (b * num_reference_points + r) * dim;
                const scalar_t* inv_cov_ptr = inv_cov_base_ptr + (b * num_reference_points + r) * (dim * dim);
                const scalar_t opacity = opacities_base_ptr[b * num_reference_points + r]; // Get opacity
                
                scalar_t dist_sq = compute_mahalanobis_sq<scalar_t>(q_ptr, r_ptr, inv_cov_ptr, dim);
                scalar_t density = std::exp(-0.5 * dist_sq); // Use std::exp
                scalar_t score = density * opacity;
                
                current_pairs.push_back({score, dist_sq, static_cast<int64_t>(r)}); // Store score, dist_sq, index
            }
        }
        
        // Find top K for each query point (can be parallelized)
        #pragma omp parallel for // Parallelize the sorting/selection step
        for (int q = 0; q < valid_query_len; q++) {
            auto& score_dist_idx_pairs = all_score_dist_idx_pairs[q];
            
             // Find the K elements with the largest scores using partial_sort
            if (valid_k > 0 && !score_dist_idx_pairs.empty()) { // Check if vector is not empty
                if (valid_k < score_dist_idx_pairs.size()) {
                    std::partial_sort(score_dist_idx_pairs.begin(), 
                                     score_dist_idx_pairs.begin() + valid_k,
                                     score_dist_idx_pairs.end()); // operator< sorts descending by score
                } else {
                    std::sort(score_dist_idx_pairs.begin(), score_dist_idx_pairs.end()); // operator< sorts descending by score
                }
            }
            
            // Store results
            for (int i = 0; i < valid_k; i++) {
                int result_idx = (b * num_query_points + q) * k + i;
                 if (!score_dist_idx_pairs.empty()) { // Check bounds
                    distances_ptr[result_idx] = score_dist_idx_pairs[i].distance; // Store Mahalanobis dist sq
                    indices_ptr[result_idx] = score_dist_idx_pairs[i].index;    // Store index
                 } else { // Should not happen if valid_k > 0, but as safeguard
                    distances_ptr[result_idx] = std::numeric_limits<scalar_t>::infinity();
                    indices_ptr[result_idx] = -1;
                 }
            }
            
            // Fill remaining slots
            for (int i = valid_k; i < k; i++) {
                int result_idx = (b * num_query_points + q) * k + i;
                distances_ptr[result_idx] = std::numeric_limits<scalar_t>::infinity();
                indices_ptr[result_idx] = -1;
            }
        }
        
        // Fill results for invalid query points in this batch
        for (int q = valid_query_len; q < num_query_points; q++) {
            for (int i = 0; i < k; i++) {
                int result_idx = (b * num_query_points + q) * k + i;
                distances_ptr[result_idx] = std::numeric_limits<scalar_t>::infinity(); // Padding value
                indices_ptr[result_idx] = -1;
            }
        }
    }
}

// 根据数据点数量自动选择最佳的实现
template <typename scalar_t>
void kdon_cpu_dispatch(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& reference_inv_covariances,
    const torch::Tensor& reference_opacities,
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
        kdon_cpu_impl<scalar_t>(
            query_points, 
            reference_points, 
            reference_inv_covariances,
            reference_opacities,
            query_lengths, 
            reference_lengths,
            distances, 
            indices, 
            k);
    } else {
        // 大规模点云使用优化实现
        kdon_cpu_optimized_impl<scalar_t>(
            query_points, 
            reference_points, 
            reference_inv_covariances,
            reference_opacities,
            query_lengths, 
            reference_lengths,
            distances, 
            indices, 
            k);
    }
}

// Exported C++ function for CPU
// std::tuple<torch::Tensor, torch::Tensor> KDenseNeighborCpu(
std::tuple<torch::Tensor, torch::Tensor> KDenseOpacityNeighborCpu( // Corrected name
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& reference_inv_covariances,
    const torch::Tensor& reference_opacities, // Added missing argument
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    int K) { // Corrected argument name K

    // --- Input validation --- 
    // Ensure tensors are on CPU
    TORCH_CHECK(!query_points.is_cuda(), "query_points must be a CPU tensor");
    TORCH_CHECK(!reference_points.is_cuda(), "reference_points must be a CPU tensor");
    TORCH_CHECK(!reference_inv_covariances.is_cuda(), "reference_inv_covariances must be a CPU tensor");
    TORCH_CHECK(!reference_opacities.is_cuda(), "reference_opacities must be a CPU tensor");
    TORCH_CHECK(!query_lengths.is_cuda(), "query_lengths must be a CPU tensor");
    TORCH_CHECK(!reference_lengths.is_cuda(), "reference_lengths must be a CPU tensor");

    // Check dimensions
    TORCH_CHECK(query_points.dim() == 3, "query_points must be 3D (B, N, D)");
    TORCH_CHECK(reference_points.dim() == 3, "reference_points must be 3D (B, M, D)");
    TORCH_CHECK(reference_inv_covariances.dim() == 4, "reference_inv_covariances must be 4D (B, M, D, D)");
    TORCH_CHECK(reference_opacities.dim() == 3 && reference_opacities.size(2) == 1, "reference_opacities must be 3D (B, M, 1)");
    TORCH_CHECK(query_lengths.dim() == 1, "query_lengths must be 1D (B)");
    TORCH_CHECK(reference_lengths.dim() == 1, "reference_lengths must be 1D (B)");

    // Check consistency
    int batch_size = query_points.size(0);
    int num_query_points = query_points.size(1);
    int dim = query_points.size(2);
    TORCH_CHECK(reference_points.size(0) == batch_size, "Batch sizes mismatch");
    TORCH_CHECK(reference_inv_covariances.size(0) == batch_size, "Batch sizes mismatch");
    TORCH_CHECK(reference_opacities.size(0) == batch_size, "Batch sizes mismatch");
    TORCH_CHECK(query_lengths.size(0) == batch_size, "Batch sizes mismatch");
    TORCH_CHECK(reference_lengths.size(0) == batch_size, "Batch sizes mismatch");
    TORCH_CHECK(reference_points.size(2) == dim, "Dimension mismatch");
    TORCH_CHECK(reference_inv_covariances.size(2) == dim && reference_inv_covariances.size(3) == dim, "Inv covariance dimension mismatch");
    TORCH_CHECK(reference_points.size(1) == reference_inv_covariances.size(1), "Num reference points mismatch");
    TORCH_CHECK(reference_points.size(1) == reference_opacities.size(1), "Num reference points mismatch");

    // Check K value
    TORCH_CHECK(K > 0, "K must be positive");
    // TORCH_CHECK(K <= reference_points.size(1), "K cannot be larger than the number of reference points"); // This check might be too strict if using lengths
    
    // Check data types
    auto q_dtype = query_points.scalar_type();
    TORCH_CHECK(reference_points.scalar_type() == q_dtype, "Data types mismatch");
    TORCH_CHECK(reference_inv_covariances.scalar_type() == q_dtype, "Data types mismatch");
    TORCH_CHECK(reference_opacities.scalar_type() == q_dtype, "Data types mismatch");
    TORCH_CHECK(query_lengths.scalar_type() == at::kLong, "query_lengths must be int64");
    TORCH_CHECK(reference_lengths.scalar_type() == at::kLong, "reference_lengths must be int64");

    // --- Prepare output tensors --- 
    auto options = query_points.options();
    auto distances = torch::empty({batch_size, num_query_points, K}, options.dtype(q_dtype));
    auto indices = torch::empty({batch_size, num_query_points, K}, options.dtype(at::kLong));

    // --- Dispatch to implementation based on data type --- 
    AT_DISPATCH_FLOATING_TYPES(query_points.scalar_type(), "KDenseOpacityNeighborCpu", ([&] {
        kdon_cpu_dispatch<scalar_t>(
            query_points, 
            reference_points, 
            reference_inv_covariances, 
            reference_opacities,
            query_lengths, 
            reference_lengths, 
            distances, 
            indices, 
            K
        );
    }));

    return std::make_tuple(distances, indices);
} 