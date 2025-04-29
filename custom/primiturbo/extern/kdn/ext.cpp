#include "kdn.h" // Include the KDN header
#include <torch/extension.h>
#include <vector>
#include <tuple>

// CPU版本的KDN函数声明 (Now defined in kdn_cpu.cpp and declared in kdn.h)
// std::tuple<torch::Tensor, torch::Tensor> KDenseNeighborCpu(...);

// CUDA版本的KDN函数声明 (Now defined in kdn.cu and declared in kdn.h)
// std::tuple<torch::Tensor, torch::Tensor> KDenseNeighborCuda(...);

// 统一接口，根据设备类型调用不同的实现
std::tuple<torch::Tensor, torch::Tensor> kdn_search(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& reference_inv_covariances, // Added inv_cov
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    int k) { // Removed norm
    
    // 检查输入张量维度
    TORCH_CHECK(query_points.dim() == 3, "query_points must be 3D (B, N, D)");
    TORCH_CHECK(reference_points.dim() == 3, "reference_points must be 3D (B, M, D)");
    TORCH_CHECK(reference_inv_covariances.dim() == 4, "reference_inv_covariances must be 4D (B, M, D, D)");
    TORCH_CHECK(query_lengths.dim() == 1, "query_lengths must be 1D (B)");
    TORCH_CHECK(reference_lengths.dim() == 1, "reference_lengths must be 1D (B)");
    
    // 检查批次大小和维度一致性
    int batch_size = query_points.size(0);
    int dim = query_points.size(2);
    TORCH_CHECK(reference_points.size(0) == batch_size, "Batch sizes of query and reference points do not match");
    TORCH_CHECK(query_lengths.size(0) == batch_size, "Batch sizes of query points and query lengths do not match");
    TORCH_CHECK(reference_lengths.size(0) == batch_size, "Batch sizes of reference points and reference lengths do not match");
    TORCH_CHECK(reference_points.size(2) == dim, "Point dimensions do not match between query and reference");
    TORCH_CHECK(reference_inv_covariances.size(0) == batch_size, "Batch sizes of points and inv_covariances do not match");
    TORCH_CHECK(reference_inv_covariances.size(1) == reference_points.size(1), "Num reference points mismatch between points and inv_covariances");
    TORCH_CHECK(reference_inv_covariances.size(2) == dim, "Inv_covariance dimension mismatch (dim 1)");
    TORCH_CHECK(reference_inv_covariances.size(3) == dim, "Inv_covariance dimension mismatch (dim 2)");

    // 检查k值
    TORCH_CHECK(k > 0, "k must be positive");
    // No need to check norm anymore
    
    // 确保所有输入张量在同一设备和类型
    auto device = query_points.device();
    auto scalar_type = query_points.scalar_type();
    
    TORCH_CHECK(reference_points.device() == device, "reference_points must be on the same device as query_points");
    TORCH_CHECK(reference_inv_covariances.device() == device, "reference_inv_covariances must be on the same device as query_points");
    TORCH_CHECK(query_lengths.device() == device, "query_lengths must be on the same device as query_points"); // Assuming lengths are moved if needed
    TORCH_CHECK(reference_lengths.device() == device, "reference_lengths must be on the same device as query_points"); // Assuming lengths are moved if needed

    TORCH_CHECK(reference_points.scalar_type() == scalar_type, "reference_points must have the same scalar type as query_points");
    TORCH_CHECK(reference_inv_covariances.scalar_type() == scalar_type, "reference_inv_covariances must have the same scalar type as query_points");
    // Lengths should be int64, checked in CPU/CUDA implementations
    
    // 根据设备类型调用CPU或CUDA实现
    if (device.is_cuda()) {
        // Ensure tensors are contiguous (important for CUDA performance, less critical for CPU but good practice)
        return KDenseNeighborCuda(
            query_points.contiguous(), 
            reference_points.contiguous(), 
            reference_inv_covariances.contiguous(), 
            query_lengths.contiguous(), // contiguous() for lengths might be overkill but safe 
            reference_lengths.contiguous(), 
            k
        );
    } else {
        return KDenseNeighborCpu(
            query_points, // CPU version might not strictly require contiguous
            reference_points, 
            reference_inv_covariances, 
            query_lengths, 
            reference_lengths, 
            k
        );
    }
}

// Remove knn_search_l1 and knn_search_l2 functions

// 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kdn_search", &kdn_search, 
          "K-Dense Neighbors Search based on Mahalanobis distance (CPU/CUDA)",
          py::arg("query_points"), 
          py::arg("reference_points"),
          py::arg("reference_inv_covariances"), // Added
          py::arg("query_lengths"),
          py::arg("reference_lengths"),
          py::arg("k")); // Removed norm argument
    // Remove bindings for knn_search_l1 and knn_search_l2
} 