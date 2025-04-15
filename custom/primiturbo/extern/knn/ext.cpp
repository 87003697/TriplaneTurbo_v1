#include <torch/extension.h>
#include <vector>
#include <tuple>

// CPU版本的KNN函数声明
std::tuple<torch::Tensor, torch::Tensor> KNearestNeighborCpu(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    int k,
    int norm);

// CUDA版本的KNN函数声明
std::tuple<torch::Tensor, torch::Tensor> KNearestNeighborCuda(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    int k,
    int norm);

// 统一接口，根据设备类型调用不同的实现
std::tuple<torch::Tensor, torch::Tensor> knn_search(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    int k,
    int norm) {
    
    // 检查输入张量
    TORCH_CHECK(query_points.dim() == 3, "query_points must be 3D");
    TORCH_CHECK(reference_points.dim() == 3, "reference_points must be 3D");
    TORCH_CHECK(query_lengths.dim() == 1, "query_lengths must be 1D");
    TORCH_CHECK(reference_lengths.dim() == 1, "reference_lengths must be 1D");
    
    // 检查批次大小和维度一致性
    TORCH_CHECK(query_points.size(0) == reference_points.size(0), 
               "Batch sizes of query and reference points do not match");
    TORCH_CHECK(query_points.size(2) == reference_points.size(2), 
               "Point dimensions do not match");
    TORCH_CHECK(query_points.size(0) == query_lengths.size(0), 
               "Batch sizes of query points and query lengths do not match");
    TORCH_CHECK(reference_points.size(0) == reference_lengths.size(0), 
               "Batch sizes of reference points and reference lengths do not match");
    
    // 检查k值
    TORCH_CHECK(k > 0, "k must be positive");
    
    // 检查距离度量类型
    TORCH_CHECK(norm == 1 || norm == 2, "norm must be 1 (L1) or 2 (L2)");
    
    // 确保所有输入张量在同一设备
    auto device = query_points.device();
    auto options = query_points.options();
    
    // 如果输入在不同设备上，将它们移动到query_points的设备
    auto query_lengths_device = query_lengths.to(device);
    auto reference_lengths_device = reference_lengths.to(device);
    auto reference_points_device = reference_points.to(device);
    
    // 根据设备类型调用CPU或CUDA实现
    if (device.is_cuda()) {
        return KNearestNeighborCuda(
            query_points, 
            reference_points_device, 
            query_lengths_device, 
            reference_lengths_device, 
            k, 
            norm
        );
    } else {
        return KNearestNeighborCpu(
            query_points, 
            reference_points_device, 
            query_lengths_device, 
            reference_lengths_device, 
            k, 
            norm
        );
    }
}

// 无论norm_type如何，使用L2范数
std::tuple<torch::Tensor, torch::Tensor> knn_search_l2(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    int k) {
    return knn_search(query_points, reference_points, query_lengths, reference_lengths, k, 2);
}

// 无论norm_type如何，使用L1范数
std::tuple<torch::Tensor, torch::Tensor> knn_search_l1(
    const torch::Tensor& query_points,
    const torch::Tensor& reference_points,
    const torch::Tensor& query_lengths,
    const torch::Tensor& reference_lengths,
    int k) {
    return knn_search(query_points, reference_points, query_lengths, reference_lengths, k, 1);
}

// 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_search", &knn_search, "K-Nearest Neighbors Search (CPU/CUDA)",
          py::arg("query_points"), 
          py::arg("reference_points"),
          py::arg("query_lengths"),
          py::arg("reference_lengths"),
          py::arg("k"),
          py::arg("norm") = 2);
          
    m.def("knn_search_l2", &knn_search_l2, "L2范数KNN搜索函数",
          py::arg("query_points"),
          py::arg("reference_points"),
          py::arg("query_lengths"),
          py::arg("reference_lengths"),
          py::arg("k"));
          
    m.def("knn_search_l1", &knn_search_l1, "L1范数KNN搜索函数",
          py::arg("query_points"),
          py::arg("reference_points"),
          py::arg("query_lengths"),
          py::arg("reference_lengths"),
          py::arg("k"));
} 