#include <torch/extension.h>
#include "kdon.h" // Include the new header
#include <vector>
#include <tuple>

// CPU版本的KDN函数声明 (Now defined in kdn_cpu.cpp and declared in kdn.h)
// std::tuple<torch::Tensor, torch::Tensor> KDenseNeighborCpu(...);

// CUDA版本的KDN函数声明 (Now defined in kdn.cu and declared in kdn.h)
// std::tuple<torch::Tensor, torch::Tensor> KDenseNeighborCuda(...);

// Function to dispatch KDON based on device
std::tuple<at::Tensor, at::Tensor> kdon(
    const at::Tensor& query_points,         // (B, N, D)
    const at::Tensor& reference_points,     // (B, M, D)
    const at::Tensor& reference_inv_covariances, // (B, M, D, D)
    const at::Tensor& reference_opacities,  // (B, M, 1) Added
    const at::Tensor& query_lengths,        // (B,)
    const at::Tensor& reference_lengths,    // (B,)
    int K) 
{
    // Check that required tensors are on the same device
    TORCH_CHECK(query_points.device() == reference_points.device(), "query_points and reference_points must be on the same device");
    TORCH_CHECK(query_points.device() == reference_inv_covariances.device(), "query_points and reference_inv_covariances must be on the same device");
    TORCH_CHECK(query_points.device() == reference_opacities.device(), "query_points and reference_opacities must be on the same device");
    TORCH_CHECK(query_points.device() == query_lengths.device(), "query_points and query_lengths must be on the same device");
    TORCH_CHECK(query_points.device() == reference_lengths.device(), "query_points and reference_lengths must be on the same device");

    // Ensure contiguous tensors for performance
    auto q_points_cont = query_points.contiguous();
    auto ref_points_cont = reference_points.contiguous();
    auto ref_inv_cov_cont = reference_inv_covariances.contiguous();
    auto ref_opacities_cont = reference_opacities.contiguous();
    auto q_len_cont = query_lengths.contiguous();
    auto ref_len_cont = reference_lengths.contiguous();

    if (query_points.is_cuda()) {
        // Call the CUDA implementation
        return KDenseOpacityNeighborCuda(q_points_cont, ref_points_cont, ref_inv_cov_cont, ref_opacities_cont, q_len_cont, ref_len_cont, K);
    } else {
        // Call the CPU implementation
        return KDenseOpacityNeighborCpu(q_points_cont, ref_points_cont, ref_inv_cov_cont, ref_opacities_cont, q_len_cont, ref_len_cont, K);
    }
}

// Remove knn_search_l1 and knn_search_l2 functions

// --- Pybind11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "kdon",                                  // Python function name
        &kdon,                                   // C++ function pointer
        "K-Dense Opacity Neighbor (KDON) computation (CUDA or CPU).\n\n"
        "Args:\n"
        "    query_points (torch.Tensor): Query points (B, N, D).\n"
        "    reference_points (torch.Tensor): Reference points (B, M, D).\n"
        "    reference_inv_covariances (torch.Tensor): Inverse covariances for reference points (B, M, D, D).\n"
        "    reference_opacities (torch.Tensor): Opacities for reference points (B, M, 1).\n"
        "    query_lengths (torch.Tensor): Actual number of query points per batch item (B,), dtype=int64.\n"
        "    reference_lengths (torch.Tensor): Actual number of reference points per batch item (B,), dtype=int64.\n"
        "    K (int): Number of neighbors to find.\n\n"
        "Returns:\n"
        "    Tuple[torch.Tensor, torch.Tensor]:\n"
        "        - distances (torch.Tensor): Squared Mahalanobis distances to the K neighbors (B, N, K).\n"
        "        - indices (torch.Tensor): Indices of the K neighbors in the reference points tensor (B, N, K), dtype=int64.",
        py::arg("query_points"),
        py::arg("reference_points"),
        py::arg("reference_inv_covariances"),
        py::arg("reference_opacities"), // Added argument
        py::arg("query_lengths"),
        py::arg("reference_lengths"),
        py::arg("K")
    );
} 