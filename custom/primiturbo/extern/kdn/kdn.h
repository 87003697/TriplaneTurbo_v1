#pragma once

#include <tuple>
#include <torch/extension.h>

// 前向传播函数 - CUDA版本 (KDN: K-Dense Neighbor)
// 返回值顺序: (distances, indices) - distances are Mahalanobis squared distances
std::tuple<at::Tensor, at::Tensor> KDenseNeighborCuda(
    const at::Tensor& query_points,         // (B, N, D)
    const at::Tensor& reference_points,     // (B, M, D)
    const at::Tensor& reference_inv_covariances, // (B, M, D, D)
    const at::Tensor& query_lengths,        // (B,)
    const at::Tensor& reference_lengths,    // (B,)
    int K,
    int version = 0); // version currently unused

// 前向传播函数 - CPU版本 (KDN: K-Dense Neighbor)
// 返回值顺序: (distances, indices) - distances are Mahalanobis squared distances
std::tuple<at::Tensor, at::Tensor> KDenseNeighborCpu(
    const at::Tensor& query_points,         // (B, N, D)
    const at::Tensor& reference_points,     // (B, M, D)
    const at::Tensor& reference_inv_covariances, // (B, M, D, D)
    const at::Tensor& query_lengths,        // (B,)
    const at::Tensor& reference_lengths,    // (B,)
    int K);

// Python绑定函数 - 根据输入的设备自动选择CPU或CUDA版本 (This might be removed as ext.cpp handles dispatch)
// 返回值顺序: (distances, indices)
// Note: This inline function is illustrative; the actual dispatch happens in ext.cpp
/*
inline std::tuple<at::Tensor, at::Tensor> KDenseNeighbor(
    const at::Tensor& query_points,
    const at::Tensor& reference_points,
    const at::Tensor& reference_inv_covariances,
    const at::Tensor& query_lengths,
    const at::Tensor& reference_lengths,
    int K) {

    if (query_points.is_cuda()) {
        // Assuming KDenseNeighborCuda is available
        return KDenseNeighborCuda(query_points, reference_points, reference_inv_covariances, query_lengths, reference_lengths, K);
    } else {
        // Assuming KDenseNeighborCpu is available
        return KDenseNeighborCpu(query_points, reference_points, reference_inv_covariances, query_lengths, reference_lengths, K);
    }
}
*/

// Rename the file itself
// mv custom/primiturbo/extern/kdn/knn.h custom/primiturbo/extern/kdn/kdn.h 