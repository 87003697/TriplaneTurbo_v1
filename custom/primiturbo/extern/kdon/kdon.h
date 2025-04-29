#pragma once

#include <tuple>
#include <torch/extension.h>

// KDON: K-Dense Opacity Neighbor

// CUDA版本 - 返回马氏距离平方
std::tuple<at::Tensor, at::Tensor> KDenseOpacityNeighborCuda(
    const at::Tensor& query_points,         // (B, N, D)
    const at::Tensor& reference_points,     // (B, M, D)
    const at::Tensor& reference_inv_covariances, // (B, M, D, D)
    const at::Tensor& reference_opacities,  // (B, M, 1) Added
    const at::Tensor& query_lengths,        // (B,)
    const at::Tensor& reference_lengths,    // (B,)
    int K,
    int version = 0); 

// CPU版本 - 返回马氏距离平方
std::tuple<at::Tensor, at::Tensor> KDenseOpacityNeighborCpu(
    const at::Tensor& query_points,         // (B, N, D)
    const at::Tensor& reference_points,     // (B, M, D)
    const at::Tensor& reference_inv_covariances, // (B, M, D, D)
    const at::Tensor& reference_opacities,  // (B, M, 1) Added
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