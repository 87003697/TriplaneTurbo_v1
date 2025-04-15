#pragma once

#include <tuple>
#include <torch/extension.h>

// 前向传播函数 - CUDA版本
// 返回值顺序: (distances, indices)
std::tuple<at::Tensor, at::Tensor> KNearestNeighborCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    int norm = 2,
    int version = 0);

// 前向传播函数 - CPU版本
// 返回值顺序: (distances, indices)
std::tuple<at::Tensor, at::Tensor> KNearestNeighborCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    int norm = 2);

// Python绑定函数 - 根据输入的设备自动选择CPU或CUDA版本
// 返回值顺序: (distances, indices)
inline std::tuple<at::Tensor, at::Tensor> KNearestNeighbor(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    int norm = 2) {
    
    if (p1.is_cuda()) {
        return KNearestNeighborCuda(p1, p2, lengths1, lengths2, K, norm);
    } else {
        return KNearestNeighborCpu(p1, p2, lengths1, lengths2, K, norm);
    }
} 