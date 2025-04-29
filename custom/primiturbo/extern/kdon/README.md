# K-Dense Opacity Neighbor (KDON) Search for PyTorch

This package provides a PyTorch extension for efficient K-Dense Opacity Neighbor (KDON) search, implemented in C++ and CUDA.

## Functionality

The core functionality is provided by the `kdon` function:

```python
import torch
from custom.primiturbo.extern.kdon import kdon

# Example Usage:
query_points = torch.randn(B, N, D, device='cuda')
reference_points = torch.randn(B, M, D, device='cuda')
# Generate positive definite inv covariances (B, M, D, D)
# ... (see test_kdon.py for example)
reference_inv_covariances = ... 
reference_opacities = torch.rand(B, M, 1, device='cuda') # (B, M, 1)
query_lengths = torch.full((B,), N, dtype=torch.int64, device='cuda')
reference_lengths = torch.full((B,), M, dtype=torch.int64, device='cuda')
K = 16

distances, indices = kdon(
    query_points,
    reference_points,
    reference_inv_covariances,
    reference_opacities, # New opacity input
    query_lengths,
    reference_lengths,
    K
)

# distances: (B, N, K) - Squared Mahalanobis distances of the top K neighbors
# indices:   (B, N, K) - Indices of the top K neighbors
```

**Key Difference from KDN:** KDON selects the top K neighbors based on a **score** calculated as `density * opacity`, where `density = exp(-0.5 * mahalanobis_sq)`. However, it still returns the **squared Mahalanobis distance** (`mahalanobis_sq`) associated with those selected neighbors, maintaining compatibility with downstream calculations that might expect distance values.

## Installation

To build and install the extension:

1.  **Ensure PyTorch is installed** with the correct CUDA version if using GPU acceleration.
2.  **Navigate to this directory:** `cd path/to/custom/primiturbo/extern/kdon`
3.  **Run the build script:** `python build_ext.py`

This will compile the C++/CUDA code and install the `cuda_kdon` package, making the `kdon` function available for import.

## Testing

To run the tests:

1.  Make sure the extension is built and installed.
2.  Navigate to this directory: `cd path/to/custom/primiturbo/extern/kdon`
3.  Run the test script: `python test_kdon.py`

This will compare the output of the C++/CUDA implementation against a pure PyTorch reference implementation for various configurations.

**Note on Float16 Precision:** The tests for the CUDA implementation using `float16` (half-precision) are currently skipped (`test_kdon_cuda_float16`). This is because `float16` has inherent precision limitations, which can lead to significant numerical differences compared to `float32` or the PyTorch reference, especially due to the `exp()` function used in the density calculation within the KDON algorithm. While the `float32` and CPU versions pass verification, use the `float16` CUDA version with awareness of potential precision deviations.

## Limitations

*   Currently, the implementation (both CPU and CUDA `compute_mahalanobis_sq`) is optimized for `D=3` dimensions.
*   Float16 precision issues as noted above.

# CUDA KDN (K-Dense Neighbor) Extension

基于马氏距离的高性能CUDA实现的K密集邻近(KDN)搜索扩展。此扩展旨在查找给定查询点在马氏距离度量下最近邻的K个参考点（例如高斯分布），优先考虑密度贡献潜力大的邻居。它提供了CPU和CUDA实现。

## 特性

- 高效的CUDA实现，支持大规模参考点集
- 基于马氏距离的搜索，更能反映点在高斯分布下的密度影响
- 支持批处理和变长序列
- 自动回退到CPU实现（当CUDA不可用或编译失败时）
- （注意：当前CPU和CUDA实现均为并行暴力搜索，未构建空间索引）

## 安装

### 方法1：使用setup.py安装

```bash
cd custom/primiturbo/extern/kdn # 进入kdn目录
python setup.py install
```

### 方法2：即时编译 (JIT)

可以在代码中直接使用，扩展会在第一次导入时自动尝试编译：

```python
import sys
import os
sys.path.append('/path/to/custom/primiturbo/extern') # 根据实际路径修改

# 尝试导入编译好的扩展或触发JIT编译
from kdn import kdn_search, CudaKDNIndex, HAS_CUDA_KDN

if not HAS_CUDA_KDN:
    print("Warning: CUDA KDN extension not available or JIT compilation failed.")
    # 这里可以添加纯CPU或纯Torch的备选方案（如果需要）
```

## 用法

```python
import torch
from kdn import kdn_search # 或使用 CudaKDNIndex 类

# 假设 B=1, N=1000 查询点, M=5000 参考点, D=3 维空间
B, N, M, D = 1, 1000, 5000, 3
query_points = torch.rand(B, N, D, device='cuda', dtype=torch.float32)
reference_points = torch.rand(B, M, D, device='cuda', dtype=torch.float32)

# --- 重要：生成或提供参考点的逆协方差矩阵 --- 
# 示例：假设每个参考点有随机的对角逆协方差
# 实际应用中应根据高斯分布的 scale 和 rotation 计算
inv_scales_sq = torch.rand(B, M, D, device='cuda', dtype=torch.float32) + 0.1 # 避免0
reference_inv_covariances = torch.diag_embed(inv_scales_sq) # (B, M, D, D)
# --------------------------------------------------

# 指定每个批次的有效长度
query_lengths = torch.tensor([N], dtype=torch.int64, device='cuda')
reference_lengths = torch.tensor([M], dtype=torch.int64, device='cuda')

# 查找16个密度最近邻
k = 16
distances_sq, indices = kdn_search(
    query_points,
    reference_points,
    reference_inv_covariances, # <<< 新增参数
    query_lengths,
    reference_lengths,
    k=k
)

# 结果形状: [B, N, K]
# distances_sq 是马氏距离的平方
print(f"距离平方形状: {distances_sq.shape}")  # [1, 1000, 16]
print(f"索引形状: {indices.shape}")    # [1, 1000, 16]

# --- 使用 CudaKDNIndex 类 (可选) ---
index = CudaKDNIndex()
index.add(reference_points, reference_inv_covariances, reference_lengths)

query_lengths_for_search = torch.tensor([N], dtype=torch.int64, device='cuda') # 通常与原始query_lengths相同
distances_sq_cls, indices_cls = index.search(query_points, query_lengths_for_search, k=k)

print(f"类方法距离平方形状: {distances_sq_cls.shape}")
print(f"类方法索引形状: {indices_cls.shape}")
```

## 输入和输出说明

### 输入 (kdn_search 函数)

- `query_points`: 形状为 `[batch_size, num_query_points, dim]` 的浮点张量，包含查询点。
- `reference_points`: 形状为 `[batch_size, num_reference_points, dim]` 的浮点张量，包含参考点（例如高斯中心）。
- `reference_inv_covariances`: 形状为 `[batch_size, num_reference_points, dim, dim]` 的浮点张量，包含每个参考点的逆协方差矩阵。
- `query_lengths`: 形状为 `[batch_size]` 的 Int64 张量，包含每个批次中有效查询点的数量。
- `reference_lengths`: 形状为 `[batch_size]` 的 Int64 张量，包含每个批次中有效参考点的数量。
- `k`: 要查找的最近邻数量。

### 输出

- `distances`: 形状为 `[batch_size, num_query_points, k]` 的浮点张量，包含每个查询点的k个最近邻的**马氏距离平方**。
- `indices`: 形状为 `[batch_size, num_query_points, k]` 的 Int64 张量，包含每个查询点的k个最近邻在 `reference_points` 中的索引。

## 性能

性能将取决于参考点数量 `M`、查询点数量 `N`、维度 `D` 和 `k` 值，以及GPU型号。由于采用了暴力搜索，当 `M` 非常大时，性能可能会受到影响。共享内存的使用对性能至关重要，因此 `k` 值的大小会影响资源占用（与原KNN类似，过大的k可能导致共享内存不足）。

## 限制

- 当前实现硬编码支持 `dim=3` 的马氏距离计算。如需支持其他维度，需要修改 `compute_mahalanobis_sq` 函数。
- 基于共享内存的 Top-K 维护对 `k` 值有限制（与原KNN类似，建议k<=32以获得较好性能和兼容性）。
- 实现方式为并行暴力搜索，没有利用空间索引结构优化，计算复杂度与参考点数量 `M` 线性相关。

## 故障排除

如果在编译过程中遇到CUDA错误，请确保：

1. 已安装与PyTorch兼容的CUDA版本。
2. 环境变量设置正确（如`CUDA_HOME`）。
3. 有足够的GPU内存用于计算。

如果扩展无法正确加载或运行出错，可以尝试使用修改后的 `test_kdn.py` 脚本进行测试和诊断。

# Rename the file itself
# mv custom/primiturbo/extern/kdn/README.md custom/primiturbo/extern/kdn/README.md
# No rename needed 