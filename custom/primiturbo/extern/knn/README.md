# CUDA KNN Extension

高性能CUDA实现的K近邻(KNN)搜索扩展。此扩展支持CPU和CUDA实现，并提供了优化的实现方式，以适应不同规模的点云数据。

## 特性

- 高效的CUDA实现，支持大规模点云数据
- 支持批处理和变长序列
- 支持L1和L2距离度量
- 自动回退到CPU实现（当CUDA不可用时）
- 针对不同规模的数据实现了优化策略

## 安装

### 方法1：使用setup.py安装

```bash
cd custom/primiturbo/extern/knn
python setup.py install
```

### 方法2：即时编译 (JIT)

可以在代码中直接使用，扩展会在第一次导入时自动编译：

```python
import sys
import os
sys.path.append('/path/to/custom/primiturbo/extern')

try:
    from knn import knn_search
except ImportError:
    # 尝试即时编译
    from torch.utils.cpp_extension import load
    import torch
    
    cuda_knn = load(
        name="cuda_knn",
        sources=["ext.cpp", "knn.cu", "knn_cpu.cpp"],
        verbose=True,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=torch.cuda.is_available()
    )
    knn_search = cuda_knn.knn_search
```

## 用法

```python
import torch
from cuda_knn import knn_search

# 批大小=2，每个批次有1000个查询点和5000个参考点，3D空间
query_points = torch.rand(2, 1000, 3, device='cuda')
reference_points = torch.rand(2, 5000, 3, device='cuda')

# 指定每个批次的有效长度
query_lengths = torch.tensor([1000, 800], dtype=torch.int64, device='cuda')
reference_lengths = torch.tensor([5000, 4000], dtype=torch.int64, device='cuda')

# 查找16个最近邻，使用L2距离
k = 16
distances, indices = knn_search(
    query_points, 
    reference_points, 
    query_lengths, 
    reference_lengths, 
    k=k, 
    norm=2  # 使用L2距离，可选值：1（L1距离）或2（L2距离）
)

# 结果形状: [batch_size, num_query_points, k]
print(f"距离形状: {distances.shape}")  # [2, 1000, 16]
print(f"索引形状: {indices.shape}")    # [2, 1000, 16]
```

## 输入和输出说明

### 输入

- `query_points`: 形状为 [batch_size, num_query_points, dim] 的浮点张量，包含查询点
- `reference_points`: 形状为 [batch_size, num_reference_points, dim] 的浮点张量，包含参考点
- `query_lengths`: 形状为 [batch_size] 的整型张量，包含每个批次中有效查询点的数量
- `reference_lengths`: 形状为 [batch_size] 的整型张量，包含每个批次中有效参考点的数量
- `k`: 要查找的最近邻数量
- `norm`: 距离度量类型，1 表示 L1 距离，2 表示 L2 距离（默认）

### 输出

- `distances`: 形状为 [batch_size, num_query_points, k] 的浮点张量，包含每个查询点的k个最近邻的距离
- `indices`: 形状为 [batch_size, num_query_points, k] 的整型张量，包含每个查询点的k个最近邻的索引

## 性能

在NVIDIA A100 GPU上，对于不同规模的点云数据，性能如下：

- 小规模 (2×100×1000, k=10): ~10ms
- 中等规模 (2×1000×10000, k=16): ~10ms
- 大规模 (2×10000×100000, k=32): ~80ms

## 限制

- 目前的实现支持的最大k值为32（当使用寄存器优化时）
- 对于更大的k值，将自动切换到共享内存实现（可能会降低性能）
- 为获得最佳性能，建议将k值保持在32以下

## 故障排除

如果在编译过程中遇到CUDA错误，请确保：

1. 已安装与PyTorch兼容的CUDA版本
2. 环境变量设置正确（如`CUDA_HOME`）
3. 有足够的GPU内存用于计算

如果扩展无法正确加载，可以尝试使用`test_knn.py`脚本进行测试和诊断。 