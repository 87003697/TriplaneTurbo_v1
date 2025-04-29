import os
import torch
import warnings
import sys
from torch.utils.cpp_extension import load

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前目录及上级目录添加到搜索路径
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

# 启用详细错误信息用于调试
try:
    # 尝试直接导入CUDA KNN扩展
    from cuda_knn import knn_search as cuda_knn_search
    CUDA_KNN_AVAILABLE = True
    print("成功导入CUDA KNN扩展")
except ImportError as e:
    # 如果直接导入失败，尝试从安装目录导入
    try:
        # 获取torch和site-packages目录
        import site
        site_packages = site.getsitepackages()
        
        # 遍历site-packages目录寻找cuda_knn egg
        for sp in site_packages:
            egg_path = None
            for item in os.listdir(sp):
                if item.startswith('cuda_knn') and item.endswith('.egg'):
                    egg_path = os.path.join(sp, item)
                    break
            
            if egg_path:
                sys.path.insert(0, egg_path)
                from cuda_knn import knn_search as cuda_knn_search
                CUDA_KNN_AVAILABLE = True
                print(f"从 {egg_path} 导入了CUDA KNN扩展")
                break
        else:
            CUDA_KNN_AVAILABLE = False
            warnings.warn(
                "未在site-packages中找到CUDA KNN扩展，使用前请运行: python -m custom.primiturbo.extern.knn.setup install"
            )
    except ImportError:
        CUDA_KNN_AVAILABLE = False
        warnings.warn(
            "导入CUDA KNN扩展失败，使用前请运行: python -m custom.primiturbo.extern.knn.setup install"
        )

# --- Try importing the compiled extension --- 
_kdon_ext = None
_kdon_ext_error = None
HAS_CUDA_KDON = False

# Try to import the pre-compiled extension 
try:
    import cuda_kdon as _kdon_ext
    # Check if the expected function exists in the loaded module
    if hasattr(_kdon_ext, 'kdon'):
        HAS_CUDA_KDON = True
    else:
        _kdon_ext = None # Reset if function not found
        _kdon_ext_error = ImportError("Imported cuda_kdon module does not have 'kdon' function.")
        warnings.warn(str(_kdon_ext_error))

except ImportError as e:
    _kdon_ext_error = e
    # Do not warn here, the main function will raise if needed
    pass

# Define the main Python function
def kdon(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    reference_inv_covariances: torch.Tensor,
    reference_opacities: torch.Tensor, # Added
    query_lengths: torch.Tensor,
    reference_lengths: torch.Tensor,
    K: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """ K-Dense Opacity Neighbor (KDON) Search.

    Finds the K reference points (Gaussians) whose contribution, weighted by
    both density (derived from Mahalanobis distance) and opacity, is highest
    for each query point.

    Args:
        query_points (torch.Tensor): Query points (B, N, D).
        reference_points (torch.Tensor): Reference points (B, M, D).
        reference_inv_covariances (torch.Tensor): Inverse covariances (B, M, D, D).
        reference_opacities (torch.Tensor): Opacities (B, M, 1).
        query_lengths (torch.Tensor): Actual number of query points per batch (B,), dtype=int64.
        reference_lengths (torch.Tensor): Actual number of ref points per batch (B,), dtype=int64.
        K (int): Number of neighbors to find.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - distances (torch.Tensor): Squared Mahalanobis distances of the top K neighbors (B, N, K).
            - indices (torch.Tensor): Indices of the top K neighbors (B, N, K).
    """
    # Check if the extension was loaded successfully
    if not HAS_CUDA_KDON or _kdon_ext is None:
        # Construct a more informative error message
        error_msg = (
            "cuda_kdon extension failed to load. "
            "Please ensure the extension is compiled correctly using 'python build_ext.py'. "
        )
        if _kdon_ext_error:
             error_msg += f"Import error details: {_kdon_ext_error}"
        else:
             error_msg += "Module imported but 'kdon' function is missing."
        raise ImportError(error_msg)

    # Input validation
    assert query_points.dim() == 3, "query_points must be 3D (B, N, D)"
    assert reference_points.dim() == 3, "reference_points must be 3D (B, M, D)"
    assert reference_inv_covariances.dim() == 4, "reference_inv_covariances must be 4D (B, M, D, D)"
    assert reference_opacities.dim() == 3 and reference_opacities.size(2) == 1, \
        "reference_opacities must be 3D with shape (B, M, 1)"
    assert query_lengths.dim() == 1, "query_lengths must be 1D (B,)"
    assert reference_lengths.dim() == 1, "reference_lengths must be 1D (B,)"
    assert K > 0, "K must be positive"
    assert query_points.size(0) == reference_points.size(0) == reference_inv_covariances.size(0) == \
           reference_opacities.size(0) == query_lengths.size(0) == reference_lengths.size(0), \
           "Batch sizes of all inputs must match"
    assert query_points.size(2) == reference_points.size(2) == reference_inv_covariances.size(2) == \
           reference_inv_covariances.size(3), "Dimension mismatch (D)"

    # Ensure lengths are int64 and on the correct device
    device = query_points.device
    q_len = query_lengths.to(device=device, dtype=torch.int64)
    ref_len = reference_lengths.to(device=device, dtype=torch.int64)

    # Ensure all tensors are on the same device and have the same float dtype
    dtype = query_points.dtype
    tensors_to_check = [reference_points, reference_inv_covariances, reference_opacities]
    for t in tensors_to_check:
        assert t.device == device, f"Tensor {t} is on device {t.device}, expected {device}"
        assert t.dtype == dtype, f"Tensor {t} has dtype {t.dtype}, expected {dtype}"

    # Call the C++/CUDA backend
    return _kdon_ext.kdon(
        query_points,             # (B, N, D)
        reference_points,         # (B, M, D)
        reference_inv_covariances,# (B, M, D, D)
        reference_opacities,      # (B, M, 1)
        q_len,                    # (B,)
        ref_len,                  # (B,)
        K                         # int
    )

# --- Helper Class (Optional, similar to original) ---
class CudaKDNIndex:
    """Helper class to encapsulate reference data for KDN search."""
    def __init__(self):
        self.reference_points = None
        self.reference_inv_covariances = None # Added
        self.reference_lengths = None
        self.is_built = False

    def add(self, 
            reference_points: torch.Tensor, 
            reference_inv_covariances: torch.Tensor, # Added
            reference_lengths: torch.Tensor):
        """
        Adds the reference data to the index.

        Args:
            reference_points (Tensor): Reference points (B, M, D).
            reference_inv_covariances (Tensor): Inverse covariances (B, M, D, D).
            reference_lengths (Tensor): Lengths of valid reference points (B,). Int64.
        """
        if reference_points.dim() != 3 or reference_inv_covariances.dim() != 4:
            raise ValueError("Inputs must be 3D (points) and 4D (inv_covariances)")
        if reference_points.shape[:2] != reference_inv_covariances.shape[:2]:
             raise ValueError("Batch and M dimensions must match between points and inv_covariances")
        if reference_points.shape[0] != reference_lengths.shape[0]:
            raise ValueError("Batch dimensions must match between points and lengths")
            
        self.reference_points = reference_points
        self.reference_inv_covariances = reference_inv_covariances # Store inv_cov
        self.reference_lengths = reference_lengths
        self.is_built = True
        # print(f"KDN Index built with {reference_points.shape[1]} reference points.") # Commented out redundant log

    def search(self, query_points: torch.Tensor, query_lengths: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs KDN search against the stored reference data.

        Args:
            query_points (Tensor): Query points (B, N, D).
            query_lengths (Tensor): Lengths of valid query points (B,). Int64.
            k (int): Number of neighbors.

        Returns:
            tuple[Tensor, Tensor]: distances (Mahalanobis sq), indices.
        """
        if not self.is_built or self.reference_points is None or self.reference_inv_covariances is None:
            raise RuntimeError("Index must be built with reference data before searching.")
        if query_points.dim() != 3 or query_lengths.dim() != 1:
             raise ValueError("Query points must be 3D, query_lengths must be 1D")
        if query_points.shape[0] != query_lengths.shape[0]:
             raise ValueError("Batch dimensions must match between query points and lengths")
        if query_points.shape[0] != self.reference_points.shape[0]:
             raise ValueError("Batch dimensions must match between query and reference data")

        # Ensure devices match (or handle moving data)
        target_device = self.reference_points.device
        if query_points.device != target_device:
            warnings.warn(f"Moving query_points to device {target_device}")
            query_points = query_points.to(target_device)
        if query_lengths.device != target_device:
             warnings.warn(f"Moving query_lengths to device {target_device}")
             query_lengths = query_lengths.to(target_device)

        return kdon(
            query_points,
            self.reference_points,
            self.reference_inv_covariances, # Pass inv_cov
            self.reference_opacities,
            query_lengths,
            self.reference_lengths,
            k
        )

# Export symbols
__all__ = ['kdon', 'CudaKDNIndex', 'HAS_CUDA_KDON']

def knn_search(query_points, reference_points, query_lengths=None, ref_lengths=None, k=10, use_cuda=True, return_sqrt_dist=False):
    """
    高效的K近邻搜索函数
    
    参数:
        query_points: 查询点，形状为(B, N, 3) 或 (N, 3)
        reference_points: 参考点，形状为(B, M, 3) 或 (M, 3)
        query_lengths: 每个批次中有效的查询点数量，形状为(B)
        ref_lengths: 每个批次中有效的参考点数量，形状为(B)
        k: 近邻数量
        use_cuda: 是否使用CUDA实现
        return_sqrt_dist: 是否返回欧氏距离(True)而不是平方欧氏距离(False)
        
    返回:
        distances: 最近邻距离，形状为(B, N, K) 或 (N, K)
        indices: 最近邻索引，形状为(B, N, K) 或 (N, K)
    """
    # 检查CUDA扩展是否可用
    if use_cuda and not CUDA_KNN_AVAILABLE:
        warnings.warn(
            "CUDA KNN扩展未安装，将使用PyTorch实现。如需使用CUDA加速，请运行: \n"
            "cd custom/primiturbo/extern/knn && "
            "python setup.py install"
        )
        use_cuda = False
    
    # 检查输入维度
    is_batched = True
    
    if query_points.dim() == 2:
        is_batched = False
        query_points = query_points.unsqueeze(0)  # 添加批次维度
        num_query = query_points.shape[1]
    
    if reference_points.dim() == 2:
        reference_points = reference_points.unsqueeze(0)  # 添加批次维度
        num_ref = reference_points.shape[1]
    
    # 获取批次大小
    batch_size = query_points.shape[0]
    
    # 确保查询点和参考点的批次大小相同
    if reference_points.shape[0] != batch_size:
        if reference_points.shape[0] == 1:
            # 将参考点扩展到与查询点相同的批次大小
            reference_points = reference_points.expand(batch_size, -1, -1)
        else:
            raise ValueError(
                f"批次大小不匹配: 查询点={query_points.shape[0]}, 参考点={reference_points.shape[0]}"
            )
    
    # 创建长度张量（表示每个批次中有效的点数）
    if query_lengths is None:
        query_lengths = torch.tensor([num_query, num_query//2], dtype=torch.int64, device=query_points.device)   
    
    if ref_lengths is None:
        ref_lengths = torch.tensor([num_ref, num_ref//2], dtype=torch.int64, device=reference_points.device)
    
    # 根据设备选择CPU或CUDA实现
    if query_points.is_cuda and reference_points.is_cuda and use_cuda and CUDA_KNN_AVAILABLE:
        # 使用CUDA实现
        distances, indices = cuda_knn_search(
            query_points, 
            reference_points,
            query_lengths, 
            ref_lengths, 
            k
        )
    else:
        # 回退到PyTorch实现
        from .test_knn import pure_pytorch_knn
        # PyTorch实现返回 (distances, indices)
        distances, indices = pure_pytorch_knn(
            query_points, reference_points, query_lengths, ref_lengths, k
        )
    
    # 如果需要返回欧氏距离而不是平方欧氏距离
    if return_sqrt_dist:
        distances = torch.sqrt(distances.clamp(min=1e-8))
    
    # 如果原始输入不是批次形式，则移除批次维度
    if not is_batched:
        indices = indices.squeeze(0)
        distances = distances.squeeze(0)
    
    return distances, indices

def build_knn_extension():
    """编译KNN CUDA扩展"""
    import subprocess
    import sys
    
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 运行setup.py install
    cmd = [sys.executable, "setup.py", "install"]
    
    try:
        subprocess.check_call(cmd, cwd=current_dir)
        print("KNN CUDA扩展编译成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"KNN CUDA扩展编译失败: {e}")
        return False 