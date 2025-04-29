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
_kdn_ext = None
_kdn_ext_error = None
HAS_CUDA_KDN = False # Renamed flag
try:
    # Try to import the compiled C++/CUDA extension
    import cuda_kdn as _kdn_ext # Changed import name
    if hasattr(_kdn_ext, 'kdn_search'): # Check for the correct function
        HAS_CUDA_KDN = True
    else:
        warnings.warn(f"Imported cuda_kdn module does not have kdn_search function.")
        _kdn_ext = None # Reset if function not found
except ImportError as e:
    _kdn_ext_error = e
    warnings.warn(f"Failed to import compiled cuda_kdn extension: {e}. Falling back to JIT or CPU.")

# --- JIT Compilation (Fallback) --- 
def _jit_compile_kdn():
    global _kdn_ext, HAS_CUDA_KDN
    if _kdn_ext is not None:
        return _kdn_ext # Already loaded or compiled
    
    print("Attempting JIT compilation for KDN...")
    try:
        # Verbose=True helps with debugging compilation issues
        _kdn_ext = load(
            name="cuda_kdn", # Changed module name
            sources=[
                os.path.join(current_dir, 'ext.cpp'),
                os.path.join(current_dir, 'kdn.cu'),     # Changed source
                os.path.join(current_dir, 'kdn_cpu.cpp') # Changed source
            ],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3'],
            verbose=True
        )
        if hasattr(_kdn_ext, 'kdn_search'):
            HAS_CUDA_KDN = True
            print("JIT compilation successful.")
            return _kdn_ext
        else:
             print("JIT compiled module missing 'kdn_search'.")
             _kdn_ext = None # Failed
             HAS_CUDA_KDN = False
             return None

    except Exception as e:
        warnings.warn(f"JIT compilation failed: {e}")
        _kdn_ext = None
        HAS_CUDA_KDN = False
        return None

# --- Main Search Function --- 
def kdn_search(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    reference_inv_covariances: torch.Tensor, # Added inv_cov
    query_lengths: torch.Tensor,
    reference_lengths: torch.Tensor,
    k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs K-Dense Neighbor search based on Mahalanobis distance.

    Args:
        query_points (Tensor): Query points (B, N, D).
        reference_points (Tensor): Reference points (B, M, D).
        reference_inv_covariances (Tensor): Inverse covariances for reference points (B, M, D, D).
        query_lengths (Tensor): Lengths of valid query points per batch (B,). Int64.
        reference_lengths (Tensor): Lengths of valid reference points per batch (B,). Int64.
        k (int): Number of neighbors to find.

    Returns:
        tuple[Tensor, Tensor]: 
            - distances (Tensor): Mahalanobis squared distances to K neighbors (B, N, K).
            - indices (Tensor): Indices of K neighbors (B, N, K). Int64.
    """
    global _kdn_ext
    if not HAS_CUDA_KDN:
        # Try JIT compilation if compiled extension wasn't found initially
        _jit_compile_kdn()
    
    if HAS_CUDA_KDN and _kdn_ext is not None:
        # Use the compiled C++/CUDA extension
        try:
            return _kdn_ext.kdn_search(
                query_points, 
                reference_points, 
                reference_inv_covariances, # Pass inv_cov
                query_lengths, 
                reference_lengths, 
                k
            )
        except Exception as e:
            warnings.warn(f"Error calling cuda_kdn.kdn_search: {e}. Falling back to manual CPU (not implemented).")
            # Fallback to a pure Python/Torch implementation if needed (currently not implemented)
            raise NotImplementedError("Pure Python/Torch KDN fallback is not implemented.") from e
            
    else:
        # Fallback if no C++/CUDA extension is available (either compiled or JIT)
        warnings.warn("CUDA KDN extension not available. No CPU fallback implemented in __init__.py.")
        # You might want to implement a pure PyTorch version here as a fallback
        # similar to the TorchKNNIndex class in few_step_one_plane_stable_diffusion.py,
        # but calculating Mahalanobis distance instead of L2.
        raise RuntimeError("Required CUDA KDN extension is not available and no fallback is implemented.")

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

        return kdn_search(
            query_points,
            self.reference_points,
            self.reference_inv_covariances, # Pass inv_cov
            query_lengths,
            self.reference_lengths,
            k
        )

# Export symbols
__all__ = ['kdn_search', 'CudaKDNIndex', 'HAS_CUDA_KDN']

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