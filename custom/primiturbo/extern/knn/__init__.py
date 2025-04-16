import os
import torch
import warnings
import sys

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