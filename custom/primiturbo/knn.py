import os
import sys
import torch
import torch.nn.functional as F

# 添加KNN扩展路径
knn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extern")
if knn_path not in sys.path:
    sys.path.append(knn_path)

# 尝试导入CUDA KNN扩展
try:
    from knn import knn_search
    print("成功导入CUDA KNN扩展")
    HAS_CUDA_KNN = True
except ImportError:
    print("CUDA KNN扩展未找到或无法加载，尝试即时编译...")
    try:
        import torch.utils.cpp_extension
        from torch.utils.cpp_extension import load
        
        # 获取KNN扩展目录
        knn_dir = os.path.join(knn_path, "knn")
        if os.path.exists(knn_dir):
            # 尝试即时编译
            sources = [
                os.path.join(knn_dir, "ext.cpp"),
                os.path.join(knn_dir, "knn.cu"),
                os.path.join(knn_dir, "knn_cpu.cpp")
            ]
            
            if all(os.path.exists(s) for s in sources):
                cuda_knn = load(
                    name="cuda_knn",
                    sources=sources,
                    verbose=True,
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3"],
                    with_cuda=torch.cuda.is_available()
                )
                knn_search = cuda_knn.knn_search
                print("成功即时编译并加载CUDA KNN扩展")
                HAS_CUDA_KNN = True
            else:
                print("KNN扩展源文件不完整，无法编译")
                HAS_CUDA_KNN = False
        else:
            print(f"KNN扩展目录不存在: {knn_dir}")
            HAS_CUDA_KNN = False
    except Exception as e:
        print(f"无法加载CUDA KNN扩展: {e}")
        HAS_CUDA_KNN = False


# 添加CudaKNNIndex类
class CudaKNNIndex(object):
    """使用我们实现的CUDA KNN扩展的KNN索引"""
    
    def __init__(self):
        self.points = None
        self.points_length = None
        self.batch_size = 0
        self.device = None
        
    def add(self, x, lengths=None):
        """添加参考点到索引中
        
        Args:
            x: 形状为[batch_size, num_points, dim]的参考点
            lengths: 每个batch中有效的点数量，形状为[batch_size]
        """
        self.points = x
        self.batch_size = x.shape[0]
        self.device = x.device
        
        if lengths is None:
            # 如果未提供长度，则所有点都有效
            self.points_length = torch.full(
                (self.batch_size,), x.shape[1], 
                dtype=torch.int64, device=self.device
            )
        else:
            self.points_length = lengths
            
    def search(self, query, k, norm = 2, lengths=None):
        """搜索最近邻
        
        Args:
            query: 形状为[batch_size, num_queries, dim]的查询点
            k: 要返回的最近邻数量
            lengths: 每个batch中有效的查询点数量，形状为[batch_size]
            
        Returns:
            distances: 形状为[batch_size, num_queries, k]的距离
            indices: 形状为[batch_size, num_queries, k]的索引
        """
        if not HAS_CUDA_KNN:
            raise ImportError("CUDA KNN扩展未安装或无法加载")
            
        if self.points is None:
            raise ValueError("必须先调用add方法添加参考点")
            
        # 确保查询点和参考点在相同设备上
        if query.device != self.device:
            query = query.to(self.device)
            
        # 创建查询长度张量
        if lengths is None:
            query_lengths = torch.full(
                (query.shape[0],), query.shape[1], 
                dtype=torch.int64, device=query.device
            )
        else:
            query_lengths = lengths
            
        # 使用CUDA KNN扩展
        distances, indices = knn_search(
            query,
            self.points,
            query_lengths,
            self.points_length,
            k,
            norm
        )
        
        return distances, indices
    
@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='sphere', min_near=0.05):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)



    if type == 'sphere_general':

        # Normalize the direction of the rays
        rays_d = F.normalize(rays_d, dim=-1)

        # Calculate the dot product of the direction of the ray and the origin of the ray
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)

        # Calculate the discriminant (b^2 - 4ac)
        discriminant = b**2 - 4.0 * (radius**2 - bound**2)

        # If the discriminant is less than 0, the ray does not intersect the sphere
        mask = discriminant >= 0
        discriminant = torch.where(mask, discriminant, torch.zeros_like(discriminant))

        # Calculate the near and far intersection distances
        near = 0.5 * (-b - torch.sqrt(discriminant))
        far = 0.5 * (-b + torch.sqrt(discriminant))

        # Ensure that 'near' is not closer than 'min_near'
        near = torch.max(near, min_near * torch.ones_like(near))

    elif type == 'sphere':

        near = radius - bound
        far = radius + bound

    return near, far
