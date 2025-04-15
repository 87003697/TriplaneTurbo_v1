import torch
import time
import sys
import os

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 首先尝试导入编译好的模块
    from cuda_knn import knn_search
    print("成功导入CUDA KNN扩展!")
except ImportError:
    print("无法导入CUDA KNN扩展，尝试直接加载...")
    try:
        import torch.utils.cpp_extension
        # 尝试即时编译并加载
        from torch.utils.cpp_extension import load
        cuda_knn = load(
            name="cuda_knn",
            sources=["ext.cpp", "knn.cu", "knn_cpu.cpp"],
            verbose=True,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=torch.cuda.is_available()
        )
        knn_search = cuda_knn.knn_search
        print("成功即时编译并加载CUDA KNN扩展!")
    except Exception as e:
        print(f"无法加载CUDA KNN扩展: {e}")
        print("使用纯PyTorch实现作为备选...")
        
        # 纯PyTorch实现作为备选
        def knn_search(query_points, ref_points, query_lengths, ref_lengths, k, norm=2):
            batch_size, num_query, dim = query_points.shape
            _, num_ref, _ = ref_points.shape
            
            # 计算欧氏距离
            distances = torch.zeros(batch_size, num_query, num_ref, device=query_points.device)
            for b in range(batch_size):
                valid_q = query_lengths[b].item()
                valid_r = ref_lengths[b].item()
                
                # 只处理有效的查询点和参考点
                q = query_points[b, :valid_q]
                r = ref_points[b, :valid_r]
                
                # 计算距离
                q_expanded = q.unsqueeze(1)  # [valid_q, 1, dim]
                r_expanded = r.unsqueeze(0)  # [1, valid_r, dim]
                
                if norm == 1:  # L1距离
                    dist = torch.abs(q_expanded - r_expanded).sum(dim=2)
                else:  # L2距离
                    dist = ((q_expanded - r_expanded) ** 2).sum(dim=2)
                
                distances[b, :valid_q, :valid_r] = dist
            
            # 找到K个最近邻
            indices = torch.zeros(batch_size, num_query, k, dtype=torch.int64, device=query_points.device)
            knn_dists = torch.zeros(batch_size, num_query, k, device=query_points.device)
            
            for b in range(batch_size):
                valid_q = query_lengths[b].item()
                valid_r = ref_lengths[b].item()
                
                # 只处理有效查询点
                if valid_q > 0 and valid_r > 0:
                    valid_k = min(k, valid_r)
                    
                    # 获取有效距离
                    valid_distances = distances[b, :valid_q, :valid_r]
                    
                    # 找到K个最近邻
                    dist_topk, idx_topk = torch.topk(valid_distances, k=valid_k, dim=1, largest=False)
                    
                    # 存储结果
                    indices[b, :valid_q, :valid_k] = idx_topk
                    knn_dists[b, :valid_q, :valid_k] = dist_topk
                    
                    # 填充剩余位置（如果k > valid_k）
                    if valid_k < k:
                        indices[b, :valid_q, valid_k:] = -1
                        knn_dists[b, :valid_q, valid_k:] = float('inf')
            
            return knn_dists, indices

# 生成测试数据
def generate_test_data(batch_size=2, num_query=1000, num_ref=5000, dim=3, device='cuda'):
    query_points = torch.rand(batch_size, num_query, dim, device=device)
    ref_points = torch.rand(batch_size, num_ref, dim, device=device)
    query_lengths = torch.tensor([num_query, num_query//2], dtype=torch.int64, device=device)
    ref_lengths = torch.tensor([num_ref, num_ref//2], dtype=torch.int64, device=device)
    return query_points, ref_points, query_lengths, ref_lengths

# 测试函数
def test_knn(batch_size=2, num_query=1000, num_ref=5000, dim=3, k=16, device='cuda'):
    print(f"测试参数: batch_size={batch_size}, num_query={num_query}, num_ref={num_ref}, dim={dim}, k={k}")
    query_points, ref_points, query_lengths, ref_lengths = generate_test_data(
        batch_size, num_query, num_ref, dim, device)
    
    # 测量性能
    torch.cuda.synchronize()
    start_time = time.time()
    
    distances, indices = knn_search(query_points, ref_points, query_lengths, ref_lengths, k)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"KNN搜索完成，用时: {(end_time - start_time) * 1000:.2f} ms")
    
    # 验证结果
    print(f"距离tensor形状: {distances.shape}")
    print(f"索引tensor形状: {indices.shape}")
    print(f"第一个点的最近邻距离: {distances[0, 0, :5]}")
    print(f"第一个点的最近邻索引: {indices[0, 0, :5]}")
    
    # 验证查询长度裁剪是否正确
    half_len = query_lengths[1].item()
    print(f"批次1的有效长度: {half_len}")
    valid_indices = indices[1, :half_len]
    invalid_indices = indices[1, half_len:]
    
    print(f"有效索引示例: {valid_indices[0]}")
    print(f"无效索引示例（应为全0）: {invalid_indices[0]}")
    
    # 验证结果准确性
    # 使用纯PyTorch实现作为参考
    _, ref_indices = pure_pytorch_knn(query_points, ref_points, query_lengths, ref_lengths, k)
    # 比较结果
    correct = (indices == ref_indices).float().mean()
    print(f"结果准确率: {correct * 100:.2f}%")
    
    return distances, indices

def pure_pytorch_knn(query_points, ref_points, query_lengths, ref_lengths, k, norm=2):
    """
    使用纯PyTorch实现KNN搜索，用于验证CUDA版本的准确性
    
    参数:
    - query_points: 形状为[batch_size, num_query, dim]的查询点张量
    - ref_points: 形状为[batch_size, num_ref, dim]的参考点张量
    - query_lengths: 形状为[batch_size]的张量，表示每个批次中实际查询点的数量
    - ref_lengths: 形状为[batch_size]的张量，表示每个批次中实际参考点的数量
    - k: 要搜索的最近邻数量
    - norm: 距离范数类型，1表示L1范数，2表示L2范数
    
    返回:
    - knn_dists: 形状为[batch_size, num_query, k]的最近邻距离
    - indices: 形状为[batch_size, num_query, k]的最近邻索引
    """
    batch_size, num_query, dim = query_points.shape
    _, num_ref, _ = ref_points.shape
    device = query_points.device
    
    # 初始化结果张量
    indices = torch.zeros(batch_size, num_query, k, dtype=torch.int64, device=device)
    knn_dists = torch.zeros(batch_size, num_query, k, device=device)
    
    # 按批次处理
    for b in range(batch_size):
        valid_q = min(query_lengths[b].item(), num_query)
        valid_r = min(ref_lengths[b].item(), num_ref)
        
        if valid_q == 0 or valid_r == 0:
            continue
            
        # 获取有效的查询点和参考点
        q_points = query_points[b, :valid_q]  # [valid_q, dim]
        r_points = ref_points[b, :valid_r]    # [valid_r, dim]
        
        # 计算距离矩阵
        # 使用高效的广播计算距离
        # 对于大规模点云，分批处理以节省内存
        
        # 内存优化: 分批计算距离
        batch_size_q = 1024  # 可根据可用内存调整
        num_batches = (valid_q + batch_size_q - 1) // batch_size_q
        
        # 为当前批次的结果分配临时存储
        all_dists = torch.zeros(valid_q, valid_r, device=device)
        
        for i in range(num_batches):
            start_idx = i * batch_size_q
            end_idx = min(start_idx + batch_size_q, valid_q)
            
            # 当前批次的查询点
            batch_q = q_points[start_idx:end_idx].unsqueeze(1)  # [batch_q, 1, dim]
            
            # 计算当前批次与所有参考点的距离
            if norm == 1:  # L1距离
                batch_dists = torch.sum(torch.abs(batch_q - r_points.unsqueeze(0)), dim=2)
            else:  # L2距离 (默认)
                batch_dists = torch.sum((batch_q - r_points.unsqueeze(0)) ** 2, dim=2)
                
            # 存储当前批次的距离
            all_dists[start_idx:end_idx] = batch_dists
        
        # 找到k个最近邻
        valid_k = min(k, valid_r)
        
        # 获取前k个最小距离及其索引
        dists, idxs = torch.topk(all_dists, k=valid_k, dim=1, largest=False)
        
        # 存储结果
        indices[b, :valid_q, :valid_k] = idxs
        knn_dists[b, :valid_q, :valid_k] = dists
        
        # 处理k > valid_r的情况
        if valid_k < k:
            indices[b, :valid_q, valid_k:] = -1  # 用-1填充无效索引
            knn_dists[b, :valid_q, valid_k:] = float('inf')  # 用无穷大填充无效距离
    
    return knn_dists, indices

if __name__ == "__main__":
    # 测试不同规模
    print("--- 小规模测试 ---")
    test_knn(batch_size=2, num_query=100, num_ref=1000, k=8)
    
    print("\n--- 中等规模测试 ---")
    test_knn(batch_size=2, num_query=1000, num_ref=10000, k=8)
    
    print("\n--- 大规模测试 ---")
    try:
        test_knn(batch_size=2, num_query=10000, num_ref=100000, k=8)
    except RuntimeError as e:
        print(f"大规模测试失败，可能是内存不足: {e}") 

    print("\n--- 超大规模测试 ---")
    try:
        test_knn(batch_size=2, num_query=100000, num_ref=1000000, k=8)
    except RuntimeError as e:
        print(f"超大规模测试失败，可能是内存不足: {e}") 
        
        