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
    
    return distances, indices

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
        
        