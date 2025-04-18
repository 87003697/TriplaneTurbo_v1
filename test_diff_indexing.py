import torch
import torch.nn.functional as F
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Custom autograd function for differentiable indexing
class DifferentiableIndexing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, indices):
        # Save tensors needed for backward
        ctx.save_for_backward(indices)
        ctx.input_size = input_tensor.size(0)
        ctx.feat_size = input_tensor.size(1) if input_tensor.dim() > 1 else 1
        
        # Simple indexing in forward pass
        return input_tensor[indices]
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        indices, = ctx.saved_tensors
        
        # Initialize gradient for input tensor
        input_grad = torch.zeros(
            ctx.input_size, ctx.feat_size, 
            device=grad_output.device, 
            dtype=grad_output.dtype
        )
        
        # Use index_add_ to scatter gradients back to input tensor
        # This is the key operation that makes indexing differentiable
        input_grad.index_add_(0, indices, grad_output)
        
        # Return gradients for each input of forward function
        # None for indices since we don't compute gradients for indices
        return input_grad, None
    
def test_differentiable_indexing():
    """测试DifferentiableIndexing操作是否正确保留梯度"""
    print("开始测试DifferentiableIndexing...")
    
    # 创建一个需要梯度的点云
    points = torch.randn(1000, 3, requires_grad=True)
    print(f"点云形状: {points.shape}")
    
    # 创建一些随机索引
    indices = torch.randint(0, 1000, (8388608*8,))
    print(f"索引形状: {indices.shape}")
    print(f"索引类型: {indices.dtype}")
    
    # 1. 测试前向传播: 比较直接索引和DifferentiableIndexing的结果
    direct_indexed = points[indices]
    diff_indexed = DifferentiableIndexing.apply(points, indices)
    
    # 验证前向传播的结果是否相同
    forward_match = torch.allclose(direct_indexed, diff_indexed)
    print(f"前向传播结果匹配: {forward_match}")
    
    # 2. 测试梯度传播
    print("\n开始测试梯度传播...")
    
    # 定义一个目标函数: 计算选择点的平均L2范数
    def compute_loss(selected_points):
        return selected_points.pow(2).sum(dim=1).mean()
    
    # 方法1: 使用直接索引 (应该没有梯度)
    direct_points = points.clone().detach().requires_grad_(True)
    direct_selected = direct_points[indices]
    direct_loss = compute_loss(direct_selected)
    print(f"直接索引的损失: {direct_loss.item()}")
    
    # 尝试反向传播
    direct_loss.backward()
    has_direct_grad = direct_points.grad is not None and direct_points.grad.abs().sum() > 0
    print(f"直接索引是否有梯度: {has_direct_grad}")
    if has_direct_grad:
        print(f"直接索引梯度的范数: {direct_points.grad.norm().item()}")
    
    # 方法2: 使用DifferentiableIndexing
    diff_points = points.clone().detach().requires_grad_(True)
    diff_selected = DifferentiableIndexing.apply(diff_points, indices)
    diff_loss = compute_loss(diff_selected)
    print(f"DifferentiableIndexing的损失: {diff_loss.item()}")
    
    # 反向传播
    if diff_points.grad is not None:
        diff_points.grad.zero_()
    diff_loss.backward()
    has_diff_grad = diff_points.grad is not None and diff_points.grad.abs().sum() > 0
    print(f"DifferentiableIndexing是否有梯度: {has_diff_grad}")
    if has_diff_grad:
        print(f"DifferentiableIndexing梯度的范数: {diff_points.grad.norm().item()}")
        # 验证梯度是否正确分配到对应索引位置
        nonzero_grads = diff_points.grad[diff_points.grad.abs().sum(dim=1) > 0]
        unique_indices = indices.unique()
        print(f"有梯度的点数量: {nonzero_grads.shape[0]}, 索引中唯一点的数量: {unique_indices.shape[0]}")
    
    # 3. 实际场景模拟: 测试在点云插值场景中的表现
    print("\n模拟点云插值的测试...")
    
    # 创建随机点云和特征
    point_cloud = torch.randn(100, 3, requires_grad=True)
    features = torch.randn(100, 8, requires_grad=True)
    
    # 创建查询点
    query_points = torch.randn(50, 3)
    
    # 假设KNN会找到这些索引
    k = 4
    knn_indices = torch.randint(0, 100, (50, k))
    
    # 使用DifferentiableIndexing获取邻域点和特征
    neighbor_positions = DifferentiableIndexing.apply(point_cloud, knn_indices.reshape(-1)).reshape(50, k, 3)
    neighbor_features = DifferentiableIndexing.apply(features, knn_indices.reshape(-1)).reshape(50, k, 8)
    
    # 计算距离
    distances = torch.norm(neighbor_positions - query_points.unsqueeze(1), dim=2)
    
    # 权重: 反距离加权
    weights = 1.0 / (distances + 1e-8)
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    # 加权平均获取插值后的特征
    interpolated_features = torch.sum(neighbor_features * weights.unsqueeze(-1), dim=1)
    
    # 计算一个简单的损失
    interp_loss = interpolated_features.pow(2).sum()
    
    # 反向传播
    interp_loss.backward()
    
    # 验证梯度流动
    has_point_grad = point_cloud.grad is not None and point_cloud.grad.abs().sum() > 0
    has_feature_grad = features.grad is not None and features.grad.abs().sum() > 0
    
    print(f"点云位置是否有梯度: {has_point_grad}")
    if has_point_grad:
        print(f"点云位置梯度的范数: {point_cloud.grad.norm().item()}")
    
    print(f"点云特征是否有梯度: {has_feature_grad}")
    if has_feature_grad:
        print(f"点云特征梯度的范数: {features.grad.norm().item()}")
    
    return forward_match and has_diff_grad and has_point_grad and has_feature_grad

if __name__ == "__main__":
    success = test_differentiable_indexing()
    print(f"\n测试{'成功' if success else '失败'}") 