import torch
import torch.nn as nn
import torch.nn.functional as F

class ImplicitMoE(nn.Module):
    """
    无中心隐式MoE (基于局部侧向抑制竞争)
    不需要显式路由网络，通过膜电位侧向抑制产生功能柱化的稀疏激活。
    
    优化: 
    - 使用稀疏矩阵存储抑制掩码，避免 O(N²) 内存开销
    - 支持动态抑制强度调节，与脉冲预算控制器联动
    """
    def __init__(self, num_neurons: int, num_groups: int, inhibition_strength: float = 0.5):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_groups = num_groups
        self.neurons_per_group = num_neurons // num_groups
        
        # 修复: 使用 nn.Parameter 存储抑制强度，支持动态调节
        self.inhibition_strength = nn.Parameter(
            torch.tensor(inhibition_strength), requires_grad=False
        )
        self.base_inhibition = inhibition_strength  # 保存基准值
        
        # 构建稀疏的组内抑制掩码 (Block Diagonal)
        # 每组有 neurons_per_group * (neurons_per_group - 1) 个连接
        indices_list = []
        for i in range(num_groups):
            start_idx = i * self.neurons_per_group
            end_idx = start_idx + self.neurons_per_group
            for src in range(start_idx, end_idx):
                for dst in range(start_idx, end_idx):
                    if src != dst:
                        indices_list.append([src, dst])
        
        if indices_list:
            indices = torch.tensor(indices_list, dtype=torch.long).T
            values = torch.ones(indices.shape[1])
            self.register_buffer("sparse_indices", indices)
            self.register_buffer("sparse_values", values)
            self._use_sparse = True
        else:
            self.register_buffer("sparse_indices", torch.empty(2, 0, dtype=torch.long))
            self.register_buffer("sparse_values", torch.empty(0))
            self._use_sparse = False
    
    def set_inhibition_strength(self, strength: float):
        """动态设置抑制强度（与脉冲预算控制器联动）"""
        self.inhibition_strength.data.fill_(strength)
    
    def adjust_inhibition(self, delta: float):
        """调整抑制强度（增量式）"""
        new_strength = (self.inhibition_strength.data + delta).clamp(0.1, 2.0)
        self.inhibition_strength.data.copy_(new_strength)
        
    def forward(self, spk: torch.Tensor) -> torch.Tensor:
        """
        计算侧向抑制电流 I_inh
        spk: (batch_size, num_neurons) 或者是 (num_neurons,) 
        """
        if not self._use_sparse:
            return torch.zeros_like(spk)
        
        if spk.dim() == 1:
            # 稀疏矩阵-向量乘法: I_inh = W @ spk
            I_inh = torch.sparse_coo_tensor(
                self.sparse_indices, self.sparse_values, 
                (self.num_neurons, self.num_neurons)
            ).to(spk.device).to(spk.dtype) @ spk
            I_inh = I_inh * self.inhibition_strength
        else:
            # 批量: I_inh = spk @ W.T
            sparse_mat = torch.sparse_coo_tensor(
                self.sparse_indices, self.sparse_values,
                (self.num_neurons, self.num_neurons)
            ).to(spk.device).to(spk.dtype).to_dense()
            I_inh = torch.matmul(spk, sparse_mat.T) * self.inhibition_strength
        return I_inh
    
    def reset(self):
        """重置状态 (稀疏矩阵是静态的，无需重置)"""
        pass
