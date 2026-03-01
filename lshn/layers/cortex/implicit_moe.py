import torch
import torch.nn as nn
import torch.nn.functional as F

class ImplicitMoE(nn.Module):
    """
    无中心隐式MoE (基于局部侧向抑制竞争)
    不需要显式路由网络，通过膜电位侧向抑制产生功能柱化的稀疏激活。
    """
    def __init__(self, num_neurons: int, num_groups: int, inhibition_strength: float = 0.5):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_groups = num_groups
        self.neurons_per_group = num_neurons // num_groups
        self.inhibition_strength = inhibition_strength
        
        # 定义组内抑制连接掩码 (Block Diagonal)
        mask = torch.zeros(num_neurons, num_neurons)
        for i in range(num_groups):
            start_idx = i * self.neurons_per_group
            end_idx = start_idx + self.neurons_per_group
            # 组内全连接抑制 (除自己外)
            mask[start_idx:end_idx, start_idx:end_idx] = 1.0
            mask[range(start_idx, end_idx), range(start_idx, end_idx)] = 0.0
            
        self.register_buffer("inhibition_mask", mask)
        
    def forward(self, spk: torch.Tensor) -> torch.Tensor:
        """
        计算侧向抑制电流 I_inh
        spk: (batch_size, num_neurons) 或者是 (num_neurons,) 
        """
        # I_inh_i = \sum_{j \in group, j \neq i} W_inh * spk_j
        if spk.dim() == 1:
            I_inh = torch.matmul(self.inhibition_mask, spk) * self.inhibition_strength
        else:
            I_inh = torch.matmul(spk, self.inhibition_mask.T) * self.inhibition_strength
        return I_inh
