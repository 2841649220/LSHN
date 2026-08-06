import torch
import torch.nn as nn
import torch.nn.functional as F

class ImplicitMoE(nn.Module):
    """
    无中心隐式MoE (基于局部侧向抑制竞争)
    不需要显式路由网络，通过膜电位侧向抑制产生功能柱化的稀疏激活。

    计算优化: I_inh_i = Σ_{j∈group(i), j≠i} W_inh * spk_j = W_inh * (S_group(i) - spk_i)
    其中 S_group = 组内脉冲总和。用组和 scatter/gather 实现, 复杂度 O(N) 而非 O(N²)。
    """
    def __init__(self, num_neurons: int, num_groups: int, inhibition_strength: float = 0.5,
                 device=None, dtype=None):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_groups = num_groups
        self.neurons_per_group = num_neurons // num_groups
        self.inhibition_strength = inhibition_strength

        # 每个神经元所属的功能柱组 id (最后一组吸收余数, 与原分组边界一致)
        # 注意: group_ids 必须与其它模块同设备创建 (构造时即指定 device 的用法
        # 曾因 CPU buffer 混入 CUDA scatter_add 而崩溃)
        group_ids = torch.arange(num_neurons, device=device, dtype=torch.long) // self.neurons_per_group
        self.register_buffer("group_ids", group_ids.clamp(0, num_groups - 1))

    def forward(self, spk: torch.Tensor) -> torch.Tensor:
        """
        计算侧向抑制电流 I_inh
        spk: (batch_size, num_neurons) 或者是 (num_neurons,)

        I_inh_i = sum_{j in group, j != i} W_inh * spk_j
        """
        is_batched = spk.dim() > 1
        if not is_batched:
            spk = spk.unsqueeze(0)

        batch_size = spk.shape[0]
        group_ids = self.group_ids.unsqueeze(0).expand(batch_size, -1)

        # 组内脉冲总和 S_group, 再按神经元取回 → (S_group(i) - spk_i) * strength
        group_sum = torch.zeros(batch_size, self.num_groups, device=spk.device, dtype=spk.dtype)
        group_sum.scatter_add_(1, group_ids, spk)
        group_total = group_sum.gather(1, group_ids)

        I_inh = (group_total - spk) * self.inhibition_strength

        if not is_batched:
            I_inh = I_inh.squeeze(0)
        return I_inh
