import torch
import torch.nn as nn
from typing import Dict, Any

class PruneGrowthModule(nn.Module):
    """
    神经元-超边协同凋亡生发与自剪枝模块
    """
    def __init__(self, max_neurons: int, max_edges: int, prune_threshold: float = 0.0, 
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.max_neurons = max_neurons
        self.max_edges = max_edges
        self.prune_threshold = prune_threshold
        
        # 掩码张量: True 表示存活
        self.register_buffer("neuron_mask", torch.ones(max_neurons, dtype=torch.bool, device=device))
        self.register_buffer("edge_mask", torch.ones(max_edges, dtype=torch.bool, device=device))
        
        # 历史因果贡献度
        self.register_buffer("contribution_history", torch.zeros(10, max_edges, **factory_kwargs))
        self.register_buffer("history_idx", torch.tensor(0, dtype=torch.long, device=device))
        
    def step_ultra_slow_evolution(self, VFE_full: float, VFE_masked_dict: Dict[int, float], 
                                  task_importance_mask: torch.Tensor = None):
        """
        [1000ms 时钟触发 / 任务边界触发]
        1. 计算因果贡献度: Contribution_e = VFE(w_e=0) - VFE(full)
        2. 超边凋亡判定
        3. 神经元凋亡判定
        4. 神经元生发判定
        """
        # 这里简化处理，VFE_masked_dict 是针对每条边被 mask 后计算得到的 VFE
        # 实际实现中可能采用一阶泰勒展开近似: Contribution_e ≈ grad(VFE)_e * w_e
        
        # 假设这里传入了每条边的贡献度估算
        # Contribution_e = VFE_masked - VFE_full
        
        contribution_e = torch.zeros(self.max_edges, device=self.edge_mask.device)
        for e_idx, vfe_val in VFE_masked_dict.items():
            if e_idx < self.max_edges:
                contribution_e[e_idx] = vfe_val - VFE_full
                
        # 记录历史
        idx = self.history_idx % 10
        self.contribution_history[idx] = contribution_e
        self.history_idx += 1
        
        mean_contribution = self.contribution_history.mean(dim=0)
        
        # 2. 超边凋亡判定
        # 连续低于阈值，且不在 task_importance_mask 保护内
        apop_edge_mask = (mean_contribution <= self.prune_threshold)
        if task_importance_mask is not None:
            apop_edge_mask = apop_edge_mask & (~task_importance_mask)
            
        self.edge_mask[apop_edge_mask] = False
        
        # 3. 神经元凋亡判定 (简化: 依赖于对应超边是否存活，实际需结合边索引图)
        # 这里仅作架构展示，如果一个神经元的全部关联边都 False，则 neuron_mask = False
        
        # 4. 生发判定 (容量补充)
        active_ratio = self.neuron_mask.float().mean()
        if active_ratio < 0.8 and VFE_full > 1.5:  # 假设 1.5 是 VFE 容忍上限
            # 找到死的神经元，复活一部分
            dead_indices = torch.where(~self.neuron_mask)[0]
            if len(dead_indices) > 0:
                num_to_grow = min(len(dead_indices), max(1, int(self.max_neurons * 0.05)))
                self.neuron_mask[dead_indices[:num_to_grow]] = True
                
        return self.neuron_mask, self.edge_mask
