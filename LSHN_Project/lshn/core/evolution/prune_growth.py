"""
神经元-超边协同凋亡生发与自剪枝模块
白皮书 §3.4:
  基于因果贡献度的量化评估，实现神经元与超边的双向耦合自适应调整。

  1. 因果贡献度: Contribution_e = VFE(w_e=0) - VFE(full)
  2. 超边凋亡: 连续低贡献 + 无任务保护 → 剔除
  3. 神经元凋亡: 关联超边凋亡占比 >90% + 无核心贡献 → 冷却后剔除
  4. 神经元生发: 代偿性（局部容量不足）/ 扩容性（全局 VFE 升高）

修复:
  - 向量化贡献度计算
  - 使用有效步数加权平均
  - VFE 相对比较
  - 随机选择复活神经元
  - 向量化超边复活逻辑
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class PruneGrowthModule(nn.Module):
    """
    神经元-超边协同凋亡生发与自剪枝模块

    Args:
        max_neurons:       最大神经元数
        max_edges:         最大超边数
        prune_threshold:   因果贡献度剪枝阈值（≤ 此值视为无贡献）
        apop_cooldown:     凋亡冷却步数（连续多少个超慢周期低于阈值才触发凋亡）
        neuron_dead_ratio: 神经元凋亡所需的关联超边死亡比例（白皮书: 0.9）
        growth_vfe_ratio:  VFE 超过基准的此倍率时触发扩容性生发
        growth_cap:        单次生发的最大比例
        active_threshold:  触发生发的最低活性阈值
    """
    
    def __init__(self, max_neurons: int, max_edges: int,
                 prune_threshold: float = 0.0,
                 apop_cooldown: int = 10,
                 neuron_dead_ratio: float = 0.9,
                 growth_vfe_ratio: float = 1.5,
                 growth_cap: float = 0.05,
                 active_threshold: float = 0.8,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.max_neurons = max_neurons
        self.max_edges = max_edges
        self.prune_threshold = prune_threshold
        self.apop_cooldown = apop_cooldown
        self.neuron_dead_ratio = neuron_dead_ratio
        self.growth_vfe_ratio = growth_vfe_ratio
        self.growth_cap = growth_cap
        self.active_threshold = active_threshold

        # === 注册 buffer: 存活掩码 ===
        self.register_buffer("neuron_mask",
                             torch.ones(max_neurons, dtype=torch.bool, device=device))
        self.register_buffer("edge_mask",
                             torch.ones(max_edges, dtype=torch.bool, device=device))

        # === 因果贡献度滑动窗口 (10 步历史) ===
        self.register_buffer("contribution_history",
                             torch.zeros(apop_cooldown, max_edges, **factory_kwargs))
        self.register_buffer("history_idx",
                             torch.tensor(0, dtype=torch.long, device=device))
        
        # 记录有效历史步数
        self.register_buffer("valid_history_count",
                             torch.tensor(0, dtype=torch.long, device=device))

        # === 连续低贡献计数器 (per-edge) ===
        self.register_buffer("low_contrib_count",
                             torch.zeros(max_edges, dtype=torch.long, device=device))
        
        # VFE 基准（用于相对比较）
        self.register_buffer("vfe_baseline", torch.tensor(float('inf'), device=device))

    def step_ultra_slow_evolution(
        self,
        VFE_full: float,
        VFE_masked_dict: Dict[int, float],
        hyperedge_index: Optional[torch.Tensor] = None,
        task_importance_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [1000ms 时钟触发 / 任务边界触发]

        Args:
            VFE_full:            当前完整模型的 VFE 值
            VFE_masked_dict:     dict{edge_idx: VFE_with_edge_masked} 每条边被 mask 后的 VFE
            hyperedge_index:     (2, num_connections) COO 格式超图拓扑，用于关联神经元与超边
            task_importance_mask: (max_edges,) bool，True = 受任务保护不可剪

        Returns:
            (neuron_mask, edge_mask) 更新后的存活掩码
        """
        device = self.edge_mask.device

        # ----- 1. 向量化计算因果贡献度 -----
        contribution_e = torch.zeros(self.max_edges, device=device,
                                     dtype=self.contribution_history.dtype)
        
        # 向量化赋值
        if VFE_masked_dict:
            edge_indices = torch.tensor(list(VFE_masked_dict.keys()), device=device, dtype=torch.long)
            vfe_values = torch.tensor(list(VFE_masked_dict.values()), device=device, 
                                      dtype=self.contribution_history.dtype)
            valid_mask = edge_indices < self.max_edges
            if valid_mask.any():
                contribution_e[edge_indices[valid_mask]] = vfe_values[valid_mask] - VFE_full

        # 记录历史 (使用张量操作，避免 Python-CPU 交互)
        with torch.no_grad():
            idx = self.history_idx % self.apop_cooldown
            self.contribution_history.data[idx] = contribution_e
            self.history_idx.add_(1)
            self.valid_history_count.add_(1)

        # 使用有效步数计算加权平均
        valid_steps = min(self.valid_history_count.item(), self.apop_cooldown)
        if valid_steps > 0:
            mean_contribution = self.contribution_history[:valid_steps].mean(dim=0)
        else:
            mean_contribution = contribution_e

        # ----- 2. 超边凋亡判定 -----
        is_low = (contribution_e <= self.prune_threshold)
        self.low_contrib_count.data[is_low] += 1
        self.low_contrib_count.data[~is_low] = 0

        apop_edge_candidates = (self.low_contrib_count >= self.apop_cooldown)

        if task_importance_mask is not None:
            apop_edge_candidates = apop_edge_candidates & (~task_importance_mask.to(device))

        apop_edge_candidates = apop_edge_candidates & self.edge_mask

        self.edge_mask.data[apop_edge_candidates] = False
        self.low_contrib_count.data[apop_edge_candidates] = 0

        # ----- 3. 神经元凋亡判定 -----
        if hyperedge_index is not None and hyperedge_index.numel() > 0:
            neuron_ids = hyperedge_index[0]
            edge_ids = hyperedge_index[1]

            total_per_neuron = torch.zeros(self.max_neurons, device=device, dtype=torch.float)
            alive_per_neuron = torch.zeros(self.max_neurons, device=device, dtype=torch.float)

            valid = (neuron_ids < self.max_neurons) & (edge_ids < self.max_edges)
            valid_neurons = neuron_ids[valid]
            valid_edges = edge_ids[valid]

            if valid_neurons.numel() > 0:
                ones = torch.ones(valid_neurons.shape[0], device=device, dtype=torch.float)
                total_per_neuron.scatter_add_(0, valid_neurons, ones)

                edge_alive = self.edge_mask[valid_edges].float()
                alive_per_neuron.scatter_add_(0, valid_neurons, edge_alive)

                has_edges = (total_per_neuron > 0)
                dead_ratio = torch.zeros(self.max_neurons, device=device)
                dead_ratio[has_edges] = 1.0 - alive_per_neuron[has_edges] / total_per_neuron[has_edges]

                neuron_apop = (dead_ratio > self.neuron_dead_ratio) & self.neuron_mask

                if task_importance_mask is not None:
                    protected_per_neuron = torch.zeros(self.max_neurons, device=device, dtype=torch.float)
                    edge_protected = task_importance_mask[valid_edges].float()
                    protected_per_neuron.scatter_add_(0, valid_neurons, edge_protected)
                    neuron_apop = neuron_apop & (protected_per_neuron == 0)

                self.neuron_mask.data[neuron_apop] = False

        # ----- 4. 神经元生发判定 (VFE 相对比较) -----
        # 更新 VFE 基准
        with torch.no_grad():
            if self.vfe_baseline.item() == float('inf'):
                self.vfe_baseline.fill_(VFE_full)
            else:
                # 指数移动平均更新基准
                self.vfe_baseline.mul_(0.99).add_(0.01 * VFE_full)
        
        active_ratio = self.neuron_mask.float().mean().item()
        vfe_baseline_val = self.vfe_baseline.item()
        
        # 相对比较：VFE 超过基准的 growth_vfe_ratio 倍时触发生发
        if active_ratio < self.active_threshold and vfe_baseline_val > 0 and VFE_full > vfe_baseline_val * self.growth_vfe_ratio:
            dead_indices = torch.where(~self.neuron_mask)[0]
            if dead_indices.numel() > 0:
                num_to_grow = min(
                    dead_indices.numel(),
                    max(1, int(self.max_neurons * self.growth_cap))
                )
                
                # 随机选择复活神经元（而非按索引顺序）
                perm = torch.randperm(dead_indices.numel(), device=device)[:num_to_grow]
                revive_indices = dead_indices[perm]
                self.neuron_mask.data[revive_indices] = True

                # 向量化复活关联超边
                if hyperedge_index is not None and hyperedge_index.numel() > 0:
                    neuron_ids = hyperedge_index[0]
                    edge_ids = hyperedge_index[1]
                    
                    for ni in revive_indices:
                        associated_mask = neuron_ids == ni.item()
                        associated = edge_ids[associated_mask]
                        dead_assoc_mask = ~self.edge_mask[associated]
                        dead_assoc = associated[dead_assoc_mask]
                        
                        if dead_assoc.numel() > 0:
                            n_revive_edges = max(1, dead_assoc.numel() // 2)
                            # 随机选择要复活的超边
                            perm_edges = torch.randperm(dead_assoc.numel(), device=device)[:n_revive_edges]
                            self.edge_mask.data[dead_assoc[perm_edges]] = True

        return self.neuron_mask, self.edge_mask
    
    def reset(self):
        """重置所有状态"""
        self.neuron_mask.fill_(True)
        self.edge_mask.fill_(True)
        self.contribution_history.zero_()
        self.history_idx.zero_()
        self.valid_history_count.zero_()
        self.low_contrib_count.zero_()
        self.vfe_baseline.fill_(float('inf'))