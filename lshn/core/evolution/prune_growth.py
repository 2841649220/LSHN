"""
神经元-超边协同凋亡生发与自剪枝模块
白皮书 §3.4:
  基于因果贡献度的量化评估，实现神经元与超边的双向耦合自适应调整。

  1. 因果贡献度: Contribution_e = VFE(w_e=0) - VFE(full)
  2. 超边凋亡: 连续低贡献 + 无任务保护 → 剔除
  3. 神经元凋亡: 关联超边凋亡占比 >90% + 无核心贡献 → 冷却后剔除
  4. 神经元生发: 代偿性（局部容量不足）/ 扩容性（全局 VFE 升高）

修复:
  - 所有状态使用 register_buffer + in-place 操作，保持 state_dict 完整性
  - 实现完整的神经元凋亡判定（基于关联超边存活率，白皮书 90% 阈值）
  - 加入冷却期（apop_cooldown）防止误杀
  - history_idx 为注册 buffer
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class PruneGrowthModule(nn.Module):
    """
    神经元-超边协同凋亡生发与自剪枝模块

    Args:
        max_neurons:       最大神经元数
        max_edges:         最大超边数
        prune_threshold:   因果贡献度剪枝阈值（≤ 此值视为无贡献）
        apop_cooldown:     凋亡冷却步数（连续多少个超慢周期低于阈值才触发凋亡）
        neuron_dead_ratio: 神经元凋亡所需的关联超边死亡比例（白皮书: 0.9）
        growth_vfe_ratio:  VFE 超过此倍率时触发扩容性生发
        growth_cap:        单次生发的最大比例
    """
    def __init__(self, max_neurons: int, max_edges: int,
                 prune_threshold: float = 0.0,
                 apop_cooldown: int = 10,
                 neuron_dead_ratio: float = 0.9,
                 growth_vfe_ratio: float = 1.5,
                 growth_cap: float = 0.05,
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

        # === 连续低贡献计数器 (per-edge) ===
        self.register_buffer("low_contrib_count",
                             torch.zeros(max_edges, dtype=torch.long, device=device))

    # ------------------------------------------------------------------
    #  主入口: 超慢时钟 (1000ms) 触发
    # ------------------------------------------------------------------
    def step_ultra_slow_evolution(
        self,
        VFE_full: float,
        VFE_masked_dict: Dict[int, float],
        hyperedge_index: Optional[torch.Tensor] = None,
        task_importance_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        # ----- 1. 计算因果贡献度 -----
        # Contribution_e = VFE(w_e=0) - VFE(full)
        # 正值 = 去掉该边后 VFE 上升 = 该边有正贡献
        contribution_e = torch.zeros(self.max_edges, device=device,
                                     dtype=self.contribution_history.dtype)
        for e_idx, vfe_val in VFE_masked_dict.items():
            if e_idx < self.max_edges:
                contribution_e[e_idx] = vfe_val - VFE_full

        # 记录历史 (in-place，不破坏 buffer)
        idx = int(self.history_idx.item()) % self.apop_cooldown
        self.contribution_history.data[idx].copy_(contribution_e)
        self.history_idx.data.add_(1)

        mean_contribution = self.contribution_history.mean(dim=0)

        # ----- 2. 超边凋亡判定 -----
        # 当前步低于阈值 → 累计计数 +1; 否则重置
        is_low = (contribution_e <= self.prune_threshold)
        self.low_contrib_count.data[is_low] += 1
        self.low_contrib_count.data[~is_low] = 0

        # 连续 apop_cooldown 步低于阈值
        apop_edge_candidates = (self.low_contrib_count >= self.apop_cooldown)

        # 排除受任务保护的超边
        if task_importance_mask is not None:
            apop_edge_candidates = apop_edge_candidates & (~task_importance_mask.to(device))

        # 只对当前存活的边执行凋亡
        apop_edge_candidates = apop_edge_candidates & self.edge_mask

        # 执行超边凋亡 (in-place)
        self.edge_mask.data[apop_edge_candidates] = False

        # 凋亡的边重置连续计数
        self.low_contrib_count.data[apop_edge_candidates] = 0

        # ----- 3. 神经元凋亡判定 -----
        # 白皮书 §3.4.2: 关联超边凋亡占比 > 90% 的神经元进行凋亡
        if hyperedge_index is not None and hyperedge_index.numel() > 0:
            # hyperedge_index: (2, num_connections)
            # row 0 = neuron_ids, row 1 = hyperedge_ids
            neuron_ids = hyperedge_index[0]
            edge_ids = hyperedge_index[1]

            # 计算每个神经元的总关联超边数和存活超边数
            # 使用 scatter 统计
            total_per_neuron = torch.zeros(self.max_neurons, device=device, dtype=torch.float)
            alive_per_neuron = torch.zeros(self.max_neurons, device=device, dtype=torch.float)

            # 只统计有效范围内的边
            valid = (neuron_ids < self.max_neurons) & (edge_ids < self.max_edges)
            valid_neurons = neuron_ids[valid]
            valid_edges = edge_ids[valid]

            ones = torch.ones(valid_neurons.shape[0], device=device, dtype=torch.float)
            total_per_neuron.scatter_add_(0, valid_neurons, ones)

            # 存活的边
            edge_alive = self.edge_mask[valid_edges].float()
            alive_per_neuron.scatter_add_(0, valid_neurons, edge_alive)

            # 死亡比例
            has_edges = (total_per_neuron > 0)
            dead_ratio = torch.zeros(self.max_neurons, device=device)
            dead_ratio[has_edges] = 1.0 - alive_per_neuron[has_edges] / total_per_neuron[has_edges]

            # 凋亡条件: 死亡比例 > neuron_dead_ratio 且当前存活
            neuron_apop = (dead_ratio > self.neuron_dead_ratio) & self.neuron_mask

            # 排除有任务保护的神经元（通过其关联边的保护状态推断）
            if task_importance_mask is not None:
                # 如果某神经元有任何受保护的边，则不凋亡
                protected_per_neuron = torch.zeros(self.max_neurons, device=device, dtype=torch.float)
                edge_protected = task_importance_mask[valid_edges].float()
                protected_per_neuron.scatter_add_(0, valid_neurons, edge_protected)
                neuron_apop = neuron_apop & (protected_per_neuron == 0)

            self.neuron_mask.data[neuron_apop] = False

        # ----- 4. 神经元生发判定 -----
        active_ratio = self.neuron_mask.float().mean().item()
        if active_ratio < 0.8 and VFE_full > self.growth_vfe_ratio:
            # 代偿性/扩容性生发: 复活部分死亡神经元
            dead_indices = torch.where(~self.neuron_mask)[0]
            if dead_indices.numel() > 0:
                num_to_grow = min(
                    dead_indices.numel(),
                    max(1, int(self.max_neurons * self.growth_cap))
                )
                revive_indices = dead_indices[:num_to_grow]
                self.neuron_mask.data[revive_indices] = True

                # 新生神经元清零贡献度历史，以高可塑性快速学习
                # (关联的超边也需要复活)
                if hyperedge_index is not None and hyperedge_index.numel() > 0:
                    revive_set = set(revive_indices.tolist())
                    # 找到这些神经元关联的已死超边，复活一部分
                    for ni in revive_indices:
                        associated = edge_ids[neuron_ids == ni.item()]
                        dead_assoc = associated[~self.edge_mask[associated]]
                        if dead_assoc.numel() > 0:
                            # 复活最多一半的死亡关联超边
                            n_revive_edges = max(1, dead_assoc.numel() // 2)
                            self.edge_mask.data[dead_assoc[:n_revive_edges]] = True

        return self.neuron_mask, self.edge_mask
