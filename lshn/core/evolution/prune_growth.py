import torch
import torch.nn as nn
from typing import Dict, Optional


class PruneGrowthModule(nn.Module):
    """
    神经元-超边协同凋亡生发与自剪枝模块
    """

    def __init__(self, max_neurons: int, max_edges: int, prune_threshold: float = 0.0,
                 min_alive_ratio: float = 0.3, growth_probability: float = 0.05,
                 min_history_samples: int = 10, min_alive_edge_ratio: float = 0.3,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.max_neurons = max_neurons
        self.max_edges = max_edges
        self.prune_threshold = prune_threshold
        self.min_alive_ratio = min_alive_ratio
        self.growth_probability = growth_probability
        # 凋亡判定所需的最少历史样本数 (白皮书语义: 连续 10 个慢时间步
        # 贡献度 ≤ 阈值才剪枝; 不足时只记录历史, 不剪)
        self.min_history_samples = min_history_samples
        # 最低超边存活比例: 剪枝后存活数跌破 max_edges * 该比例时回退本次剪枝
        self.min_alive_edge_ratio = min_alive_edge_ratio

        # 掩码张量: True 表示存活
        self.register_buffer("neuron_mask", torch.ones(max_neurons, dtype=torch.bool, device=device))
        self.register_buffer("edge_mask", torch.ones(max_edges, dtype=torch.bool, device=device))

        # 历史因果贡献度 (NaN 初始化: 未被评估的边不参与均值,
        # 避免零初始化稀释真实贡献)
        # 环形容量取 max(10, min_history_samples): 默认 10 槽与白皮书
        # "连续 10 步"一致; 更大的 min_history_samples 同步扩大容量。
        self.register_buffer(
            "contribution_history",
            torch.full((max(10, min_history_samples), max_edges), float("nan"), **factory_kwargs)
        )
        self.register_buffer("history_idx", torch.tensor(0, dtype=torch.long, device=device))

    def step_ultra_slow_evolution(self, VFE_full: float, VFE_masked_dict: Dict[int, float],
                                  task_importance_mask: Optional[torch.Tensor] = None,
                                  hyperedge_index: Optional[torch.Tensor] = None):
        """
        [1000ms 时钟触发 / 任务边界触发]
        1. 计算因果贡献度: Contribution_e = VFE(w_e=0) - VFE(full)
        2. 超边凋亡判定
        3. 神经元凋亡判定 (依赖超图拓扑: 无任何存活关联边的神经元凋亡)
        4. 神经元生发判定 (活跃比例低于 min_alive_ratio 时复活)
        """
        # 无任何贡献度数据时不进行凋亡判定
        # (否则全零贡献会被误判为低于阈值而全量剪枝)
        if not VFE_masked_dict:
            return self.neuron_mask, self.edge_mask

        # 未被评估的超边不参与凋亡判定
        contribution_e = torch.full(
            (self.max_edges,), float("nan"), device=self.edge_mask.device
        )
        for e_idx, vfe_val in VFE_masked_dict.items():
            if e_idx < self.max_edges:
                contribution_e[e_idx] = vfe_val - VFE_full

        # 记录历史
        idx = self.history_idx % self.contribution_history.shape[0]
        self.contribution_history[idx] = contribution_e
        self.history_idx += 1

        # 历史积累: 环形索引累计数未写满 min_history_samples 槽时只记录历史,
        # 跳过凋亡判定 —— 否则 1 个样本时 nanmean 退化为单样本贡献度,
        # 训练初期 (批次/任务未稳定) 会误剪整批超边。
        if self.history_idx < self.min_history_samples:
            return self.neuron_mask, self.edge_mask

        # NaN 安全均值: 未被评估的边 (NaN) 不参与平均
        mean_contribution = torch.nanmean(self.contribution_history, dim=0)

        # 2. 超边凋亡判定
        # 连续低于阈值, 且不在 task_importance_mask 保护内
        apop_edge_mask = (mean_contribution <= self.prune_threshold) & torch.isfinite(mean_contribution)
        if task_importance_mask is not None:
            task_importance_mask = task_importance_mask.to(self.edge_mask.device)
            task_importance_mask = task_importance_mask.bool()  # 显式 bool, 防御非 bool 张量
            apop_edge_mask = apop_edge_mask & (~task_importance_mask)

        # 最低超边存活比例保护: 剪枝后存活数跌破 max_edges * min_alive_edge_ratio
        # 时回退本次剪枝 (不置 False), 防止结构过度收缩后无法恢复。
        alive_before = self.edge_mask.sum().item()
        dead_candidates = apop_edge_mask & self.edge_mask
        alive_after = alive_before - dead_candidates.sum().item()
        if alive_after >= self.max_edges * self.min_alive_edge_ratio:
            self.edge_mask[dead_candidates] = False

        # 3. 神经元凋亡判定 (拓扑感知):
        # 仅当提供 hyperedge_index 时执行 —— 没有存活关联边的神经元凋亡。
        # 保护: 即使部分神经元凋亡, 存活比例不低于 min_alive_ratio。
        if hyperedge_index is not None:
            self._prune_neurons_by_topology(hyperedge_index)

        # 4. 生发判定 (容量补充) — 联动复活关联超边 (传 hyperedge_index)
        self._grow_neurons(hyperedge_index)

        return self.neuron_mask, self.edge_mask

    def _prune_neurons_by_topology(self, hyperedge_index: torch.Tensor):
        """按超图拓扑凋亡神经元: 无任何存活关联边的神经元标记为死亡"""
        if hyperedge_index.shape[0] < 2 or hyperedge_index.shape[1] == 0:
            return
        src, edge_ids = hyperedge_index[0], hyperedge_index[1]
        alive_conn = self.edge_mask[edge_ids.clamp(0, self.max_edges - 1)]
        alive_src = src[alive_conn].unique().to(self.neuron_mask.device)

        new_mask = torch.zeros(self.max_neurons, dtype=torch.bool, device=self.neuron_mask.device)
        new_mask[alive_src[alive_src < self.max_neurons]] = True

        # 保护: 保留已存活神经元中仍被拓扑连接的
        # (若拓扑连接集为空, 不做任何凋亡, 避免全灭)
        if new_mask.any():
            self.neuron_mask.data.copy_(new_mask)

    def _grow_neurons(self, hyperedge_index: Optional[torch.Tensor] = None):
        """
        生发: 活跃比例低于阈值时复活死亡神经元

        联动复活关联超边 (修复生发空转): 死神经元的关联边全部为死
        (edge_mask 为 False), 若只复活 neuron_mask, 复活神经元无输入,
        下一超慢周期又被拓扑凋亡杀死 → 无限震荡。因此同时复活以该
        神经元为源节点的超边 (edge_mask 置 True); 双势阱概率 s_e 的重置
        由调用方 model.py 处理 (本模块不持有 s_e)。

        Args:
            hyperedge_index: (2, N_connections) 超图拓扑 [src_nodes, edge_ids]
                提供时联动复活复活神经元为源节点的超边, 否则只复活神经元。
        """
        active_ratio = self.neuron_mask.float().mean().item()
        if active_ratio >= self.min_alive_ratio:
            return

        dead_indices = torch.where(~self.neuron_mask)[0]
        if dead_indices.numel() == 0:
            return

        num_to_grow = min(
            dead_indices.numel(),
            max(1, int(self.max_neurons * self.growth_probability)),
        )
        revived = dead_indices[:num_to_grow]
        self.neuron_mask[revived] = True

        # ---- 联动复活关联超边 ----
        if (hyperedge_index is not None and hyperedge_index.shape[0] >= 2
                and hyperedge_index.shape[1] > 0):
            src = hyperedge_index[0].to(self.neuron_mask.device)
            edge_ids = hyperedge_index[1].to(self.edge_mask.device)
            # 复活神经元为源节点的超边 (去重, 剔除越界)
            candidate = edge_ids[torch.isin(src, revived)]
            candidate = candidate[(candidate >= 0) & (candidate < self.max_edges)].unique()
            if candidate.numel() > 0:
                # 复活边数上限 = 复活神经元数 × 每条边平均成员数
                # (平均成员数 = 总连接数 / 唯一边数, 网络整体均值)
                unique_edges = torch.unique(edge_ids.clamp(0, self.max_edges - 1)).numel()
                avg_members = hyperedge_index.shape[1] / max(unique_edges, 1)
                max_revive_edges = max(1, int(num_to_grow * avg_members))
                # 优先复活贡献度历史较高的边; 无历史 (NaN) 视为高贡献,
                # 训练早期历史不足时即等价于"全部候选边复活"。
                hist = torch.nanmean(self.contribution_history, dim=0)
                hist = torch.nan_to_num(hist, nan=float("inf"))
                order = torch.argsort(hist[candidate], descending=True)
                revive_edges = candidate[order[:max_revive_edges]]
                self.edge_mask[revive_edges] = True
