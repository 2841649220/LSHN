"""
测试: 凋亡生发与自剪枝模块 (PruneGrowthModule)
覆盖: 白皮书 §4.1 结构演化 (超慢时钟)

行为契约 (修复后):
- 历史不足 min_history_samples (默认 10) 槽只记录, 不剪枝
- 连续低贡献 (均值 ≤ prune_threshold) 的超边被剪
- 剪枝后存活数跌破 max_edges × min_alive_edge_ratio 时回退本次剪枝
- task_importance_mask 内的边受保护不剪
- 神经元凋亡依赖超图拓扑: 无任何存活关联边的神经元死亡
- 生发联动复活关联超边 (防止复活神经元无输入震荡)
- 空 VFE_masked_dict 不做任何判定
"""
import torch
import pytest
from lshn.core.evolution.prune_growth import PruneGrowthModule


class TestPruneGrowthModule:

    def test_history_insufficient_no_prune(self):
        """① 历史不足 10 槽不剪枝: 低贡献输入下 edge_mask 保持不变"""
        pg = PruneGrowthModule(max_neurons=16, max_edges=8)
        VFE_full = 1.0
        VFE_masked = {e: 0.1 for e in range(8)}  # 贡献度 = 0.1 - 1.0 = -0.9 ≤ 0

        pg.step_ultra_slow_evolution(VFE_full, VFE_masked)
        assert torch.all(pg.edge_mask)
        assert pg.history_idx.item() == 1

        # 第 2 次调用仍不足 10 槽 → 只记录历史, 不剪
        pg.step_ultra_slow_evolution(VFE_full, VFE_masked)
        assert torch.all(pg.edge_mask)
        assert pg.history_idx.item() == 2

    def test_prune_low_contribution_after_full_history(self):
        """② 历史满 10 槽后低贡献边被剪, 高贡献边保留"""
        pg = PruneGrowthModule(max_neurons=16, max_edges=8)
        VFE_full = 1.0
        # 边 0-3 低贡献 (贡献度 -0.9), 边 4-7 高贡献 (贡献度 +1.0)
        VFE_masked = {e: (0.1 if e < 4 else 2.0) for e in range(8)}

        for _ in range(10):
            pg.step_ultra_slow_evolution(VFE_full, VFE_masked)

        assert pg.history_idx.item() == 10
        # 低贡献边被剪, 高贡献边保留
        assert torch.all(~pg.edge_mask[:4])
        assert torch.all(pg.edge_mask[4:])
        # 剪后存活 4 条 ≥ max_edges × min_alive_edge_ratio (8 × 0.3 = 2.4) ✓

    def test_min_alive_edge_ratio_rollback(self):
        """③ 全部低贡献时剪枝会跌破最低存活比例 → 回退, mask 不变"""
        pg = PruneGrowthModule(max_neurons=16, max_edges=8, min_alive_edge_ratio=0.5)
        VFE_masked = {e: 0.0 for e in range(8)}  # 全部低贡献

        for _ in range(10):
            pg.step_ultra_slow_evolution(1.0, VFE_masked)

        # 剪 8 条 → 存活 0 < 8 × 0.5 = 4 → 回退本次剪枝
        assert torch.all(pg.edge_mask)

    def test_task_importance_mask_protection(self):
        """④ task_importance_mask 受保护的边不剪"""
        pg = PruneGrowthModule(max_neurons=16, max_edges=8, min_alive_edge_ratio=0.1)
        importance = torch.zeros(8, dtype=torch.bool)
        importance[2] = True  # 边 2 受保护

        VFE_masked = {e: 0.0 for e in range(8)}  # 全部低贡献
        for _ in range(10):
            pg.step_ultra_slow_evolution(1.0, VFE_masked,
                                         task_importance_mask=importance)

        # 受保护边保留, 其余 7 条剪掉 (存活 1 ≥ 8 × 0.1 = 0.8)
        assert pg.edge_mask[2]
        assert torch.sum(~pg.edge_mask) == 7

    def test_topology_aware_neuron_apoptosis(self):
        """⑤ 拓扑感知凋亡: 无存活关联边的神经元死亡"""
        pg = PruneGrowthModule(max_neurons=8, max_edges=4)
        # 边 0,1 连神经元 0-3; 边 2,3 连神经元 4-7
        hyperedge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 1, 1, 2, 2, 3, 3],
        ])
        VFE_masked = {0: 2.0, 1: 2.0, 2: 0.0, 3: 0.0}  # 边 2,3 低贡献

        for _ in range(10):
            pg.step_ultra_slow_evolution(1.0, VFE_masked,
                                         hyperedge_index=hyperedge_index)

        # 边 2,3 被剪 (存活 2 ≥ 4 × 0.3 = 1.2) → 神经元 4-7 无存活关联边 → 凋亡
        assert torch.all(pg.neuron_mask[:4])
        assert torch.all(~pg.neuron_mask[4:])

    def test_growth_revives_linked_edges(self):
        """⑥ 生发联动复活: 神经元复活时其关联超边同步复活"""
        pg = PruneGrowthModule(
            max_neurons=8, max_edges=4,
            min_alive_ratio=0.6, growth_probability=0.5,
            min_alive_edge_ratio=0.25,
        )
        # 边 0 ← 神经元 0; 边 1 ← 神经元 1; 边 2 ← 神经元 2; 边 3 ← 神经元 3
        hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        VFE_masked = {0: 2.0, 1: 0.0, 2: 0.0, 3: 0.0}  # 仅边 0 高贡献

        for _ in range(10):
            pg.step_ultra_slow_evolution(1.0, VFE_masked,
                                         hyperedge_index=hyperedge_index)

        # 边 1,2,3 被剪 (存活 1 ≥ 4 × 0.25 = 1.0) → 神经元 1-3 无存活边凋亡
        # 存活比例 1/8 = 0.125 < 0.6 → 生发复活神经元 1,2,3 (及 4) 并联动复活边 1,2,3
        assert pg.neuron_mask[0]
        assert pg.neuron_mask[1] and pg.neuron_mask[2] and pg.neuron_mask[3]
        assert torch.all(pg.edge_mask)  # 边 1,2,3 被联动复活

    def test_empty_vfe_dict_no_prune(self):
        """⑦ 空 VFE_masked_dict: 不判定不记录, 掩码全保留"""
        pg = PruneGrowthModule(max_neurons=8, max_edges=4)
        pg.step_ultra_slow_evolution(1.0, {})
        assert torch.all(pg.edge_mask)
        assert torch.all(pg.neuron_mask)
        assert pg.history_idx.item() == 0  # 未记录历史

    def test_neuron_growth_boundary(self):
        """生发仅当活跃比例低于 min_alive_ratio 时发生 (补充断言)"""
        pg = PruneGrowthModule(max_neurons=8, max_edges=4, min_alive_ratio=0.3)
        # 神经元 4-7 全部死亡但活跃比例 0.5 ≥ 0.3 → 不触发生发
        pg.neuron_mask[4:] = False
        pg._grow_neurons()
        assert torch.sum(pg.neuron_mask) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
