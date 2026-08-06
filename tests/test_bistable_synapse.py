"""
测试: 双势阱脉冲超图突触 (BistableHypergraphSynapse)
覆盖: 白皮书 §3.3, §3.5.3, §3.5.4
"""
import torch
import pytest
from lshn.core.synapses.bistable_hypergraph import BistableHypergraphSynapse


class TestBistableHypergraphSynapse:
    
    @pytest.fixture
    def synapse(self):
        """创建标准测试用突触"""
        return BistableHypergraphSynapse(in_channels=10, out_channels=10)
    
    @pytest.fixture
    def edge_index(self):
        """创建标准超边索引"""
        return torch.tensor([
            [0, 1, 2, 3],  # 节点 index
            [0, 0, 1, 1]   # 超边 index
        ])
    
    def test_init_shapes(self, synapse):
        """测试初始化后参数和缓冲区形状"""
        assert synapse.w_hat.shape == (10,)
        assert synapse.s_e.shape == (10,)
        assert synapse.e_trace.shape == (10,)
        assert synapse.pre_trace.shape == (10,)
        assert synapse.post_trace.shape == (10,)
        assert synapse.coact_window.shape == (10, 10)
    
    def test_init_s_e_range(self, synapse):
        """测试 s_e 初始值在 [0, 1]"""
        assert torch.all(synapse.s_e >= 0.0)
        assert torch.all(synapse.s_e <= 1.0)
    
    def test_step_fast_basic(self, synapse, edge_index):
        """测试快时钟基本前向传播"""
        x_in = torch.randn(5, 10)
        out = synapse.step_fast(x_in, edge_index)
        assert out.shape == (5, 10)
    
    def test_step_fast_with_post_spk(self, synapse, edge_index):
        """测试带后突触脉冲的快时钟步进"""
        x_in = torch.randn(5, 10)
        post_spk = (torch.rand(10) > 0.5).float()
        out = synapse.step_fast(x_in, edge_index, post_spk=post_spk)
        assert out.shape == (5, 10)
    
    def test_step_fast_with_g_slow(self, synapse, edge_index):
        """测试带 g_slow 门控的快时钟步进 (多跳资格迹)"""
        x_in = torch.randn(5, 10)
        g_slow = torch.rand(10) * 0.5
        
        # 需要设置 local_group_adj 才能激活多跳
        adj = torch.eye(10) * 0.5
        synapse.set_local_group_adjacency(adj)
        
        out = synapse.step_fast(x_in, edge_index, g_slow=g_slow)
        assert out.shape == (5, 10)
    
    def test_step_fast_updates_traces(self, synapse, edge_index):
        """测试快时钟步进更新 STDP 迹"""
        x_in = torch.randn(5, 10)

        old_pre = synapse.pre_trace.clone()
        old_e = synapse.e_trace.clone()

        # 纯共发放迹 (e = λ·e + y_pre·y_post): 需要后突触活动
        post_spk = (torch.rand(10) > 0.3).float()
        # 保证有连接的超边 (0,1) 有后活动: 随机发放恰好使 0,1 静默时
        # coact 恒为 0, e_trace 不变 → 原测试 flaky
        post_spk[0] = 1.0
        post_spk[1] = 1.0
        synapse.step_fast(x_in, edge_index, post_spk=post_spk)

        # pre_trace 应该被更新 (衰减 + 新输入)
        assert not torch.allclose(synapse.pre_trace, old_pre)
        # e_trace 应该被更新 (共发放 → 迹增长)
        assert not torch.allclose(synapse.e_trace, old_e)
    
    def test_step_fast_updates_coact_window(self, synapse, edge_index):
        """测试快时钟步进自动记录共发放"""
        # 使用确定性的前后活动: 共发放 = pre_spk_per_edge * post_trace
        # 有确定输入时必然非零, 窗口被真实写入 (原测试随机输入 + 无后脉冲时
        # coact 恒为 0, 用 "or True" 恒真断言掩盖)
        x_in = torch.ones(5, 10)
        post_spk = torch.ones(10)
        synapse.step_fast(x_in, edge_index, post_spk=post_spk)

        assert synapse.window_idx.item() == 1
        # 共发放非零: 第一条超边 (节点 0,1) 有前活动 × 后迹 > 0
        assert synapse.coact_window[0].abs().sum().item() > 0.0
    
    def test_step_slow_structure(self, synapse):
        """测试慢时钟双势阱结构更新"""
        # 先填充一些共发放数据
        for i in range(10):
            synapse.coact_window[i] = torch.rand(10) * 0.1
        
        old_s_e = synapse.s_e.clone()
        
        synapse.step_slow_structure(M_global=1.0, R_replay=0.5, T_temp=0.1)
        
        # s_e 应被更新
        assert not torch.allclose(old_s_e, synapse.s_e)
        # s_e 仍在 [0, 1]
        assert torch.all(synapse.s_e >= 0.0)
        assert torch.all(synapse.s_e <= 1.0)
    
    def test_step_slow_structure_with_dt(self, synapse):
        """测试慢时钟可配置时间步长"""
        old_s_e = synapse.s_e.clone()
        synapse.step_slow_structure(M_global=0.5, R_replay=0.3, T_temp=0.05, dt_slow=0.01)
        # 更小的 dt_slow 应导致更小的变化
        assert torch.all(synapse.s_e >= 0.0)
        assert torch.all(synapse.s_e <= 1.0)
    
    def test_bistable_convergence(self, synapse):
        """测试双势阱最终使 s_e 趋向0或1 (两个稳定不动点)"""
        # 200 步时随机游走噪声 (σ=0.01/步) 尚未被势阱漂移压过, 各随机种子
        # 下 near_boundary ∈ [0.10, 0.80], 原断言 ≥0.3 为 flaky;
        # 400 步后漂移累计 ~1.5 且 s_e 被 clamp 至 1, 收敛与随机种子无关
        for _ in range(400):
            synapse.step_slow_structure(M_global=0.0, R_replay=0.0, T_temp=0.001, dt_slow=0.05)

        near_boundary = (synapse.s_e < 0.15) | (synapse.s_e > 0.85)
        assert near_boundary.float().mean() >= 0.3
    
    def test_get_effective_weights(self, synapse):
        """测试有效权重计算"""
        w_eff = synapse.get_effective_weights()
        assert w_eff.shape == (10,)
        # w_e = w_max * s_e * w_hat
        expected = synapse.w_max * synapse.s_e * synapse.w_hat
        assert torch.allclose(w_eff, expected)
    
    def test_get_alive_mask(self, synapse):
        """测试存活超边掩码"""
        # 默认 s_e=0.5 > threshold=0.05，全部存活
        mask = synapse.get_alive_mask()
        assert mask.shape == (10,)
        assert torch.all(mask)
        
        # 手动设置一些 s_e 很低
        synapse.s_e.data[0:3] = 0.01
        mask = synapse.get_alive_mask(threshold=0.05)
        assert not mask[0] and not mask[1] and not mask[2]
        assert mask[3]
    
    def test_record_coact_compat(self, synapse):
        """测试手动 record_coact 兼容接口"""
        coact = torch.rand(10)
        synapse.record_coact(coact)
        assert synapse.window_idx.item() == 1
    
    def test_multihop_eligibility_trace(self):
        """测试多跳资格迹传播 (白皮书 §3.5.3)"""
        synapse = BistableHypergraphSynapse(in_channels=8, out_channels=8)

        # 设置局部组邻接
        adj = torch.eye(8) * 0.3
        adj[0, 1] = 0.5  # 0→1 传播
        adj[1, 2] = 0.5  # 1→2 传播
        synapse.set_local_group_adjacency(adj)

        edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
        x_in = torch.randn(5, 8)
        g_slow = torch.ones(8) * 0.8
        post_spk = (torch.rand(8) > 0.3).float()
        # 保证有连接的超边 (0,1) 有后突触活动: 若随机发放恰好使 0,1 都静默,
        # coact = pre_spk_per_edge * post_trace 恒为 0, 资格迹为零 → 原测试 flaky
        post_spk[0] = 1.0
        post_spk[1] = 1.0

        # 先建立一些资格迹 (纯共发放迹需要前后突触活动)
        for _ in range(5):
            synapse.step_fast(x_in, edge_index, post_spk=post_spk, g_slow=g_slow)

        # 资格迹不应为零 (共发放 + 多跳传播)
        assert synapse.e_trace.abs().sum() > 0.0

    def test_multihop_trace_decay_factor(self):
        """多跳项系数 = σ(ḡ)·(1−trace_decay)·组内均值 (精确断言)"""
        synapse = BistableHypergraphSynapse(in_channels=4, out_channels=4, trace_decay=0.8)
        # 块对角邻接: 组 {0,1} 与 {2,3}, 组内 1/2 (归一化)
        adj = torch.zeros(4, 4)
        adj[0:2, 0:2] = 0.5
        adj[2:4, 2:4] = 0.5
        synapse.set_local_group_adjacency(adj)

        edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        # 确定性权重: effective_w = w_max·s_e·w_hat = 0.5
        synapse.w_hat.data.fill_(0.5)
        synapse.s_e.data.fill_(1.0)
        synapse.e_trace.zero_()
        synapse.e_trace[0] = 1.0  # 仅边 0 有迹

        # 输入全零: coact = pre_edge × post_trace = 0, 更新纯由多跳项驱动
        # g_slow.mean() = 2 → σ = sigmoid(2) ≈ 0.8808
        g_slow = torch.ones(4) * 2.0
        synapse.step_fast(torch.zeros(1, 4), edge_index, g_slow=g_slow)

        sigma = torch.sigmoid(torch.tensor(2.0)).item()
        multihop = sigma * (1.0 - 0.8) * (0.5 * 1.0 / 2.0)  # 组内均值 = w·e / 2
        assert synapse.e_trace[0].item() == pytest.approx(0.8 * 1.0 + multihop, abs=1e-6)
        assert synapse.e_trace[1].item() == pytest.approx(multihop, abs=1e-6)
        assert synapse.e_trace[2].item() == pytest.approx(0.0)
        assert synapse.e_trace[3].item() == pytest.approx(0.0)

    def test_aggregate_spikes_to_edges_mean_and_sum(self):
        """聚合函数: mean = sum/K (确定性拓扑手算)"""
        synapse = BistableHypergraphSynapse(in_channels=6, out_channels=4)
        # 边 0 ← 节点 0,1,2 (K=3); 边 1 ← 节点 0,1 (K=2)
        edge_index = torch.tensor([[0, 1, 2, 0, 1], [0, 0, 0, 1, 1]])
        spk = torch.tensor([[1.0, 2.0, 4.0, 0.0, 0.0, 0.0]])

        mean_out = synapse.aggregate_spikes_to_edges(spk, edge_index, reduction="mean")
        sum_out = synapse.aggregate_spikes_to_edges(spk, edge_index, reduction="sum")

        # 边 0: sum = 1+2+4 = 7, mean = 7/3; 边 1: sum = 3, mean = 3/2
        assert mean_out[0].item() == pytest.approx(7.0 / 3.0)
        assert mean_out[1].item() == pytest.approx(3.0 / 2.0)
        assert sum_out[0].item() == pytest.approx(7.0)
        assert sum_out[1].item() == pytest.approx(3.0)
        assert torch.all(mean_out[2:] == 0.0)
        assert torch.all(sum_out[2:] == 0.0)

    def test_aggregate_empty_index_returns_zeros(self):
        """聚合函数: 空超图索引返回全零"""
        synapse = BistableHypergraphSynapse(in_channels=6, out_channels=4)
        empty = torch.zeros(2, 0, dtype=torch.long)
        spk = torch.ones(2, 6)

        out = synapse.aggregate_spikes_to_edges(spk, empty)
        out_batch = synapse.aggregate_spikes_to_edges_batch(spk, empty)
        assert out.shape == (4,)
        assert torch.all(out == 0.0)
        assert out_batch.shape == (2, 4)
        assert torch.all(out_batch == 0.0)

    def test_aggregate_ignores_out_of_range(self):
        """聚合函数: 越界 edge_id 被忽略, 越界 src 钳位到节点范围"""
        synapse = BistableHypergraphSynapse(in_channels=6, out_channels=4)
        # edge_id=99 越界 (max_edges=4) → 忽略; src=100 越界 → 钳位到节点 5
        edge_index = torch.tensor([[0, 5, 100], [0, 99, 1]])
        spk = torch.ones(1, 6)

        out = synapse.aggregate_spikes_to_edges(spk, edge_index)
        assert out[0].item() == pytest.approx(1.0)  # src 0 → 边 0
        assert out[1].item() == pytest.approx(1.0)  # src 100 → 节点 5 → 边 1
        assert out[2].item() == pytest.approx(0.0)
        assert out[3].item() == pytest.approx(0.0)

    def test_delayed_pre_output_path(self):
        """延迟通路: 输出 = delayed_pre × 有效权重"""
        synapse = BistableHypergraphSynapse(in_channels=8, out_channels=4)
        synapse.w_hat.data.fill_(0.5)
        synapse.s_e.data.fill_(1.0)  # effective_w = 1.0 × 1.0 × 0.5

        edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
        delayed_pre = torch.tensor([[2.0, 3.0, 0.0, 0.0]])

        out = synapse.step_fast(torch.zeros(1, 8), edge_index, delayed_pre=delayed_pre)

        expected = delayed_pre * (synapse.w_max * synapse.s_e * synapse.w_hat)
        assert torch.allclose(out, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
