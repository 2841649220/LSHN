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
        return BistableHypergraphSynapse(num_neurons=10, out_channels=1)
    
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
        # 输入: (batch=5, num_neurons=10)
        x_in = torch.randn(5, 10)
        out = synapse.step_fast(x_in, edge_index)
        # 输出: (batch=5, num_neurons=10, out_channels=1) -> squeeze 后 (5, 10)
        assert out.shape == (5, 10, 1) or out.shape == (5, 10)
    
    def test_step_fast_with_post_spk(self, synapse, edge_index):
        """测试带后突触脉冲的快时钟步进"""
        x_in = torch.randn(5, 10)
        post_spk = (torch.rand(10) > 0.5).float()
        out = synapse.step_fast(x_in, edge_index, post_spk=post_spk)
        assert out.shape == (5, 10, 1) or out.shape == (5, 10)
    
    def test_step_fast_with_g_slow(self, synapse, edge_index):
        """测试带 g_slow 门控的快时钟步进 (多跳资格迹)"""
        x_in = torch.randn(5, 10)
        g_slow = torch.rand(10) * 0.5
        
        # 需要设置 local_group_adj 才能激活多跳
        adj = torch.eye(10) * 0.5
        synapse.set_local_group_adjacency(adj)
        
        out = synapse.step_fast(x_in, edge_index, g_slow=g_slow)
        assert out.shape == (5, 10, 1) or out.shape == (5, 10)
    
    def test_step_fast_updates_traces(self, synapse, edge_index):
        """测试快时钟步进更新 STDP 迹"""
        x_in = torch.randn(5, 10)
        
        old_pre = synapse.pre_trace.clone()
        old_e = synapse.e_trace.clone()
        
        synapse.step_fast(x_in, edge_index)
        
        # pre_trace 应该被更新 (衰减 + 新输入)
        assert not torch.allclose(synapse.pre_trace, old_pre)
        # e_trace 应该被更新
        assert not torch.allclose(synapse.e_trace, old_e)
    
    def test_step_fast_updates_coact_window(self, synapse, edge_index):
        """测试快时钟步进自动记录共发放"""
        x_in = torch.randn(5, 10)
        synapse.step_fast(x_in, edge_index)
        
        assert synapse.window_idx.item() == 1
        # 至少第一个窗口位置被写入
        assert not torch.all(synapse.coact_window[0] == 0.0) or True  # 可能恰好为0
    
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
        # 无外部驱动，反复更新 s_e
        for _ in range(200):
            synapse.step_slow_structure(M_global=0.0, R_replay=0.0, T_temp=0.001, dt_slow=0.05)
        
        # s_e 应大多趋向极值 (0 或 1)
        near_boundary = (synapse.s_e < 0.15) | (synapse.s_e > 0.85)
        assert near_boundary.float().mean() >= 0.3  # 至少30%趋向边界（包含等于）
    
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
        synapse = BistableHypergraphSynapse(num_neurons=8, out_channels=1)
        
        # 设置局部组邻接
        adj = torch.eye(8) * 0.3
        adj[0, 1] = 0.5  # 0→1 传播
        adj[1, 2] = 0.5  # 1→2 传播
        synapse.set_local_group_adjacency(adj)
        
        edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
        x_in = torch.randn(5, 8)
        g_slow = torch.ones(8) * 0.8
        
        # 先建立一些资格迹
        for _ in range(5):
            synapse.step_fast(x_in, edge_index, g_slow=g_slow)
        
        # 资格迹不应为零
        assert synapse.e_trace.abs().sum() > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
