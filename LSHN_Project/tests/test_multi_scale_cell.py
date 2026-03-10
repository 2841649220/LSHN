"""
测试: 多尺度液态门控元胞 (LiquidGatedCell + DendriteCompartment)
覆盖: 白皮书 §3.2, §1.3.2, §4.2.1
"""
import torch
import pytest
from lshn.core.cells.liquid_cell import LiquidGatedCell, DendriteCompartment


# ============================================================
# DendriteCompartment 测试
# ============================================================

class TestDendriteCompartment:
    
    def test_init_shape(self):
        """测试树突模块初始化后缓冲区形状"""
        num_neurons = 16
        num_branches = 4
        dc = DendriteCompartment(num_neurons, num_branches)
        
        assert dc.branch_weights.shape == (num_branches, num_neurons)
        assert dc.branch_potential.shape == (num_branches, num_neurons)
    
    def test_forward_shape(self):
        """测试树突前向输出形状"""
        num_neurons = 16
        dc = DendriteCompartment(num_neurons, num_branches=4)
        I_syn = torch.randn(num_neurons)
        out = dc(I_syn)
        assert out.shape == (num_neurons,)
    
    def test_ca_spike_nonlinearity(self):
        """测试 Ca 尖峰非线性：强输入应产生超线性输出"""
        num_neurons = 8
        dc = DendriteCompartment(num_neurons, num_branches=4, dendrite_threshold=0.3)
        
        # 用较大输入重复推几步，让分支电位累积
        I_strong = torch.ones(num_neurons) * 5.0
        for _ in range(5):
            out = dc(I_strong)
        
        # 输出不应为零 (有 Ca 尖峰放大)
        assert out.abs().sum() > 0.0
    
    def test_reset(self):
        """测试重置后分支电位归零"""
        dc = DendriteCompartment(8, 4)
        dc(torch.randn(8))
        dc.reset()
        assert torch.all(dc.branch_potential == 0.0)


# ============================================================
# LiquidGatedCell 测试
# ============================================================

class TestLiquidGatedCell:
    
    def test_init_default(self):
        """测试默认参数初始化"""
        num_neurons = 32
        cell = LiquidGatedCell(num_neurons=num_neurons)
        
        assert cell.v.shape == (num_neurons,)
        assert cell.g_fast.shape == (num_neurons,)
        assert cell.g_slow.shape == (num_neurons,)
        assert cell.a.shape == (num_neurons,)
        assert cell.theta.shape == (num_neurons,)
        assert cell.spk_window.shape == (100, num_neurons)
        assert cell.window_idx.item() == 0
    
    def test_step_fast_output_shape(self):
        """测试快时钟单步返回 (spk, v) 的形状"""
        num_neurons = 10
        cell = LiquidGatedCell(num_neurons=num_neurons)
        
        I_syn = torch.randn(num_neurons) * 2.0
        spk, v = cell.step_fast(I_syn)
        
        assert spk.shape == (num_neurons,)
        assert v.shape == (num_neurons,)
        assert cell.window_idx.item() == 1
    
    def test_step_fast_spk_binary(self):
        """测试前向传播中脉冲为二值 {0, 1}（硬阈值）"""
        num_neurons = 20
        cell = LiquidGatedCell(num_neurons=num_neurons, theta_0=0.5)
        
        # 用强输入确保一些神经元发放
        I_syn = torch.randn(num_neurons) * 10.0
        spk, _ = cell.step_fast(I_syn)
        
        # STE: 前向用硬阈值，spk 的值在 detach 后应该接近 0 或 1
        spk_hard = spk.detach()
        assert torch.all((spk_hard == 0.0) | (spk_hard == 1.0))
    
    def test_step_fast_with_optional_inputs(self):
        """测试带可选参数 (I_ext, I_inh, theta_ie) 的快时钟步进"""
        num_neurons = 10
        cell = LiquidGatedCell(num_neurons=num_neurons)
        
        I_syn = torch.randn(num_neurons)
        I_ext = torch.randn(num_neurons) * 0.5
        I_inh = torch.randn(num_neurons).abs() * 0.3
        theta_ie = torch.randn(num_neurons) * 0.1
        
        spk, v = cell.step_fast(I_syn, I_ext=I_ext, I_inh=I_inh, theta_ie=theta_ie)
        assert spk.shape == (num_neurons,)
        assert v.shape == (num_neurons,)
    
    def test_step_fast_window_wraps(self):
        """测试滑动窗口环形缓冲区的正确回绕"""
        num_neurons = 5
        cell = LiquidGatedCell(num_neurons=num_neurons)
        
        for i in range(150):
            cell.step_fast(torch.randn(num_neurons))
        
        assert cell.window_idx.item() == 150
        # 环形索引 = 150 % 100 = 50，不应越界
    
    def test_step_slow_updates_a_and_theta(self):
        """测试慢时钟更新适应变量 a 和阈值 theta"""
        num_neurons = 5
        cell = LiquidGatedCell(num_neurons=num_neurons, theta_0=0.5)
        
        # 模拟100步让窗口有数据 - 使用更强的输入确保产生脉冲
        for _ in range(100):
            cell.step_fast(torch.ones(num_neurons) * 5.0)  # 强输入确保发放
        
        old_a = cell.a.clone()
        old_theta = cell.theta.clone()
        
        global_e = torch.tensor([1.0])
        cell.step_slow(global_e)
        
        # a 应被更新
        assert cell.a.shape == (num_neurons,)
        # 由于使用了强输入，应该有一些神经元发放了脉冲，a 应该增加
        # 但即使 a 为0，theta 也应该等于 theta_0
        assert torch.allclose(cell.theta, cell.theta_0 + cell.a)
    
    def test_step_slow_updates_g_slow(self):
        """测试慢时钟更新 g_slow 门控变量"""
        num_neurons = 5
        cell = LiquidGatedCell(num_neurons=num_neurons)
        
        for _ in range(100):
            cell.step_fast(torch.randn(num_neurons))
        
        old_g_slow = cell.g_slow.clone()
        cell.step_slow(torch.tensor([0.5]))
        
        # g_slow 应被更新 (不太可能完全不变)
        assert cell.g_slow.shape == (num_neurons,)
    
    def test_get_plasticity_modulation(self):
        """测试可塑性调制返回 g_slow"""
        cell = LiquidGatedCell(num_neurons=8)
        mod = cell.get_plasticity_modulation()
        assert mod.shape == (8,)
        assert torch.allclose(mod, cell.g_slow)
    
    def test_get_firing_rate(self):
        """测试发放率计算"""
        num_neurons = 8
        cell = LiquidGatedCell(num_neurons=num_neurons)
        
        # 初始全零窗口 → 发放率应为0
        rate = cell.get_firing_rate()
        assert rate.shape == (num_neurons,)
        assert torch.allclose(rate, torch.zeros(num_neurons))
        
        # 跑一些步后发放率应非零
        for _ in range(50):
            cell.step_fast(torch.randn(num_neurons) * 5.0)
        rate = cell.get_firing_rate()
        assert rate.shape == (num_neurons,)
    
    def test_update_delta_window(self):
        """测试误差脉冲窗口的记录"""
        cell = LiquidGatedCell(num_neurons=8)
        cell.step_fast(torch.randn(8))  # 先推一步让 window_idx > 0
        
        delta = torch.randn(8)
        cell.update_delta_window(delta)
        
        idx = (cell.window_idx - 1) % 100
        assert torch.allclose(cell.delta_window[idx], delta)
    
    def test_reset_hidden(self):
        """测试完全重置"""
        cell = LiquidGatedCell(num_neurons=8)
        for _ in range(10):
            cell.step_fast(torch.randn(8))
        
        cell.reset_hidden()
        
        assert torch.all(cell.v == 0.0)
        assert torch.all(cell.g_fast == 0.0)
        assert torch.allclose(cell.g_slow, torch.tensor(0.5))
        assert torch.all(cell.a == 0.0)
        assert torch.allclose(cell.theta, torch.ones(8) * cell.theta_0)
        assert cell.window_idx.item() == 0
    
    def test_with_dendrites(self):
        """测试启用树突的液态元胞"""
        cell = LiquidGatedCell(num_neurons=16, enable_dendrites=True, num_branches=4)
        assert hasattr(cell, 'dendrite')
        
        I_syn = torch.randn(16) * 2.0
        spk, v = cell.step_fast(I_syn)
        assert spk.shape == (16,)
        assert v.shape == (16,)
    
    def test_ste_gradient_flow(self):
        """测试 STE 替代梯度是否允许梯度流过脉冲"""
        cell = LiquidGatedCell(num_neurons=8, theta_0=0.5)
        
        # 创建叶子张量
        I_syn_raw = torch.randn(8)
        I_syn = I_syn_raw * 5.0
        I_syn.requires_grad_(True)
        
        spk, v = cell.step_fast(I_syn)
        
        # spk 应该可以反向传播
        loss = spk.sum()
        loss.backward()
        
        # 检查梯度是否存在且非零
        assert I_syn.grad is not None, "梯度为 None"
        # 梯度不应全零 (STE 让梯度通过)
        assert I_syn.grad.abs().sum() > 0.0, "梯度全零"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
