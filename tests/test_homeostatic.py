"""
测试: 稳态可塑性模块 (SynapticScaling, IntrinsicExcitabilityPlasticity, HomeostaticController)
覆盖: 白皮书 §2.3, §4.1
"""
import torch
import pytest
from lshn.core.plasticity.homeostatic import (
    SynapticScaling,
    IntrinsicExcitabilityPlasticity,
    HomeostaticController,
)


# ============================================================
# SynapticScaling 测试
# ============================================================

class TestSynapticScaling:
    
    @pytest.fixture
    def scaler(self):
        return SynapticScaling(num_neurons=16, target_rate=0.05)
    
    def test_init(self, scaler):
        """测试初始状态"""
        assert scaler.firing_rate_ema.shape == (16,)
        # 初始 EMA = target_rate
        assert torch.allclose(
            scaler.firing_rate_ema,
            torch.ones(16) * 0.05
        )
    
    def test_update_rates(self, scaler):
        """测试发放率 EMA 更新"""
        spk = torch.zeros(16)
        spk[0] = 1.0
        spk[5] = 1.0
        
        old_ema = scaler.firing_rate_ema.clone()
        scaler.update_rates(spk)
        
        # 发放的神经元 EMA 应增大
        assert scaler.firing_rate_ema[0] > old_ema[0]
        assert scaler.firing_rate_ema[5] > old_ema[5]
        # 未发放的神经元 EMA 应因衰减而降低
        assert scaler.firing_rate_ema[1] < old_ema[1]
    
    def test_scaling_factors_at_target(self):
        """测试在目标发放率时缩放因子 ≈ 1"""
        scaler = SynapticScaling(num_neurons=8, target_rate=0.05, scaling_strength=0.1)
        # EMA 初始 = target_rate，所以缩放因子应为1
        scale = scaler.compute_scaling_factors()
        assert torch.allclose(scale, torch.ones(8), atol=1e-5)
    
    def test_scaling_up_for_silent_neurons(self):
        """测试沉默神经元的突触上调"""
        scaler = SynapticScaling(num_neurons=8, target_rate=0.05, scaling_strength=0.1)
        
        # 模拟神经元完全不发放，使 EMA 降到接近0
        for _ in range(500):
            scaler.update_rates(torch.zeros(8))
        
        scale = scaler.compute_scaling_factors()
        # 所有缩放因子应 > 1 (上调)
        assert torch.all(scale > 1.0)
    
    def test_scaling_down_for_overactive_neurons(self):
        """测试过度活跃神经元的突触下调"""
        scaler = SynapticScaling(num_neurons=8, target_rate=0.05, scaling_strength=0.1)
        
        # 模拟神经元总是发放
        for _ in range(500):
            scaler.update_rates(torch.ones(8))
        
        scale = scaler.compute_scaling_factors()
        # 所有缩放因子应 < 1 (下调)
        assert torch.all(scale < 1.0)
    
    def test_scaling_clamp(self):
        """测试缩放因子被限制在 [0.5, 2.0]"""
        scaler = SynapticScaling(num_neurons=8, target_rate=0.05, scaling_strength=0.5)
        
        # 极端情况
        scaler.firing_rate_ema.fill_(1e-5)
        scale = scaler.compute_scaling_factors()
        assert torch.all(scale <= 2.0)
        
        scaler.firing_rate_ema.fill_(0.99)
        scale = scaler.compute_scaling_factors()
        assert torch.all(scale >= 0.5)
    
    def test_apply_scaling_global(self, scaler):
        """测试全局平均缩放应用"""
        w_hat = torch.randn(16)
        scaled = scaler.apply_scaling(w_hat, neuron_to_edge_map=None)
        assert scaled.shape == (16,)
    
    def test_apply_scaling_per_neuron(self, scaler):
        """测试按神经元缩放"""
        w_hat = torch.randn(16)
        # 每条边映射到对应的后突触神经元
        neuron_map = torch.arange(16)
        scaled = scaler.apply_scaling(w_hat, neuron_to_edge_map=neuron_map)
        assert scaled.shape == (16,)


# ============================================================
# IntrinsicExcitabilityPlasticity 测试
# ============================================================

class TestIntrinsicExcitabilityPlasticity:
    
    @pytest.fixture
    def ie(self):
        return IntrinsicExcitabilityPlasticity(
            num_neurons=16, target_rate=0.05, ie_learning_rate=0.001
        )
    
    def test_init(self, ie):
        """测试初始阈值调整为零"""
        assert ie.theta_ie.shape == (16,)
        assert torch.allclose(ie.theta_ie, torch.zeros(16))
    
    def test_step_slow_output_shape(self, ie):
        """测试慢时钟输出形状"""
        rate = torch.ones(16) * 0.05
        theta = ie.step_slow(rate)
        assert theta.shape == (16,)
    
    def test_high_rate_increases_threshold(self, ie):
        """测试高发放率 → 阈值增大 → 抑制兴奋性"""
        high_rate = torch.ones(16) * 0.5  # 远高于 target=0.05
        
        for _ in range(100):
            ie.step_slow(high_rate)
        
        # theta_ie 应为正 (增大阈值)
        assert torch.all(ie.theta_ie > 0.0)
    
    def test_low_rate_decreases_threshold(self, ie):
        """测试低发放率 → 阈值减小 → 提升兴奋性"""
        low_rate = torch.ones(16) * 0.001  # 远低于 target=0.05
        
        for _ in range(100):
            ie.step_slow(low_rate)
        
        # theta_ie 应为负 (减小阈值)
        assert torch.all(ie.theta_ie < 0.0)
    
    def test_at_target_no_change(self, ie):
        """测试在目标发放率时阈值不变"""
        target_rate = torch.ones(16) * 0.05
        ie.step_slow(target_rate)
        
        # Δθ = η * (0.05 - 0.05) = 0
        assert torch.allclose(ie.theta_ie, torch.zeros(16), atol=1e-7)
    
    def test_theta_ie_clamped(self, ie):
        """测试阈值调整被限制在 [-0.5, 1.0]"""
        # 极端高发放率
        for _ in range(10000):
            ie.step_slow(torch.ones(16))
        assert torch.all(ie.theta_ie <= 1.0)
        
        # 重置
        ie.theta_ie.zero_()
        
        # 极端低发放率
        for _ in range(10000):
            ie.step_slow(torch.zeros(16))
        assert torch.all(ie.theta_ie >= -0.5)


# ============================================================
# HomeostaticController 测试
# ============================================================

class TestHomeostaticController:
    
    @pytest.fixture
    def controller(self):
        return HomeostaticController(num_neurons=32, target_rate=0.05)
    
    def test_init(self, controller):
        """测试控制器包含两个子模块"""
        assert hasattr(controller, 'synaptic_scaling')
        assert hasattr(controller, 'ie_plasticity')
    
    def test_step_fast(self, controller):
        """测试快时钟更新发放率"""
        spk = (torch.rand(32) > 0.9).float()
        controller.step_fast(spk)
        # 不应报错
    
    def test_step_slow_returns_dict(self, controller):
        """测试慢时钟返回缩放因子和阈值调整"""
        # 先快时钟更新一些数据
        for _ in range(50):
            controller.step_fast((torch.rand(32) > 0.8).float())
        
        result = controller.step_slow()
        
        assert "scaling_factors" in result
        assert "theta_ie" in result
        assert result["scaling_factors"].shape == (32,)
        assert result["theta_ie"].shape == (32,)
    
    def test_apply_to_weights(self, controller):
        """测试权重缩放应用"""
        w = torch.randn(32)
        scaled = controller.apply_to_weights(w)
        assert scaled.shape == (32,)
    
    def test_full_cycle(self, controller):
        """测试完整的快+慢周期"""
        # 100个快时间步
        for _ in range(100):
            spk = (torch.rand(32) > 0.7).float()
            controller.step_fast(spk)
        
        # 1个慢时间步
        result = controller.step_slow()
        
        # 过度活跃 (>0.7的发放概率) → 缩放因子应 < 1
        assert result["scaling_factors"].mean() < 1.0
        # 过度活跃 → 阈值应增大
        assert result["theta_ie"].mean() > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
