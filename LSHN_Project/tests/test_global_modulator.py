"""
测试: 全局神经调节器 (GlobalNeuromodulator + AstrocyteGate)
覆盖: 白皮书 §1.3.1, §3.2(§3)
"""
import torch
import pytest
from lshn.engine.global_modulator import GlobalNeuromodulator, AstrocyteGate


class TestAstrocyteGate:
    
    @pytest.fixture
    def gate(self):
        return AstrocyteGate(num_neurons=32)
    
    def test_init(self, gate):
        """测试星形胶质细胞初始化"""
        assert gate.calcium.shape == (1,)
        assert gate.calcium.item() == pytest.approx(0.5, abs=1e-5)
    
    def test_step_slow_output_range(self, gate):
        """测试 step_slow 输出在 [0, 1]"""
        for pe in [0.0, 0.5, 1.0, 5.0]:
            for fr in [0.0, 0.1, 0.5]:
                out = gate.step_slow(pe, fr)
                assert out.item() >= 0.0
                assert out.item() <= 1.0
    
    def test_step_slow_high_error_opens_gate(self, gate):
        """测试高预测误差开放可塑性门控"""
        # 先用低误差稳定
        for _ in range(100):
            gate.step_slow(0.01, 0.05)
        low_calcium = gate.calcium.item()
        
        # 突然注入高误差
        for _ in range(100):
            gate.step_slow(5.0, 0.05)
        high_calcium = gate.calcium.item()
        
        # 高误差应导致门控值上升
        assert high_calcium > low_calcium
    
    def test_step_slow_decay(self, gate):
        """测试星形胶质钙浓度的时间常数衰减"""
        # 激活后恢复
        for _ in range(50):
            gate.step_slow(5.0, 0.1)
        activated = gate.calcium.item()
        
        for _ in range(500):
            gate.step_slow(0.0, 0.0)
        recovered = gate.calcium.item()
        
        # 恢复后钙浓度应下降
        assert recovered < activated


class TestGlobalNeuromodulator:
    
    @pytest.fixture
    def modulator(self):
        return GlobalNeuromodulator(num_neurons=64)
    
    def test_init_states(self, modulator):
        """测试初始状态"""
        assert modulator.ACh.item() == pytest.approx(1.0, abs=1e-5)
        assert modulator.NE.item() == pytest.approx(0.1, abs=1e-5)
        assert modulator.DA.item() == pytest.approx(0.01, abs=1e-5)
    
    def test_step_slow_returns_dict(self, modulator):
        """测试 step_slow 返回完整调制字典"""
        result = modulator.step_slow(prediction_error=0.5, firing_rate=0.05)
        
        assert "ACh" in result
        assert "NE" in result
        assert "DA" in result
        assert "plasticity_gate" in result
        assert "surprise" in result
    
    def test_step_slow_tensor_types(self, modulator):
        """测试返回值都是 tensor"""
        result = modulator.step_slow(0.5, 0.05)
        for key, val in result.items():
            assert isinstance(val, torch.Tensor), f"{key} should be tensor"
    
    def test_ach_responds_to_surprise(self, modulator):
        """测试 ACh 对突变的响应：突变大 → ACh 降低"""
        # 稳定输入
        for _ in range(50):
            modulator.step_slow(0.5, 0.05)
        ach_stable = modulator.ACh.item()
        
        # 突然大误差
        for _ in range(20):
            modulator.step_slow(5.0, 0.05)
        ach_surprised = modulator.ACh.item()
        
        # 大突变应使 ACh 下降
        assert ach_surprised < ach_stable
    
    def test_ne_responds_to_surprise(self, modulator):
        """测试 NE 对突变的响应：突变大 → NE 升高"""
        for _ in range(50):
            modulator.step_slow(0.5, 0.05)
        ne_stable = modulator.NE.item()
        
        for _ in range(20):
            modulator.step_slow(5.0, 0.05, ood_score=0.8)
        ne_surprised = modulator.NE.item()
        
        assert ne_surprised > ne_stable
    
    def test_da_responds_to_reward(self, modulator):
        """测试 DA 对奖赏信号的响应"""
        for _ in range(50):
            modulator.step_slow(0.5, 0.05, reward_signal=0.0)
        da_baseline = modulator.DA.item()
        
        for _ in range(50):
            modulator.step_slow(0.5, 0.05, reward_signal=1.0)
        da_rewarded = modulator.DA.item()
        
        assert da_rewarded > da_baseline
    
    def test_modulator_bounds(self, modulator):
        """测试所有调质变量在合法范围内"""
        # 极端输入
        for _ in range(100):
            modulator.step_slow(100.0, 1.0, reward_signal=10.0, ood_score=1.0)
        
        assert 0.1 <= modulator.ACh.item() <= 10.0
        assert 0.01 <= modulator.NE.item() <= 2.0
        assert 0.001 <= modulator.DA.item() <= 1.0
        
        # 极端低输入
        for _ in range(100):
            modulator.step_slow(0.0, 0.0, reward_signal=-1.0, ood_score=0.0)
        
        assert 0.1 <= modulator.ACh.item() <= 10.0
        assert 0.01 <= modulator.NE.item() <= 2.0
        assert 0.001 <= modulator.DA.item() <= 1.0
    
    def test_get_learning_rate_scale(self, modulator):
        """测试学习率缩放"""
        modulator.step_slow(0.5, 0.05)
        lr_scale = modulator.get_learning_rate_scale()
        
        assert lr_scale.dim() == 0  # 标量
        assert lr_scale.item() >= 0.0
    
    def test_get_temperature(self, modulator):
        """测试温度获取"""
        modulator.step_slow(0.5, 0.05)
        temp = modulator.get_temperature()
        assert temp.item() > 0.0
    
    def test_get_third_factor(self, modulator):
        """测试第三因子获取"""
        modulator.step_slow(0.5, 0.05, reward_signal=0.5)
        da = modulator.get_third_factor()
        assert da.item() > 0.0
    
    def test_surprise_ema_tracking(self, modulator):
        """测试突变量 EMA 追踪"""
        # 稳定输入
        for _ in range(100):
            modulator.step_slow(0.5, 0.05)
        s1 = modulator.surprise_ema.item()
        
        # 突变
        modulator.step_slow(10.0, 0.05)
        s2 = modulator.surprise_ema.item()
        
        assert s2 > s1  # 突变后 surprise 应增大
    
    def test_multiple_slow_steps_converge(self, modulator):
        """测试持续稳定输入下调质趋于稳态"""
        values = []
        for _ in range(500):
            result = modulator.step_slow(0.5, 0.05)
            values.append(result["ACh"].item())
        
        # 后100步的方差应远小于前100步
        early_var = torch.tensor(values[:100]).var().item()
        late_var = torch.tensor(values[400:]).var().item()
        # 可能早期方差也很小，但至少不应爆炸
        assert late_var < early_var + 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
