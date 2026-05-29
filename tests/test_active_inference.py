"""
测试: 主动推理与预期自由能模块 (ActiveInferenceEngine)
覆盖: 白皮书 §3.1.1, §4.2.4
"""
import torch
import pytest
from lshn.engine.active_inference import ActiveInferenceEngine


class TestActiveInferenceEngine:
    
    @pytest.fixture
    def engine(self):
        """创建标准 AIE 实例"""
        return ActiveInferenceEngine(
            state_dim=16, obs_dim=10, num_policies=4, gamma=1.0
        )
    
    def test_init(self, engine):
        """测试初始化参数"""
        assert engine.state_dim == 16
        assert engine.obs_dim == 10
        assert engine.num_policies == 4
        assert len(engine.transition_models) == 4
        assert engine.log_preference.shape == (10,)
        assert engine.belief_mean.shape == (16,)
        assert engine.belief_logvar.shape == (16,)
    
    def test_update_belief(self, engine):
        """测试信念更新"""
        old_mean = engine.belief_mean.clone()
        old_logvar = engine.belief_logvar.clone()
        
        obs = torch.randn(10)
        pred_error = torch.randn(16) * 0.5
        
        engine.update_belief(obs, pred_error)
        
        # 信念均值应被更新
        assert not torch.allclose(engine.belief_mean, old_mean)
    
    def test_compute_efe_shape(self, engine):
        """测试 EFE 计算的输出形状"""
        state = torch.randn(16)
        G, components = engine.compute_efe(state)
        
        assert G.shape == (4,)  # num_policies
        assert "risk" in components
        assert "ambiguity" in components
        assert "info_gain" in components
        assert components["risk"].shape == (4,)
        assert components["ambiguity"].shape == (4,)
        assert components["info_gain"].shape == (4,)
    
    def test_compute_efe_batched(self, engine):
        """测试批量状态输入的 EFE 计算"""
        state = torch.randn(3, 16)  # batch=3
        G, components = engine.compute_efe(state)
        
        assert G.shape == (4,)
    
    def test_efe_decomposition_identity(self, engine):
        """测试 G = Risk + Ambiguity - Info_Gain"""
        state = torch.randn(16)
        G, comp = engine.compute_efe(state)
        
        reconstructed = comp["risk"] + comp["ambiguity"] - comp["info_gain"]
        assert torch.allclose(G, reconstructed, atol=1e-5)
    
    def test_select_policy(self, engine):
        """测试策略选择"""
        state = torch.randn(16)
        selected, info = engine.select_policy(state)
        
        assert isinstance(selected, int)
        assert 0 <= selected < 4
        
        assert "G" in info
        assert "policy_probs" in info
        assert "selected" in info
        assert "risk" in info
        assert "ambiguity" in info
        assert "info_gain" in info
        
        # 概率和为1
        assert torch.allclose(info["policy_probs"].sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_select_policy_deterministic_with_high_gamma(self):
        """测试高 gamma (低温) 下策略选择趋向确定性"""
        engine = ActiveInferenceEngine(
            state_dim=8, obs_dim=6, num_policies=4, gamma=100.0
        )
        state = torch.randn(8)
        
        # 高 gamma 下策略概率应集中在某一个策略上
        _, info = engine.select_policy(state)
        probs = info["policy_probs"]
        assert probs.max() > 0.5  # 至少一个策略占比超过50%
    
    def test_get_exploration_signal(self, engine):
        """测试探索信号"""
        state = torch.randn(16)
        signal = engine.get_exploration_signal(state)
        
        assert signal.dim() == 0  # 标量
        assert signal.dtype == torch.float32
    
    def test_exploration_signal_detached(self, engine):
        """测试探索信号不携带梯度"""
        state = torch.randn(16, requires_grad=True)
        signal = engine.get_exploration_signal(state)
        assert not signal.requires_grad
    
    def test_risk_finite(self, engine):
        """测试 Risk (KL散度) 始终为有限值"""
        for _ in range(10):
            state = torch.randn(16) * 5.0
            G, comp = engine.compute_efe(state)
            assert torch.all(torch.isfinite(comp["risk"]))
    
    def test_gamma_scales_policy_probs(self):
        """测试 gamma 对策略概率分布的影响"""
        state = torch.randn(8)
        
        # 低温
        engine_cold = ActiveInferenceEngine(state_dim=8, obs_dim=6, num_policies=4, gamma=10.0)
        _, info_cold = engine_cold.select_policy(state)
        
        # 高温
        engine_hot = ActiveInferenceEngine(state_dim=8, obs_dim=6, num_policies=4, gamma=0.01)
        # 使用相同权重
        for i in range(4):
            engine_hot.transition_models[i].load_state_dict(
                engine_cold.transition_models[i].state_dict()
            )
        engine_hot.likelihood_model.load_state_dict(engine_cold.likelihood_model.state_dict())
        engine_hot.log_preference.data.copy_(engine_cold.log_preference.data)
        
        _, info_hot = engine_hot.select_policy(state)
        
        # 低温分布更尖锐 (最大概率更高)
        # 高温分布更平坦 (最大概率更低)
        assert info_cold["policy_probs"].max() >= info_hot["policy_probs"].max() - 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
