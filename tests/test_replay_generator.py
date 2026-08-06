"""
测试: 可控离线采样动力学回放生成器 (ReplayGenerator)
覆盖: 白皮书 §3.6.2

行为契约:
- 未 init_state 调用 generate_step / inject_pattern 抛 RuntimeError
- init_state 后 generate_step 输出二值伪脉冲, 状态轨迹逐步变化
- inject_pattern 以 inject_rate 混合牵引状态 (λ·pattern 吸引项)
- reset() 清零状态与速度
- 温度参数生效: 高温 → 噪声 std 更大 → 状态变化幅度更大
"""
import torch
import pytest
import torch.nn as nn
from lshn.layers.hippocampus.replay_generator import ReplayGenerator


class TestReplayGenerator:

    def test_generate_step_before_init_raises(self):
        """① 未 init 调 generate_step 抛 RuntimeError"""
        rg = ReplayGenerator(hidden_dim=8)
        decoder = nn.Linear(8, 8)
        with pytest.raises(RuntimeError):
            rg.generate_step(decoder)

    def test_inject_pattern_before_init_raises(self):
        """未 init 调 inject_pattern 抛 RuntimeError"""
        rg = ReplayGenerator(hidden_dim=8)
        with pytest.raises(RuntimeError):
            rg.inject_pattern(torch.ones(1, 8))

    def test_generate_step_output_and_trajectory(self):
        """② init_state 后 generate_step 输出二值、形状正确、状态轨迹变化"""
        torch.manual_seed(0)
        rg = ReplayGenerator(hidden_dim=16, leakage=0.1, momentum=0.9)
        rg.init_state(batch_size=1)
        assert rg._state.shape == (1, 16)
        assert rg._velocity.shape == (1, 16)

        decoder = nn.Linear(16, 16)
        pseudo = rg.generate_step(decoder, temperature=0.1)
        assert pseudo.shape == (1, 16)
        assert torch.all((pseudo == 0.0) | (pseudo == 1.0))

        # 两次调用间状态轨迹变化 (泄漏-动量二阶动力学推进)
        s1 = rg._state.clone()
        rg.generate_step(decoder, temperature=0.1)
        assert not torch.allclose(rg._state, s1)

    def test_inject_pattern_attracts_state(self):
        """③ inject_pattern 牵引状态: 注入全 1 pattern 后状态均值上升"""
        torch.manual_seed(0)
        rg = ReplayGenerator(hidden_dim=32)
        rg.init_state(batch_size=1)
        s0 = rg._state.clone()
        mean0 = s0.mean().item()

        rg.inject_pattern(torch.ones(1, 32), inject_rate=0.3)
        # state = 0.7·state + 0.3·pattern → 向 1 移动
        assert rg._state.mean().item() > mean0
        expected = 0.7 * s0 + 0.3 * torch.ones(1, 32)
        assert torch.allclose(rg._state, expected)

    def test_reset_zeros_state(self):
        """④ reset() 清零状态与速度"""
        torch.manual_seed(0)
        rg = ReplayGenerator(hidden_dim=8)
        rg.init_state(batch_size=1)
        decoder = nn.Linear(8, 8)
        rg.generate_step(decoder, temperature=0.1)
        rg.inject_pattern(torch.ones(1, 8), inject_rate=0.5)

        rg.reset()
        assert torch.all(rg._state == 0.0)
        assert torch.all(rg._velocity == 0.0)

    def test_temperature_scales_noise(self):
        """⑤ 温度参数生效: 高温噪声 std 更大 (固定种子, 状态变化幅度更大)"""
        torch.manual_seed(0)
        rg_low = ReplayGenerator(hidden_dim=64)
        rg_high = ReplayGenerator(hidden_dim=64)
        rg_low.init_state(batch_size=1)
        rg_high.init_state(batch_size=1)
        # 两生成器状态/速度归零: 泄漏项 −leakage·state = 0,
        # 状态变化纯由噪声驱动 Δstate = velocity = (1−momentum)·noise
        rg_low._state.zero_()
        rg_low._velocity.zero_()
        rg_high._state.zero_()
        rg_high._velocity.zero_()
        decoder = nn.Linear(64, 64)

        rg_low.generate_step(decoder, temperature=0.01)   # noise std = 0.001
        low_change = rg_low._state.abs().mean().item()

        rg_high.generate_step(decoder, temperature=1.0)   # noise std = 0.010
        high_change = rg_high._state.abs().mean().item()

        # 噪声 std 相差 10 倍 → 高温状态变化幅度显著更大
        assert high_change > 3.0 * low_change


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
