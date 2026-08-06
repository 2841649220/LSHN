"""
测试: 三因素可塑性 (ThreeFactorPlasticity) 与泊松误差编码 (PoissonErrorEncoder)
覆盖: 白皮书 §3.5.3 三因素学习规则

行为契约 (修复后):
- 误差符号: error = target − pred (正误差 = 欠预测 → 权重增大, 负反馈环)
- 泊松编码全程 no_grad, 输出不携带梯度
- Δw = lr · e_trace · error_spk · neuromodulator · plasticity_gate
- w_hat clamp ±1 (有界更新防发散)
- plasticity_gate = 0 时权重不变
"""
import torch
import pytest
from lshn.core.plasticity.three_factor import ThreeFactorPlasticity, PoissonErrorEncoder


class TestPoissonErrorEncoder:

    def test_error_sign_under_prediction(self):
        """① 符号: target > pred (欠预测) → error_spk 均值 > 0 → 权重上调"""
        torch.manual_seed(0)
        enc = PoissonErrorEncoder(f_max=1.0)
        pred = torch.zeros(1000)
        target = torch.ones(1000)

        error_spk = enc(pred, target)
        # |error| = 1.0 → freq = 1.0 → 全 +1
        assert error_spk.mean().item() > 0.0
        assert torch.all(error_spk == 1.0)

    def test_error_sign_over_prediction(self):
        """① 符号: target < pred (过预测) → error_spk 均值 < 0 → 权重下调"""
        torch.manual_seed(0)
        enc = PoissonErrorEncoder(f_max=1.0)
        pred = torch.ones(1000)
        target = torch.zeros(1000)

        error_spk = enc(pred, target)
        assert error_spk.mean().item() < 0.0

    def test_error_magnitude_proportional(self):
        """误差幅值 ∝ |target − pred| (零基准泊松率)"""
        torch.manual_seed(0)
        enc = PoissonErrorEncoder(f_max=1.0)
        # 误差 0.5 → freq = 0.5 → E[error_spk] = 0.5
        pred = torch.zeros(2000)
        target = torch.ones(2000) * 0.5
        error_spk = enc(pred, target)
        assert error_spk.mean().item() == pytest.approx(0.5, abs=0.05)

    def test_no_grad_output(self):
        """② no_grad: 输出 requires_grad=False (泊松采样不可微)"""
        enc = PoissonErrorEncoder(f_max=1.0)
        pred = torch.randn(10, requires_grad=True)
        target = torch.randn(10, requires_grad=True)
        error_spk = enc(pred, target)
        assert not error_spk.requires_grad


class TestThreeFactorPlasticity:

    def test_delta_w_formula(self):
        """③ 公式: Δw = lr · e_trace · error_spk · mod · gate (精确断言)"""
        pl = ThreeFactorPlasticity(learning_rate=0.01, trace_decay=0.9)
        w_hat = torch.tensor([0.5, -0.5, 0.0])
        e_trace = torch.tensor([1.0, 2.0, 3.0])
        error_spk = torch.tensor([1.0, -1.0, 0.5])
        mod = torch.tensor(2.0)

        old = w_hat.clone()
        pl(w_hat, e_trace, error_spk, neuromodulator=mod, plasticity_gate=1.0)

        expected = old + 0.01 * e_trace * error_spk * mod
        expected.clamp_(-1.0, 1.0)
        assert torch.allclose(w_hat, expected)

    def test_error_sign_positive_increases_weight(self):
        """③ 符号: 正误差 (欠预测) → w_hat 增大 (负反馈环方向)"""
        pl = ThreeFactorPlasticity(learning_rate=0.01)
        w_hat = torch.tensor([0.0])
        pl(w_hat, torch.ones(1), torch.ones(1))
        assert w_hat.item() > 0.0

    def test_w_hat_clamped(self):
        """④ 大误差多次更新后 w_hat 仍在 ±1 界内"""
        pl = ThreeFactorPlasticity(learning_rate=10.0)
        w_up = torch.zeros(1)
        for _ in range(50):
            pl(w_up, torch.ones(1), torch.ones(1))       # 正误差
        assert w_up.item() == 1.0

        w_down = torch.zeros(1)
        for _ in range(50):
            pl(w_down, torch.ones(1), -torch.ones(1))    # 负误差
        assert w_down.item() == -1.0

    def test_plasticity_gate_zero(self):
        """⑤ plasticity_gate = 0 时 w_hat 不变"""
        pl = ThreeFactorPlasticity(learning_rate=1.0)
        w_hat = torch.tensor([0.3, -0.2])
        pl(w_hat, torch.ones(2), torch.ones(2), plasticity_gate=0.0)
        assert torch.allclose(w_hat, torch.tensor([0.3, -0.2]))

    def test_mod_factor_amplifies(self):
        """neuromodulator 作为第三因子放大更新"""
        pl = ThreeFactorPlasticity(learning_rate=0.01)
        w1 = torch.tensor([0.0])
        w2 = torch.tensor([0.0])
        pl(w1, torch.ones(1), torch.ones(1), neuromodulator=torch.tensor(0.5))
        pl(w2, torch.ones(1), torch.ones(1), neuromodulator=torch.tensor(1.0))
        assert w2.item() > w1.item() > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
