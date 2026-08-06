"""
测试: 能量/脉冲预算控制器 (SpikeBudgetController)
覆盖: 白皮书 §2.3, §3.1, §4.2.4

行为契约:
- 超标 (current > target) → theta_adj/inh_adj 为正 (增大阈值/抑制)
- 不足 (current < target) → theta_adj/inh_adj 为负
- 积分项 anti-windup clamp ±max_integral
- reset() 归零全部状态
- 稳态: current == target → budget_error ≈ 0
"""
import torch
import pytest
from lshn.engine.budget_control import SpikeBudgetController


class TestSpikeBudgetController:

    def test_over_budget_positive_adjustments(self):
        """① 超标 → theta_adj/inh_adj 为正"""
        ctrl = SpikeBudgetController(target_spikes_per_step=50, kp=0.01, ki=0.001)
        result = ctrl.step_control(80.0)  # 超出预算 30

        assert result["budget_error"] == pytest.approx(30.0)
        assert result["theta_adj"] > 0.0
        assert result["inh_adj"] > 0.0
        # control = kp·e + ki·∫e = 0.01×30 + 0.001×30 = 0.33
        assert result["theta_adj"] == pytest.approx(0.33 * ctrl.theta_adj_scale)
        assert result["inh_adj"] == pytest.approx(0.33 * ctrl.inh_adj_scale)

    def test_under_budget_negative_adjustments(self):
        """① 不足 → theta_adj/inh_adj 为负"""
        ctrl = SpikeBudgetController(target_spikes_per_step=50, kp=0.01, ki=0.001)
        result = ctrl.step_control(20.0)  # 低于预算 30

        assert result["budget_error"] == pytest.approx(-30.0)
        assert result["theta_adj"] < 0.0
        assert result["inh_adj"] < 0.0

    def test_integral_anti_windup_clamp(self):
        """② 积分项 anti-windup clamp ±max_integral"""
        ctrl = SpikeBudgetController(target_spikes_per_step=50,
                                     kp=0.01, ki=0.001, max_integral=10.0)
        # 持续超标: 积分收敛到 +10 后不再增长
        for _ in range(100):
            ctrl.step_control(100.0)
        assert ctrl.integral_error == pytest.approx(10.0)

        # 持续不足: 积分收敛到 -10
        ctrl2 = SpikeBudgetController(target_spikes_per_step=50,
                                      kp=0.01, ki=0.001, max_integral=10.0)
        for _ in range(100):
            ctrl2.step_control(0.0)
        assert ctrl2.integral_error == pytest.approx(-10.0)

    def test_reset_zeros_state(self):
        """③ reset() 归零"""
        ctrl = SpikeBudgetController(target_spikes_per_step=50)
        ctrl.step_control(100.0)
        ctrl.step_control(100.0)  # 同向误差 → 积分累积非零
        assert ctrl.integral_error != 0.0

        ctrl.reset()
        assert ctrl.integral_error == 0.0
        assert ctrl.theta_adj == 0.0
        assert ctrl.inh_adj == 0.0
        assert ctrl.lambda_E_adj == 0.0

    def test_steady_state_zero_error(self):
        """④ 稳态: step_control(target) 返回 budget_error ≈ 0"""
        ctrl = SpikeBudgetController(target_spikes_per_step=50)
        result = ctrl.step_control(50.0)
        assert result["budget_error"] == pytest.approx(0.0, abs=1e-12)
        assert result["theta_adj"] == 0.0
        assert result["inh_adj"] == 0.0
        # 积分项不受影响 (0 误差不积累)
        assert ctrl.integral_error == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
