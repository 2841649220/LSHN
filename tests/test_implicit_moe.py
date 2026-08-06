"""
测试: 无中心隐式MoE (ImplicitMoE)
覆盖: 白皮书 §4.2 分区表征 / 局部侧向抑制竞争

行为契约:
- I_inh_i = W_inh × (组内脉冲总和 − spk_i), 组内自抑制排除自身
- 批量输入逐样本独立计算 (不跨样本混合)
- 最后一组吸收余数: num_neurons 不能被 num_groups 整除时组边界正确
"""
import torch
import pytest
from lshn.layers.cortex.implicit_moe import ImplicitMoE


class TestImplicitMoE:

    def test_deterministic_math_single_batch(self):
        """① 确定性数学: I_inh = (组内脉冲和 − spk_i) × strength (手算断言)"""
        moe = ImplicitMoE(num_neurons=6, num_groups=2, inhibition_strength=0.5)
        # neurons_per_group = 3 → 组0 = {0,1,2}, 组1 = {3,4,5}
        spk = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
        # 组0 和 = 2 → (2 − spk) × 0.5 = [0.5, 1.0, 0.5]
        # 组1 和 = 1 → (1 − spk) × 0.5 = [0.5, 0.0, 0.5]
        expected = torch.tensor([[0.5, 1.0, 0.5, 0.5, 0.0, 0.5]])
        assert torch.allclose(moe(spk), expected)

    def test_batch_shape_and_sample_independence(self):
        """② 批输入形状与逐样本独立性"""
        moe = ImplicitMoE(num_neurons=6, num_groups=2, inhibition_strength=0.5)
        spk = torch.tensor([
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ])
        out = moe(spk)
        assert out.shape == (2, 6)
        # 样本 0: 组0 和=2, 组1 和=1
        expected_0 = torch.tensor([0.5, 1.0, 0.5, 0.5, 0.0, 0.5])
        # 样本 1: 组0 (神经元0,1,2) 和=1 → [0.5, 0.0, 0.5]; 组1 和=2 → [0.5, 1.0, 0.5]
        expected_1 = torch.tensor([0.5, 0.0, 0.5, 0.5, 1.0, 0.5])
        assert torch.allclose(out[0], expected_0)
        assert torch.allclose(out[1], expected_1)

    def test_remainder_group_boundary(self):
        """③ 余数组边界: num_neurons=7, num_groups=3 → 最后一组 3 个神经元"""
        moe = ImplicitMoE(num_neurons=7, num_groups=3, inhibition_strength=0.5)
        # neurons_per_group = 2 → group_ids = [0,0,1,1,2,2,2] (余数吸收进最后一组)
        assert moe.group_ids.tolist() == [0, 0, 1, 1, 2, 2, 2]

        spk = torch.ones(1, 7)
        out = moe(spk)
        # 组0 (神经元 0,1): 和=2 → I = 0.5
        assert out[0, 0].item() == pytest.approx(0.5)
        assert out[0, 1].item() == pytest.approx(0.5)
        # 组1 (神经元 2,3): 和=2 → I = 0.5
        assert out[0, 2].item() == pytest.approx(0.5)
        # 组2 (神经元 4,5,6): 和=3 → I = (3−1)×0.5 = 1.0
        assert out[0, 4].item() == pytest.approx(1.0)
        assert out[0, 5].item() == pytest.approx(1.0)
        assert out[0, 6].item() == pytest.approx(1.0)

    def test_non_batched_input(self):
        """单样本输入 (num_neurons,) 输出形状 (num_neurons,)"""
        moe = ImplicitMoE(num_neurons=6, num_groups=2, inhibition_strength=0.5)
        spk = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        out = moe(spk)
        assert out.shape == (6,)
        assert torch.allclose(out, torch.tensor([0.5, 1.0, 0.5, 0.5, 0.0, 0.5]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
