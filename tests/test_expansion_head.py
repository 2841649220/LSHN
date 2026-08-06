"""
测试: 动态扩容输出解码层 (DynamicExpansionHead)
覆盖: 白皮书 §4.1 输出解码层持续学习扩容

行为契约 (单参数重建 + add_param_group):
- expand(n) -> [weight, bias] 当前完整参数对象
- n <= 0 返回 []
- 新行 std ≈ 0.01, bias 新行 = 0
- 旧行数值不变 (只追加)
"""
import torch
import pytest
from lshn.layers.io.dynamic_expansion_head import DynamicExpansionHead


class TestDynamicExpansionHead:

    def test_expand_zero_returns_empty(self):
        """① expand(0) / expand(负) 返回 []"""
        head = DynamicExpansionHead(in_features=8, initial_classes=2)
        assert head.expand(0) == []
        assert head.expand(-3) == []
        assert head.num_classes == 2

    def test_expand_returns_weight_bias(self):
        """② expand(2) 返回 [weight, bias] 且为当前参数对象"""
        head = DynamicExpansionHead(in_features=8, initial_classes=2)
        result = head.expand(2)
        assert len(result) == 2
        assert result[0] is head.weight
        assert result[1] is head.bias
        assert head.weight.shape == (4, 8)
        assert head.bias.shape == (4,)
        assert head.num_classes == 4
        assert head.weight.requires_grad and head.bias.requires_grad

    def test_expand_preserves_old_rows(self):
        """③ 旧行数值不变, 新行 std ≈ 0.01, bias 新行为 0"""
        torch.manual_seed(0)
        head = DynamicExpansionHead(in_features=8, initial_classes=2)
        old_weight = head.weight.detach().clone()
        old_bias = head.bias.detach().clone()

        head.expand(2)

        assert torch.allclose(head.weight[:2], old_weight)
        assert torch.allclose(head.bias[:2], old_bias)
        # 新行 std ≈ 0.01 (对齐架构文档 W_new ~ N(0, 0.01))
        assert head.weight[2:].std().item() == pytest.approx(0.01, abs=0.004)
        assert torch.all(head.bias[2:] == 0.0)

    def test_forward_after_expand(self):
        """④ 扩容后前向形状正确"""
        torch.manual_seed(0)
        head = DynamicExpansionHead(in_features=8, initial_classes=2)
        x = torch.randn(3, 8)
        assert head(x).shape == (3, 2)
        head.expand(3)
        assert head(x).shape == (3, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
