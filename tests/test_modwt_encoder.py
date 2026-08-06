"""
测试: 多尺度可学习编码前端 (MODWTEncoder)
覆盖: 白皮书 §4.1 输入编码层

行为契约:
- 前向输出形状 (batch, out_features)
- 硬脉冲 + STE 直通: 梯度可流通到 scale_extractors/fusion/threshold
- 可学习阈值偏置: 存在且初始值 ≈ 1.0 (压低基线发放率)
- 有阈值版本发放率显著低于无阈值版本 (同权重同输入同种子)
"""
import torch
import pytest
from lshn.layers.io.modwt_encoder import MODWTEncoder


class TestMODWTEncoder:

    def test_forward_shape(self):
        """① 前向输出形状 (batch, out)"""
        torch.manual_seed(0)
        enc = MODWTEncoder(in_features=16, out_features=32, num_scales=3)
        x = torch.randn(8, 16)
        spk = enc(x)
        assert spk.shape == (8, 32)

    def test_spike_binary_and_ste_gradient(self):
        """② 输出二值 (detach 后) + STE 梯度流通到全部可学习参数"""
        torch.manual_seed(0)
        enc = MODWTEncoder(in_features=16, out_features=32, num_scales=3)
        x = torch.randn(8, 16, requires_grad=True)
        spk = enc(x)

        # 硬脉冲部分二值
        assert torch.all((spk.detach() == 0.0) | (spk.detach() == 1.0))

        spk.sum().backward()
        # STE: 梯度经 spk_soft 残差直通到各层
        assert x.grad is not None
        for i, extractor in enumerate(enc.scale_extractors):
            assert extractor.weight.grad is not None, f"scale_extractor[{i}] 无梯度"
        assert enc.fusion.weight.grad is not None
        assert enc.threshold.grad is not None

    def test_threshold_bias_init(self):
        """③ 可学习阈值偏置存在且 init ≈ 1.0"""
        enc = MODWTEncoder(in_features=16, out_features=32)
        assert enc.threshold.shape == (1,)
        assert enc.threshold.item() == pytest.approx(1.0, abs=1e-6)
        assert enc.threshold.requires_grad

    def test_threshold_lowers_firing_rate(self):
        """④ 有阈值版本发放率显著低于无阈值版本 (同权重同输入同种子)"""
        torch.manual_seed(0)
        enc_th = MODWTEncoder(in_features=16, out_features=64, num_scales=3)
        enc_nt = MODWTEncoder(in_features=16, out_features=64, num_scales=3)
        # 同步权重, 仅阈值不同 (1.0 vs 0.0)
        enc_nt.load_state_dict(enc_th.state_dict())
        enc_nt.threshold.data.fill_(0.0)

        x = torch.randn(16, 16)

        torch.manual_seed(42)
        spk_th = enc_th(x)
        torch.manual_seed(42)
        spk_nt = enc_nt(x)

        rate_th = spk_th.float().mean().item()
        rate_nt = spk_nt.float().mean().item()
        # sigmoid 中心平移 1.0 → 基线发放率 ~27% vs ~50%
        assert rate_th < rate_nt * 0.9
        assert rate_nt > 0.3  # 无阈值版本应有明显发放


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
