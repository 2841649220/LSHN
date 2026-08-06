"""
测试: 海马体脉冲自编码器 (SpikingAutoEncoder)
覆盖: 白皮书 §3.2.2 海马体快速学习层

行为契约:
- step_fast 输出二值脉冲, 形状 (batch, hidden_dim)
- reconstruction_loss(spk_hidden, target_spk) = MSE(decoder(spk), target),
  对 decoder_linear 产生梯度
- 解码重构方向: 同输入多次 forward 重构损失有限
"""
import torch
import pytest
from lshn.layers.hippocampus.spiking_ae import SpikingAutoEncoder


class TestSpikingAutoEncoder:

    def test_step_fast_binary_shape(self):
        """① step_fast 输出二值形状 (batch, hidden)"""
        torch.manual_seed(0)
        ae = SpikingAutoEncoder(input_dim=16, hidden_dim=32, input_gain=2.0)
        x = torch.rand(4, 16)  # 脉冲输入 [0, 1]
        spk = ae.step_fast(x)
        assert spk.shape == (4, 32)
        # 硬脉冲部分二值 (STE 残差 detach 后为 0/1)
        assert torch.all((spk.detach() == 0.0) | (spk.detach() == 1.0))

    def test_reconstruction_loss_grad_to_decoder(self):
        """② reconstruction_loss 对 decoder 产生梯度"""
        torch.manual_seed(0)
        ae = SpikingAutoEncoder(input_dim=16, hidden_dim=32)
        # 含脉冲的隐层脉冲张量 (输入无梯度需求, 直接构造确定有脉冲)
        spk_hidden = (torch.rand(4, 32) > 0.5).float()
        target_spk = torch.rand(4, 16)

        loss = ae.reconstruction_loss(spk_hidden, target_spk)
        assert torch.isfinite(loss)
        assert loss.shape == ()  # 标量

        loss.backward()
        assert ae.decoder_linear.weight.grad is not None
        assert ae.decoder_linear.weight.grad.abs().sum().item() > 0.0

    def test_decode_direction_finite(self):
        """③ 同输入两次 forward 重构 loss 有限 (解码通路稳定)"""
        torch.manual_seed(0)
        ae = SpikingAutoEncoder(input_dim=16, hidden_dim=32, input_gain=2.0)
        x = torch.rand(4, 16)
        for _ in range(3):
            spk = ae.step_fast(x)
            loss = ae.reconstruction_loss(spk, x)
            assert torch.isfinite(loss).item()
        # decode 接口与 reconstruction_loss 内部一致
        recon = ae.decode(spk)
        assert recon.shape == (4, 16)
        assert torch.isfinite(recon).all().item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
