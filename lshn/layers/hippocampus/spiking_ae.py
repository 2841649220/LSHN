import torch
import torch.nn as nn
from lshn.core.cells.liquid_cell import LiquidGatedCell

class SpikingAutoEncoder(nn.Module):
    """
    海马体脉冲自编码器 (Hippocampus Spiking AutoEncoder)
    高可塑性，快速编码新知识。

    双通路结构 (白皮书 §3.2.2):
    - 编码通路: encoder_linear + 高可塑性元胞, 将输入脉冲压缩为隐脉冲
      (step_fast), 供皮层/双势阱使用;
    - 解码通路: decoder_linear 将隐脉冲重构回输入模式 (decode /
      reconstruction_loss), 经重构训练后供 ReplayGenerator 生成
      有意义的伪样本用于皮层巩固。
    """
    def __init__(self, input_dim: int, hidden_dim: int, input_gain: float = 1.0,
                 device=None, dtype=None):
        super().__init__()
        self.encoder_linear = nn.Linear(input_dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.decoder_linear = nn.Linear(hidden_dim, input_dim, bias=False, device=device, dtype=dtype)

        # 简单的高可塑性元胞 (缩短慢适应常数)
        self.cell = LiquidGatedCell(
            num_neurons=hidden_dim,
            tau_g_slow=50.0,   # 相比皮层更快
            tau_a=20.0,
            input_gain=input_gain,
            device=device, dtype=dtype
        )
        
    def step_fast(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        x_in: (batch_size, input_dim) 脉冲输入
        """
        I_syn = self.encoder_linear(x_in)
        spk_out, mem_out = self.cell.step_fast(I_syn)
        return spk_out
        
    def decode(self, spk_hidden: torch.Tensor) -> torch.Tensor:
        """
        重构输入
        """
        return self.decoder_linear(spk_hidden)

    def reconstruction_loss(self, spk_hidden: torch.Tensor, target_spk: torch.Tensor) -> torch.Tensor:
        """
        重构损失: MSE(decoder(spk_hidden), target_spk)。
        target_spk 为编码侧输入脉冲 (spk_encoded)。

        解码通路 (§3.2.2) 语义: 海马体学习将隐脉冲重构回输入模式,
        回放生成器用训练后的 decoder 生成有意义伪样本。
        """
        recon = self.decoder_linear(spk_hidden)
        return torch.mean((recon - target_spk) ** 2)
