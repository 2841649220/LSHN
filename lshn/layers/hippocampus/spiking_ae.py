import torch
import torch.nn as nn
from lshn.core.cells.liquid_cell import LiquidGatedCell

class SpikingAutoEncoder(nn.Module):
    """
    海马体脉冲自编码器 (Hippocampus Spiking AutoEncoder)
    高可塑性，快速编码新知识。
    """
    def __init__(self, input_dim: int, hidden_dim: int, device=None, dtype=None):
        super().__init__()
        self.encoder_linear = nn.Linear(input_dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.decoder_linear = nn.Linear(hidden_dim, input_dim, bias=False, device=device, dtype=dtype)
        
        # 简单的高可塑性元胞 (缩短慢适应常数)
        self.cell = LiquidGatedCell(
            num_neurons=hidden_dim, 
            tau_g_slow=50.0,   # 相比皮层更快
            tau_a=20.0,
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
