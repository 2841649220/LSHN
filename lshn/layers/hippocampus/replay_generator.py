import torch
import torch.nn as nn

class ReplayGenerator(nn.Module):
    """
    可控离线采样动力学 (Leakage & Second-Order Momentum)
    生成用于皮层巩固的伪脉冲数据。
    """
    def __init__(self, hidden_dim: int, leakage: float = 0.1, momentum: float = 0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.leakage = leakage
        self.momentum = momentum
        
        self._state = None
        self._velocity = None
        
    def init_state(self, batch_size: int, device=None, dtype=None):
        need_reinit = (
            self._state is None
            or self._state.shape[0] != batch_size
            or self._state.shape[1] != self.hidden_dim
            or (device is not None and self._state.device != device)
        )
        if need_reinit:
            self._state = torch.randn(batch_size, self.hidden_dim, device=device, dtype=dtype)
            self._velocity = torch.zeros_like(self._state)
        else:
            self._state.normal_()
            self._velocity.zero_()
        
    def generate_step(self, ae_decoder: nn.Module) -> torch.Tensor:
        """
        基于二阶动力学采样生成一步回放数据。
        """
        if self._state is None:
            raise RuntimeError("ReplayGenerator.init_state() must be called before generate_step()")
        
        with torch.no_grad():
            noise = torch.randn_like(self._state) * 0.1
            
            force = -self.leakage * self._state + noise
            
            self._velocity = self.momentum * self._velocity + (1 - self.momentum) * force
            self._state = self._state + self._velocity
            
            out = ae_decoder(torch.sigmoid(self._state))
            
            pseudo_spk = (out > 0.5).float()
        return pseudo_spk
