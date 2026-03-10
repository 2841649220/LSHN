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
        
        # 注册为 buffer 以确保 .to(device), state_dict() 等正确追踪
        # 初始化为单样本大小, init_state() 时会按需重建
        self.register_buffer("velocity", torch.zeros(1, hidden_dim))
        self.register_buffer("state", torch.randn(1, hidden_dim))
        self._initialized = False
        
    def init_state(self, batch_size: int, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        state_new = torch.randn(batch_size, self.hidden_dim, **factory_kwargs)
        velocity_new = torch.zeros(batch_size, self.hidden_dim, **factory_kwargs)
        # 重新注册 buffer (大小可能改变)
        # 使用 register_buffer 保证 buffer 身份不丢失
        self.state = state_new
        self.velocity = velocity_new
        # 重新注册以更新 _buffers dict
        self.register_buffer("state", self.state)
        self.register_buffer("velocity", self.velocity)
        self._initialized = True
        
    def generate_step(self, ae_decoder: nn.Module) -> torch.Tensor:
        """
        基于二阶动力学采样生成一步回放数据。
        """
        if not self._initialized:
            raise RuntimeError("ReplayGenerator.init_state() must be called before generate_step()")
        
        # 欠阻尼 Langevin 近似
        noise = torch.randn_like(self.state) * 0.1
        
        # 引入 leakage
        force = -self.leakage * self.state + noise
        
        # 更新速度和位置 (in-place 保护 buffer)
        self.velocity.data.mul_(self.momentum).add_((1 - self.momentum) * force)
        self.state.data.add_(self.velocity)
        
        # 经过解码器并转为脉冲概率
        out = ae_decoder(torch.sigmoid(self.state))
        
        # 转为伪脉冲
        pseudo_spk = (out > 0.5).float()
        return pseudo_spk
    
    def reset(self):
        """重置所有状态"""
        self._initialized = False
        self.velocity.zero_()
        self.state.zero_()
