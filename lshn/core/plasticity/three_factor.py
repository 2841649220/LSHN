import torch
import torch.nn as nn
from typing import Optional

class ThreeFactorPlasticity(nn.Module):
    """
    三因素可塑性与预测误差反向脉冲学习规则
    1. Pre-synaptic trace (前突触)
    2. Post-synaptic trace (后突触)
    3. Global/Local Error Neuromodulator (第三因子: 神经调质/反向误差脉冲)
    """
    def __init__(self, learning_rate: float = 0.01, trace_decay: float = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        
    def forward(self, w_hat: torch.Tensor, e_trace: torch.Tensor, 
                error_spk: torch.Tensor, neuromodulator: Optional[torch.Tensor] = None):
        """
        w_hat: 当前快权重
        e_trace: 资格迹 (由于共发放累积得到)
        error_spk: 泊松编码的误差反向脉冲 (局部的或自顶向下的)
        neuromodulator: 全局多巴胺/调节信号 (如 DA_signal \in [-1, 1])
        """
        # Eq: \Delta w_hat = \eta * e_trace * error_spk * neuromodulator
        
        mod_factor = 1.0
        if neuromodulator is not None:
            mod_factor = neuromodulator
            
        # 实际操作中，e_trace 维度需与 w_hat 对齐
        # 这里简化为直接广播乘积
        delta_w = self.learning_rate * e_trace * error_spk * mod_factor
        
        # In-place 更新快权重
        w_hat.data.add_(delta_w)
        
        return w_hat

class PoissonErrorEncoder(nn.Module):
    """
    将预测误差编码为泊松脉冲序列 (反向信息流)
    """
    def __init__(self, f_max: float = 1.0):
        super().__init__()
        self.f_max = f_max
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, precision: float = 1.0):
        """
        pred: 预测脉冲率或膜电位
        target: 目标
        precision: 精度矩阵(或标量)
        """
        error = pred - target
        freq = torch.sigmoid(torch.abs(error) * precision) * self.f_max
        # 泊松采样
        prob = torch.rand_like(freq)
        error_spk = (prob < freq).float() * torch.sign(error)
        return error_spk
