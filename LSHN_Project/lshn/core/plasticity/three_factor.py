"""
三因素可塑性与预测误差反向脉冲学习规则
白皮书 §3.5.2, §3.5.3

修复:
- 提供可微和在线两种模式
- 添加 delta_w 范围限制
- 添加 device/dtype 参数
- precision 饱和保护
"""
import torch
import torch.nn as nn
from typing import Optional


class ThreeFactorPlasticity(nn.Module):
    """
    三因素可塑性与预测误差反向脉冲学习规则
    
    三因素:
    1. Pre-synaptic trace (前突触迹)
    2. Post-synaptic trace (后突触迹/误差脉冲)
    3. Global/Local Error Neuromodulator (第三因子: 神经调质/DA信号)
    
    模式:
    - online: 在线可塑性，直接修改权重，不参与反向传播梯度计算
    - differentiable: 可微模式，返回更新后的权重，参与梯度计算
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01, 
                 trace_decay: float = 0.9,
                 max_delta: float = 0.1,
                 mode: str = "online",
                 device=None, 
                 dtype=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        self.max_delta = max_delta  # delta_w 的最大绝对值
        self.mode = mode  # "online" 或 "differentiable"
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        
    def forward(self, w_hat: torch.Tensor, e_trace: torch.Tensor, 
                error_spk: torch.Tensor, neuromodulator: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        计算并应用三因素权重更新
        
        Args:
            w_hat: 当前快权重 (max_edges,)
            e_trace: 资格迹 (max_edges,)
            error_spk: 泊松编码的误差反向脉冲 (max_edges,)
            neuromodulator: 全局多巴胺/调节信号 (标量或向量)
            
        Returns:
            更新后的 w_hat
        """
        # 修复: 自动对齐维度，支持不同形状的输入
        target_shape = w_hat.shape
        
        # 对齐 e_trace
        if e_trace.shape != target_shape:
            if e_trace.shape[0] < target_shape[0]:
                e_trace = torch.nn.functional.pad(e_trace, (0, target_shape[0] - e_trace.shape[0]))
            else:
                e_trace = e_trace[:target_shape[0]]
        
        # 对齐 error_spk
        if error_spk.shape != target_shape:
            if error_spk.shape[0] < target_shape[0]:
                error_spk = torch.nn.functional.pad(error_spk, (0, target_shape[0] - error_spk.shape[0]))
            else:
                error_spk = error_spk[:target_shape[0]]
        
        # Eq: Δw_hat = η * e_trace * error_spk * neuromodulator
        mod_factor = 1.0
        if neuromodulator is not None:
            if isinstance(neuromodulator, torch.Tensor):
                # 支持标量或向量形式的 neuromodulator
                if neuromodulator.numel() == 1:
                    mod_factor = neuromodulator.float().item()
                else:
                    # 向量形式: 对齐维度
                    if neuromodulator.shape[0] != target_shape[0]:
                        if neuromodulator.shape[0] < target_shape[0]:
                            neuromodulator = torch.nn.functional.pad(
                                neuromodulator, (0, target_shape[0] - neuromodulator.shape[0]))
                        else:
                            neuromodulator = neuromodulator[:target_shape[0]]
                    mod_factor = neuromodulator.float()
            else:
                mod_factor = float(neuromodulator)
            
        # 计算 delta_w
        delta_w = self.learning_rate * e_trace * error_spk * mod_factor
        
        # 范围限制，防止权重爆炸
        delta_w = delta_w.clamp(-self.max_delta, self.max_delta)
        
        if self.mode == "online":
            # 在线可塑性模式：不参与梯度计算
            with torch.no_grad():
                w_hat_new = w_hat + delta_w
                # 权重范围限制 [-1, 1]
                w_hat_new = w_hat_new.clamp(-1.0, 1.0)
                w_hat.data.copy_(w_hat_new)
                return w_hat
        else:
            # 可微模式：返回更新后的权重，参与梯度计算
            w_hat_new = w_hat + delta_w
            return w_hat_new.clamp(-1.0, 1.0)


class PoissonErrorEncoder(nn.Module):
    """
    将预测误差编码为泊松脉冲序列 (反向信息流)
    
    将预测误差转换为二值脉冲 {−1, 0, +1}:
    - +1: 正误差 (预测低于目标)
    - -1: 负误差 (预测高于目标)
    - 0: 无误差或误差过小
    """
    
    def __init__(self, 
                 f_max: float = 1.0,
                 max_precision: float = 10.0,
                 device=None,
                 dtype=None):
        super().__init__()
        self.f_max = f_max
        self.max_precision = max_precision  # 防止 sigmoid 饱和
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, precision: float = 1.0
                ) -> torch.Tensor:
        """
        将预测误差编码为泊松脉冲
        
        Args:
            pred: 预测脉冲率或输出 (batch, num_classes)
            target: 目标值 (batch, num_classes)
            precision: 精度参数 (控制编码灵敏度)
            
        Returns:
            error_spk: 误差脉冲 {−1, 0, +1} (batch, num_classes)
        """
        error = pred - target
        
        # 限制 precision 范围，防止 sigmoid 饱和
        precision_safe = min(abs(precision), self.max_precision) if precision != 0 else 1.0
        
        # 计算发放频率
        freq = torch.sigmoid(torch.abs(error) * precision_safe) * self.f_max
        
        # 泊松采样
        prob = torch.rand_like(freq)
        error_spk = (prob < freq).float() * torch.sign(error)
        
        return error_spk