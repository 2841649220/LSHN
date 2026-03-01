"""
轴突延迟学习模块 (Axonal Delay Learning)
白皮书 §1.3.4, §3.5.4: 可学习传导延迟 × STDP/eligibility 的时序信用分配
把可学习传导延迟纳入可塑性闭环，增强序列、语音、事件流任务的时间表征能力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AxonalDelayModule(nn.Module):
    """
    轴突传导延迟模块
    
    为每条超边维护一个可学习的传导延迟 d_e ∈ [d_min, d_max]（离散化为整数步），
    实现方式为环形缓冲区（delay buffer），延迟后的脉冲再参与 STDP 和权重更新。
    
    延迟学习规则：
        Δd_e = η_d * ∂L/∂d_e ≈ η_d * e_trace * (pre_spike_delayed - post_spike) 
    利用资格迹和前后脉冲时差的梯度近似来调整延迟。
    """
    
    def __init__(self, max_edges: int, max_delay: int = 20, min_delay: int = 1,
                 delay_lr: float = 0.001, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.max_edges = max_edges
        self.max_delay = max_delay
        self.min_delay = min_delay
        self.delay_lr = delay_lr
        
        # 可学习的连续延迟值（用于梯度更新，后续离散化）
        init_delay = torch.ones(max_edges, **factory_kwargs) * ((max_delay + min_delay) / 2.0)
        self.delay_continuous = nn.Parameter(init_delay, requires_grad=False)
        
        # 离散化后的延迟索引
        self.register_buffer(
            "delay_discrete",
            torch.ones(max_edges, dtype=torch.long, device=device) * min_delay
        )
        
        # 环形缓冲区：存储最近 max_delay 步的前突触脉冲
        # shape: (max_delay, max_edges) — 每条边独立的脉冲历史
        self.register_buffer(
            "spike_buffer",
            torch.zeros(max_delay, max_edges, **factory_kwargs)
        )
        self.register_buffer(
            "buffer_ptr",
            torch.tensor(0, dtype=torch.long, device=device)
        )
        
        # 延迟相关的STDP迹
        self.register_buffer(
            "pre_trace_delayed",
            torch.zeros(max_edges, **factory_kwargs)
        )
        self.register_buffer(
            "post_trace",
            torch.zeros(max_edges, **factory_kwargs)
        )
        
        self.trace_decay = 0.95
        
    def reset(self):
        """重置所有状态"""
        self.spike_buffer.zero_()
        self.buffer_ptr.zero_()
        self.pre_trace_delayed.zero_()
        self.post_trace.zero_()
        
    def _discretize_delays(self):
        """将连续延迟离散化为整数步"""
        self.delay_discrete.data.copy_(torch.clamp(
            torch.round(self.delay_continuous).long(),
            self.min_delay,
            self.max_delay - 1
        ))
    
    def step_fast(self, pre_spk: torch.Tensor, post_spk: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        快时钟 (1ms) 前向
        
        Args:
            pre_spk: (max_edges,) 或 (batch, max_edges) 当前步的前突触脉冲
            post_spk: (max_edges,) 或 (batch, max_edges) 后突触脉冲
            
        Returns:
            delayed_spk: (max_edges,) 经过延迟的前突触脉冲
            stdp_delta: (max_edges,) 延迟相关的STDP信号
        """
        # 处理 batch 维度: 如果是 batch，则在当前步取平均以简化
        if pre_spk.dim() > 1:
            pre_spk = pre_spk.mean(dim=0)
        if post_spk.dim() > 1:
            post_spk = post_spk.mean(dim=0)
            
        # 1. 将当前脉冲存入环形缓冲区
        ptr = self.buffer_ptr % self.max_delay
        self.spike_buffer[ptr] = pre_spk
        
        # 2. 从缓冲区读取延迟后的脉冲
        self._discretize_delays()
        # 读取位置 = (当前指针 - 延迟) mod max_delay
        read_idx = (self.buffer_ptr - self.delay_discrete) % self.max_delay
        delayed_spk = self.spike_buffer[read_idx, torch.arange(self.max_edges, device=pre_spk.device)]
        
        # 3. 更新延迟相关的STDP迹 (in-place 操作保护 buffer 注册)
        self.pre_trace_delayed.data.mul_(self.trace_decay).add_(delayed_spk)
        self.post_trace.data.mul_(self.trace_decay).add_(post_spk)
        
        # 4. 计算延迟敏感的STDP信号
        # LTP: delayed_pre * post_trace (先前后后 → 增强)
        # LTD: post * pre_trace_delayed (先后后前 → 抑制)
        stdp_delta = delayed_spk * self.post_trace - post_spk * self.pre_trace_delayed
        
        # 5. 前进指针
        self.buffer_ptr += 1
        
        return delayed_spk, stdp_delta
    
    def update_delays(self, e_trace: torch.Tensor, timing_error: torch.Tensor):
        """
        慢时钟 (100ms) 延迟学习
        
        根据资格迹和时序误差调整延迟值:
            Δd = η_d * e_trace * timing_error
        
        timing_error > 0: 脉冲到达太早 → 增大延迟
        timing_error < 0: 脉冲到达太晚 → 减小延迟
        
        Args:
            e_trace: (max_edges,) 资格迹
            timing_error: (max_edges,) 时序误差信号
        """
        delta_d = self.delay_lr * e_trace * timing_error
        self.delay_continuous.data.add_(delta_d)
        self.delay_continuous.data.clamp_(float(self.min_delay), float(self.max_delay - 1))
        self._discretize_delays()
    
    def get_delay_stats(self) -> dict:
        """返回延迟分布的统计信息（用于可解释监控）"""
        d = self.delay_continuous.detach()
        return {
            "delay_mean": d.mean().item(),
            "delay_std": d.std().item(),
            "delay_min": d.min().item(),
            "delay_max": d.max().item(),
            "delay_entropy": -torch.sum(
                F.softmax(d, dim=0) * F.log_softmax(d, dim=0)
            ).item()
        }
