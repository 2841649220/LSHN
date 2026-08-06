"""
稳态可塑性与突触缩放模块 (Homeostatic Plasticity & Synaptic Scaling)
白皮书 §2.3, §4.1:
  将突触缩放/归一化作为稳态约束环，与 STDP/三因素共同维持长期在线的可学性。
  
两个核心机制:
  1. 突触缩放 (Synaptic Scaling): 维持总突触强度稳定，防止突触漂移
  2. 内在兴奋性可塑性 (Intrinsic Excitability Plasticity): 维持目标发放率

参考: [R9] Effects of Introducing Synaptic Scaling on SNN Learning (2026)
"""
import torch
import torch.nn as nn
from typing import Optional, Dict


class SynapticScaling(nn.Module):
    """
    突触缩放 (Synaptic Scaling)
    
    对每个神经元的所有传入突触权重进行乘性缩放:
        w_{ij} *= (target_rate / actual_rate_i)^β
        
    其中:
    - target_rate: 目标发放率 (稳态设定值)
    - actual_rate_i: 神经元 i 的实际平均发放率
    - β: 缩放强度参数 (< 1 以避免振荡)
    
    这对应 FEP 中的复杂度约束: 防止表征无限膨胀。
    """
    
    def __init__(self, num_neurons: int, target_rate: float = 0.05,
                 scaling_strength: float = 0.1, ema_decay: float = 0.99,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.num_neurons = num_neurons
        self.target_rate = target_rate
        self.scaling_strength = scaling_strength

        # 维护每个神经元的发放率 EMA
        self.register_buffer(
            "firing_rate_ema",
            torch.ones(num_neurons, **factory_kwargs) * target_rate
        )
        self.ema_decay = ema_decay
        
    def update_rates(self, spk: torch.Tensor):
        """
        更新发放率 EMA（每个快时间步调用）
        spk: (num_neurons,) 当前步的脉冲 {0, 1}
        """
        # 形状防御: 传入 batched (batch, num_neurons) 时按 batch 平均
        if spk.dim() > 1:
            spk = spk.mean(dim=0)
        self.firing_rate_ema.data.mul_(self.ema_decay).add_(
            (1.0 - self.ema_decay) * spk.detach()
        )
    
    def compute_scaling_factors(self) -> torch.Tensor:
        """
        计算每个神经元的突触缩放因子
        
        Returns:
            scale: (num_neurons,) 乘性缩放因子
                > 1: 神经元活动不足，需要上调突触
                < 1: 神经元过度活动，需要下调突触
                = 1: 达到目标发放率
        """
        # 防止除零
        rate = self.firing_rate_ema.clamp(min=1e-6)
        
        # 乘性缩放: (target / actual)^β
        ratio = self.target_rate / rate
        scale = ratio.pow(self.scaling_strength)
        
        # 限制缩放幅度，避免突变
        scale = scale.clamp(0.5, 2.0)
        
        return scale
    
    def apply_scaling(self, w_hat: torch.Tensor, neuron_to_edge_map: Optional[torch.Tensor] = None,
                      alive_neuron_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对权重应用突触缩放

        Args:
            w_hat: (max_edges,) 快权重
            neuron_to_edge_map: (max_edges,) 每条边对应的后突触神经元索引
                如果 None，则对所有边使用平均缩放因子
            alive_neuron_mask: (num_neurons,) 神经元存活掩码 (bool)
                提供时, 映射到死神经元的边缩放因子强制 1.0 —— 防止死神经元
                关联边被 scale=2.0 持续放大到 ±1 饱和, 复活时爆发;
                当 neuron_to_edge_map 为 None 时, 全局均值只统计存活神经元
                (默认 None 时行为不变)。

        Returns:
            scaled_w_hat: 缩放后的权重
        """
        scale = self.compute_scaling_factors()

        if neuron_to_edge_map is not None:
            # 每条边使用其后突触神经元的缩放因子
            edge_scale = scale[neuron_to_edge_map]
            if alive_neuron_mask is not None:
                # 死神经元关联边缩放强制 1.0 (不放大也不缩小),
                # 使死神经元的边权重停在当前值, 复活时从稳定点继续而非爆发
                alive_map = alive_neuron_mask.to(scale.device)[neuron_to_edge_map]
                edge_scale = torch.where(alive_map, edge_scale, torch.ones_like(edge_scale))
            return w_hat * edge_scale
        else:
            # 简化: 使用全局平均缩放 (提供 mask 时剔除死神经元再求均值)
            if alive_neuron_mask is not None:
                alive = alive_neuron_mask.to(scale.device)
                if alive.any():
                    mean_scale = scale[alive].mean()
                else:
                    mean_scale = scale.mean()
            else:
                mean_scale = scale.mean()
            return w_hat * mean_scale


class IntrinsicExcitabilityPlasticity(nn.Module):
    """
    内在兴奋性可塑性 (Intrinsic Excitability Plasticity)
    
    通过调节每个神经元的发放阈值来维持目标发放率:
        Δθ_i = η_ie * (actual_rate_i - target_rate)
        
    发放率高 → 阈值升高 → 降低兴奋性
    发放率低 → 阈值降低 → 提升兴奋性
    
    这与白皮书中适应变量 a_i 的机制互补: a_i 是短时适应（秒级），
    而内在兴奋性是中长期适应（分钟-小时级）。
    """
    
    def __init__(self, num_neurons: int, target_rate: float = 0.05,
                 ie_learning_rate: float = 0.001,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.num_neurons = num_neurons
        self.target_rate = target_rate
        self.ie_learning_rate = ie_learning_rate
        
        # 阈值调整量 (叠加到 theta_0 + a_i 之上)
        self.register_buffer(
            "theta_ie",
            torch.zeros(num_neurons, **factory_kwargs)
        )
        
    def step_slow(self, firing_rate_ema: torch.Tensor) -> torch.Tensor:
        """
        慢时钟更新阈值
        
        Args:
            firing_rate_ema: (num_neurons,) 发放率 EMA
            
        Returns:
            theta_ie: (num_neurons,) 阈值调整量
        """
        delta_theta = self.ie_learning_rate * (firing_rate_ema - self.target_rate)
        self.theta_ie.data.add_(delta_theta)
        self.theta_ie.data.clamp_(-0.5, 1.0)
        
        return self.theta_ie


class HomeostaticController(nn.Module):
    """
    稳态可塑性控制器（整合突触缩放和内在兴奋性）
    
    作为白皮书中 "无中心隐式MoE" 的重要约束之一:
    与抑制竞争、双势阱结构固化共同完成分区与干扰隔离。
    """
    
    def __init__(self, num_neurons: int, target_rate: float = 0.05,
                 scaling_strength: float = 0.1, ie_lr: float = 0.001,
                 tau_rate: float = 100.0,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # tau_rate (ms) → 每快时钟步 (1ms) 的 EMA 衰减系数
        ema_decay = 1.0 - 1.0 / max(tau_rate, 1.0)
        self.synaptic_scaling = SynapticScaling(
            num_neurons, target_rate, scaling_strength=scaling_strength,
            ema_decay=ema_decay, **factory_kwargs
        )
        self.ie_plasticity = IntrinsicExcitabilityPlasticity(
            num_neurons, target_rate, ie_learning_rate=ie_lr, **factory_kwargs
        )

    def reset(self):
        """重置发放率 EMA 与阈值调整 (保留网络权重)"""
        self.synaptic_scaling.firing_rate_ema.fill_(self.synaptic_scaling.target_rate)
        self.ie_plasticity.theta_ie.zero_()

    def step_fast(self, spk: torch.Tensor):
        """快时钟: 更新发放率 EMA"""
        self.synaptic_scaling.update_rates(spk)
    
    def step_slow(self) -> Dict[str, torch.Tensor]:
        """
        慢时钟: 计算突触缩放因子和阈值调整
        
        Returns:
            dict with 'scaling_factors' and 'theta_ie'
        """
        scaling = self.synaptic_scaling.compute_scaling_factors()
        theta_ie = self.ie_plasticity.step_slow(
            self.synaptic_scaling.firing_rate_ema
        )
        
        return {
            "scaling_factors": scaling,
            "theta_ie": theta_ie
        }
    
    def apply_to_weights(self, w_hat: torch.Tensor,
                         neuron_to_edge_map: Optional[torch.Tensor] = None,
                         alive_neuron_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """对权重施加突触缩放 (透传 alive_neuron_mask 到 SynapticScaling)"""
        return self.synaptic_scaling.apply_scaling(w_hat, neuron_to_edge_map, alive_neuron_mask)
