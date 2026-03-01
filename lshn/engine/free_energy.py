"""
变分自由能(VFE)全局计算引擎
白皮书 §3.1, §3.1.1, §3.1.2

核心功能:
1. VFE = Accuracy + KL_weight * Complexity
2. 能量正则化: J = F + λ_E * E[SynapticEvents]  (白皮书 §3.1)
3. 自由能分解可解释监控 (白皮书 §3.1.2)
4. 神经调质反馈 (集成 GlobalNeuromodulator)
"""
import torch
from typing import Dict, Optional


class FreeEnergyEngine:
    """
    变分自由能(VFE)全局计算引擎
    
    F(q,θ) = E_q(s)[ log q(s) - log p_θ(o,s) ]
            = -E_q(s)[log p_θ(o|s)] + D_KL(q(s)||p_θ(s))
              ↑ 预测误差/不相容        ↑ 复杂度
              
    完整目标函数 (含能量预算):
        J = F + λ_E * E[#SynapticEvents]
    """
    def __init__(self, kl_weight: float = 0.01, energy_lambda: float = 0.001):
        self.kl_weight = kl_weight
        self.energy_lambda = energy_lambda
        
        # 历史记录 (用于可解释监控)
        self.history = {
            "vfe_total": [],
            "J_total": [],
            "accuracy_loss": [],
            "complexity_loss": [],
            "energy_cost": [],
        }
        self.max_history = 1000
        
    def compute_vfe(self, prediction_error: torch.Tensor, s_e_tensor: torch.Tensor, 
                    active_neurons_ratio: float,
                    synaptic_events: Optional[int] = None,
                    precision: float = 1.0) -> Dict[str, float]:
        """
        计算全局变分自由能和完整目标函数 J
        
        Args:
            prediction_error: 预测误差张量
            s_e_tensor: 结构变量 s_e
            active_neurons_ratio: 活跃神经元比例
            synaptic_events: 当前时间窗的突触事件数 (脉冲数)
            precision: 精度参数 (由 ACh 调制)
        """
        # === Accuracy 项 (精度加权的预测误差) ===
        # 对应白皮书: -E_q[log p(o|s)], 精度高 → 误差被放大
        accuracy_term = precision * torch.mean(prediction_error ** 2).item()
        
        # === Complexity 项 ===
        # D_KL(q(s)||p(s)) 的近似:
        # 1. 结构复杂度: s_e 的 KL (鼓励稀疏结构)
        structure_kl = torch.mean(
            s_e_tensor * torch.log(s_e_tensor.clamp(min=1e-6) / 0.5)
            + (1 - s_e_tensor) * torch.log((1 - s_e_tensor).clamp(min=1e-6) / 0.5)
        ).item()
        # 2. 活跃度复杂度
        activity_cost = active_neurons_ratio
        complexity_term = abs(structure_kl) + activity_cost
        
        # === VFE ===
        vfe_total = accuracy_term + self.kl_weight * complexity_term
        
        # === 能量正则化: J = F + λ_E * E[SynapticEvents] (白皮书 §3.1) ===
        energy_cost = 0.0
        if synaptic_events is not None:
            energy_cost = float(synaptic_events)
        
        J_total = vfe_total + self.energy_lambda * energy_cost
        
        result = {
            "vfe_total": vfe_total,
            "J_total": J_total,
            "accuracy_loss": accuracy_term,
            "complexity_loss": complexity_term,
            "structure_kl": structure_kl,
            "energy_cost": energy_cost,
            "precision": precision
        }
        
        # 记录历史
        self._record_history(result)
        
        return result
    
    def compute_energy_regularization_gradient(self, 
                                                current_events: int,
                                                target_budget: int) -> float:
        """
        计算能量正则化对 λ_E 的自适应调整
        (反馈控制: 当实际事件超出预算时增大 λ_E)
        
        Args:
            current_events: 当前突触事件数
            target_budget: 目标预算
            
        Returns:
            λ_E 的调整量
        """
        budget_error = current_events - target_budget
        # 如果超出预算, 增大惩罚; 低于预算, 减小惩罚
        delta_lambda = 0.0001 * budget_error
        self.energy_lambda = max(0.0, min(0.1, self.energy_lambda + delta_lambda))
        return self.energy_lambda
    
    def get_decomposition_report(self) -> Dict[str, float]:
        """
        返回自由能分解报告 (白皮书 §3.1.2 的硬性交付)
        
        包含:
        - 预测误差（按模块分组的可扩展）
        - 复杂度（结构KL、连接数、活跃超边数）
        - 精度/温度
        - 不确定性指标
        """
        if not self.history["vfe_total"]:
            return {}
        
        recent_n = min(10, len(self.history["vfe_total"]))
        
        return {
            "vfe_recent_mean": sum(self.history["vfe_total"][-recent_n:]) / recent_n,
            "J_recent_mean": sum(self.history["J_total"][-recent_n:]) / recent_n,
            "accuracy_trend": sum(self.history["accuracy_loss"][-recent_n:]) / recent_n,
            "complexity_trend": sum(self.history["complexity_loss"][-recent_n:]) / recent_n,
            "energy_trend": sum(self.history["energy_cost"][-recent_n:]) / recent_n,
            "vfe_total_history_len": len(self.history["vfe_total"])
        }
    
    def _record_history(self, result: dict):
        """记录历史"""
        for key in ["vfe_total", "J_total", "accuracy_loss", "complexity_loss", "energy_cost"]:
            if key in result:
                self.history[key].append(result[key])
                if len(self.history[key]) > self.max_history:
                    self.history[key] = self.history[key][-self.max_history:]
