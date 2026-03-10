"""
主动推理与预期自由能模块 (Active Inference / Expected Free Energy)
白皮书 §3.1.1, §4.2.4:
  G(π) = E_q(o,s|π)[ log q(s|π) - log p(o,s) ]
  分解为: 风险(偏好不满足) + 模糊性(观测不确定) - 信息增益(揭示隐藏状态)
  
用于驱动探索-利用平衡，当系统需要做动作/感知选择时使用。
参考 pymdp 的 EFE 计算范式。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ActiveInferenceEngine(nn.Module):
    """
    主动推理引擎
    
    实现 Expected Free Energy (EFE) 的计算与策略选择:
      G(π) = Risk + Ambiguity - Information_Gain
    
    其中:
    - Risk (风险): E_q[D_KL(q(o|π) || p(o))] — 预测观测偏离偏好分布
    - Ambiguity (模糊性): E_q[H[p(o|s)]] — 观测给定状态的不确定性  
    - Information Gain (信息增益): E_q[D_KL(q(s|o,π) || q(s|π))] — 减少隐状态不确定性
    """
    
    def __init__(self, state_dim: int, obs_dim: int, num_policies: int = 8,
                 gamma: float = 1.0, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.num_policies = num_policies
        self.gamma = gamma  # EFE 温度（精度）
        
        # 状态转移模型: q(s'|s, π) — 对每个策略学习一个转移概率
        self.transition_models = nn.ModuleList([
            nn.Linear(state_dim, state_dim, **factory_kwargs)
            for _ in range(num_policies)
        ])
        
        # 观测似然模型: p(o|s) — 从隐状态到观测的映射
        self.likelihood_model = nn.Linear(state_dim, obs_dim, **factory_kwargs)
        
        # 偏好先验: log p(o) — 对期望观测的偏好分布（可学习）
        self.log_preference = nn.Parameter(
            torch.zeros(obs_dim, **factory_kwargs)
        )
        
        # 后验信念: q(s) 的参数（均值和对数方差）
        self.register_buffer("belief_mean", torch.zeros(state_dim, **factory_kwargs))
        self.register_buffer("belief_logvar", torch.zeros(state_dim, **factory_kwargs))
        
    def update_belief(self, observation: torch.Tensor, prediction_error: torch.Tensor):
        """
        基于新观测更新后验信念 q(s)
        使用简化的贝叶斯滤波:
            q(s) ∝ p(o|s) * q_prior(s)
        """
        lr = 0.1
        pred_error_scaled = lr * prediction_error
        new_mean = self.belief_mean * (1.0 - lr) + pred_error_scaled
        self.belief_mean.copy_(new_mean)
        error_abs = prediction_error.abs().clamp(min=1e-6)
        new_logvar = self.belief_logvar + lr * (error_abs.log() - self.belief_logvar)
        self.belief_logvar.copy_(new_logvar.clamp(-5, 2))
    
    def compute_efe(self, current_state: torch.Tensor
                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算所有策略的预期自由能 G(π)
        
        Args:
            current_state: (state_dim,) 或 (batch, state_dim) 当前隐状态
            
        Returns:
            G: (num_policies,) 每个策略的 EFE
            components: 分解项字典 {risk, ambiguity, info_gain}
        """
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
        batch_size = current_state.shape[0]
        
        G_all = []
        risk_all = []
        ambiguity_all = []
        info_gain_all = []
        
        # 偏好分布 (softmax 归一化)
        log_pref = F.log_softmax(self.log_preference, dim=-1)
        
        for pi_idx in range(self.num_policies):
            # 1. 预测下一状态: q(s'|s, π)
            next_state_mean = self.transition_models[pi_idx](current_state)
            
            # 2. 预测观测: q(o|π) = E_q(s')[p(o|s')]
            pred_obs_logits = self.likelihood_model(next_state_mean)
            pred_obs_log_prob = F.log_softmax(pred_obs_logits, dim=-1)
            pred_obs_prob = pred_obs_log_prob.exp()
            
            # 3. Risk = D_KL(q(o|π) || p(o))
            # F.kl_div expects log-prob as first arg, so we compute KL(q||p) as:
            # KL(q||p) = sum(q * log(q/p)) = sum(q * (log(q) - log(p)))
            # = H(q) - sum(q * log(p)) = H(q) + F.kl_div(log_p, q, reduction='none')
            risk = (pred_obs_prob * (pred_obs_log_prob - log_pref)).sum(dim=-1).mean()
            
            # 4. Ambiguity = H[p(o|s')] — 观测条件熵
            # H(p) = -sum(p * log(p))
            ambiguity = -(pred_obs_prob * pred_obs_log_prob).sum(dim=-1).mean()
            
            # 5. Information Gain = 状态不确定性的预期减少
            # Gaussian entropy: H = 0.5 * d * log(2*pi*e) + 0.5 * sum(log_var)
            d = self.state_dim
            current_entropy = 0.5 * d * torch.log(torch.tensor(2 * 3.14159265359 * 2.71828182846)) + 0.5 * self.belief_logvar.sum()
            # 预期精度越高，信息增益越大
            pred_precision = pred_obs_prob.max(dim=-1).values.mean()
            expected_entropy_reduction = torch.log1p(pred_precision.clamp(min=1e-6))
            info_gain = expected_entropy_reduction
            
            # G(π) = Risk + Ambiguity - Info_Gain
            G = risk + ambiguity - info_gain
            
            G_all.append(G)
            risk_all.append(risk)
            ambiguity_all.append(ambiguity)
            info_gain_all.append(info_gain)
        
        G_tensor = torch.stack(G_all)
        
        return G_tensor, {
            "risk": torch.stack(risk_all),
            "ambiguity": torch.stack(ambiguity_all),
            "info_gain": torch.stack(info_gain_all)
        }
    
    def select_policy(self, current_state: torch.Tensor) -> Tuple[int, Dict]:
        """
        基于 EFE 的 softmax 策略选择
            π* = softmax(-γ * G(π))
            
        Returns:
            selected_policy: int 选中的策略索引
            info: 包含 EFE 分解和策略概率的字典
        """
        G, components = self.compute_efe(current_state)
        
        # softmax 策略选择
        policy_probs = F.softmax(-self.gamma * G, dim=0)
        selected = torch.multinomial(policy_probs, 1).item()
        
        return selected, {
            "G": G.detach(),
            "policy_probs": policy_probs.detach(),
            "selected": selected,
            **{k: v.detach() for k, v in components.items()}
        }
    
    def get_exploration_signal(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        从 EFE 分解中提取全局探索信号
        当 information gain 高时，驱动系统探索。
        当 risk 高时，驱动系统回避或适应。
        
        Returns:
            exploration_signal: 标量，正值鼓励探索，负值鼓励利用
        """
        _, components = self.compute_efe(current_state)
        
        avg_info_gain = components["info_gain"].mean()
        avg_risk = components["risk"].mean()
        
        # 信息增益高 → 探索（正信号）；风险高 → 适应/保守（负信号）
        exploration = avg_info_gain - 0.5 * avg_risk
        
        return exploration.detach()
    
    def reset(self):
        """重置信念状态"""
        self.belief_mean.zero_()
        self.belief_logvar.zero_()
