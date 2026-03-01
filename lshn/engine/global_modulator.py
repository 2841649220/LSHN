"""
全局神经调节器模块 (Global Neuromodulator)
白皮书 §1.3.1, §3.2(§3): 
  将多巴胺/乙酰胆碱/去甲肾上腺素等神经调质抽象为可计算的全局调制变量,
  星形胶质三方突触慢变量承接跨时间尺度的 metaplasticity。

三个闭环量:
  - ACh (预期不确定性/注意) → 精度上调与学习率窗口放大
  - NE  (意外不确定性/突变检测) → 温度上调与结构可塑性开启
  - DA  (价值/偏好满足) → 三因素第三因子门控
  
还包含星形胶质全局门控: O(1) 复杂度的持续学习优化
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class AstrocyteGate(nn.Module):
    """
    星形胶质细胞全局门控
    
    模拟星形胶质细胞的钙波扩散同步:
    整合全网络的平均预测误差与神经元活动，动态调节全局可塑性。
    稳定状态下抑制权重更新，分布变化时启用适应。
    
    更新规则 (慢时间步):
        astro_state(t+1) = τ_astro^{-1} * astro_state(t) + (1-τ_astro^{-1}) * f(inputs)
    """
    
    def __init__(self, num_neurons: int, tau_astro: float = 500.0,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.tau_astro = tau_astro
        
        # 星形胶质细胞的内部钙浓度状态
        self.register_buffer(
            "calcium", 
            torch.ones(1, **factory_kwargs) * 0.5
        )
        
        # 整合预测误差和活动率的权重
        self.W_error = nn.Parameter(torch.tensor(0.3, **factory_kwargs))
        self.W_activity = nn.Parameter(torch.tensor(-0.2, **factory_kwargs))
        self.bias = nn.Parameter(torch.tensor(0.0, **factory_kwargs))
        
    def step_slow(self, mean_prediction_error: float, mean_firing_rate: float) -> torch.Tensor:
        """
        慢时钟更新星形胶质状态
        
        Returns:
            plasticity_gate: 标量 [0, 1], 控制全局可塑性
                ~0: 抑制所有权重更新（稳定状态）
                ~1: 完全开放可塑性（分布变化/新任务）
        """
        decay = 1.0 - (1.0 / self.tau_astro)
        
        drive = torch.sigmoid(
            self.W_error * mean_prediction_error 
            + self.W_activity * mean_firing_rate 
            + self.bias
        )
        
        self.calcium = decay * self.calcium + (1.0 - decay) * drive
        
        return self.calcium.clamp(0.0, 1.0)


class GlobalNeuromodulator(nn.Module):
    """
    全局神经调节器
    
    将三种神经调质 (ACh, NE, DA) 和星形胶质门控统一为一个调制层。
    所有输出量均为可微的标量/向量，可直接插入到各模块的更新规则中。
    """
    
    def __init__(self, num_neurons: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.num_neurons = num_neurons
        
        # 星形胶质门控
        self.astrocyte = AstrocyteGate(num_neurons, **factory_kwargs)
        
        # 三种神经调质的状态 (慢变量)
        self.register_buffer("ACh", torch.tensor(1.0, **factory_kwargs))   # 精度/注意
        self.register_buffer("NE", torch.tensor(0.1, **factory_kwargs))    # 温度/探索
        self.register_buffer("DA", torch.tensor(0.01, **factory_kwargs))   # 第三因子/奖赏
        
        # 调质更新的时间常数
        self.tau_ACh = 200.0
        self.tau_NE = 100.0
        self.tau_DA = 150.0
        
        # 调质的驱动权重 (可学习)
        self.register_buffer("surprise_ema", torch.tensor(0.0, **factory_kwargs))
        self.surprise_decay = 0.95
        
        # 历史平均误差 (用于突变检测)
        self.register_buffer("error_ema", torch.tensor(0.5, **factory_kwargs))
        self.error_ema_decay = 0.99
        
    def step_slow(self, prediction_error: float, firing_rate: float,
                  reward_signal: float = 0.0, 
                  ood_score: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        慢时钟 (100ms) 更新全部神经调质
        
        Args:
            prediction_error: 当前预测误差 (标量)
            firing_rate: 平均发放率 (标量)
            reward_signal: 外部奖赏/偏好信号 [-1, 1]
            ood_score: OOD/分布突变检测分数 [0, 1]
            
        Returns:
            modulation: 包含所有调制量的字典
        """
        device = self.ACh.device
        
        # ---- 突变/意外检测 ----
        error_t = torch.tensor(prediction_error, device=device, dtype=self.ACh.dtype)
        self.error_ema = self.error_ema_decay * self.error_ema + (1 - self.error_ema_decay) * error_t
        surprise = torch.abs(error_t - self.error_ema)  # 突变量 = 当前误差偏离 EMA
        self.surprise_ema = self.surprise_decay * self.surprise_ema + (1 - self.surprise_decay) * surprise
        
        # ---- ACh: 预期不确定性 / 注意 ----
        # 误差适中且稳定 → ACh 升高(精度上调)，误差激增 → ACh 下降(降低精度门槛)
        ach_drive = 1.0 / (1.0 + surprise * 5.0)  # surprise 越大 ACh 越低
        decay_ach = 1.0 - (1.0 / self.tau_ACh)
        self.ACh = decay_ach * self.ACh + (1.0 - decay_ach) * ach_drive * 5.0
        self.ACh = self.ACh.clamp(0.1, 10.0)
        
        # ---- NE: 意外不确定性 / 温度 ----
        # surprise 或 OOD 高 → NE 升高 → 温度升高 → 结构可塑性开启
        ne_drive = surprise + torch.tensor(ood_score, device=device, dtype=self.ACh.dtype)
        decay_ne = 1.0 - (1.0 / self.tau_NE)
        self.NE = decay_ne * self.NE + (1.0 - decay_ne) * torch.sigmoid(ne_drive * 3.0)
        self.NE = self.NE.clamp(0.01, 2.0)
        
        # ---- DA: 价值/偏好满足 → 三因素第三因子 ----
        # 正奖赏 → DA 升高 → 巩固当前行为的突触
        # 预测误差降低 → 内生好奇满足 → DA 轻微升高
        curiosity = torch.max(torch.tensor(0.0, device=device), -surprise + 0.1)
        da_drive = torch.tensor(reward_signal, device=device, dtype=self.ACh.dtype) + 0.5 * curiosity
        decay_da = 1.0 - (1.0 / self.tau_DA)
        self.DA = decay_da * self.DA + (1.0 - decay_da) * torch.sigmoid(da_drive)
        self.DA = self.DA.clamp(0.001, 1.0)
        
        # ---- 星形胶质门控 ----
        plasticity_gate = self.astrocyte.step_slow(prediction_error, firing_rate)
        
        return {
            "ACh": self.ACh.clone(),         # 精度 (用于 VFE 的精度加权)
            "NE": self.NE.clone(),           # 温度 (用于双势阱 Langevin 噪声)
            "DA": self.DA.clone(),           # 第三因子 (用于三因素可塑性)
            "plasticity_gate": plasticity_gate.clone(),  # 星形胶质门控 [0,1]
            "surprise": self.surprise_ema.clone()        # 突变量 (用于监控)
        }
    
    def get_learning_rate_scale(self) -> torch.Tensor:
        """
        返回基于 ACh 和星形胶质门的学习率缩放因子
        高 ACh + 高 plasticity_gate → 高学习率
        """
        return ((self.ACh / 5.0) * self.astrocyte.calcium).squeeze()
    
    def get_temperature(self) -> torch.Tensor:
        """返回 NE 对应的全局温度参数"""
        return self.NE
    
    def get_third_factor(self) -> torch.Tensor:
        """返回 DA 作为三因素可塑性的第三因子"""
        return self.DA
