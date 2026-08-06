"""
能量/脉冲预算控制器
白皮书 §2.3, §3.1, §4.2.4:
  将"突触事件数/脉冲率"作为反馈信号,
  动态调节阈值、抑制强度、稀疏激活比例与正则系数。

以反馈控制方式维护预算:
  在每个慢时间窗，根据当前突触事件数与目标预算的偏差，
  自适应调节稀疏正则系数、阈值/适应变量、抑制强度。

改进:
  - PI 控制器加入 anti-windup 防止积分饱和
  - 双向调整 (超标时增大阈值/抑制, 不足时减小)
  - 输出实际应用到系统中
"""


class SpikeBudgetController:
    """
    能量/脉冲预算 PI 控制器
    
    控制信号用于:
    1. 调节发放阈值 (theta_adj) —— 由 model.py 消费
    2. 调节侧向抑制强度 (inh_adj) —— 由 model.py 消费
    3. 调节自由能引擎的能量正则系数 (lambda_E_adj) —— 遗留输出, 未被消费
       (λ_E 自适应由 FreeEnergyEngine.compute_energy_regularization_gradient 完成)
    """
    def __init__(self, target_spikes_per_step: int, kp: float = 0.01, ki: float = 0.001,
                 max_integral: float = 100.0,
                 theta_adj_scale: float = 0.1, inh_adj_scale: float = 0.05):
        self.target_budget = target_spikes_per_step
        self.kp = kp
        self.ki = ki
        self.max_integral = max_integral  # anti-windup 上限
        self.theta_adj_scale = theta_adj_scale
        self.inh_adj_scale = inh_adj_scale

        self.integral_error = 0.0

        # 输出调整量
        self.theta_adj = 0.0
        self.inh_adj = 0.0
        # 注意: lambda_E_adj 为历史遗留输出, 当前未被任何调用方消费 (死代码)——
        # λ_E 的自适应已由 model.py 中的 vfe_engine.compute_energy_regularization_gradient()
        # 完成; model.py 仅读取 theta_adj 与 inh_adj。
        # 保留该字段仅为兼容外部潜在读取, 请勿依赖其语义。
        self.lambda_E_adj = 0.0

    def step_control(self, current_spikes: float) -> dict:
        """
        基于 PI 控制调整 (每个慢时间步或固定间隔调用)

        Args:
            current_spikes: 当前时间窗内的平均脉冲数 (已按窗口长度归一化到"每快时钟步")

        Returns:
            包含所有调整量的字典
        """
        error = current_spikes - self.target_budget

        # 积分项 (含 anti-windup)
        self.integral_error += error
        self.integral_error = max(-self.max_integral,
                                   min(self.max_integral, self.integral_error))

        # PI 控制信号
        control_signal = self.kp * error + self.ki * self.integral_error

        # === 双向调整 ===
        # 正 control_signal: 超出预算 → 增大阈值/抑制
        # 负 control_signal: 低于预算 → 减小阈值/抑制
        self.theta_adj = control_signal * self.theta_adj_scale
        self.inh_adj = control_signal * self.inh_adj_scale
        # (遗留输出, 未被消费; λ_E 自适应由 vfe_engine.compute_energy_regularization_gradient 负责)
        self.lambda_E_adj = max(0.0, control_signal * 0.001)  # 能量惩罚系数只增不减到0

        return {
            "theta_adj": self.theta_adj,
            "inh_adj": self.inh_adj,
            "lambda_E_adj": self.lambda_E_adj,
            "budget_error": error,
            "integral_error": self.integral_error
        }

    def reset(self):
        """重置控制器状态"""
        self.integral_error = 0.0
        self.theta_adj = 0.0
        self.inh_adj = 0.0
        self.lambda_E_adj = 0.0
