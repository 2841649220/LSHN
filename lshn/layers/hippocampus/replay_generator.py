import math

import torch
import torch.nn as nn


class ReplayGenerator(nn.Module):
    """
    可控离线采样动力学 (Leakage + Second-Order Momentum + Pattern Attraction)
    生成用于皮层巩固的伪脉冲数据。

    完整采样动力学 (白皮书 §3.6.2):
    1) 向零泄漏: force = -leakage·state + noise, 位置有界;
    2) 二阶动量: velocity = momentum·velocity + (1-momentum)·force,
       平滑力, 避免每步独立随机 draw;
    3) 模式吸引: inject_pattern 将近期编码模式混合进 state
       (λ·W_dec^T·S_hippo 的简化: state = (1-λ_inj)·state + λ_inj·pattern),
       使回放与近期记忆相关而非纯噪声;
    解码器经重构训练后输出有意义, 固定阈值 0.5 二值化为伪脉冲。
    """

    def __init__(self, hidden_dim: int, leakage: float = 0.1, momentum: float = 0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.leakage = leakage
        self.momentum = momentum

        # 状态注册为 buffer (persistent=False: 不进入 state_dict, 属瞬态运行状态);
        # init 时注册空张量, init_state 中按需重建 (重复 register_buffer 同名会替换)。
        self.register_buffer("_state", torch.empty(0), persistent=False)
        self.register_buffer("_velocity", torch.empty(0), persistent=False)

    def init_state(self, batch_size: int, device=None, dtype=None):
        """
        初始化/校验回放状态。

        只在首次调用或形状/设备不匹配时重建状态;
        重复调用 (如每个慢时钟) 不重新随机化 ——
        否则泄漏-动量二阶动力学永远只走一步, 生成器退化为
        "每次独立随机高斯 draw", 白皮书 §3.6.2 的采样动力学无法展开。
        """
        # 兼容旧式置 None (外部 `_state = None` 会移除 buffer 变成普通属性,
        # 此时需按 buffer 形状检查重建)
        need_reinit = (
            self._state is None
            or self._state.numel() == 0
            or self._state.shape[0] != batch_size
            or self._state.shape[1] != self.hidden_dim
            or (device is not None and self._state.device != device)
            or (dtype is not None and self._state.dtype != dtype)
        )
        if need_reinit:
            self.register_buffer(
                "_state",
                torch.randn(batch_size, self.hidden_dim, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                "_velocity",
                torch.zeros_like(self._state),
                persistent=False,
            )

    def inject_pattern(self, pattern: torch.Tensor, inject_rate: float = 0.3):
        """
        注入近期编码模式到生成器状态 (白皮书 §3.6.2 吸引项 λ·W_dec^T·S_hippo):
        state = (1-λ_inj)·state + λ_inj·pattern。pattern 形状 (batch, hidden_dim)。
        """
        if self._state is None or self._state.numel() == 0:
            raise RuntimeError("ReplayGenerator.init_state() must be called before inject_pattern()")
        self._state.mul_(1.0 - inject_rate).add_(inject_rate * pattern)

    def generate_step(self, ae_decoder: nn.Module, temperature: float = 0.1) -> torch.Tensor:
        """
        基于二阶动力学采样生成一步回放数据。

        temperature: 噪声温度, 噪声 std = 0.1·sqrt(temperature);
        调低温度压回放多样性, 调高则增加探索性。
        """
        if self._state is None or self._state.numel() == 0:
            raise RuntimeError("ReplayGenerator.init_state() must be called before generate_step()")

        with torch.no_grad():
            noise = torch.randn_like(self._state) * (0.1 * math.sqrt(temperature))

            force = -self.leakage * self._state + noise

            self._velocity.mul_(self.momentum).add_((1 - self.momentum) * force)
            self._state.add_(self._velocity)

            out = ae_decoder(torch.sigmoid(self._state))

            # 固定阈值 0.5 二值化: 解码器经重构训练 (reconstruction_loss) 后
            # 输出有意义 (重构的是脉冲输入, 值域为 sigmoid 输出);
            # 固定阈值使 R_replay (伪脉冲活动度) 有真实动态范围,
            # 而非按行均值自适应阈值导致的恒 ≈0.5 信息量为零。
            pseudo_spk = (out > 0.5).float()
        return pseudo_spk

    def reset(self):
        """清零回放状态 (速度与位置归零; 下一轮 init_state 重新初始化)。"""
        if self._state is not None and self._state.numel() > 0:
            self._state.zero_()
        if self._velocity is not None and self._velocity.numel() > 0:
            self._velocity.zero_()
