"""
MODWT 多尺度小波编码与泊松脉冲前端
白皮书 §4.1 输入编码层:
  基于小波多尺度分析（MODWT）与泊松编码，
  将连续信号转化为多尺度脉冲序列。

实现: 纯 PyTorch MODWT（Maximal Overlap Discrete Wavelet Transform）
  - 支持 Haar 和 DB4（Daubechies-4）小波基
  - 非下采样（保持信号长度），平移不变
  - 圆周卷积实现，适配 GPU 加速
  - STE（Straight-Through Estimator）使泊松采样可反向传播
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  小波滤波器系数（已归一化为 MODWT 版本: 除以 sqrt(2)）
# ============================================================

def _haar_filters() -> tuple[torch.Tensor, torch.Tensor]:
    """Haar 小波 MODWT 滤波器 (长度 2)"""
    # 原始 DWT Haar: h=[1/√2, 1/√2], g=[1/√2, -1/√2]
    # MODWT 缩放: 除以 √2 → h=[1/2, 1/2], g=[1/2, -1/2]
    h = torch.tensor([0.5, 0.5])       # 低通 (scaling)
    g = torch.tensor([0.5, -0.5])      # 高通 (wavelet)
    return h, g


def _db4_filters() -> tuple[torch.Tensor, torch.Tensor]:
    """Daubechies-4 小波 MODWT 滤波器 (长度 8)"""
    # 标准 DB4 scaling 系数 (DWT 版)
    sqrt2 = math.sqrt(2.0)
    h_dwt = torch.tensor([
        -0.010597401784997278,
         0.032883011666982945,
         0.030841381835986965,
        -0.18703481171888114,
        -0.027983769416983849,
         0.63088076792959036,
         0.71484657055254153,
         0.23037781330885523,
    ])
    # MODWT 缩放: 除以 √2
    h = h_dwt / sqrt2

    # 高通滤波器: 由低通通过 QMF 关系导出
    # g[n] = (-1)^n * h[L-1-n]
    L = len(h)
    g = h.flip(0).clone()
    signs = torch.tensor([(-1.0) ** n for n in range(L)])
    g = g * signs

    return h, g


# ============================================================
#  MODWT 核心: 纯 PyTorch 圆周卷积实现
# ============================================================

def _modwt_circular_conv(signal: torch.Tensor, filt: torch.Tensor,
                         scale: int) -> torch.Tensor:
    """
    MODWT 圆周卷积（à trous 算法 / 带孔卷积）。

    Args:
        signal: (batch, 1, N) 输入信号
        filt:   (filter_len,) 滤波器系数
        scale:  当前分解层级 j (从 0 开始), 步长 = 2^j

    Returns:
        (batch, 1, N) 卷积结果（与输入等长）
    """
    L = filt.shape[0]
    stride = 2 ** scale  # à trous 步长

    # 构建带孔滤波器: 在系数间插入 (stride-1) 个零
    if stride > 1:
        dilated = torch.zeros(1 + (L - 1) * stride, device=signal.device, dtype=signal.dtype)
        dilated[::stride] = filt.to(signal.device, signal.dtype)
    else:
        dilated = filt.to(signal.device, signal.dtype)

    # 翻转为卷积核 (conv1d 执行互相关，翻转后等效卷积)
    kernel = dilated.flip(0).reshape(1, 1, -1)

    # 圆周填充: 左侧填充 (kernel_size - 1)，确保因果对齐且输出等长
    pad_len = kernel.shape[-1] - 1
    # 使用 circular padding
    padded = F.pad(signal, (pad_len, 0), mode='circular')

    return F.conv1d(padded, kernel)


def modwt_decompose(signal: torch.Tensor, num_scales: int,
                    wavelet: str = 'haar') -> list[torch.Tensor]:
    """
    执行 MODWT 多尺度分解。

    Args:
        signal:     (batch, N) 输入一维信号
        num_scales: 分解层数 J
        wavelet:    'haar' 或 'db4'

    Returns:
        coeffs: 长度为 (num_scales + 1) 的列表
                [d_1, d_2, ..., d_J, a_J]
                每个元素形状 (batch, N)
                d_j = 第 j 层细节系数, a_J = 最粗尺度近似系数
    """
    if wavelet == 'haar':
        h, g = _haar_filters()
    elif wavelet == 'db4':
        h, g = _db4_filters()
    else:
        raise ValueError(f"不支持的小波基: {wavelet}, 可选: 'haar', 'db4'")

    h = h.to(signal.device, signal.dtype)
    g = g.to(signal.device, signal.dtype)

    # (batch, N) → (batch, 1, N)
    approx = signal.unsqueeze(1)
    details = []

    for j in range(num_scales):
        d_j = _modwt_circular_conv(approx, g, scale=j)   # 细节
        approx = _modwt_circular_conv(approx, h, scale=j) # 近似
        details.append(d_j.squeeze(1))  # (batch, N)

    details.append(approx.squeeze(1))  # 最粗尺度近似 a_J
    return details  # [d_1, ..., d_J, a_J]


# ============================================================
#  STE 泊松采样（可反向传播）
# ============================================================

class _PoissonSTE(torch.autograd.Function):
    """Straight-Through Estimator for Poisson spike sampling."""

    @staticmethod
    def forward(ctx, rates: torch.Tensor) -> torch.Tensor:
        # 泊松采样: 发放概率 = rates, 二值输出
        spikes = (torch.rand_like(rates) < rates).float()
        return spikes

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # STE: 梯度直通
        return grad_output


poisson_spike_ste = _PoissonSTE.apply


# ============================================================
#  MODWTEncoder: 完整编码模块
# ============================================================

class MODWTEncoder(nn.Module):
    """
    MODWT 多尺度小波编码 + 泊松脉冲前端

    流程:
      1. 输入 x: (batch, in_features) 视为一维信号
      2. MODWT 分解为 num_scales 层细节 + 1 层近似 → (num_scales+1) 个 (batch, in_features) 系数
      3. 每层系数经可学习线性映射 → (batch, out_features)
      4. 多尺度融合: concat → linear → (batch, out_features)
      5. sigmoid → 泊松 STE 采样 → 脉冲输出

    接口与原版完全兼容:
        MODWTEncoder(in_features, out_features, num_scales=3)
        forward(x) → spikes: (batch, out_features)
    """

    def __init__(self, in_features: int, out_features: int,
                 num_scales: int = 3, wavelet: str = 'haar'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_scales = num_scales
        self.wavelet = wavelet
        self.num_bands = num_scales + 1  # J 层细节 + 1 层近似

        # 每个频带的可学习投影 (系数 → 隐层)
        self.band_projections = nn.ModuleList([
            nn.Linear(in_features, out_features)
            for _ in range(self.num_bands)
        ])

        # 多尺度融合层
        self.fusion = nn.Linear(out_features * self.num_bands, out_features)

        # 可学习的尺度注意力权重（自动学习各频带重要性）
        self.scale_logits = nn.Parameter(torch.zeros(self.num_bands))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_features) 连续输入信号

        Returns:
            spikes: (batch_size, out_features) 泊松脉冲 {0, 1}
        """
        # 1. MODWT 多尺度分解
        # coeffs: list of (batch, in_features), 长度 = num_scales + 1
        coeffs = modwt_decompose(x, self.num_scales, self.wavelet)

        # 2. 尺度注意力加权
        scale_weights = torch.softmax(self.scale_logits, dim=0)

        # 3. 各频带投影 + 加权
        band_features = []
        for i, (coeff, proj) in enumerate(zip(coeffs, self.band_projections)):
            feat = torch.relu(proj(coeff)) * scale_weights[i]
            band_features.append(feat)

        # 4. 融合: concat → linear
        fused = self.fusion(torch.cat(band_features, dim=-1))

        # 5. 转为发放率 [0, 1] → 泊松采样
        rates = torch.sigmoid(fused)

        if self.training:
            spikes = poisson_spike_ste(rates)
        else:
            # 推理时使用确定性阈值，避免随机性
            spikes = (rates > 0.5).float()

        return spikes
