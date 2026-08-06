import torch
import torch.nn as nn


class MODWTEncoder(nn.Module):
    """
    多尺度可学习编码与泊松编码前端 (Multi-Scale Learnable Encoder & Poisson)

    注意: 本模块为"多尺度线性提取 + 可学习阈值 + 泊松编码"的可学习前端,
    并非严格意义的 MODWT (最大重叠离散小波变换) —— 真实小波分解需要
    时间维输入, 而当前数据管线输入为单帧特征 (batch, input_dim), 无时间轴
    可做多分辨率分解。若未来引入时间序列输入, 可在此处前置小波滤波器组
    (db2/db4, 预留扩展点)。
    """
    def __init__(self, in_features: int, out_features: int, num_scales: int = 3,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_scales = num_scales

        self.scale_extractors = nn.ModuleList([
            nn.Linear(in_features, out_features, device=device, dtype=dtype)
            for _ in range(num_scales)
        ])
        # 注: 各尺度保持独立 Linear, 不合并为块对角单 GEMM —— 块对角合并会使
        # 参数量 ×num_scales 且引入零块, 改变初始化分布与训练行为, 收益有限。

        self.fusion = nn.Linear(out_features * num_scales, out_features,
                                device=device, dtype=dtype)

        # 可学习阈值偏置: 初始 1.0 压低基线发放率 (sigmoid 中心移至 1,
        # 默认发放率由 ~50% 降至 ~27%), 使编码层脉冲能耗进入可控范围;
        # 训练中自适应 (阈值参与 STE 梯度, sigmoid 输入减阈值即可, 可微)。
        self.threshold = nn.Parameter(
            torch.full((1,), 1.0, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_features)
        返回泊松脉冲: (batch_size, out_features)
        使用 STE (直通估计器) 保持梯度流通
        """
        features = []
        for i in range(self.num_scales):
            features.append(torch.relu(self.scale_extractors[i](x)))

        fused = self.fusion(torch.cat(features, dim=-1))

        # 减去可学习阈值, 压低基线发放率
        rates = torch.sigmoid(fused - self.threshold)

        prob = torch.rand_like(rates)
        # STE 硬脉冲: `.to(rates.dtype)` 避免 autocast 下 bf16 被提升为 fp32;
        # STE 直通: 硬脉冲 + 软-硬残差 (spk_soft - spk_soft.detach()),
        # 等价于 rates + (spk_hard - rates).detach(), 保持梯度从 rates 流通。
        spk_hard = (prob < rates).to(rates.dtype)
        spk_soft = rates
        spikes = spk_hard + (spk_soft - spk_soft.detach())
        return spikes
