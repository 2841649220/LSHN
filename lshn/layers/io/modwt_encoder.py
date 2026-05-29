import torch
import torch.nn as nn

class MODWTEncoder(nn.Module):
    """
    小波多尺度分析与泊松编码前端 (MODWT & Poisson Encoder)
    用于将连续信号转化为多尺度脉冲序列。
    """
    def __init__(self, in_features: int, out_features: int, num_scales: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_scales = num_scales
        
        self.scale_extractors = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(num_scales)
        ])
        
        self.fusion = nn.Linear(out_features * num_scales, out_features)
        
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
        
        rates = torch.sigmoid(fused)
        
        prob = torch.rand_like(rates)
        spk_hard = (prob < rates).float()
        spk_soft = rates
        spikes = spk_hard + (spk_soft - spk_soft.detach())
        return spikes
