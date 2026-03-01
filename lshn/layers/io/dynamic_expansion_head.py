import torch
import torch.nn as nn
from typing import List

class DynamicExpansionHead(nn.Module):
    """
    动态扩容输出解码层 (Dynamic Expansion Decoding Head)
    当持续学习遇到新类别时，自动增加输出神经元并建立连接。
    """
    def __init__(self, in_features: int, initial_classes: int = 2):
        super().__init__()
        self.in_features = in_features
        self.num_classes = initial_classes
        
        # 使用 nn.Parameter 存储权重以便动态调整
        self.weight = nn.Parameter(torch.randn(initial_classes, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(initial_classes))
        
    def expand(self, num_new_classes: int) -> List[nn.Parameter]:
        """
        扩容输出头
        
        Returns:
            new_params: 新增的参数列表，调用方需要将其加入 optimizer:
                optimizer.add_param_group({'params': new_params, 'lr': lr})
        """
        old_classes = self.num_classes
        self.num_classes += num_new_classes
        
        device = self.weight.device
        dtype = self.weight.dtype
        
        new_weight = torch.randn(num_new_classes, self.in_features, device=device, dtype=dtype) * 0.1
        new_bias = torch.zeros(num_new_classes, device=device, dtype=dtype)
        
        # 拼接旧权重和新权重
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_weight], dim=0))
        self.bias = nn.Parameter(torch.cat([self.bias.data, new_bias], dim=0))
        
        # 返回新参数以便调用方加入 optimizer
        return [self.weight, self.bias]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        解码脉冲率为类别 logits (或直接解码脉冲)
        """
        return torch.nn.functional.linear(x, self.weight, self.bias)
