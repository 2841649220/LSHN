"""
动态扩容输出解码层 (Dynamic Expansion Head)
白皮书 §4.1 / 架构文档 §3.4: 持续学习遇到新类别时自动增加输出神经元。

扩容策略 (单参数重建 + add_param_group):
- 扩容时重建 Parameter (拼接旧权重副本 + 新行), 返回 [weight, bias] 供调用方
  `optimizer.add_param_group` 注册。重建后新参数对象进入优化器时 Adam 矩
  从零开始 —— 实测 (合成数据 5 任务) 这构成任务边界的"优化器热启动":
  新任务梯度对所有类别行 (含旧行) 施加正常化的初始大步, 梯度流动顺畅,
  新任务学习速度显著快于"旧行保留旧 Adam 矩"的分块方案 (分块方案实测
  新任务准确率 100% → ~52%, 因旧行矩的梯度尺度与当前任务不匹配)。
  注意: 旧 Parameter 对象 (重建前) 仍留在原 param_group 中, 但已脱离
  计算图: backward 不为其产生 grad, zero_grad/step 均为 no-op, 无副作用。
- 防遗忘的"旧权重冻结" (架构文档 §3.4) 当前未落地: 实测冻结旧行会使
  新任务学习速度大幅下降 (共享皮层表征下稳定性-可塑性难以兼得),
  属研究级开放问题, 见 docs/fix_plan_260806.md 与架构文档的已知限制。
"""
import torch
import torch.nn as nn
from typing import List


class DynamicExpansionHead(nn.Module):
    """
    动态扩容输出解码层

    扩容契约:
        expand(n) -> [weight, bias] (当前完整参数对象)
        n <= 0 时返回 [] (no-op)
    """

    def __init__(self, in_features: int, initial_classes: int = 2,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.num_classes = initial_classes

        # 初始行 std 0.1 (与 v0.1.0 一致): 初始 logits 有足够量级驱动
        # 学习; 扩容新行 std 0.01 (架构文档 W_new ~ N(0, 0.01))。
        self.weight = nn.Parameter(
            torch.randn(initial_classes, in_features, device=device, dtype=dtype) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(initial_classes, device=device, dtype=dtype))

    def expand(self, num_new_classes: int) -> List[nn.Parameter]:
        """
        扩容输出头, 追加 num_new_classes 个新类别行。

        返回: [self.weight, self.bias] 当前完整参数 (调用方 add_param_group
            注册; 重建后新参数对象 Adam 矩从零开始 —— 任务边界热启动,
            见模块 docstring)。

        注意: 必须重建 nn.Parameter 对象, 不能重赋 `.data` —— 直接替换
        `.data` 会使旧 autograd 图保存的张量形状失效, 扩容后任何一次
        backward 都会报 `AddmmBackward0 invalid gradient`。

        新行初始化 std=0.01 (架构文档 §3.4: W_new ~ N(0, 0.01)),
        bias 新行为 0。
        """
        if num_new_classes <= 0:
            return []
        self.num_classes += num_new_classes

        device = self.weight.device
        dtype = self.weight.dtype

        with torch.no_grad():
            new_weight = torch.randn(
                num_new_classes, self.in_features, device=device, dtype=dtype
            ) * 0.01
            new_bias = torch.zeros(num_new_classes, device=device, dtype=dtype)

        # 旧权重值保留 (副本), 只追加新行
        self.weight = nn.Parameter(
            torch.cat([self.weight.detach(), new_weight], dim=0)
        )
        self.bias = nn.Parameter(
            torch.cat([self.bias.detach(), new_bias], dim=0)
        )
        return [self.weight, self.bias]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        解码脉冲率为类别 logits (或直接解码脉冲)
        """
        return torch.nn.functional.linear(x, self.weight, self.bias)
