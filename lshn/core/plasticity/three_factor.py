import torch
import torch.nn as nn
from typing import Optional

class ThreeFactorPlasticity(nn.Module):
    """
    三因素可塑性与预测误差反向脉冲学习规则
    1. Pre-synaptic trace (前突触)
    2. Post-synaptic trace (后突触)
    3. Global/Local Error Neuromodulator (第三因子: 神经调质/反向误差脉冲)
    """
    def __init__(self, learning_rate: float = 0.01, trace_decay: float = 0.9,
                 importance_lambda: float = 0.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        # 结构重要性保护强度 (EWC-lite, 白皮书 §3.4 "结构版 EWC"):
        # 0.0 = 关闭 (所有边同塑性); >0 时高重要性 (旧任务已固化) 边的
        # 三因素更新按 1/(1+λ·imp) 衰减, 保护旧任务结构不被新任务覆盖。
        self.importance_lambda = importance_lambda

    def forward(self, w_hat: torch.Tensor, e_trace: torch.Tensor,
                error_spk: torch.Tensor, neuromodulator: Optional[torch.Tensor] = None,
                plasticity_gate: float = 1.0,
                importance: Optional[torch.Tensor] = None):
        """
        w_hat: 当前快权重
        e_trace: 资格迹 (由于共发放累积得到)
        error_spk: 泊松编码的误差反向脉冲 (局部的或自顶向下的)
        neuromodulator: 全局多巴胺/调节信号 (如 DA_signal in [-1, 1])
        plasticity_gate: 星形胶质全局可塑性门控 [0, 1] (默认 1.0 = 不门控)
        importance: (max_edges,) 归一化到 [0,1] 的边重要性 (可选; 提供时
            高重要性边 Δw 按 1/(1+λ_imp·importance) 衰减 —— EWC-lite)
        """
        # Eq: \Delta w_hat = \eta * e_trace * error_spk * neuromodulator * gate
        #
        # 误差符号约定 (与 PoissonErrorEncoder 一致):
        #   error_spk > 0 = 欠预测 (target > pred) → Δw > 0 使权重增大、
        #     输出逼近目标;
        #   error_spk < 0 = 过预测 → Δw < 0 使权重减小。
        # 三因素因此构成负反馈环 (而非放大误差的正反馈)。

        mod_factor = 1.0
        if neuromodulator is not None:
            mod_factor = neuromodulator

        # 实际操作中，e_trace 维度需与 w_hat 对齐
        # 这里简化为直接广播乘积
        delta_w = self.learning_rate * e_trace * error_spk * mod_factor * plasticity_gate

        # EWC-lite: 重要性保护。高重要性边 (旧任务已固化的结构) 的
        # 可塑性按 1/(1+λ·imp) 衰减 —— 新任务学习主要发生在低重要性
        # (未固化) 边上, 旧任务结构不被覆盖 (白皮书 §3.4 结构版 EWC)。
        if importance is not None and self.importance_lambda > 0.0:
            delta_w = delta_w / (1.0 + self.importance_lambda * importance)

        # 手动更新快权重 (绕过 autograd; 梯度路径由 STE 前向提供)。
        # 注意: 该更新发生在 backward 之前, 梯度相对前向评估点存在
        # O(Δw) 的偏差 —— 这是"在线可塑性 + 批内 BPTT"混合训练的固有属性,
        # Δw 有界 (w_hat clamp ±1) 时偏差有界, 属可接受的近似。
        with torch.no_grad():
            w_hat.data.add_(delta_w)
            # w_hat 有界更新, 与类声明 w_hat ∈ [-1, 1] 一致,
            # 防止 e_trace 正反馈下无界爆炸 (实测曾达 2 万量级)。
            # clamp 同时把误差反馈环的更新幅度限制在有界范围:
            # 在线可塑性 + 批内 BPTT 混合训练下, 权重有界即保证
            # 前向评估点与梯度点之间的 O(Δw) 偏差有界, 误差信号
            # 不会被持续放大而失稳。
            w_hat.data.clamp_(-1.0, 1.0)

        return w_hat

class PoissonErrorEncoder(nn.Module):
    """
    将预测误差编码为泊松脉冲序列 (反向信息流)

    误差符号: error = target - pred (正 = 欠预测)。
      正误差 → 正脉冲 → 三因素 Δw = +η·e_trace·error_spk·... 上调权重,
      使输出逼近目标; 负误差 (过预测) 下调权重 —— 构成负反馈, 而非
      放大误差的正反馈。
    幅值: 泊松发放率 f_max·min(1, |error|·precision) 编码误差大小。
    泊松采样不可微, 整个编码在 no_grad 下进行, 不构建无用 autograd 图。
    """
    def __init__(self, f_max: float = 1.0):
        super().__init__()
        self.f_max = f_max

    def forward(self, pred: torch.Tensor, target: torch.Tensor, precision: float = 1.0):
        """
        pred: 预测脉冲率或膜电位
        target: 目标
        precision: 精度矩阵(或标量)

        返回: error_spk ∈ {-1, 0, +1} (正 = 欠预测, 方向沿 target - pred)
        """
        # 泊松采样不可微, 采样与符号均无需梯度, 全程 no_grad
        with torch.no_grad():
            # 误差符号: 正误差 = 欠预测 (target > pred) → 权重增大 → 输出逼近目标。
            # (符号修正: 原实现 pred - target 使 Δw 与误差同号放大 ——
            # 过预测→权重↑→输出↑→误差更大, 是正反馈; 现改为 target - pred)
            error = target - pred
            # 零基准泊松率: freq = f_max · min(1, |error|·precision)
            # 原实现 sigmoid(|error|·precision) ∈ [0.5, 1), 微小误差也以 ~50% 概率
            # 发放 ±1 脉冲, 误差信号被符号主导、与误差大小脱钩;
            # 零基准使 E[error_spk] ∝ error, 误差信号随误差大小平滑缩放。
            freq = self.f_max * torch.clamp(torch.abs(error) * precision, max=1.0)
            # 泊松采样
            prob = torch.rand_like(freq)
            error_spk = (prob < freq).float() * torch.sign(error)
            return error_spk
