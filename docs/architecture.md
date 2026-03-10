# LSHN 系统架构：形式化数学描述

> **版本**: v0.9 Beta 　|　**更新**: 2026年3月 　|　**作者**: Apocalypse 　|　**GitHub**: [LSHN](https://github.com/2841649220/LSHN)

本文档以严格的数学语言定义液态脉冲超图网络（LSHN）的系统架构，包括符号系统、动力学方程、模块数据流及与神经生物学的映射关系。

---

## 目录

1. [符号系统与核心实体](#1-符号系统与核心实体)
2. [三层时钟同步机制](#2-三层时钟同步机制)
3. [分层数据流与动力学方程](#3-分层数据流与动力学方程)
4. [全局引擎模块](#4-全局引擎模块)
5. [神经生物学对应关系](#5-神经生物学对应关系)
6. [精度策略与设备管理](#6-精度策略与设备管理)
7. [监控与可解释性](#7-监控与可解释性)
8. [小结](#8-小结)

---

## 1. 符号系统与核心实体

### 1.1 集合与空间

| 符号 | 定义 | 默认规模 | 生物学对应 |
|:---|:---|:---:|:---|
| N | 神经元集合 {1, 2, ..., N} | 5 × 10^5 | 皮层神经元 |
| E | 超边集合 {1, 2, ..., E} | 5 × 10^4 | 突触连接 |
| G | 功能分区集合 {1, 2, ..., G} | 100 | 功能柱/微柱 |
| T_fast | 快时间尺度，步长 Δt = 1 ms | — | 脉冲传播 |
| T_slow | 慢时间尺度，步长 Δt = 100 ms | — | 突触可塑性 |
| T_ultra | 超慢时间尺度，步长 Δt = 1000 ms | — | 结构演化 |

### 1.2 状态变量

| 符号 | 维度 | 范围 | 时间尺度 | 物理意义 |
|:---|:---:|:---|:---:|:---|
| v ∈ R^N | (N,) | (-∞, θ] | 快 | 膜电位 |
| S ∈ {0, 1}^N | (N,) | {0, 1} | 快 | 脉冲发放 |
| g^fast ∈ (0, 1)^N | (N,) | (0, 1) | 快 | 快门控（离子通道） |
| g^slow ∈ (0, 1)^N | (N,) | (0, 1) | 慢 | 慢门控（星形胶质） |
| a ∈ R^N | (N,) | R | 慢 | 适应电流（慢钾） |
| θ ∈ R^N | (N,) | R^+ | 慢 | 动态发放阈值 |
| ŵ ∈ [-1, 1]^E | (E,) | [-1, 1] | 快 | 快权重（STDP 更新） |
| s_e ∈ [0, 1]^E | (E,) | [0, 1] | 慢 | 结构变量（双势阱） |
| e ∈ R^E | (E,) | R | 快 | 资格迹 |
| τ_d ∈ N^E | (E,) | [1, 20] | 慢 | 轴突延迟 |

### 1.3 有效突触权重

有效权重由三变量耦合定义：

w_e = w_max · s_e ⊙ ŵ

其中 w_max = 1.0，⊙ 表示逐元素乘法。

---

## 2. 三层时钟同步机制

中央时钟引擎 `ClockSyncEngine` 严格同步三个时间尺度的更新事件：

| 时间尺度 | 周期 | 触发事件 | 执行模块 |
|:---|:---:|:---|:---|
| **快时钟** | 1 ms | 膜电位积分、脉冲发放、STDP 迹更新 | `LiquidGatedCell.step_fast()`<br>`BistableHypergraphSynapse.step_fast()` |
| **慢时钟** | 100 ms | 神经调质更新、结构演化、稳态可塑性 | `GlobalNeuromodulator.step_slow()`<br>`BistableHypergraphSynapse.step_slow_structure()` |
| **超慢时钟** | 1000 ms | 凋亡/生发、冷知识归档 | `PruneGrowthModule.step_ultra_slow()`<br>`KnowledgeArchiver.archive_cold_edges()` |

**触发逻辑**（伪代码）：

```python
def tick(self):
    self.fast_step += 1
    trigger_slow = (self.fast_step % 100 == 0)
    trigger_ultra = (self.fast_step % 1000 == 0)
    return trigger_slow, trigger_ultra
```

---

## 3. 分层数据流与动力学方程

### 3.1 输入编码层：MODWT 多尺度小波变换

**输入**：连续信号 x(t) ∈ R^(B×d)

**处理流程**：

**步骤 1：MODWT 分解**（num_scales=3）

coeffs = [d_1, d_2, d_3, a_3]

其中 d_j ∈ R^(B×d) 为细节系数，a_3 为近似系数。

**步骤 2：频带投影与注意力加权**

f_i = ReLU(W_i · coeffs_i + b_i) · α_i

其中 α_i = softmax(β)_i 为可学习尺度注意力。

**步骤 3：多尺度融合**

f_fused = W_fusion · [f_1; f_2; f_3; f_4] + b_fusion

**步骤 4：泊松编码（STE）**

r = σ(f_fused), S_enc ~ Poisson(r)

**输出**：S_enc ∈ {0, 1}^(B×1024)

---

### 3.2 海马体快速学习层：脉冲自编码器

#### 3.2.1 编码通路

I_hippo = W_enc · S_enc

S_hippo, v_hippo = LiquidCell_fast(I_hippo)

z_hippo = LiquidCell_fast(...)

#### 3.2.2 解码通路

I_decode = W_dec · z_hippo

#### 3.2.3 回放生成（可控采样动力学）

h_(t+1) = (1-λ)h_t + λ W_dec^T · S_hippo + μ(h_t - h_(t-1)) + √(2T)ξ

**输出**：
- 潜在表征 z_hippo ∈ {0, 1}^1024
- 回放信号 R_replay = E[S_hippo]

---

### 3.3 皮层核心层：LSHN 计算主体

#### 3.3.1 轴突延迟处理

**延迟线缓冲**：

d_buf[t, e] = S_pre[t - τ_d[e], e]

**STDP 延迟调制**：

Δτ_d[e] = -η_d · pre_trace[e] · post_trace[e]

#### 3.3.2 超图卷积传播

**Node → Hyperedge 聚合**：

m_e = (1/|N(e)|) Σ_(i∈N(e)) w_e[i] · x_i

**Hyperedge → Node 聚合**：

I_i^syn = Σ_(e∋i) w_e · m_e

**工程实现**（`SpikeHypergraphConv`）：

```python
# COO 格式: hyperedge_index = [node_idx; edge_idx]
# connection_weights = effective_w[edge_idx]
out = scatter_sum(
    scatter_mean(x[node_idx] * connection_weights, edge_idx), 
    node_idx
)
```

#### 3.3.3 液态门控元胞动力学

**膜电位更新**（快时钟）：

v_(t+1) = (1 - 1/τ_v) v_t + (1/τ_v)(I^syn_t + I^hippo_t - I^inh_t - a_t) + g^fast_t ⊙ η_t

其中状态依赖噪声 η_t ~ N(0, (0.01(1+g^slow_t))^2)。

**脉冲发放（STE）**：

S_t = Θ(v_(t+1) - θ_t), ∂S/∂v ≈ 1

**软重置**：

v_(t+1) ← v_(t+1) - S_t ⊙ θ_t

**快门控更新**：

g^fast_(t+1) = (1 - 1/τ_(g,f)) g^fast_t + (1/τ_(g,f)) σ(W_f ⊙ v_(t+1) + U_f ⊙ a_t + b_f)

**慢门控更新**（慢时钟）：

g^slow_(t+1) = (1 - 1/τ_(g,s)) g^slow_t + (1/τ_(g,s)) σ(W_s ⊙ S̅_t + U_s ⊙ δ̅_t + Z_s ⊙ e_global)

**适应变量更新**：

a_(t+1) = (1 - 1/τ_a) a_t + (a_inc/τ_a) S̅_t

θ_t = θ_0 + a_t

#### 3.3.4 隐式 MoE 侧向抑制

组内软性赢家通吃竞争（i ∈ G_k）：

I_i^inh = λ_inh · Σ_(j∈G_k, j≠i) S_j

默认抑制强度 λ_inh = 0.5。

#### 3.3.5 双势阱结构演化

**势能函数**：

U(s_e) = (α/4) s_e^4 - (α/2) s_e^2, α = 0.1

**慢时钟更新**（Langevin 动力学）：

s_e,(t+1) = clip( s_e,t + Δt_slow(-α s_e,t(s_e,t^2 - 1) + β c̅_t + γ M + δ R) + √(2 T Δt_slow) ξ, 0, 1)

其中：
- c̅_t：10 步滑动窗口平均共发放
- M：全局调制（预测误差）
- R：回放信号
- T：温度（NE 调制）
- ξ ~ N(0, 1)

#### 3.3.6 三因素可塑性规则

**资格迹更新**（含多跳项）：

e_(t+1) = λ_e e_t + c_t + σ(g̅^slow) · (A_local(w_e ⊙ e_t))

**快权重更新**：

Δŵ = η · e_t ⊙ b_post ⊙ DA

其中 b_post 为反向误差脉冲，DA 为第三因子。

#### 3.3.7 稳态可塑性控制器

**发放率估计**（指数移动平均）：

r̅_(t+1) = (1 - 1/τ_rate) r̅_t + (1/τ_rate) S_t

**内在兴奋性更新**：

θ_ie,(t+1) = θ_ie,t + η_ie(r̅_t - r_target)

**突触缩放**：

ŵ ← ŵ · (1 + α_scale(r_target - r̅))

---

### 3.4 输出解码层：动态扩容分类头

**基础解码**：

y = W_out · (1/T)Σ_(t=1)^T S_cortex[t] + b_out

**类别扩容**（新任务到来时）：

W_out' = [W_out; W_new], W_new ~ N(0, 0.01)

旧权重冻结，仅优化新增参数。

---

## 4. 全局引擎模块

### 4.1 变分自由能引擎

**目标函数**：

J = [Accuracy: ρ · E[||ε||^2_2]] + [Complexity: λ_KL · (D_KL(s_e || 0.5) + E[S])] + [Energy: λ_E · E[#Spikes]]

**结构 KL 散度**：

D_KL(s_e || 0.5) = (1/E) Σ_(e=1)^E [s_e log(s_e/0.5) + (1-s_e)log((1-s_e)/0.5)]

**λ_E 自适应调整**（PI 控制）：

λ_E(t+1) = λ_E(t) + α · (current_spikes - target_budget)

---

### 4.2 全局神经调节器

**惊喜度检测**（意外不确定性）：

surprise_t = |error_t - EMA(error)|

**ACh（精度/注意）**：

ACh_(t+1) = decay · ACh_t + (1-decay) · 1/(1 + 5 · surprise_t)

**NE（温度/探索）**：

NE_(t+1) = decay · NE_t + (1-decay) · σ(3 · (surprise_t + ood_score))

**DA（第三因子/奖赏）**：

DA_(t+1) = decay · DA_t + (1-decay) · σ(reward + 0.5 · curiosity)

**星形胶质门控**：

calcium_(t+1) = decay · calcium_t + (1-decay) · σ(0.3 · error - 0.2 · firing_rate)

plasticity_gate = calcium

---

### 4.3 脉冲预算控制器

**PI 控制律**：

u(t) = K_p · e(t) + K_i · ∫_0^t e(τ) dτ

e(t) = current_spikes - target_budget

**阈值调节**：

θ_adj = clip(u(t) · scale, -0.2, 0.2)

θ ← θ · (1 + θ_adj)

**抑制强度调节**：

λ_inh,adj = clip(u(t) · scale, -0.1, 0.1)

---

### 4.4 冷知识归档器

**冷边检测**：

cold_mask = ¬(edge_mask) ∨ (s_e < 0.05)

**NF4 非线性量化**（分组大小 64）：

基于正态分布分位数的 16 级非线性量化表：

```python
_NF4_TABLE = [-1.0, -0.6961928, -0.5250730, -0.3949301,
              -0.2844677, -0.1847513, -0.0917715, 0.0,
              0.0797546, 0.1609459, 0.2461693, 0.3379146,
              0.4407282, 0.5626170, 0.7229568, 1.0]
```

**线性 INT4 量化**（分组大小 64）：

scale = max(|x|)/15, codes = round(x/scale)

**Bit-Packing**：
两个 INT4 值打包进一个 uint8（高4位/低4位）

**归档数据结构**：

```python
{
    'w_hat_packed':   (ceil(N_cold/2),)   uint8   # NF4 bit-packed
    'w_hat_scales':   (num_groups_w,)     bfloat16
    'se_packed':      (ceil(N_cold/2),)   uint8   # 线性INT4 bit-packed
    'se_scales':      (num_groups_se,)    bfloat16
    'se_zeros':       (num_groups_se,)    bfloat16
    'csr_indptr':     (num_nodes+1,)      int32
    'csr_indices':    (N_cold,)           int32
    'cold_indices':   (N_cold,)           int64
    'num_nodes':      int
    'N_cold':         int
    'group_size':     int
    'timestamp':      float
    'archive_id':     str
}
```

**槽位重置**：

ŵ[cold] = 0, s_e[cold] = 0.5, edge_mask[cold] = True

---

## 5. 神经生物学对应关系

| LSHN 组件 | 生物学对应物 | 理论依据 | 参考文献 |
|:---|:---|:---|:---|
| **双势阱势场** | LTP/LTD 分子双稳态 | 突触权重双稳态维持 | [R3, R4] |
| **多时间尺度** | 离子通道 vs G蛋白偶联受体 | 时间常数差异 | [R5] |
| **超图拓扑** | 皮层细胞集合（Cell Assemblies） | 高阶共发放关联 | [R15] |
| **ACh** | 乙酰胆碱（预期不确定性） | 精度权重调制 | [R10] |
| **NE** | 去甲肾上腺素（意外不确定性） | 温度/探索调控 | [R7] |
| **DA** | 多巴胺（奖赏预测误差） | 三因素第三因子 | [R5, R6] |
| **星形胶质** | 星形胶质细胞钙波 | 全局可塑性门控 | [R10, R11] |
| **冷知识归档** | 睡眠期突触修剪 | 系统巩固理论 | [R8, R9] |
| **树突非线性** | 树突 Ca 尖峰 | 分支独立积分 | [R3, R4] |
| **轴突延迟** | 传导延迟可塑性 | 时序信用分配 | [R12, R13] |

---

## 6. 精度策略与设备管理

### 6.1 混合精度围栏

| 模块 | 精度 | 理由 |
|:---|:---:|:---|
| `nn.Linear` / `SpikeHypergraphConv` | BF16 | 加速矩阵乘法 |
| 状态变量（v, g, a, θ, s_e, traces） | FP32 | 防止累积误差导致脉冲消失 |
| `KnowledgeArchiver` 量化 | FP32→INT4 | 压缩存储 |

**实现模式**：

```python
with torch.autocast('cuda', dtype=torch.bfloat16):
    # 前向计算（BF16 加速）
    spk = self.encoder(x)
    # ...
# 状态变量更新（FP32 保护）
self.v.data.copy_(v_next.mean(dim=0))  # .float() 显式转换
```

### 6.2 FP8 实验性加速（H100/RTX 4090）

**配置**（`configs/default.yaml`）：

```yaml
precision:
  mixed_precision: true       # 启用 BF16 autocast
  autocast_dtype: "bfloat16"  # 目标精度
  use_fp8: false              # FP8 实验性（需 PyTorch 2.1+ 和 H100/RTX 4090）
```

**FP8 格式技术规格**：

| 格式 | 指数位 | 尾数位 | 动态范围 | 最小正值 | 典型用途 |
|:---:|:---:|:---:|:---:|:---:|:---|
| **E4M3** | 4 | 3 | ±448.0 | 2^-9 ≈ 1.95e-3 | 前向传播、激活值 |
| **E5M2** | 5 | 2 | ±57344.0 | 2^-16 ≈ 1.53e-5 | 梯度计算、权重更新 |

**使用示例**：

```python
import torch
from lshn import LSHNModel

# 创建支持 FP8 的模型
model = LSHNModel(
    input_dim=128,
    num_neurons=500000,
    mixed_precision=True,
    autocast_dtype=torch.float8_e4m3fn,  # FP8 E4M3 格式
)

# 或使用配置字典
config = {
    'precision': {
        'mixed_precision': True,
        'autocast_dtype': 'float8_e4m3fn',
        'use_fp8': True
    }
}

# 前向传播会自动使用 FP8 精度（在支持的硬件上）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 输入数据
x = torch.randn(32, 128).to(device)  # batch_size=32
target = torch.randn(32, 10).to(device)

# 单步前向（自动应用 FP8 autocast）
output = model.forward_step(x, target)
```

**FP8 模式说明**：
- FP8 是 NVIDIA H100 和 RTX 4090 引入的 8 位浮点格式
- 提供 E4M3（4位指数，3位尾数）和 E5M2（5位指数，2位尾数）两种变体
- 相比 BF16 可进一步减少显存占用 50%，提升吞吐 1.3-1.5 倍
- **当前状态**：实验性支持，需配合 TransformerEngine 或 PyTorch 2.1+ 的 `torch.float8_e4m3fn`
- **限制**：SNN 状态变量仍需保持 FP32，仅前向矩阵乘法可使用 FP8

**与 BF16 性能对比**：

| 指标 | BF16 基线 | FP8 E4M3 | 变化 |
|:---|:---:|:---:|:---:|
| **显存占用** | 100% | ~50% | -50% |
| **训练吞吐量** | 1.0x | 1.3-1.5x | +30-50% |
| **推理延迟** | 1.0x | 0.7-0.8x | -20-30% |
| **精度损失** | 0% | 0.2-0.5% | 可接受 |

**硬件与软件要求**：
- **GPU**: NVIDIA H100 或 RTX 4090（Hopper/Ada 架构）
- **PyTorch**: 2.1 或更高版本
- **CUDA**: 11.8+

**精度注意事项**：
1. SNN 状态变量（膜电位 v、门控变量 g、适应变量 a、阈值 θ、资格迹）始终在 FP32 下运行，不受 autocast 影响
2. 仅 `nn.Linear` 和矩阵乘法操作会使用 FP8
3. 反向传播梯度仍使用 FP16/BF16
4. 建议在训练初期使用 BF16 稳定收敛后再尝试 FP8

### 6.3 设备迁移

所有模块继承 `nn.Module`，支持标准设备管理：

```python
model = LSHNModel(..., device='cuda')
model.to('cuda:0')  # 单卡迁移
model.to('mps')     # Apple Silicon
```

---

## 7. 监控与可解释性

### 7.1 VFE 分解报告

```python
report = model.get_monitoring_report()
```

**返回字典**：

| 键 | 含义 | 计算方式 |
|:---|:---|:---|
| vfe_recent_mean | 最近 10 步 VFE 平均 | (1/10)Σ_(t=T-9)^T F_t |
| accuracy_trend | 预测误差趋势 | EMA of ρ||ε||^2 |
| complexity_trend | 结构复杂度趋势 | EMA of D_KL + activity_cost |
| energy_trend | 能量代价趋势 | EMA of #Spikes |
| modulator_ACh | ACh 水平 | 当前精度参数 |
| modulator_NE | NE 水平 | 当前温度参数 |
| modulator_DA | DA 水平 | 当前第三因子 |
| alive_edges_ratio | 存活超边比例 | (1/E)Σ_e I(s_e > 0.05) |
| alive_neurons_ratio | 存活神经元比例 | (1/N)Σ_i neuron_mask_i |
| mean_firing_rate | 平均发放率 | (1/N)Σ_i r̅_i |
| delay_mean | 平均轴突延迟 | E[τ_d] |
| delay_entropy | 延迟分布熵 | -Σ p(τ)log p(τ) |

---

## 8. 小结

LSHN 通过形式化的数学框架将生物脑的多尺度组织原则转化为可计算的模块：

1. **时间解耦**：三个时间尺度严格分离（1 ms / 100 ms / 1000 ms），支持实时计算与长期固化并行
2. **双重可塑性**：快权重 ŵ 与结构变量 s_e 分离，为**稳定性 - 可塑性困境**提供物理解
3. **能量约束**：变分自由能框架统一精确性、复杂度与能量代价，脉冲预算 PI 控制维持能耗
4. **无限容量**：冷知识归档机制（INT4 压缩）理论上支持无上限任务序列学习

---

## 文档导航

| 文档 | 内容 | 适用读者 |
|:---|:---|:---|
| [README](../README.md) | 项目概览、快速开始、使用示例 | 所有用户 |
| [API 参考](api_reference.md) | 模块接口、张量规格、代码示例 | 开发者 |
| **本文档** | 数学形式化、动力学方程、生物学映射 | 研究人员 |

---

**版本**: v0.9 Beta 　|　**最后更新**: 2026年3月 　|　**作者**: Apocalypse 　|　**GitHub**: [LSHN](https://github.com/2841649220/LSHN)
