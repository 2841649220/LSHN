# LSHN 系统架构：形式化数学描述

> **版本**: v0.1.1 　|　**更新**: 2026年8月 　|　**作者**: Apocalypse 　|　**GitHub**: [LSHN](https://github.com/2841649220/LSHN)

> **实现偏差说明**: 本文档给出设计语义; 与 `lshn/` 实现的已知偏差
> 已在对应小节标注 (如 §3.3.3 适应项、§3.3.5 噪声标定、§3.4 扩容冻结)。

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
| $\mathcal{N}$ | 神经元集合 $\{1, 2, ..., N\}$ | $5 \times 10^5$ | 皮层神经元 |
| $\mathcal{E}$ | 超边集合 $\{1, 2, ..., E\}$ | $5 \times 10^4$ | 突触连接 |
| $\mathcal{G}$ | 功能分区集合 $\{1, 2, ..., G\}$ | 100 | 功能柱/微柱 |
| $\mathbb{T}_{\text{fast}}$ | 快时间尺度，步长 $\Delta t = 1$ ms | — | 脉冲传播 |
| $\mathbb{T}_{\text{slow}}$ | 慢时间尺度，步长 $\Delta t = 100$ ms | — | 突触可塑性 |
| $\mathbb{T}_{\text{ultra}}$ | 超慢时间尺度，步长 $\Delta t = 1000$ ms | — | 结构演化 |

### 1.2 状态变量

| 符号 | 维度 | 范围 | 时间尺度 | 物理意义 |
|:---|:---:|:---:|:---:|:---|
| $\mathbf{v} \in \mathbb{R}^N$ | $(N,)$ | $(-\infty, \theta]$ | 快 | 膜电位 |
| $\mathbf{S} \in \{0, 1\}^N$ | $(N,)$ | $\{0, 1\}$ | 快 | 脉冲发放 |
| $\mathbf{g}^{\text{fast}} \in (0, 1)^N$ | $(N,)$ | $(0, 1)$ | 快 | 快门控（离子通道） |
| $\mathbf{g}^{\text{slow}} \in (0, 1)^N$ | $(N,)$ | $(0, 1)$ | 慢 | 慢门控（星形胶质） |
| $\mathbf{a} \in \mathbb{R}^N$ | $(N,)$ | $\mathbb{R}$ | 慢 | 适应电流（慢钾） |
| $\boldsymbol{\theta} \in \mathbb{R}^N$ | $(N,)$ | $\mathbb{R}^+$ | 慢 | 动态发放阈值 |
| $\hat{\mathbf{w}} \in [-1, 1]^E$ | $(E,)$ | $[-1, 1]$ | 快 | 快权重（STDP 更新） |
| $\mathbf{s}_e \in [0, 1]^E$ | $(E,)$ | $[0, 1]$ | 慢 | 结构变量（双势阱） |
| $\mathbf{e} \in \mathbb{R}^E$ | $(E,)$ | $\mathbb{R}$ | 快 | 资格迹 |
| $\boldsymbol{\tau}_d \in \mathbb{N}^E$ | $(E,)$ | $[1, 20]$ | 慢 | 轴突延迟 |

### 1.3 有效突触权重

有效权重由三变量耦合定义：

$$\mathbf{w}_e = w_{\text{max}} \cdot \mathbf{s}_e \odot \hat{\mathbf{w}}$$

其中 $w_{\text{max}} = 1.0$，$\odot$ 表示逐元素乘法。

---

## 2. 三层时钟同步机制

中央时钟引擎 `ClockSyncEngine` 严格同步三个时间尺度的更新事件：

| 时间尺度 | 周期 | 触发事件 | 执行模块 |
|:---|:---:|:---|:---|
| **快时钟** | 1 ms | 膜电位积分、脉冲发放、STDP 迹更新 | `LiquidGatedCell.step_fast()`<br>`BistableHypergraphSynapse.step_fast()` |
| **慢时钟** | 100 ms | 神经调质更新、结构演化、稳态可塑性 | `GlobalNeuromodulator.step_slow()`<br>`BistableHypergraphSynapse.step_slow_structure()` |
| **超慢时钟** | 1000 ms | 凋亡/生发、冷知识归档 | `PruneGrowthModule.step_ultra_slow()`<br>`KnowledgeArchiver.archive_cold_edges()` |

**触发逻辑**（伪代码，周期从配置 `clocks` 读取，默认 1/100/1000ms）：

```python
def tick(self):
    self.fast_step += 1
    self.steps_since_slow += 1
    trigger_slow = (self.fast_step % (self.slow_ms // self.fast_ms) == 0)
    trigger_ultra = (self.fast_step % (self.ultra_slow_ms // self.fast_ms) == 0)
    if trigger_slow:
        self.slow_steps += 1
        self.steps_since_slow = 0   # 窗口长度供脉冲计数归一化
    return trigger_slow, trigger_ultra
```

---

## 3. 分层数据流与动力学方程

### 3.1 输入编码层：MODWT 多尺度小波变换

**输入**：连续信号 $\mathbf{x}(t) \in \mathbb{R}^{B \times d}$

**处理流程**：

**步骤 1：MODWT 分解**（`num_scales=3`）

$$\text{coeffs} = [\mathbf{d}_1, \mathbf{d}_2, \mathbf{d}_3, \mathbf{a}_3]$$

其中 $\mathbf{d}_j \in \mathbb{R}^{B \times d}$ 为细节系数，$\mathbf{a}_3$ 为近似系数。

**步骤 2：频带投影与注意力加权**

$$\mathbf{f}_i = \text{ReLU}(\mathbf{W}_i \cdot \text{coeffs}_i + \mathbf{b}_i) \cdot \alpha_i$$

其中 $\alpha_i = \text{softmax}(\beta)_i$ 为可学习尺度注意力。

**步骤 3：多尺度融合**

$$\mathbf{f}_{\text{fused}} = \mathbf{W}_{\text{fusion}} \cdot [\mathbf{f}_1; \mathbf{f}_2; \mathbf{f}_3; \mathbf{f}_4] + \mathbf{b}_{\text{fusion}}$$

**步骤 4：泊松编码（STE）**

$$\mathbf{r} = \sigma(\mathbf{f}_{\text{fused}}), \quad \mathbf{S}_{\text{enc}} \sim \text{Poisson}(\mathbf{r})$$

**输出**：$\mathbf{S}_{\text{enc}} \in \{0, 1\}^{B \times 1024}$

---

### 3.2 海马体快速学习层：脉冲自编码器

#### 3.2.1 编码通路

$$
\mathbf{I}_{\text{hippo}} = \mathbf{W}_{\text{enc}} \mathbf{S}_{\text{enc}}
$$

$$
\mathbf{S}_{\text{hippo}}, \mathbf{v}_{\text{hippo}} = \text{LiquidCell}_{\text{fast}}(\mathbf{I}_{\text{hippo}})
$$

$$
\mathbf{z}_{\text{hippo}} = \text{LiquidCell}_{\text{fast}}(\cdots)
$$

#### 3.2.2 解码通路

$$\mathbf{I}_{\text{decode}} = \mathbf{W}_{\text{dec}} \mathbf{z}_{\text{hippo}}$$

#### 3.2.3 回放生成（可控采样动力学）

**在线回放**（每慢时钟 100ms 触发，实现为泄漏-动量 + 模式吸引的混合动力学）：

$$\mathbf{h}_{t+1} = (1-\lambda_{\text{inj}})\mathbf{h}_t + \lambda_{\text{inj}} \mathbf{W}_{\text{dec}}^\top \bar{\mathbf{S}}_{\text{hippo}} - \lambda_{\text{leak}} \mathbf{h}_t + \mathbf{v}_{t+1}$$

$$\mathbf{v}_{t+1} = m \cdot \mathbf{v}_t + (1-m) \cdot \text{force}_t, \quad \text{force}_t = -\lambda_{\text{leak}} \mathbf{h}_t + \sqrt{2T}\xi$$

（$\lambda_{\text{inj}}=0.3$ 为模式吸引注入率，$\bar{\mathbf{S}}_{\text{hippo}}$ 为最近海马体隐层脉冲的批量均值 —— 回放状态被牵引到最近编码模式附近，修复"回放与记忆无关"的退化）

**离线回放**（每超慢时钟 1000ms 触发）：生成 K=5 步伪脉冲经 `hippo_to_cortex` 投射后传入皮层，巩固旧知识对应超边的共发放结构。

**输出**：
- 潜在表征 $\mathbf{z}_{\text{hippo}} \in \{0, 1\}^{\text{hidden\_dim}}$
- 回放信号 $R_{\text{replay}} = \mathbb{E}[\mathbf{S}_{\text{pseudo}}]$（固定阈值 0.5 伪脉冲的活动度，有真实动态范围）

**海马体重构训练**（实现补充）：解码器经重构损失训练（$\mathcal{L}_{\text{recon}} = \text{MSE}(\mathbf{W}_{\text{dec}}\mathbf{z}_{\text{hippo}}, \mathbf{S}_{\text{enc}})$，并入训练总损失，权重 `training.recon_weight`），使解码投影 $\mathbf{W}_{\text{dec}}$ 成为有意义的模式重构器。

---

### 3.3 皮层核心层：LSHN 计算主体

#### 3.3.1 轴突延迟处理

**延迟线缓冲**：

$$\mathbf{d}_{\text{buf}}[t, e] = \mathbf{S}_{\text{pre}}[t - \tau_d[e], e]$$

**STDP 延迟调制**：

$$\Delta \tau_d[e] = -\eta_d \cdot \text{pre\_trace}[e] \cdot \text{post\_trace}[e]$$

#### 3.3.2 超图卷积传播

**Node → Hyperedge 聚合**：

$$\mathbf{m}_{e} = \frac{1}{|\mathcal{N}(e)|} \sum_{i \in \mathcal{N}(e)} \mathbf{w}_e[i] \cdot \mathbf{x}_i$$

**Hyperedge → Node 聚合**：

$$\mathbf{I}_i^{\text{syn}} = \sum_{e \ni i} \mathbf{w}_e \cdot \mathbf{m}_e$$

**工程实现**（手动 scatter 消息传递，COO 格式: `hyperedge_index = [node_idx; edge_idx]`，源节点聚合 → 按连接数均值归一化 → × 有效权重）：

```python
# hyperedge_index: (2, N_conn) [src_nodes; edge_ids]
edge_out = scatter_add(x[src], edge_ids) / count[edge_ids] * effective_w[edge_ids]
```

**拓扑持久化**（实现补充）：默认拓扑（每条超边 K=8 个随机源节点）注册为持久 buffer 随检查点保存（固定种子 2026 生成）——训练/评估/续训连接一致，可复现。

#### 3.3.3 液态门控元胞动力学

**膜电位更新**（快时钟）：

$$\mathbf{v}_{t+1} = \left(1 - \frac{1}{\tau_v}\right) \mathbf{v}_t + \frac{1}{\tau_v} (\mathbf{I}^{\text{syn}}_t + \mathbf{I}^{\text{hippo}}_t - \mathbf{I}^{\text{inh}}_t - \mathbf{a}_t) + \mathbf{g}^{\text{fast}}_t \odot \boldsymbol{\eta}_t$$

其中状态依赖噪声 $\boldsymbol{\eta}_t \sim \mathcal{N}(0, (0.01(1+\mathbf{g}^{\text{slow}}_t))^2)$。

**实现偏差**: 适应项 $-\mathbf{a}_t$ 未直接进入膜电位方程 —— 适应变量折入动态阈值 $\theta_t = \theta_0 + \mathbf{a}_t$（定性等效：都提高发放难度；实现选择阈值折入以避免抑制电流的直流偏移）。输入电流经 `input_gain / std_ema` 自适应归一化（EMA 仅训练模式更新）。

**脉冲发放（STE）**：

$$\mathbf{S}_t = \Theta(\mathbf{v}_{t+1} - \boldsymbol{\theta}_t), \quad \frac{\partial \mathbf{S}}{\partial \mathbf{v}} \approx \mathbf{1}$$

**软重置**：

$$\mathbf{v}_{t+1} \leftarrow \mathbf{v}_{t+1} - \mathbf{S}_t \odot \boldsymbol{\theta}_t$$

**快门控更新**：

$$\mathbf{g}^{\text{fast}}_{t+1} = \left(1 - \frac{1}{\tau_{g,f}}\right) \mathbf{g}^{\text{fast}}_t + \frac{1}{\tau_{g,f}} \sigma(\mathbf{W}_f \odot \mathbf{v}_{t+1} + \mathbf{U}_f \odot \mathbf{a}_t + \mathbf{b}_f)$$

**慢门控更新**（慢时钟）：

$$\mathbf{g}^{\text{slow}}_{t+1} = \left(1 - \frac{1}{\tau_{g,s}}\right) \mathbf{g}^{\text{slow}}_t + \frac{1}{\tau_{g,s}} \sigma(\mathbf{W}_s \odot \bar{\mathbf{S}}_t + \mathbf{U}_s \odot \bar{\boldsymbol{\delta}}_t + \mathbf{Z}_s \odot \mathbf{e}_{\text{global}})$$

**适应变量更新**：

$$\mathbf{a}_{t+1} = \left(1 - \frac{1}{\tau_a}\right) \mathbf{a}_t + \frac{a_{\text{inc}}}{\tau_a} \bar{\mathbf{S}}_t$$

$$\boldsymbol{\theta}_t = \theta_0 + \mathbf{a}_t$$

#### 3.3.4 隐式 MoE 侧向抑制

组内软性赢家通吃竞争（$i \in \mathcal{G}_k$）：

$$I_i^{\text{inh}} = \lambda_{\text{inh}} \cdot \sum_{j \in \mathcal{G}_k, j \neq i} \mathbf{S}_j$$

默认抑制强度 $\lambda_{\text{inh}} = 0.5$。

#### 3.3.5 双势阱结构演化

**势能函数**：

$$U(s_e) = \frac{\alpha}{4} s_e^4 - \frac{\alpha}{2} s_e^2, \quad \alpha = 0.1$$

**慢时钟更新**（Langevin 动力学）：

$$\mathbf{s}_{e, t+1} = \text{clip}\left( \mathbf{s}_{e,t} + \Delta t_{\text{slow}} \left( -\alpha \mathbf{s}_{e,t} (\mathbf{s}_{e,t}^2 - 1) + \beta \bar{\mathbf{c}}_t + \gamma M + \delta R \right) + \sqrt{2 T \Delta t_{\text{slow}} \Delta U} \, \xi, 0, 1 \right)$$

其中：
- $\bar{\mathbf{c}}_t$：10 步滑动窗口平均共发放
- $M$：全局调制（预测误差）
- $R$：回放信号
- $T$：温度（NE 调制，clamp $[10^{-6}, 0.5]$）
- $\Delta U = \alpha/4$：势阱深度（噪声按势垒深度标定 —— 实现偏差修正：原公式 $\sqrt{2T\Delta t}$ 的噪声尺度是势垒深度的 ~12 倍，双稳性被噪声淹没；现按 $\sigma = \sqrt{2\Delta t \cdot T \cdot \Delta U}$ 标定）
- $\xi \sim \mathcal{N}(0, 1)$

#### 3.3.6 三因素可塑性规则

**资格迹更新**（含多跳项）：

$$\mathbf{e}_{t+1} = \text{clamp}\left( \lambda_e \mathbf{e}_t + \mathbf{c}_t + (1-\lambda_e) \cdot \sigma(\bar{g}^{\text{slow}}) \cdot (\mathbf{A}_{\text{local}} (\mathbf{w}_e \odot \mathbf{e}_t)), \pm 10 \right)$$

其中多跳项乘 $(1-\lambda_e)$ 保证压缩性（$\|\cdot\|$ 因子 ≤ 1，防资格迹指数发散），整体 clamp ±10 作为硬保险。$\mathbf{A}_{\text{local}}$ 为组内归一化邻接（实现为组均值 scatter，O(E)）。

**快权重更新**（误差符号约定: 正误差 = 欠预测）：

$$\Delta \hat{\mathbf{w}} = \eta \cdot \mathbf{e}_t \odot \mathbf{b}_{\text{post}} \odot \text{DA} \cdot \text{gate}\,/\,(1 + \lambda_{\text{imp}} \cdot \text{imp}_e)$$

其中 $\mathbf{b}_{\text{post}}$ 为反向误差脉冲（$\propto \mathbf{W}_{\text{out}}^\top(\text{target} - \text{pred})$，实现偏差修正：原实现符号相反构成正反馈，v0.1.1 修正为负反馈纠错），DA 为第三因子，gate 为星形胶质可塑性门控，$\text{imp}_e$ 为归一化边重要性（EWC-lite 结构保护，`importance_lambda>0` 时启用，默认关闭）。

**资格迹语义**：实现采用衰减而非"更新后重置"（白皮书 §3.5），$\lambda_e=0.9$ 的衰减即时间信用分配，效果等价且更稳定。

#### 3.3.7 稳态可塑性控制器

**发放率估计**（指数移动平均）：

$$\bar{\mathbf{r}}_{t+1} = \left(1 - \frac{1}{\tau_{\text{rate}}}\right) \bar{\mathbf{r}}_t + \frac{1}{\tau_{\text{rate}}} \mathbf{S}_t$$

**内在兴奋性更新**：

$$\theta_{\text{ie}, t+1} = \theta_{\text{ie}, t} + \eta_{\text{ie}} (\bar{\mathbf{r}}_t - r_{\text{target}})$$

**突触缩放**（实现为乘性幂缩放，与白皮书 §2.3 一致；clamp [0.5, 2.0]，死神经元关联边缩放强制 1.0）：

$$\hat{\mathbf{w}} \leftarrow \hat{\mathbf{w}} \cdot \left(\frac{r_{\text{target}}}{\bar{\mathbf{r}}}\right)^{\beta}$$

---

### 3.4 输出解码层：动态扩容分类头

**基础解码**：

$$\mathbf{y} = \mathbf{W}_{\text{out}} \cdot \frac{1}{T}\sum_{t=1}^T \mathbf{S}_{\text{cortex}}[t] + \mathbf{b}_{\text{out}}$$

**类别扩容**（新任务到来时）：

$$\mathbf{W}_{\text{out}}' = \begin{bmatrix} \mathbf{W}_{\text{out}} \\ \mathbf{W}_{\text{new}} \end{bmatrix}, \quad \mathbf{W}_{\text{new}} \sim \mathcal{N}(0, 0.01)$$

实现为 Parameter 重建（必须重建而非重赋 `.data`，否则 autograd 形状失效崩溃），`expand(n)` 返回新增参数由调用方 `add_param_group` 注册（旧参数 Adam 矩保留、脱离计算图后为 no-op，无需重建优化器）。

**实现偏差说明**: 文档所述"旧权重冻结，仅优化新增参数"当前**未落地**——实测冻结旧行使新任务学习速度大幅下降（共享皮层表征下稳定性-可塑性难以兼得，100% → ~52%）。v0.1.1 默认采用全塑性 + 扩容矩重置（"优化器热启动"），跨任务保留率提升为开放研究方向（见 api_reference.md §7）。

---

## 4. 全局引擎模块

### 4.1 变分自由能引擎

**目标函数**：

$$\mathcal{J} = \underbrace{\rho \cdot \mathbb{E}[\|\boldsymbol{\epsilon}\|^2_2]}_{\text{Accuracy}} + \lambda_{KL} \cdot \underbrace{\left( D_{KL}(\mathbf{s}_e \| 0.5) + \mathbb{E}[\bar{\mathbf{r}}] \right)}_{\text{Complexity}} + \lambda_E \cdot \underbrace{\mathbb{E}[\#\text{Spikes}]}_{\text{Energy}}$$

其中复杂度项的活动度取皮层平均发放率 $\mathbb{E}[\bar{\mathbf{r}}]$（实现修正：原实现用存活神经元比例，在剪枝/生发稳态下 ≈1.0 使该项失去信号意义）；能量项 $\mathbb{E}[\#\text{Spikes}]$ 为**每样本每快步平均脉冲数**（实现修正：原实现为 100 步窗口累加和，与 per-step 目标错配 100 倍，导致预算 PI 控制器与 λ_E 自适应失效）。

**结构 KL 散度**：

$$D_{KL}(\mathbf{s}_e \| 0.5) = \frac{1}{E} \sum_{e=1}^E \left[ s_e \log \frac{s_e}{0.5} + (1-s_e)\log\frac{1-s_e}{0.5} \right]$$

**$\lambda_E$ 自适应调整**（PI 控制）：

$$\lambda_E(t+1) = \lambda_E(t) + \alpha \cdot (\text{current\_spikes} - \text{target\_budget})$$

---

### 4.2 全局神经调节器

**惊喜度检测**（意外不确定性）：

$$\text{surprise}_t = |\text{error}_t - \text{EMA}(\text{error})|$$

**ACh（精度/注意）**：

$$\text{ACh}_{t+1} = \text{decay} \cdot \text{ACh}_t + (1-\text{decay}) \cdot \frac{1}{1 + 5 \cdot \text{surprise}_t}$$

**NE（温度/探索）**：

$$\text{NE}_{t+1} = \text{decay} \cdot \text{NE}_t + (1-\text{decay}) \cdot \sigma(3 \cdot (\text{surprise}_t + \text{ood\_score}))$$

**DA（第三因子/奖赏）**：

$$\text{DA}_{t+1} = \text{decay} \cdot \text{DA}_t + (1-\text{decay}) \cdot \sigma(\text{reward} + 0.5 \cdot \text{curiosity})$$

**星形胶质门控**：

$$\text{calcium}_{t+1} = \text{decay} \cdot \text{calcium}_t + (1-\text{decay}) \cdot \sigma(0.3 \cdot \text{error} - 0.2 \cdot \text{firing\_rate})$$

$$\text{plasticity\_gate} = \text{calcium}$$

---

### 4.3 脉冲预算控制器

**PI 控制律**：

$$u(t) = K_p \cdot e(t) + K_i \cdot \int_0^t e(\tau) d\tau$$

$$e(t) = \text{current\_spikes} - \text{target\_budget}$$

**阈值调节**（实现为叠加而非乘性: $\boldsymbol{\theta}_{\text{eff}} = \boldsymbol{\theta}_{\text{ie}} + \theta_{\text{adj}}$，叠加于稳态可塑性阈值调整量之上）：

$$\theta_{\text{adj}} = u(t) \cdot \text{scale}$$

**抑制强度调节**：

$$\lambda_{\text{inh, adj}} = u(t) \cdot \text{scale}$$

其中 $u(t) = K_p e(t) + K_i \int e(\tau)d\tau$（anti-windup: 积分 clamp $\pm \text{integral\_max}$）；输入 $e(t)$ 为**每样本每快步平均脉冲数**与目标之差（同单位比较）。

---

### 4.4 冷知识归档器

**冷边检测**：

$$\text{cold\_mask} = \neg(\text{edge\_mask}) \lor (\mathbf{s}_e < 0.05)$$

**NF4 非线性量化**（分组大小 64）：

基于正态分布分位数的 16 级非线性量化表：

```python
_NF4_TABLE = [-1.0, -0.6961928, -0.5250730, -0.3949301,
              -0.2844677, -0.1847513, -0.0917715, 0.0,
              0.0797546, 0.1609459, 0.2461693, 0.3379146,
              0.4407282, 0.5626170, 0.7229568, 1.0]
```

**线性 INT4 量化**（分组大小 64）：

$$\text{scale} = \frac{\max(|\mathbf{x}|)}{15}, \quad \text{codes} = \text{round}\left(\frac{\mathbf{x}}{\text{scale}}\right)$$

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

$$\hat{\mathbf{w}}[\text{cold}] = 0, \quad \mathbf{s}_e[\text{cold}] = 0.5, \quad \text{edge\_mask}[\text{cold}] = \text{True}$$

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
| 状态变量（$\mathbf{v}, \mathbf{g}, \mathbf{a}, \boldsymbol{\theta}, \mathbf{s}_e, \text{traces}$） | FP32 | 防止累积误差导致脉冲消失 |
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
1. SNN 状态变量（膜电位 $v$、门控变量 $g$、适应变量 $a$、阈值 $\theta$、资格迹）始终在 FP32 下运行，不受 autocast 影响
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
| `vfe_recent_mean` | 最近 10 步 VFE 平均 | $\frac{1}{10}\sum_{t=T-9}^T \mathcal{F}_t$ |
| `accuracy_trend` | 预测误差趋势 | EMA of $\rho\|\boldsymbol{\epsilon}\|^2$ |
| `complexity_trend` | 结构复杂度趋势 | EMA of $D_{KL} + \text{activity\_cost}$ |
| `energy_trend` | 能量代价趋势 | EMA of $\#\text{Spikes}$ |
| `modulator_ACh` | ACh 水平 | 当前精度参数 |
| `modulator_NE` | NE 水平 | 当前温度参数 |
| `modulator_DA` | DA 水平 | 当前第三因子 |
| `alive_edges_ratio` | 存活超边比例 | $\frac{1}{E}\sum_e \mathbb{I}(s_e > 0.05)$ |
| `alive_neurons_ratio` | 存活神经元比例 | $\frac{1}{N}\sum_i \text{neuron\_mask}_i$ |
| `mean_firing_rate` | 平均发放率 | $\frac{1}{N}\sum_i \bar{\mathbf{r}}_i$ |
| `delay_mean` | 平均轴突延迟 | $\mathbb{E}[\boldsymbol{\tau}_d]$ |
| `delay_entropy` | 延迟分布熵 | $-\sum p(\tau)\log p(\tau)$ |

---

## 8. 小结

LSHN 通过形式化的数学框架将生物脑的多尺度组织原则转化为可计算的模块：

1. **时间解耦**：三个时间尺度严格分离（1 ms / 100 ms / 1000 ms），支持实时计算与长期固化并行
2. **双重可塑性**：快权重 $\hat{w}$ 与结构变量 $s_e$ 分离，为**稳定性-可塑性困境**提供物理解
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

**版本**: v0.1.1 　|　**最后更新**: 2026年8月 　|　**作者**: Apocalypse 　|　**GitHub**: [LSHN](https://github.com/2841649220/LSHN)
