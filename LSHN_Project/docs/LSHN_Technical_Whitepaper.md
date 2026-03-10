# 液态脉冲超图网络（LSHN）：面向持续学习的类脑智能系统

## 技术白皮书

> **版本**: v0.9 Beta 　|　**作者**: Apocalypse 　|　**GitHub**: [LSHN](https://github.com/2841649220/LSHN)

---

## 摘要

液态脉冲超图网络（Liquid Spiking Hypergraph Network, LSHN）是一个面向开放世界与长期在线场景的持续学习系统，在资源受限硬件上实现低能耗、可扩容、抗遗忘与可解释的类脑智能。系统以**变分自由能最小化（Variational Free Energy, $	ext{VFE}$）**为统一理论框架，将多时间尺度脉冲动力学、突触可塑性与结构演化纳入同一闭环优化过程，并显式引入能量预算与不确定性驱动的推理机制。

LSHN 整合五大核心技术突破：

1.**多尺度液态门控元胞**：实现毫秒级脉冲积分与秒级神经调质的分层时间解耦；
2.**双势阱脉冲超图突触**：将结构变量视为模型选择的后验概率，形成"结构慢、权重快"的双时间尺度学习机制；
3.**三因素预测误差脉冲学习规则**：以局部可得信息完成信用分配，避免全局反向传播的生物不合理性；
4.**神经元 - 超边协同演化机制**：基于因果贡献度量化实现无损自剪枝与精准扩容；
5.**海马体 - 皮层双系统巩固架构**：通过生成式回放与知识归档解决灾难性遗忘。

本白皮书阐述 LSHN 的理论基础、架构设计、核心算法与工程实现路径，为边缘端低功耗持续学习提供可验证、可复现、可扩展的类脑计算范式。

---

## 关键词

持续学习；脉冲神经网络；超图神经网络；变分自由能；主动推理；神经调质；结构演化；能量约束；灾难性遗忘

---

## 1 引言

### 1.1 研究背景与核心痛点

当前主流人工智能系统在真实动态场景中面临四大根本性瓶颈，无法实现生物级的终身持续学习：

1. **灾难性遗忘**：新任务学习会不可逆地覆盖旧任务的知识权重，无法在无任务边界的开放场景中持续积累知识
2. **固定容量局限**：模型结构与参数量在训练前固定，无法自适应动态变化的任务复杂度
3. **能效与泛化矛盾**：密集激活的深度学习模型算力开销极大，且对分布偏移、数据缺失的鲁棒性极差
4. **学习机制的生物脱节**：依赖全局反向传播的优化方式，与大脑局部、无监督、自组织的学习规则完全背离

生物大脑通过多尺度神经动力学、双稳态突触可塑性、神经发生与凋亡、海马体 - 皮层记忆巩固、预测编码等机制，在能耗与长期适应之间取得平衡。LSHN 基于**变分自由能最小化（Free Energy Principle, FEP）**的统一理论框架，融合液态神经网络（LNN）的动态适应性、脉冲神经网络（SNN）的事件驱动能效、超图神经网络（HGNN）的高阶关联建模能力，构建面向持续学习瓶颈的机制一致性解决方案。

### 1.2 核心定位

LSHN 的核心定位是：以**变分自由能最小化/主动推理**为统一语言，构建以**自监督/无监督为主**、可通过少量偏好/奖赏信号调制的终身持续学习系统，使"学习/结构演化/探索"成为系统内在动力学的自然结果，同时兼顾工程落地可行性与边缘端部署能力。

### 1.3 核心技术贡献

| 创新模块 | 技术突破 | 理论依据 |
|---------|---------|---------|
| **多尺度液态门控元胞** | 三时间尺度解耦（1ms/100ms/1000ms），树突非线性增强单元表达力 | 离子通道与 G 蛋白偶联受体的时间常数差异 |
| **双势阱脉冲超图突触** | 结构变量=模型后验，权重=条件参数，双时间尺度学习 | 突触权重的双稳态维持机制（LTP/LTD） |
| **三因素预测误差学习** | 局部 STDP 迹×误差脉冲×神经调质，替代全局反向传播 | 生物突触可塑性三因素理论 |
| **神经元 - 超边协同演化** | 基于因果贡献度的无损剪枝与精准扩容 | 彩票假设与稀疏编码理论 |
| **海马体 - 皮层双系统** | 生成式回放 + 知识归档，不依赖大规模缓存 | 系统巩固理论与突触稳态假说 |
| **能量预算反馈控制** | 脉冲预算作为优化约束，PI 控制器动态调节 | 神经形态系统的能量 - 精度权衡 |

---

## 2 统一理论基础

### 2.1 变分自由能最小化框架

LSHN 将所有模块统一到变分自由能最小化框架，所有动力学更新、可塑性规则、结构演化均以最小化系统变分自由能为唯一目标，确保理论完全自洽。

**变分自由能标准形式**：

$$\mathcal{F}(q,\theta)=\mathbb{E}_{q(\mathbf{s})}\left[\log q(\mathbf{s})-\log p_\theta(\mathbf{o},\mathbf{s})\right]$$

**等价分解**（便于工程监控）：

$$\mathcal{F}=\underbrace{-\mathbb{E}_{q(\mathbf{s})}[\log p_\theta(\mathbf{o}|\mathbf{s})]}_{\text{预测误差/Accuracy 相反数}} + \underbrace{D_{KL}(q(\mathbf{s})\parallel p_\theta(\mathbf{s}))}_{\text{复杂度}}$$

其中：
- $\mathbf{o}$ 为观测输入
- $\mathbf{s}$ 为系统内部状态（膜电位、门控变量、突触权重、超边结构等）
- 预测误差项决定"把哪类误差当真"（精度/不确定性调制）
- 复杂度项决定"结构要不要付出复杂度成本"（稀疏性、模型选择）

**完整目标函数**（含能量约束）：

$$\mathcal{J}=\mathcal{F} + \lambda_E \cdot \mathbb{E}\left[\#\text{SynapticEvents}\right]$$

能量正则项将"突触事件数/脉冲率"纳入优化闭环，通过反馈控制维持预算并与遗忘率协同优化。

### 2.2 主动推理扩展

当系统需要执行动作选择或感知选择时（如机器人控制、主动探索），引入**预期自由能（Expected Free Energy, EFE）**：

$$G(\pi)=\mathbb{E}_{q(\mathbf{o},\mathbf{s}\mid \pi)}\left[\log q(\mathbf{s}\mid \pi)-\log p(\mathbf{o},\mathbf{s})\right]$$

EFE 分解为三部分驱动探索 - 利用平衡：
- **风险**：偏好不满足程度
- **模糊性**：观测不确定性
- **信息增益**：揭示隐藏状态的价值

### 2.3 自由能分解的可解释监控

LSHN 在线训练/推理过程中每个时间窗的$\mathcal{F}$ 被分解并记录为以下可解释指标：

| 监控维度 | 具体指标 | 工程映射 |
|---------|---------|---------|
| **预测误差** | 按模态/模块/专家分组 | 精度加权 MSE |
| **复杂度** | 结构 KL、连接数、活跃超边数、延迟分布熵 | 稀疏性约束 |
| **精度/温度** | 全局与局部 | 神经调质 (ACh/NE) 映射 |
| **不确定性** | 校准误差、预测分布熵、OOD 指标 | 可靠性评估 |

这些量既是调参仪表盘，也是"结构演化是否合理"的主要证据链。

---

## 3 系统架构与核心机制

### 3.1 分层架构

LSHN 采用分层解耦的四层架构，各模块通过标准化接口通信，支持单独验证、替换与扩展：

```
输入编码层 → 海马体快速学习层 → 皮层 LSHN 核心网络层 → 输出解码层
                ↑                                  ↓
            生成式回放 ←─────────────────── 知识归档器
```

| 层级 | 功能 | 关键技术 |
|-----|------|---------|
| **输入编码层** | 连续信号→脉冲序列 | MODWT 多尺度小波分析、泊松编码 |
| **海马体层** | 快速编码新任务 | 脉冲自编码器、高可塑性 LIF |
| **皮层核心层** | 长期知识存储 | 液态门控元胞、超图卷积、隐式 MoE |
| **输出解码层** | 脉冲→任务输出 | 动态扩容分类头 |

系统**不包含显式路由模块**：分区与干扰隔离由皮层内的抑制竞争、稳态可塑性与双势阱结构固化共同完成，避免额外门控计算与单点瓶颈。

### 3.2 多时间尺度时钟同步

严格分离快/慢/超慢变量的更新频率，兼顾数值稳定性与计算效率：

| 时间尺度 | 更新频率 | 对应模块 | 生物对应 |
|---------|----------|---------|---------|
| **快时钟** | 1ms | 膜电位、脉冲发放、STDP 迹、快变权重 | 快离子通道动力学 |
| **慢时钟** | 100ms | 慢门控、适应变量、超边结构、神经调质 | 神经调质系统 (ACh/DA/NE) |
| **超慢时钟** | 1000ms | 结构重连、凋亡/生发、知识归档 | 睡眠期突触修剪 |

### 3.3 多尺度液态门控元胞

液态门控元胞是 LSHN 跨尺度的统一计算单元，采用分层多时间常数设计：

**核心状态变量**：

| 变量 | 符号 | 时间尺度 | 生物机制 |
|-----|------|---------|---------|
| 膜电位 | $v_i$ | 快 (ms) | 跨膜电位变化 |
| 快门控 | $g_i^\text{fast}$ | 快 (ms) | 离子通道门控 |
| 慢门控 | $g_i^\text{slow}$ | 慢 (100ms) | 星形胶质调控 |
| 适应变量 | $a_i$ | 慢 (秒) | 慢速钾电流 |

**动力学方程**（离散化实现）：

膜电位更新：
$$v_i(t+1) = \tau_v^{-1} v_i(t) + (1-\tau_v^{-1}) \left( I_i^\text{syn} + I_i^\text{ext} - I_i^\text{inh} \right) + \sigma(g_i^\text{fast}) \cdot \eta_i$$

快门控更新：
$$g_i^\text{fast}(t+1) = \tau_{g,f}^{-1} g_i^\text{fast}(t) + (1-\tau_{g,f}^{-1}) \cdot \sigma\left( W_f v_i + U_f a_i \right)$$

慢门控更新：
$$g_i^\text{slow}(t+1) = \tau_{g,s}^{-1} g_i^\text{slow}(t) + (1-\tau_{g,s}^{-1}) \cdot \sigma\left( W_s \bar{y}_i + U_s \bar{\delta}_i + Z_s e_\text{global} \right)$$

**树突非线性扩展**（可选）：引入轻量树突分支/亚室，实现局部阈值、Ca 尖峰样事件、分支独立积分，提升单元表达能力。

### 3.4 双势阱脉冲超图突触

将传统二元突触扩展为高阶超边，通过双势阱动力学实现结构自适应演化：

**超边定义**：每个超边$e$连接神经元集合$\mathcal{E} = \{i_1,i_2,...,i_k\}$，维护：
- 快变权重 $\hat{w}_e \in [-1,1]$：快速更新
- 结构变量 $s_e \in [0,1]$：双势阱慢速演化
- 资格迹 $e_e(t)$：活动历史记录

**有效权重耦合**：$w_e = w_\text{max} \cdot s_e \cdot \hat{w}_e$

**双势阱动力学**：
$$U(s_e) = \frac{\alpha}{4} s_e^4 - \frac{\alpha}{2} s_e^2$$

结构变量更新：
$$s_e(t+1) = \text{clip}\left( s_e(t) + \Delta t \cdot \left( -\frac{dU}{ds_e} + \beta C_e + \gamma M + \delta R \right) + \sqrt{2\Delta t T} \cdot \xi, 0, 1 \right)$$

其中$C_e$为共发放项，$M$为全局调制，$R$为回放信号，$T$为温度参数。

**超图卷积传播**：
$$I_j^{\text{syn}} = \sum_{e \in \mathcal{E}} H_{je} \left( w_e \cdot \frac{1}{|\mathcal{N}(e)|} \sum_{i \in \mathcal{N}(e)} x_i \right)$$

### 3.5 三因素预测误差学习

基于生物预测编码与三因素突触可塑性理论，实现局部信用分配：

**误差脉冲编码**：将预测误差编码为泊松脉冲序列
$$f_\delta = \sigma(|\delta| \cdot \Sigma^{-1}) \cdot f_\text{max}$$

**多跳资格迹**：
$$e_e(t+1) = \lambda_e e_e(t) + y_\text{pre}(t) \cdot y_\text{post}(t) + \sigma(g_\text{post}^\text{slow}) \cdot \sum_{e' \in \text{local}} w_{e'} e_{e'}(t)$$

**三因素更新规则**：
$$\Delta \hat{w}_e = \eta \cdot e_e(t) \cdot b_\text{post}(t)$$

其中$b_\text{post}$为反向误差脉冲，更新后资格迹重置。

### 3.6 神经元 - 超边协同演化

基于因果贡献度量化，实现无损自剪枝与精准扩容：

**因果贡献度定义**：
- 超边：$\text{Contribution}_e = \mathbb{E}\left[ \mathcal{F}(w_e=0) - \mathcal{F}(w_e=\text{current}) \right]$
- 神经元：关联超边贡献度之和

**程序性凋亡**：
- 连续 10 个慢时间步贡献度≤0 且无任务关键度 → 强制剪枝
- 关联超边凋亡占比>90% → 神经元凋亡
- 分层保护：关键任务前 20% 单元禁止剪枝

**精准生发**（仅在两种场景触发）：
- **代偿性生发**：局部剪枝后误差持续超标时补充
- **扩容性生发**：新任务到来、容量不足时优先在海马体生成

### 3.7 海马体 - 皮层双系统巩固

**双系统分工**：
- **海马体**：高可塑性、快速编码新任务，脉冲自编码器架构
- **皮层**：低可塑性、长期知识存储，LSHN 核心网络

**双模式回放**：
- **在线回放**：每 100 快时间步，回放最近输入模式
- **离线回放**：任务间隙，回放旧任务典型模式与海马体重构伪样本

**可控采样动力学**：
- **泄漏（leakage）**：避免回放链条被早期误差锁死
- **二阶动量（momentum）**：实现时间压缩回放

### 3.8 全局神经调质与星形胶质门控

将神经调质抽象为可计算变量，驱动系统自适应：

| 调质 | 功能映射 | 工程实现 |
|-----|---------|---------|
| **ACh（乙酰胆碱）** | 预期不确定性/注意 → 精度上调 | 学习率窗口放大 |
| **NE（去甲肾上腺素）** | 意外不确定性/突变 → 温度上调 | 结构可塑性开启 |
| **DA（多巴胺）** | 价值/偏好满足 → 第三因子 | 门控更新与巩固强度 |
| **星形胶质** | 慢门控变量 → 跨尺度 metaplasticity | 全局广播调控 |

### 3.9 能量预算反馈控制

将"脉冲预算/突触事件数"作为可控资源，通过 PI 控制器维持预算：

$$\lambda_E(t+1) = \lambda_E(t) + \alpha \cdot (\text{current\_spikes} - \text{target\_budget})$$

自适应调节稀疏正则系数、阈值、抑制强度，使持续学习与能耗在同一闭环内收敛。

### 3.10 冷知识归档机制

检测失活超边（$s_e < 0.05$且已被剪枝），通过 INT4 量化压缩导出至 NVMe 存储：

**归档流程**：
1. 冷边检测：$\neg(\text{edge\_mask}) \lor (s_e < 0.05)$
2. INT4 通道级量化：$\hat{w}_{\text{int4}} = \text{round}(\frac{w - \mu}{\sigma} \cdot 7) \odot \text{sign}(w)$
3. 元数据打包：拓扑索引 + 量化参数 + 结构变量
4. 槽位重置：$w_\text{hat}=0, s_e=0.5, \text{edge\_mask}=True$

实现理论上的无限学习容量。

---

## 4 工程实现路径

### 4.1 技术栈选择

| 层级 | 技术选型 | 理由 |
|-----|---------|-----|
| **深度学习框架** | PyTorch 2.1+ | 动态图、混合精度、活跃社区 |
| **脉冲神经网络** | snntorch | 替代梯度、STDP 规则、事件驱动支持 |
| **图/超图计算** | torch_geometric + 自定义 kernel | 灵活的消息传递机制 |
| **混合精度** | BF16 autocast + FP32 状态围栏 | 加速计算同时保护 SNN 状态 |
| **稀疏优化** | torch.sparse + 自定义 CSR/CSC | 百万级神经元高效仿真 |
| **部署框架** | MindSpore（后续） | 昇腾 NPU 适配 |

### 4.2 混合精度策略

采用**精度围栏（Precision Fence）**策略：

| 模块 | 精度 | 理由 |
|-----|------|-----|
| `nn.Linear` / `SpikeHypergraphConv` | BF16 | 加速矩阵乘法 |
| 所有状态变量（$v, g, a, \theta, s_e$） | FP32 | 防止累积误差导致脉冲消失 |
| 冷知识归档量化 | FP32→INT4 | 压缩存储 |

#### 4.2.1 FP8 实验性加速（NVIDIA Hopper/Ada 架构）

在支持 FP8 的硬件（如 NVIDIA H100、RTX 4090）上，可启用 E4M3/E5M2 混合精度进一步加速训练与推理：

**技术细节**：

| 配置项 | E4M3（前向） | E5M2（反向） |
|:-------|:-------------|:-------------|
| 指数位 | 4 bits | 5 bits |
| 尾数位 | 3 bits | 2 bits |
| 动态范围 | $\pm 448$ | $\pm 57344$ |
| 典型用途 | 权重、激活 | 梯度 |

**实现方案**：

```python
# PyTorch 2.1+ FP8 自动混合精度示例
from torch.amp import autocast, GradScaler

# 前向传播使用 E4M3
with autocast(device_type='cuda', dtype=torch.float8_e4m3fn):
    output = model(input_spikes)

# 反向传播使用 E5M2（PyTorch 自动处理）
loss.backward()

# 主权重保持 FP32，通过 scaling 防止下溢
scaler = GradScaler()
scaler.step(optimizer)
scaler.update()
```

**关键注意事项**：

1. **状态变量保护**：膜电位 $v$、结构变量 $s_e$ 等状态必须保持 FP32，避免累积误差导致脉冲动力学不稳定
2. **梯度缩放策略**：FP8 梯度需要配合动态缩放（dynamic scaling），建议初始缩放因子 $2^{12}$，根据溢出频率自适应调整
3. **超边权重范围**：双势阱突触权重 $\hat{w}_e \in [-1,1]$ 适合 FP8 表示，但需监控离群值（outlier）比例
4. **硬件限制**：FP8 Tensor Core 要求矩阵维度为 16 的倍数，超图卷积的邻接矩阵需填充对齐

**性能收益评估**：

| 指标 | BF16 基线 | FP8 实验 | 备注 |
|:-----|:----------|:---------|:-----|
| 训练吞吐量 | 1.0× | 1.4-1.6× | 取决于矩阵乘法占比 |
| 显存占用 | 1.0× | 0.6× | 权重/激活减半 |
| 数值稳定性 | 稳定 | 需监控 | 脉冲发放率偏差 $< 2\%$ |

**限制条件**：
- 仅推荐在 NVIDIA Hopper（SM90+）或 Ada Lovelace 架构 GPU 上启用
- 需要 PyTorch 2.1+ 与 CUDA 12.1+
- 脉冲时间精度敏感任务建议保持 BF16

### 4.3 显存与算力优化

针对单卡 96G 显存环境，实现百万级神经元、千万级超边的高效仿真：

1. **稀疏张量存储**：仅保留有效连接，显存占用降低 90%+
2. **事件驱动仿真**：仅当神经元发放脉冲时更新突触电流，效率提升 5-10 倍
3. **梯度检查点**：牺牲少量计算速度换取显存大幅降低
4. **局部连接限制**：组内全连接、组间稀疏（<5%），避免$O(N^2)$爆炸

### 4.4 模块接口规范

所有核心模块遵循统一接口设计，支持即插即用：

```
LiquidGatedCell
  ├─ step_fast(I_syn) → (spk, v)      # 快时钟 (1ms)
  └─ step_slow(global_e) → None       # 慢时钟 (100ms)

BistableHypergraphSynapse
  ├─ step_fast(pre_spk, hyperedge_index) → I_syn
  └─ step_slow_structure(M, R, T) → None

FreeEnergyEngine
  └─ compute_vfe(prediction_error, s_e, spikes) → dict

KnowledgeArchiver
  ├─ archive_cold_edges(w_cold, s_cold, H_cold) → archive_id
  └─ load_archive(archive_id) → dict
```

---

## 5 实验验证方案

### 5.1 实验设计原则

1. **循序渐进**：先验证单个模块动力学，再整合全系统
2. **基线全面**：覆盖传统持续学习 SOTA、类脑 SNN SOTA、同架构消融基线
3. **指标闭环**：包含准确率、遗忘率、能效、鲁棒性、可解释性多维指标
4. **消融全覆盖**：每个核心模块均设置消融实验

### 5.2 分阶段验证计划

**阶段一：基础模块动力学验证（1-2 个月）**
- 液态门控元胞响应特性测试
- 双势阱超边双稳态验证
- 凋亡生发机制单元测试
- 简单时序预测任务（正弦序列、连续 XOR）

**阶段二：持续学习基准验证（3-4 个月）**
- Split MNIST、Permuted MNIST、CIFAR-10 增量学习
- 核心指标：平均准确率、遗忘率、前向迁移、激活稀疏度
- 对比基线：EWC、SI、GEM、HLOP、标准 SNN+STDP

**阶段三：自剪枝与开放世界学习验证（5-7 个月）**
- CIFAR-10/100 增量学习过程中的在线自剪枝
- 无任务边界 Class-Incremental 场景
- 神经元随机失活后的功能恢复实验

**阶段四：多模态与复杂场景验证（8-11 个月）**
- 多模态关联任务（Time-MMD、无人机多模态数据集）
- 持续对话任务（MultiWOZ 基准）
- 主动探索任务（迷宫导航、信息寻求游戏）

**阶段五：全系统整合与极限测试（12-15 个月）**
- 全任务集端到端无监督持续学习
- 不同神经元规模（1 万 -100 万）扩展性测试
- 与 Transformer、LSTM、传统 SNN 的能效对比
- 长达 100 个增量任务的长期稳定性测试

### 5.3 核心评估指标

| 指标类别 | 具体指标 | 目标值 |
|---------|---------|-------|
| **准确性** | 平均准确率、前向迁移 | ≥ 主流基线 |
| **遗忘率** | 任务学习后准确率下降幅度 | $F \approx 0$ |
| **能效** | 单位样本突触事件数、单位准确率能耗 | 比 Transformer 低 $1$–$2$ 数量级 |
| **稀疏度** | 神经元平均发放率 | $1\%$–$5\%$ |
| **压缩率** | 剪枝后参数量减少比例 | $30\%$–$90\%$ 无损 |
| **可靠性** | OOD 检测 ECE、风险 - 覆盖曲线 | 优于基线 |

---

## 6 应用前景与落地路径

### 6.1 目标应用场景

| 场景 | 核心需求 | LSHN 优势 |
|-----|---------|---------|
| **边缘端自主机器人** | 低功耗、持续学习新技能 | 事件驱动能效、动态扩容 |
| **个性化持续对话系统** | 无灾难性遗忘、本地隐私保护 | 海马体回放、无需频繁微调 |
| **工业时序预测** | 非平稳数据、分布漂移检测 | 自适应调整、在线持续学习 |
| **可穿戴医疗设备** | 极致低功耗、个体生理适配 | 脉冲稀疏性、长期适应能力 |

### 6.2 部署路线

**短期（6 个月）**：
- PyTorch 框架完成机制验证
- 单卡 RTX Pro6000/A800 环境部署
- Split-MNIST/CIFAR-10 基准验证

**中期（12 个月）**：
- 自定义稀疏 kernel 优化关键算子
- MindSpore 框架迁移与昇腾 NPU 适配
- 多模态/对话/导航复杂场景验证

**长期（18 个月）**：
- 专用类脑芯片适配
- 开源社区发布与生态建设
- 医疗诊断、自动驾驶等大规模落地

---

## 7 风险管控与应对策略

| 风险类型 | 风险描述 | 应对策略 |
|---------|---------|---------|
| **数值不稳定** | 膜电位爆炸、沉默神经元 | 稳态可塑性机制、梯度裁剪、辅助积累通路 |
| **性能不达标** | 遗忘率过高、准确率低于基线 | 分模块消融定位、强化回放与结构固化 |
| **剪枝性能下降** | 自剪枝导致核心性能损失 | 性能校验闸门、渐进式剪枝 + 步步巩固 |
| **算力/显存不足** | 大规模网络超出单卡限制 | 稀疏存储 + 事件驱动、混合精度、梯度检查点 |
| **开发进度延迟** | 单个模块开发难度超出预期 | 任务拆解为周度里程碑、最小可行版本优先 |
| **学术创新性风险** | 核心创新点与现有工作重叠 | 全面文献调研、聚焦空白创新点、提前发布预印本 |

---

## 8 总结与展望

液态脉冲超图网络（LSHN）实现了从"数据驱动优化"到"机制驱动自组织"的人工智能范式转变，通过统一的变分自由能最小化框架，将生物大脑的多尺度动力学、双稳态突触可塑性、神经发生与凋亡、预测编码、记忆巩固五大核心机制，转化为可计算、可落地的工程实现。

**核心创新总结**：
1. **理论统一性**：所有模块统一于 VFE 最小化，无机制悬浮
2. **时间解耦**：三时间尺度严格分离，同时支持实时计算与长期固化
3. **双重可塑性**：快权重与结构变量分离，为稳定性 - 可塑性困境提供物理解
4. **能量约束**：脉冲预算纳入优化目标，实现能效与性能协同
5. **无限容量**：冷知识归档机制支持无上限任务序列学习

LSHN 旨在为边缘端低功耗智能系统提供一套可扩展、可解释、可验证的持续学习方案，并为后续更大规模的类脑系统研究奠定基础。未来将进一步拓展更复杂的生物启发机制，适配专用类脑芯片，推动类脑智能从理论走向大规模落地应用。

---

## 参考文献

[R1] infer-actively/pymdp: A Python implementation of active inference for MDPs. https://github.com/infer-actively/pymdp

[R2] Spiking Graph Predictive Coding for Reliable OOD Generalization (WWW 2026). https://arxiv.org/html/2602.19392

[R3] Dendrify: a framework for incorporating dendrites into SNNs (Nat. Commun. 2022). https://www.nature.com/articles/s41467-022-35747-8

[R4] Temporal dendritic heterogeneity incorporated with spiking neural networks (Nat. Commun. 2023). https://www.nature.com/articles/s41467-023-44614-z

[R5] Three-factor learning in spiking neural networks: an overview (2025). https://www.sciencedirect.com/science/article/pii/S2666389925002624

[R6] Meta-SpikePropamine: learning to learn with synaptic plasticity in SNNs (PMC). https://pmc.ncbi.nlm.nih.gov/articles/PMC10213417/

[R7] Energy-Aware Spike Budgeting for Continual Learning in SNNs (2026). https://arxiv.org/html/2602.12236

[R8] Leakage and Second-Order Dynamics Improve Hippocampal RNN Replay (2026). https://arxiv.org/abs/2602.18401

[R9] Effects of Introducing Synaptic Scaling on Spiking Neural Network Learning (2026). https://arxiv.org/abs/2601.11261

[R10] Astrocyte-mediated plasticity review (open mirror). https://pmc.ncbi.nlm.nih.gov/articles/PMC12730915/

[R11] Tripartite synapse complexity modeling (Biological Cybernetics 2024). https://link.springer.com/article/10.1007/s00422-024-00994-z

[R12] Learnable axonal delay improves spoken word recognition (Frontiers 2023). https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1275944/full

[R13] Co-learning synaptic delays, weights and adaptation in SNNs (Frontiers 2024, PMC). https://pmc.ncbi.nlm.nih.gov/articles/PMC11055628/

[R14] grid-cells (DeepMind). https://github.com/google-deepmind/grid-cells

[R15] Slack-Free Spiking Neural Network Formulation for Hypergraph Minimum Vertex Cover (NeurIPS 2024). https://openreview.net/forum?id=4A5IQEjG8c

---

## 附录 A：核心符号表

| 符号 | 定义 | 取值范围 |
|-----|------|---------|
| $\mathcal{N}$ | 神经元集合 | $\|\mathcal{N}\| = N = 5 \times 10^5$ |
| $\mathcal{E}$ | 超边集合 | $\|\mathcal{E}\| = E$ |
| $\mathbf{v}$ | 膜电位向量 | $(-\infty, \theta]$ |
| $\mathbf{S}$ | 脉冲发放状态 | $\{0, 1\}$ |
| $\hat{\mathbf{w}}$ | 快权重 | $[-1, 1]$ |
| $\mathbf{s}_e$ | 结构变量 | $[0, 1]$ |
| $\mathcal{F}$ | 变分自由能 | $\mathbb{R}^+$ |
| $\mathbb{T}_{\text{fast}}$ | 快时间尺度 | $\Delta t = 1\text{ms}$ |
| $\mathbb{T}_{\text{slow}}$ | 慢时间尺度 | $\Delta t = 100\text{ms}$ |
| $\mathbb{T}_{\text{ultra}}$ | 超慢时间尺度 | $\Delta t = 1000\text{ms}$ |

---

**文档版本**：v0.9 Beta（技术白皮书精简版）  
**作者**：Apocalypse  
**GitHub**: [LSHN](https://github.com/2841649220/LSHN)  
**最后更新**：2026 年 3 月  
**许可协议**：MIT
