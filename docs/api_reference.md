# LSHN API 参考文档

> **版本**: v0.9 Beta 　|　**更新**: 2026年3月 　|　**作者**: Apocalypse 　|　**GitHub**: [LSHN](https://github.com/2841649220/LSHN)

本文档提供 LSHN（液态脉冲超图网络）所有核心模块的完整 API 参考，包括类定义、方法签名、参数说明和使用示例。

---

## 目录

1. [LSHNModel 主类](#lshnmodel-主类)
2. [核心模块 (Core)](#核心模块-core)
3. [全局引擎 (Engine)](#全局引擎-engine)
4. [网络层 (Layers)](#网络层-layers)
5. [工具函数 (Utils)](#工具函数-utils)
6. [配置参数](#配置参数)
7. [FP8 实验性加速](#fp8-实验性加速)

---

## LSHNModel 主类

### 类定义

```python
class LSHNModel(nn.Module)
```

液态脉冲超图网络端到端模型，实现白皮书描述的分层解耦四层架构。

### 构造函数

```python
def __init__(
    self,
    input_dim: int = 128,
    hidden_dim: int = 1024,
    num_neurons: int = 500000,
    num_groups: int = 100,
    max_edges: int = 50000,
    initial_classes: int = 2,
    enable_dendrites: bool = False,
    enable_active_inference: bool = False,
    target_spikes_per_step: int = 5000,
    mixed_precision: bool = True,
    autocast_dtype: torch.dtype = torch.bfloat16,
    archive_dir: str = "./cold_archive",
    archive_group_size: int = 64,
    cold_threshold: float = 0.05,
    device=None,
    dtype=None
)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_dim` | int | 128 | 原始输入维度 |
| `hidden_dim` | int | 1024 | 海马体/编码器隐层维度 |
| `num_neurons` | int | 500000 | 皮层神经元数（50万） |
| `num_groups` | int | 100 | 皮层功能分区数（隐式MoE列数） |
| `max_edges` | int | 50000 | 超图最大超边数 |
| `initial_classes` | int | 2 | 初始分类头类别数 |
| `enable_dendrites` | bool | False | 是否启用树突区室非线性 |
| `enable_active_inference` | bool | False | 是否启用主动推理 |
| `target_spikes_per_step` | int | 5000 | 目标脉冲数/步 |
| `mixed_precision` | bool | True | 是否启用BF16混合精度 |
| `autocast_dtype` | torch.dtype | torch.bfloat16 | autocast目标精度 |
| `archive_dir` | str | "./cold_archive" | 冷知识归档目录 |
| `archive_group_size` | int | 64 | INT4量化分组大小 |
| `cold_threshold` | float | 0.05 | 冷边判定阈值 |

### 主要方法

#### forward_step

```python
def forward_step(
    self,
    x: torch.Tensor,
    target: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]
```

单步前向传播（1ms快时钟）。

**参数：**
- `x`: (`batch`, `input_dim`) 原始输入
- `target`: (`batch`, `num_classes`) 目标（用于误差驱动学习）

**返回：**
字典包含：
- `output`: 模型输出
- `spk_cortex`: 皮层脉冲
- `spk_hippo`: 海马体脉冲
- `spk_encoded`: 编码后脉冲

#### expand_classes

```python
def expand_classes(self, num_new_classes: int) -> List[nn.Parameter]
```

动态扩容输出类别，用于持续学习场景。

**返回：** 新增参数列表，需要调用方加入optimizer。

#### get_monitoring_report

```python
def get_monitoring_report(self) -> Dict[str, float]
```

返回可解释监控报告。

**返回：** 包含VFE分解、调质状态、预算状态、结构统计的字典。

#### reset

```python
def reset(self)
```

重置所有状态（时钟、隐藏状态、缓存等）。

---

## 核心模块 (Core)

### LiquidGatedCell

**文件路径：** `lshn/core/cells/liquid_cell.py`

多尺度液态门控元胞。

#### 构造函数

```python
def __init__(
    self,
    num_neurons: int,
    tau_v: float = 10.0,
    tau_g_fast: float = 5.0,
    tau_g_slow: float = 200.0,
    tau_a: float = 100.0,
    theta_0: float = 1.0,
    enable_dendrites: bool = False,
    num_branches: int = 4,
    device=None,
    dtype=None
)
```

**核心状态变量：**
- `v`: 膜电位（快，ms级）
- `g_fast`: 快门控（快，ms级，调制离子通道）
- `g_slow`: 慢门控（慢，100ms级，调制可塑性和噪声）
- `a`: 适应变量（慢，秒级，模拟慢速钾电流）

#### 主要方法

| 方法 | 说明 |
|------|------|
| `step_fast(I_syn, I_ext=None, I_inh=None, theta_ie=None)` | 快时钟（1ms）前向更新 |
| `step_slow(global_e)` | 慢时钟（100ms）更新 |
| `reset_hidden()` | 重置所有状态 |
| `get_plasticity_modulation()` | 返回g_slow作为可塑性调制因子 |
| `get_firing_rate()` | 返回当前滑动窗口内的平均发放率 |

### BistableHypergraphSynapse

**文件路径：** `lshn/core/synapses/bistable_hypergraph.py`

双势阱脉冲超图突触。

#### 构造函数

```python
def __init__(
    self,
    num_neurons: int,
    out_channels: int = 1,
    max_edges: Optional[int] = None,
    w_max: float = 1.0,
    alpha: float = 0.1,
    beta: float = 0.05,
    trace_decay: float = 0.9,
    device=None,
    dtype=None
)
```

**核心变量：**
- `w_hat`: 快变权重 ∈ [-1,1]，由三因素规则快速更新
- `s_e`: 结构变量 ∈ [0,1]，双势阱慢速演化
- `e_trace`: 资格迹，记录超边的活动历史

**有效权重公式：** `w_e = w_max * s_e * w_hat`

#### 主要方法

| 方法 | 说明 |
|------|------|
| `step_fast(x_in, hyperedge_index, post_spk=None, g_slow=None)` | 快时钟前向 |
| `step_slow_structure(M_global, R_replay, T_temp, dt_slow=0.1)` | 慢时钟结构双势阱更新 |
| `get_effective_weights()` | 返回有效权重 |
| `get_alive_mask(threshold=0.05)` | 返回存活超边掩码 |

### ThreeFactorPlasticity

**文件路径：** `lshn/core/plasticity/three_factor.py`

三因素可塑性与预测误差反向脉冲学习规则。

```python
def __init__(self, learning_rate: float = 0.01, trace_decay: float = 0.9)

def forward(
    self,
    w_hat: torch.Tensor,
    e_trace: torch.Tensor,
    error_spk: torch.Tensor,
    neuromodulator: Optional[torch.Tensor] = None
)
```

**三因素：**
1. Pre-synaptic trace（前突触）
2. Post-synaptic trace（后突触）
3. Global/Local Error Neuromodulator（第三因子）

### HomeostaticController

**文件路径：** `lshn/core/plasticity/homeostatic.py`

稳态可塑性控制器。

```python
def __init__(
    self,
    num_neurons: int,
    target_rate: float = 0.05,
    device=None,
    dtype=None
)
```

**包含模块：**
- `SynapticScaling`: 突触缩放，维持总突触强度稳定
- `IntrinsicExcitabilityPlasticity`: 内在兴奋性可塑性，维持目标发放率

#### 主要方法

| 方法 | 说明 |
|------|------|
| `step_fast(spk)` | 快时钟：更新发放率EMA |
| `step_slow()` | 慢时钟：计算突触缩放因子和阈值调整 |
| `apply_to_weights(w_hat, neuron_to_edge_map=None)` | 对权重施加突触缩放 |

### PruneGrowthModule

**文件路径：** `lshn/core/evolution/prune_growth.py`

神经元-超边协同凋亡生发与自剪枝模块。

```python
def __init__(
    self,
    max_neurons: int,
    max_edges: int,
    prune_threshold: float = 0.0,
    apop_cooldown: int = 10,
    neuron_dead_ratio: float = 0.9,
    growth_vfe_ratio: float = 1.5,
    growth_cap: float = 0.05,
    device=None,
    dtype=None
)
```

#### 主要方法

```python
def step_ultra_slow_evolution(
    self,
    VFE_full: float,
    VFE_masked_dict: Dict[int, float],
    hyperedge_index: Optional[torch.Tensor] = None,
    task_importance_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

### AxonalDelayModule

**文件路径：** `lshn/core/synapses/axonal_delay.py`

轴突传导延迟模块。

```python
def __init__(
    self,
    max_edges: int,
    max_delay: int = 20,
    min_delay: int = 1,
    delay_lr: float = 0.001,
    device=None,
    dtype=None
)
```

#### 主要方法

| 方法 | 说明 |
|------|------|
| `step_fast(pre_spk, post_spk)` | 快时钟前向，返回延迟后的脉冲 |
| `update_delays(e_trace, timing_error)` | 慢时钟延迟学习 |
| `get_delay_stats()` | 返回延迟分布统计信息 |
| `reset()` | 重置所有状态 |

---

## 全局引擎 (Engine)

### FreeEnergyEngine

**文件路径：** `lshn/engine/free_energy.py`

变分自由能(VFE)全局计算引擎。

```python
def __init__(
    self,
    kl_weight: float = 0.01,
    energy_lambda: float = 0.001
)
```

**核心公式：**
- F(q,θ) = -E_q(s)[log p_θ(o|s)] + D_KL(q(s)||p_θ(s))
- J = F + λ_E · E[#SynapticEvents]

#### 主要方法

| 方法 | 说明 |
|------|------|
| `compute_vfe(prediction_error, s_e_tensor, active_neurons_ratio, synaptic_events, precision)` | 计算VFE和J |
| `compute_energy_regularization_gradient(current_events, target_budget)` | 自适应调整λ_E |
| `get_decomposition_report()` | 返回自由能分解报告 |

### GlobalNeuromodulator

**文件路径：** `lshn/engine/global_modulator.py`

全局神经调节器。

```python
def __init__(self, num_neurons: int, device=None, dtype=None)
```

**三种神经调质：**
- **ACh (乙酰胆碱)**: 精度/注意，控制学习率窗口
- **NE (去甲肾上腺素)**: 温度/探索，控制结构可塑性
- **DA (多巴胺)**: 第三因子/奖赏，门控三因素可塑性

#### 主要方法

```python
def step_slow(
    self,
    prediction_error: float,
    firing_rate: float,
    reward_signal: float = 0.0,
    ood_score: float = 0.0
) -> Dict[str, torch.Tensor]
```

**返回字典键：** `ACh`, `NE`, `DA`, `plasticity_gate`, `surprise`

### ClockSyncEngine

**文件路径：** `lshn/engine/clock_sync.py`

多时间尺度时钟同步器。

```python
def __init__(self)

def tick(self) -> Tuple[bool, bool]
```

**时钟周期：**
- 快时钟：1ms（膜电位、脉冲发放、快门控）
- 慢时钟：100ms（慢门控、适应变量、双势阱、能量控制）
- 超慢时钟：1000ms（因果贡献度、凋亡生发、回放）

**返回：** `(trigger_slow, trigger_ultra_slow)`

### SpikeBudgetController

**文件路径：** `lshn/engine/budget_control.py`

能量/脉冲预算PI控制器。

```python
def __init__(
    self,
    target_spikes_per_step: int,
    kp: float = 0.01,
    ki: float = 0.001,
    max_integral: float = 100.0,
    lambda_E_base: float = 0.01
)
```

#### 主要方法

```python
def step_control(self, current_spikes: int) -> dict
```

**返回字典键：** `theta_adj`, `inh_adj`, `lambda_E_adj`, `lambda_E_effective`, `budget_error`, `integral_error`

### KnowledgeArchiver

**文件路径：** `lshn/engine/knowledge_archiver.py`

冷知识INT4归档器。

```python
def __init__(
    self,
    archive_dir: str = "./cold_archive",
    group_size: int = 64
)
```

**压缩方案：**
- `w_hat ∈ [-1, 1]`: NF4非线性分块量化
- `s_e ∈ [0, 1]`: 线性INT4分块量化
- 两个INT4值bit-pack进一个uint8字节

#### 主要方法

| 方法 | 说明 |
|------|------|
| `archive_cold_edges(cold_indices, w_hat_cold, s_e_cold, hyperedge_index_cold, num_nodes)` | 归档冷超边 |
| `retrieve_archived_edges(archive_id, device=None)` | 解压恢复冷超边 |
| `list_archives()` | 返回所有归档条目 |
| `total_cold_edges()` | 返回累计归档冷超边总数 |
| `delete_archive(archive_id)` | 删除指定归档 |

### ActiveInferenceEngine

**文件路径：** `lshn/engine/active_inference.py`

主动推理与预期自由能模块。

```python
def __init__(
    self,
    state_dim: int,
    obs_dim: int,
    num_policies: int = 8,
    gamma: float = 1.0,
    device=None,
    dtype=None
)
```

**EFE公式：** `G(π) = Risk + Ambiguity - Information_Gain`

#### 主要方法

| 方法 | 说明 |
|------|------|
| `update_belief(observation, prediction_error)` | 基于新观测更新后验信念 |
| `compute_efe(current_state)` | 计算所有策略的预期自由能 |
| `select_policy(current_state)` | 基于EFE的softmax策略选择 |
| `get_exploration_signal(current_state)` | 提取全局探索信号 |

---

## 网络层 (Layers)

### CorticalLayer

**文件路径：** `lshn/layers/cortex/cortical_layer.py`

皮层LSHN核心网络层。

```python
def __init__(
    self,
    in_channels: int,
    num_neurons: int,
    num_groups: int,
    max_edges: int,
    enable_dendrites: bool = False,
    inhibition_strength: float = 0.5,
    device=None,
    dtype=None
)
```

**整合模块：**
- 多尺度液态门控元胞
- 双势阱超图突触
- 轴突延迟模块
- 隐式MoE（横向抑制）
- 三因素可塑性
- 稳态可塑性控制
- 凋亡生发机制

#### 主要方法

| 方法 | 说明 |
|------|------|
| `step_fast(x_in, hyperedge_index, theta_ie=None)` | 1ms前向步 |
| `step_slow(global_e, M_global, R_replay, T_temp)` | 100ms更新慢变量 |
| `step_ultra_slow(VFE_full, VFE_masked_dict, task_importance_mask=None)` | 1000ms凋亡生发 |
| `apply_plasticity(error_spk, neuromodulator=None)` | 应用三因素可塑性 |
| `get_spike_count_and_reset()` | 获取并重置脉冲计数 |

### MODWTEncoder

**文件路径：** `lshn/layers/io/modwt_encoder.py`

MODWT多尺度小波编码与泊松脉冲前端。

```python
def __init__(
    self,
    in_features: int,
    out_features: int,
    num_scales: int = 3,
    wavelet: str = 'haar'
)
```

**支持小波基：** `haar`, `db4` (Daubechies-4)

#### 主要方法

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**输入：** (B × d_in) 连续输入信号

**输出：** (B × d_out) 泊松脉冲 {0, 1}

### DynamicExpansionHead

**文件路径：** `lshn/layers/io/dynamic_expansion_head.py`

动态扩容输出解码层。

```python
def __init__(self, in_features: int, initial_classes: int = 2)

def expand(self, num_new_classes: int) -> List[nn.Parameter]
```

### SpikingAutoEncoder

**文件路径：** `lshn/layers/hippocampus/spiking_ae.py`

海马体脉冲自编码器。

```python
def __init__(
    self,
    input_dim: int,
    hidden_dim: int,
    device=None,
    dtype=None
)
```

#### 主要方法

| 方法 | 说明 |
|------|------|
| `step_fast(x_in)` | 快时钟前向 |
| `decode(spk_hidden)` | 重构输入 |

### ReplayGenerator

**文件路径：** `lshn/layers/hippocampus/replay_generator.py`

可控离线采样动力学。

```python
def __init__(
    self,
    hidden_dim: int,
    leakage: float = 0.1,
    momentum: float = 0.9
)

def init_state(self, batch_size: int, device=None, dtype=None)

def generate_step(self, ae_decoder: nn.Module) -> torch.Tensor
```

### ImplicitMoE

**文件路径：** `lshn/layers/cortex/implicit_moe.py`

无中心隐式MoE（基于局部侧向抑制竞争）。

```python
def __init__(
    self,
    num_neurons: int,
    num_groups: int,
    inhibition_strength: float = 0.5
)

def forward(self, spk: torch.Tensor) -> torch.Tensor
```

---

## 工具函数 (Utils)

### sparse_event_driven_matmul

**文件路径：** `lshn/utils/sparse_kernel.py`

事件驱动稀疏矩阵乘法。

```python
def sparse_event_driven_matmul(
    spk: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor
```

**说明：** 仅当神经元发放脉冲(spk=1)时才提取相应权重进行计算。

**参数：**
- `spk`: (B × d_in) 或 (d_in,) 脉冲张量
- `weight`: (out_features, in_features) 权重矩阵

**返回：** (B × d_out) 或 (d_out,)

### ContinualLearningMetrics

**文件路径：** `lshn/utils/metrics.py`

持续学习指标评估模块。

```python
def __init__(self, num_tasks: int)
```

#### 主要方法

| 方法 | 说明 |
|------|------|
| `update_accuracy(trained_task_idx, eval_task_idx, acc)` | 更新准确率矩阵 |
| `record_spike_sparsity(spk)` | 记录脉冲稀疏度 |
| `average_accuracy(current_task_idx)` | 平均准确率 |
| `forgetting_measure(current_task_idx)` | 平均遗忘率 |
| `get_average_sparsity()` | 平均稀疏度 |
| `report(current_task_idx)` | 返回完整报告 |

---

## 配置参数

### 完整配置示例（default.yaml）

```yaml
# ---------- 模型架构 ----------
model:
  input_dim: 128              # 原始输入维度
  hidden_dim: 1024            # 海马体/编码器隐层维度
  num_neurons: 500000         # 皮层神经元数 (50万)
  num_groups: 100             # 皮层功能分区数
  max_edges: 50000            # 超图最大超边数
  initial_classes: 2          # 初始分类头类别数
  enable_dendrites: true      # 是否启用树突区室非线性
  enable_active_inference: false  # 是否启用主动推理

# ---------- 多时间尺度时钟 ----------
clocks:
  fast_ms: 1                  # 快时钟 (脉冲传播)
  slow_ms: 100                # 慢时钟 (突触/调质更新)
  ultra_slow_ms: 1000         # 超慢时钟 (结构演化)

# ---------- 液态门控细胞 ----------
cell:
  tau_fast: 10.0              # 快膜时间常数 (ms)
  tau_slow: 100.0             # 慢门控时间常数 (ms)
  threshold: 1.0              # 脉冲阈值
  noise_std: 0.01             # 基础噪声标准差
  dendrite_threshold: 0.5     # 树突Ca-spike阈值
  dendrite_branches: 4        # 树突分支数

# ---------- 双稳态超图突触 ----------
synapse:
  tau_pre: 20.0               # STDP前突触迹时间常数 (ms)
  tau_post: 20.0              # STDP后突触迹时间常数 (ms)
  tau_slow_se: 500.0          # s_e双势阱时间常数 (ms)
  A_plus: 0.01                # STDP增强幅度
  A_minus: 0.012              # STDP抑制幅度
  tau_elig: 1000.0            # 资格迹时间常数 (ms)

# ---------- 轴突延迟 ----------
axonal_delay:
  max_delay: 20               # 最大延迟步数
  num_connections: 50000      # 连接数
  delay_lr: 0.001             # 延迟学习率
  min_delay: 1                # 最小延迟步数

# ---------- 三因素可塑性 ----------
three_factor:
  lr: 0.001                   # 可塑性学习率
  f_max: 1.0                  # 泊松编码器最大频率

# ---------- 稳态可塑性 ----------
homeostatic:
  target_rate: 0.05           # 目标放电率
  tau_rate: 1000.0            # 放电率估计时间常数 (ms)
  scaling_strength: 0.01      # 突触缩放强度
  ie_lr: 0.001                # 内在兴奋性学习率

# ---------- 隐式MoE (侧抑制) ----------
implicit_moe:
  inhibition_strength: 0.5    # 侧抑制强度

# ---------- 结构演化 (剪枝/生长) ----------
evolution:
  prune_threshold: 0.01       # 剪枝阈值
  growth_probability: 0.05    # 生长概率
  min_alive_ratio: 0.3        # 最低存活比例

# ---------- 全局神经调节器 ----------
neuromodulator:
  tau_ach: 200.0              # ACh时间常数 (ms)
  tau_ne: 300.0               # NE时间常数 (ms)
  tau_da: 500.0               # DA时间常数 (ms)
  ach_gain: 0.05              # ACh增益
  ne_gain: 0.05               # NE增益
  da_gain: 0.1                # DA增益

# ---------- 星形胶质门控 ----------
astrocyte:
  tau_ca: 5000.0              # 钙动力学时间常数 (ms)
  ip3_gain: 0.02              # IP3增益

# ---------- 变分自由能引擎 ----------
free_energy:
  kl_weight: 0.01             # KL散度权重 β
  energy_lambda: 0.001        # 能量正则化 λ_E
  energy_lambda_lr: 0.0001    # λ_E 自适应学习率

# ---------- 脉冲预算控制器 ----------
budget:
  target_spikes_per_step: 5000  # 目标脉冲数/步
  kp: 0.01                    # 比例增益
  ki: 0.001                   # 积分增益
  integral_max: 10.0          # 积分防饱和上限

# ---------- 主动推理 ----------
active_inference:
  num_policies: 8             # 策略数
  state_dim: 1024             # 隐状态维度
  obs_dim: 128                # 观测维度
  planning_horizon: 5         # 规划时域
  efe_temperature: 1.0        # EFE softmax温度

# ---------- 海马体 ----------
hippocampus:
  replay_interval: 100        # 回放间隔 (快时钟步数)
  replay_steps: 1             # 每次回放步数

# ---------- MODWT编码器 ----------
encoder:
  num_scales: 3               # 小波分解尺度数
  wavelet: "db4"              # 小波基函数

# ---------- 混合精度 ----------
precision:
  mixed_precision: true       # 是否启用 BF16 autocast
  autocast_dtype: "bfloat16"  # autocast 目标精度
  use_fp8: false              # FP8 实验性加速

# ---------- 冷知识归档器 ----------
archiver:
  enabled: true               # 是否启用 INT4 冷归档
  archive_dir: "./cold_archive"  # 归档目录
  group_size: 64              # INT4 量化分组大小
  cold_threshold: 0.05        # 冷边判定阈值
```

---

## FP8 实验性加速

### 概述

FP8是NVIDIA H100和RTX 4090引入的8位浮点格式，可进一步加速前向计算。

**注意：** SNN状态变量（v, g_fast, g_slow, a, theta, traces）仍需保持FP32，仅前向矩阵乘法可使用FP8。

### 配置参数

```yaml
precision:
  mixed_precision: true       # 启用混合精度
  autocast_dtype: "bfloat16"  # 基础精度
  use_fp8: false              # FP8实验性加速（需PyTorch 2.1+和H100/RTX4090）
```

### 使用示例

```python
import torch
from lshn import LSHNModel

# 创建支持FP8的模型
model = LSHNModel(
    input_dim=128,
    hidden_dim=1024,
    num_neurons=500000,
    mixed_precision=True,
    autocast_dtype=torch.float8_e4m3fn,  # FP8 E4M3格式
)

# 或使用配置字典
config = {
    'precision': {
        'mixed_precision': True,
        'autocast_dtype': 'float8_e4m3fn',
        'use_fp8': True
    }
}

# 前向传播会自动使用FP8精度（在支持的硬件上）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 输入数据
x = torch.randn(32, 128).to(device)  # batch_size=32
target = torch.randn(32, 10).to(device)

# 单步前向（自动应用FP8 autocast）
output = model.forward_step(x, target)
```

### 硬件要求

- **GPU**: NVIDIA H100 或 RTX 4090
- **PyTorch**: 2.1+
- **CUDA**: 11.8+

### 精度注意事项

1. SNN状态变量始终在FP32下运行，不受autocast影响
2. 仅`nn.Linear`和矩阵乘法操作会使用FP8
3. 反向传播梯度仍使用FP16/BF16
4. 建议在训练初期使用BF16稳定收敛后再尝试FP8

### FP8 格式对比

| 格式 | 指数位 | 尾数位 | 数值范围 | 适用场景 |
|:---:|:---:|:---:|:---:|:---|
| **E4M3** | 4 位 | 3 位 | ±448 | 前向激活、权重 |
| **E5M2** | 5 位 | 2 位 | ±57344 | 梯度计算 |

---

## 版本信息

- **版本**: v0.9 Beta
- **最后更新**: 2026年3月
- **作者**: Apocalypse
- **GitHub**: [LSHN](https://github.com/2841649220/LSHN)
- **许可证**: MIT

---

## 文档导航

| 文档 | 内容 | 适用读者 |
|:---|:---|:---|
| [README](../README.md) | 项目概览、快速开始、使用示例 | 所有用户 |
| [技术白皮书](LSHN_Technical_Whitepaper.md) | 完整理论框架与技术细节 | 研究人员 |
| [架构文档](architecture.md) | 数学形式化、动力学方程、生物学映射 | 算法工程师 |
| **本文档** | 模块接口、张量规格、代码示例 | 开发者 |
