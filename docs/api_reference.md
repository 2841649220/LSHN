# LSHN API 参考文档

> **版本**: v0.1.1 　|　**更新**: 2026年8月 　|　**作者**: Apocalypse 　|　**GitHub**: [LSHN](https://github.com/2841649220/LSHN)

本文档提供 LSHN（液态脉冲超图网络）所有核心模块的完整 API 参考，包括类定义、方法签名、参数说明和使用示例。本文档与 `lshn/` 源码逐一对应，请以源码为准。

---

## 目录

1. [LSHNModel 主类](#lshnmodel-主类)
2. [核心模块 (Core)](#核心模块-core)
3. [全局引擎 (Engine)](#全局引擎-engine)
4. [网络层 (Layers)](#网络层-layers)
5. [工具函数 (Utils)](#工具函数-utils)
6. [配置参数](#配置参数)
7. [已知限制与研究方向](#已知限制与研究方向)

---

## LSHNModel 主类

### 类定义

```python
class LSHNModel(nn.Module)
```

液态脉冲超图网络端到端模型：输入编码层 → 海马体快速学习层 → 皮层核心网络层 → 输出解码层，外加五大全局引擎（时钟同步 / 自由能 / 神经调质 / 预算控制 / 回放巩固）。

### 构造函数

```python
def __init__(
    self,
    input_dim: int = 128,          # 原始输入维度
    hidden_dim: int = 256,         # 海马体/编码器隐层维度
    num_neurons: int = 1000,       # 皮层神经元数
    num_groups: int = 10,          # 皮层功能分区数 (隐式MoE列数)
    max_edges: int = 500,          # 超图最大超边数
    initial_classes: int = 2,      # 初始分类头类别数
    enable_dendrites: bool = False,# 是否启用树突区室非线性
    enable_active_inference: bool = False,  # 预留: 主动推理引擎尚未接入主流程 (启用会告警)
    target_spikes_per_step: int = 50,      # 脉冲预算目标 (每样本每快步)
    device=None, dtype=None,       # factory kwargs
    cfg: Optional[dict] = None,    # 配置字典 (与 configs/default.yaml 同构)
)
```

**cfg 结构**（完整键清单见 §6 配置参数）：`encoder / hippocampus / cortex / cell / synapse / axonal_delay / three_factor / homeostatic / implicit_moe / evolution / neuromodulator / astrocyte / free_energy / budget / clocks`。

### 主要方法

#### forward_step

```python
def forward_step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]
```

单步前向（1ms 快时钟）。完整数据流：`x → MODWTEncoder → SpikingAutoEncoder → hippo_to_cortex → CorticalLayer → DynamicExpansionHead`。

- `x`: (batch, input_dim) 原始输入
- `target`: (batch, num_classes) one-hot 目标（提供时触发误差脉冲生成与三因素可塑性）

返回 dict：
| 键 | 形状 | 说明 |
|---|---|---|
| `output` | (batch, num_classes) | 类别 logits（解码器输出） |
| `spk_cortex` | (batch, num_neurons) | 皮层脉冲（detached） |
| `spk_hippo` | (batch, hidden_dim) | 海马体脉冲（detached） |
| `spk_encoded` | (batch, hidden_dim) | 编码层脉冲（detached） |
| `recon_loss` | 标量（可微） | 海马体重构损失 `MSE(decoder(spk_hippo), spk_encoded)`，**仅训练模式且有 target 时存在** |

每 100 快步触发慢时钟（VFE/调质/结构双势阱/回放/预算 PI），每 1000 快步触发超慢时钟（凋亡生发/离线回放）。

#### expand_classes

```python
def expand_classes(self, num_new_classes: int)
```

动态扩容输出头。内部调用 `decoder.expand(n)`；**新增参数需由调用方注册进优化器**：

```python
model.expand_classes(2)
optimizer.add_param_group({"params": model.decoder.expand(2)})
```

#### get_monitoring_report

```python
def get_monitoring_report(self) -> Dict[str, float]
```

可解释监控报告（白皮书 §3.1.2 硬性交付）。键包括：
- VFE 分解：`vfe_recent_mean / J_recent_mean / accuracy_trend / complexity_trend / energy_trend / vfe_total_history_len`
- 调质：`modulator_ACh / modulator_NE / modulator_DA / modulator_plasticity_gate / modulator_surprise`
- 预算：`budget_theta_adj / budget_inh_adj / budget_error / budget_integral_error / budget_lambda_E_adj`
- 结构：`alive_edges_ratio`（s_e>0.05）/ `alive_edges_mask_ratio`（结构剪枝掩码）/ `alive_neurons_ratio` / `mean_firing_rate` / `e_trace_abs_max`（资格迹发散监控）
- 时钟：`clock_fast_steps / clock_slow_steps`
- 延迟：`delay_mean / delay_std / delay_min / delay_max / delay_entropy`

#### reset / reset_sample_state

```python
def reset(self)
# 任务边界全量重置: 时钟/膜电位/迹/稳态/引擎 EMA/λ_E/预算积分 (保留权重与结构知识)

def reset_sample_state(self)
# 样本级瞬态清理: 膜电位/树突电位/延迟环形缓冲/prev_spk (保留权重、迹与引擎状态)。
# 训练脚本在每个 batch 起点调用, 保证"每样本 fast_steps 快步"语义独立。
```

### 使用示例

```python
import torch
from lshn import LSHNModel

model = LSHNModel(input_dim=128, hidden_dim=256, num_neurons=1000,
                  num_groups=10, max_edges=500, initial_classes=2,
                  device='cuda').train().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(64, 128, device='cuda')
y = torch.zeros(64, 2, device='cuda'); y[:, 0] = 1.0

for _ in range(20):
    out = model.forward_step(x, y)
loss = torch.nn.CrossEntropyLoss()(out['output'], torch.zeros(64, dtype=torch.long, device='cuda'))
loss.backward()
optimizer.step()
print(model.get_monitoring_report()['mean_firing_rate'])
```

---

## 核心模块 (Core)

### LiquidGatedCell — 多尺度液态门控元胞

```python
class LiquidGatedCell(nn.Module)
LiquidGatedCell(num_neurons: int, tau_v: float = 10.0, tau_g_fast: float = 5.0,
                tau_g_slow: float = 200.0, tau_a: float = 100.0, theta_0: float = 1.0,
                enable_dendrites: bool = False, num_branches: int = 4,
                dendrite_threshold: float = 0.3, noise_std: float = 0.01,
                input_gain: float = 1.0, device=None, dtype=None)
```

- `step_fast(I_syn, I_ext=None, I_inh=None, theta_ie=None) -> (spk, v)`：膜电位更新 + STE 脉冲 + 软重置 + 快门控。膜电位按样本持有 (batch, N)；输入电流经 `input_gain / _input_std_ema` 自适应归一化（EMA 仅训练模式更新，纯 GPU 侧 torch.where，无设备同步）。
- `step_slow(global_e)`：慢门控 g_slow + 适应变量 a + 阈值 θ = θ₀ + a。
- `update_delta_window(delta_spk)`：记录误差脉冲到 100 步窗口（GPU 索引写入）。
- `get_plasticity_modulation()` / `get_firing_rate()`：g_slow 与发放率读取。

状态：`v`（persistent=False，batch 变化时重建）、`g_fast / g_slow / a / theta`、`spk_window / delta_window`（100 步环形）。

### DendriteCompartment — 树突亚室非线性（可选）

```python
DendriteCompartment(num_neurons, num_branches=4, dendrite_threshold=0.3, device=None, dtype=None)
```

分支独立积分 + 局部 Ca 尖峰（超过阈值部分 ×2 放大），输出汇聚到胞体。分支电位为均值场状态（batch 平均回写）。

### BistableHypergraphSynapse — 双势阱脉冲超图突触

```python
class BistableHypergraphSynapse(nn.Module)
BistableHypergraphSynapse(in_channels: int, out_channels: int, w_max: float = 1.0,
                          alpha: float = 0.1, beta: float = 0.05, trace_decay: float = 0.9,
                          tau_pre: float = 20.0, tau_post: float = 20.0, device=None, dtype=None)
```

- 变量：快权重 `w_hat` (max_edges,)、结构变量 `s_e`（双势阱慢演化，requires_grad=False）、资格迹 `e_trace`（多跳传播，clamp ±10）、STDP 迹 `pre_trace / post_trace`、共发放窗口 `coact_window`（10 慢步环形）。
- 有效权重：`w_e = w_max · s_e · w_hat`。
- `step_fast(x_in, hyperedge_index, post_spk=None, g_slow=None, delayed_pre=None) -> (batch, max_edges)`：STDP 迹 + 多跳资格迹（`g_slow 门控 × (1−λ) × 组均值`，scatter 实现）+ 消息传递输出。
- `step_slow_structure(M_global, R_replay, T_temp, dt_slow=0.1)`：双势阱 Langevin 更新，噪声按势垒深度标定 `σ = √(2·dt·T·α/4)`（防止噪声淹没双稳性）。
- `aggregate_spikes_to_edges(spk, hyperedge_index, reduction="mean")`：节点脉冲 → 超边级聚合（batch 均值语义）。
- `set_local_group_adjacency(adj)`：多跳传播的组邻接（同时推导 edge_group_ids）。
- `get_effective_weights()` / `get_alive_mask(threshold=0.05)` / `reset()`。

**hyperedge_index 约定**：形状 (2, N_connections)，`index[0]` = 源神经元 idx，`index[1]` = 超边 idx（< max_edges）。默认拓扑每条超边连接 K=8 个随机源神经元，随模型检查点持久化（固定种子 2026 生成，训练/评估一致）。

### AxonalDelayModule — 轴突延迟学习

```python
AxonalDelayModule(max_edges: int, max_delay: int = 20, min_delay: int = 1,
                  delay_lr: float = 0.001, device=None, dtype=None)
```

- 每条超边维护连续延迟 `delay_continuous`（离散化 `delay_discrete`，仅慢时钟 `update_delays` 时重算）。
- `step_fast(pre_spk, post_spk) -> (delayed_spk, stdp_delta)`：环形缓冲 (batch, max_delay, max_edges) 存储前突触脉冲，按延迟读取（gather 实现）；延迟学习迹 `pre_trace_delayed / post_trace`。
- `update_delays(e_trace, timing_error)`：`Δd = lr · e_trace · timing_error`（慢时钟）。
- `get_delay_stats()`：延迟分布统计（监控报告用）。

### ThreeFactorPlasticity — 三因素可塑性

```python
ThreeFactorPlasticity(learning_rate: float = 0.01, trace_decay: float = 0.9,
                      importance_lambda: float = 0.0)
forward(w_hat, e_trace, error_spk, neuromodulator=None, plasticity_gate=1.0, importance=None)
```

`Δw = η · e_trace · error_spk · mod · gate / (1 + λ_imp · importance)`，就地更新并 clamp ±1。`importance_lambda>0` 时启用 EWC-lite 结构重要性保护（可选实验机制，默认关闭）。

### PoissonErrorEncoder — 泊松误差编码

```python
PoissonErrorEncoder(f_max: float = 1.0)
forward(pred, target, precision: float = 1.0) -> error_spk ∈ {-1, 0, +1}
```

**误差符号：`error = target - pred`**（正 = 欠预测 → 权重增大 → 负反馈纠错）。零基准泊松率 `f_max·min(1, |error|·precision)`。全程 no_grad（采样不可微）。

### HomeostaticController — 稳态可塑性控制

```python
HomeostaticController(num_neurons, target_rate=0.05, scaling_strength=0.1, ie_lr=0.001,
                      tau_rate=100.0, device=None, dtype=None)
```

- `step_fast(spk)`：发放率 EMA（tau_rate 以快时钟步计）。
- `step_slow()`：突触缩放因子 + 内在兴奋性阈值调整 `Δθ_ie = lr·(rate − target)`。
- `apply_to_weights(w_hat, neuron_to_edge_map=None, alive_neuron_mask=None)`：乘性幂缩放 `(target/rate)^β`（clamp [0.5, 2.0]）；`alive_neuron_mask` 提供时死神经元关联边缩放强制 1.0（防饱和放大）。

### PruneGrowthModule — 凋亡生发

```python
PruneGrowthModule(max_neurons, max_edges, prune_threshold=0.0, min_alive_ratio=0.3,
                  growth_probability=0.05, min_history_samples=10, min_alive_edge_ratio=0.3,
                  device=None, dtype=None)
```

- `step_ultra_slow_evolution(VFE_full, VFE_masked_dict, task_importance_mask=None, hyperedge_index=None)`：
  - 因果贡献度 `Contribution_e = VFE(w_e=0) − VFE(full)`，NaN 安全均值（未评估边不参与）
  - **历史满 min_history_samples 槽才执行剪枝**（防训练初期单样本误剪）
  - **最低超边存活比例保护**（剪后低于 min_alive_edge_ratio 回退本次剪枝）
  - `task_importance_mask` 保护（显式 bool 转换）
  - 拓扑感知神经元凋亡（无存活关联边的神经元）
  - 生发联动边复活：复活神经元同时复活其关联超边（edge_mask）

### ImplicitMoE — 无中心隐式 MoE（侧向抑制）

```python
ImplicitMoE(num_neurons, num_groups, inhibition_strength=0.5, device=None, dtype=None)
forward(spk) -> I_inh
```

`I_inh_i = strength · (S_group(i) − spk_i)`，组和 scatter 实现，O(N)。批输入逐样本竞争（皮层 prev_spk 为逐样本 buffer）。

---

## 全局引擎 (Engine)

### ClockSyncEngine — 三层时钟同步

```python
ClockSyncEngine(fast_ms: int = 1, slow_ms: int = 100, ultra_slow_ms: int = 1000)
tick() -> (trigger_slow, trigger_ultra_slow)
```

属性：`fast_steps / slow_steps / ultra_slow_steps / steps_since_slow / steps_per_slow`（窗口长度，脉冲归一化用）。

### FreeEnergyEngine — 变分自由能

```python
FreeEnergyEngine(kl_weight=0.01, energy_lambda=0.001, energy_lambda_lr=0.0001)
compute_vfe(prediction_error, s_e_tensor, activity, synaptic_events=None, precision=1.0) -> dict
compute_energy_regularization_gradient(current_events, target_budget) -> λ_E
get_decomposition_report() -> dict
reset()
```

`VFE = precision·E[err²] + kl_weight·(D_KL(s_e‖0.5) + activity)`；`J = VFE + λ_E·E[synaptic_events]`。λ_E 自适应 clamp [0, 0.1]；`synaptic_events` 为**每样本每快步平均脉冲数**（与预算目标同单位）。

### SpikeBudgetController — 脉冲预算 PI 控制

```python
SpikeBudgetController(target_spikes_per_step, kp=0.01, ki=0.001, max_integral=100.0,
                      theta_adj_scale=0.1, inh_adj_scale=0.05)
step_control(current_spikes) -> dict
```

输入为**每样本每快步平均脉冲数**（`cortex.get_spike_count_and_reset()` 的返回值，已按窗口长度归一化）。PI 控制 + anti-windup，输出 `theta_adj / inh_adj`（供 model 应用）与监控量。

### GlobalNeuromodulator — ACh/NE/DA + 星形胶质门控

```python
GlobalNeuromodulator(num_neurons, tau_ach=200.0, tau_ne=100.0, tau_da=150.0, tau_ca=500.0,
                     device=None, dtype=None)
step_slow(prediction_error, firing_rate, reward_signal=0.0, ood_score=0.0) -> dict
```

- **时间常数以慢时钟步计**（每步 = 100ms；τ=200 步 ≈ 20s）
- ACh ∈ [0.1, 2.0]（精度，直接作误差编码 precision）；NE ∈ [0.01, 2.0]（温度）；DA ∈ [0.001, 1.0]（第三因子，curiosity = max(0, surprise−0.1) 驱动探索）
- 返回：`ACh / NE / DA / plasticity_gate / surprise`（标量张量）

### AstrocyteGate — 星形胶质门控

```python
AstrocyteGate(num_neurons, tau_astro=500.0, device=None, dtype=None)
step_slow(mean_prediction_error, mean_firing_rate) -> plasticity_gate ∈ [0,1]
```

钙浓度慢变量（固定门控权重 buffer），`gate = sigmoid(0.3·err − 0.2·rate + bias)` 的指数平滑。

### ActiveInferenceEngine — 主动推理（预留）

```python
ActiveInferenceEngine(state_dim, obs_dim, num_policies=8, gamma=1.0, device=None, dtype=None)
update_belief(observation, prediction_error)
compute_efe(current_state) -> (G, components)
select_policy(current_state) -> (idx, info)
```

**尚未接入 LSHNModel 主流程**（`enable_active_inference=True` 仅产生告警）。独立可用；接入前需重写 info_gain 与信念更新（当前公式为占位性质，见 §7）。

---

## 网络层 (Layers)

### MODWTEncoder — 多尺度编码前端

```python
MODWTEncoder(in_features, out_features, num_scales=3, device=None, dtype=None)
forward(x: (batch, in)) -> spikes: (batch, out) 二值脉冲 (STE)
```

**注意**：当前实现为"多尺度线性提取 + 可学习阈值偏置 + 泊松编码"，**并非严格 MODWT 小波分解**（真实小波需时间维输入，预留扩展）。可学习阈值 `threshold`（init 1.0）压低基线发放率，使编码层脉冲能耗进入可控范围。

### DynamicExpansionHead — 动态扩容输出头

```python
DynamicExpansionHead(in_features, initial_classes=2, device=None, dtype=None)
expand(num_new_classes) -> [weight, bias]   # 重建 Parameter, 返回当前参数对象
forward(x) -> logits
```

- 初始行 std 0.1；扩容新行 std 0.01（架构文档 §3.4 `W_new ~ N(0, 0.01)`），bias 新行为 0。
- **必须重建 Parameter 而非重赋 `.data`**（否则 `AddmmBackward0 invalid gradient`）。
- 调用方契约：`optimizer.add_param_group({"params": model.decoder.expand(n)})`；旧 Parameter 对象留在原 group 中为 no-op。扩容重建使新参数 Adam 矩从零开始——实测构成任务边界"优化器热启动"，新任务学习速度显著优于保留旧矩的分块方案。

### SpikingAutoEncoder — 海马体脉冲自编码器

```python
SpikingAutoEncoder(input_dim, hidden_dim, input_gain=1.0, device=None, dtype=None)
step_fast(x_in) -> spk   # 编码通路
decode(spk_hidden) -> recon   # 解码通路
reconstruction_loss(spk_hidden, target_spk) -> MSE
```

解码器经 `model.forward_step` 的 `recon_loss` 训练（并入训练总损失，权重 `training.recon_weight`），使回放生成器使用有意义的解码投影。

### ReplayGenerator — 可控采样动力学回放

```python
ReplayGenerator(hidden_dim, leakage=0.1, momentum=0.9)
init_state(batch_size, device, dtype)
inject_pattern(pattern, inject_rate=0.3)   # 模式吸引: state = (1-λ)·state + λ·pattern
generate_step(ae_decoder, temperature=0.1) -> pseudo_spk (二值, 固定阈值 0.5)
reset()
```

泄漏 + 二阶动量 + 模式吸引的采样动力学（白皮书 §3.6.2）。`_state / _velocity` 为 persistent=False buffer。在线回放（每慢时钟）注入 `W_dec^T·S_hippo` 吸引项；离线回放（每超慢时钟）伪脉冲经皮层传播巩固共发放结构。

### CorticalLayer — 皮层核心网络层

```python
CorticalLayer(in_channels, num_neurons, num_groups, max_edges, enable_dendrites=False,
              inhibition_strength=0.5, input_gain=1.0, device=None, dtype=None, cfg=None)
```

整合元胞/超图突触/轴突延迟/隐式MoE/三因素可塑性/稳态控制/凋亡生发。

- `step_fast(x_in, hyperedge_index, theta_ie=None, inh_scale=1.0) -> spk`：输入聚合 → 延迟 → 超图消息传递 → edge_to_neuron 映射 → 侧抑制（逐样本 prev_spk）→ 元胞动力学。
- `apply_plasticity(error_spk, error_neuron=None, neuromodulator=None, plasticity_gate=1.0, importance=None)`：三因素更新 + delta_window。
- `step_slow(global_e, M_global, R_replay, T_temp)`：g_slow / 双势阱 / 稳态缩放（死神经元跳过）/ 延迟学习。
- `step_ultra_slow(VFE_full, VFE_masked_dict, task_importance_mask=None, hyperedge_index=None)`：凋亡生发。
- `get_spike_count_and_reset() -> float`：**每样本每快步平均脉冲数**（窗口累加 ÷ 窗口步数，仅训练模式累加）。

---

## 工具函数 (Utils)

### ContinualLearningMetrics — 持续学习指标

```python
ContinualLearningMetrics(num_tasks: int)
update_accuracy(trained_task_idx, eval_task_idx, acc)
record_spike_sparsity(spk) / record_sparsity_value(sparsity)
average_accuracy(current_task_idx) / forgetting_measure(current_task_idx)
get_average_sparsity() / report(current_task_idx)
```

R 矩阵 (num_tasks, num_tasks)，`R[i,j]` = 学习任务 i 后在任务 j 上的准确率。遗忘度 `F_k = 1/(k−1)·Σ_j (max_{l≤k−1} R[l,j] − R[k,j])`。单检查点评估（R 仅一行）时遗忘度无意义，eval.py 会跳过输出。

---

## 配置参数

完整配置文件：`configs/default.yaml`（21 个配置节，全部被代码消费；历史遗留无消费键已删除并在注释中说明去向）。

| 配置节 | 关键键 | 语义 |
|---|---|---|
| `model` | num_neurons / max_edges / enable_dendrites 等 | 模型规模 |
| `clocks` | fast_ms / slow_ms / ultra_slow_ms | 三层时钟周期（被 ClockSyncEngine 消费） |
| `cortex` | input_gain | 皮层有效输入驱动量级 |
| `cell` | tau_fast / tau_slow / threshold / noise_std / dendrite_* | 元胞动力学（tau_slow 以慢时钟步计） |
| `synapse` | tau_pre / tau_post | STDP 迹时间常数 |
| `axonal_delay` | max_delay / delay_lr / min_delay | 延迟学习 |
| `three_factor` | lr / f_max / importance_lambda | 三因素学习率 / 误差编码频率 / EWC-lite 强度（默认 0 关闭） |
| `homeostatic` | target_rate / tau_rate / scaling_strength / ie_lr | 稳态可塑性（tau_rate 以快时钟步=ms 计） |
| `implicit_moe` | inhibition_strength | 侧抑制强度 |
| `evolution` | prune_threshold / growth_probability / min_alive_ratio | 剪枝生发 |
| `neuromodulator` | tau_ach / tau_ne / tau_da | 调质时间常数（以慢时钟步计） |
| `astrocyte` | tau_ca | 钙动力学（慢时钟步） |
| `free_energy` | kl_weight / energy_lambda / energy_lambda_lr | VFE 分解 |
| `budget` | target_spikes_per_step / kp / ki / integral_max | 脉冲预算（目标 = 每样本每快步） |
| `hippocampus` | input_gain | 海马体驱动量级 |
| `encoder` | num_scales | 多尺度提取器数 |
| `training` | batch_size / lr / epochs / fast_steps_per_sample / seed / recon_weight / gradient_clip | 训练 |
| `continual` | tasks / classes_per_task | 持续学习任务序列 |
| `data` | data_dir / num_workers / pin_memory | 数据加载 |
| `device` | cuda | 设备开关 |

**时间常数单位**：快时钟步 = 1ms（`homeostatic.tau_rate` 等）；慢时钟步 = 100ms（`cell.tau_slow`、`neuromodulator.tau_*`、`astrocyte.tau_ca` 等）。配置注释已逐项标注。

---

## 已知限制与研究方向

1. **跨任务遗忘（旗舰研究问题）**：当前共享皮层表征 + 全局误差信号的架构在合成数据 5 任务基准上**逐任务学习可达 ~100%**，但旧任务在新任务训练后显著遗忘（见 experiments/stage2 的 R 矩阵）。已实现的候选机制：
   - 输出头"旧权重冻结"（分块参数设计曾实现，实测新任务学习 100% → ~52%，稳定性-可塑性难以兼得，已回退为单参数重建）
   - EWC-lite 结构重要性保护（`three_factor.importance_lambda`，实测保留率提升有限且牺牲新任务学习速度，默认关闭）
   - 回放巩固（在线注入 + 离线皮层传播）已接线，但海马体仅编码当前任务，无法回放旧任务模式
   - **下一步方向**：任务条件化/稀疏专家分区、旧任务生成式回放（海马体多任务保留）、按任务冻结 + 弹性共享（如 PackNet/Progressive Networks 思路）
2. **主动推理未接线**：`ActiveInferenceEngine` 独立可用，但其 info_gain 公式为占位性质（非真实后验熵差），接入前需重写。
3. **MODWT 为线性多尺度近似**：真实小波分解需时间维输入；时间序列任务可在编码层前置小波滤波器组（db2/db4）。
4. **混合精度未实现**：SNN 状态变量需 FP32 保护，`device.mixed_precision` 配置已移除；后续版本可引入 autocast + 状态围栏。
5. **Triton 算子路径未启用**：热路径已做纯 PyTorch 优化（消除每步设备同步、kernel 启动合并、缓存），大规模（50 万神经元 +）场景可在 Linux GPU 服务器启用 Triton 融合 kernel（液态元胞动力学整段、多跳组归约、泊松采样）。

---

**版本**: v0.1.1 　|　**最后更新**: 2026年8月 　|　**作者**: Apocalypse 　|　**GitHub**: [LSHN](https://github.com/2841649220/LSHN)
