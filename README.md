# 液态脉冲超图网络 (LSHN)

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg" alt="PyTorch 2.2+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/2841649220/LSHNN"><img src="https://img.shields.io/badge/GitHub-LSHNN-green.svg" alt="GitHub"></a>
</p>

<p align="center">
  <strong>面向持续学习的类脑智能系统</strong> |
  <strong>变分自由能最小化驱动</strong> |
  <strong>事件驱动低功耗</strong>
</p>

<p align="center">
  <strong>版本</strong>: v0.9 Beta | <strong>作者</strong>: Apocalypse
</p>

---

## 快速预览

```python
from lshn import LSHNModel

# 初始化 50 万神经元规模的持续学习系统
model = LSHNModel(
    input_dim=128,
    hidden_dim=1024,
    num_neurons=500000,      # 50 万皮层神经元
    num_groups=100,          # 100 个功能分区（隐式 MoE）
    max_edges=50000,         # 5 万条超边
    enable_dendrites=True,   # 启用树突非线性
)

# 训练循环：每个样本运行 20 个快时间步（20ms）
for x, target in dataloader:
    for _ in range(20):
        output = model.forward_step(x, target)

    # 获取可解释监控报告（VFE 分解、神经调质状态、预算状态）
    report = model.get_monitoring_report()
```

---

## 核心特性

### 类脑持续学习机制

| 机制 | 技术实现 | 生物学对应 |
|:---:|:---|:---|
| **多时间尺度解耦** | 快 (1ms) / 慢 (100ms) / 超慢 (1000ms) 三层时钟 | 离子通道 vs 神经调质 vs 结构可塑性 |
| **双势阱突触** | 结构变量 $s_e \in [0,1]$ 在双势阱中演化 | LTP/LTD 分子双稳态 |
| **三因素学习规则** | 资格迹 × 误差脉冲 × 神经调质 | 突触可塑性三因素理论 |
| **海马体-皮层双系统** | 快速编码 + 生成式回放 | 系统巩固理论 |
| **神经元-超边协同演化** | 基于因果贡献度的无损剪枝 | 神经发生/凋亡机制 |

### 工程优化特性

| 特性 | 描述 | 效果 |
|:---:|:---|:---:|
| **事件驱动能效** | 仅当神经元发放时更新突触电流 | 功耗降低 5-10× |
| **混合精度训练** | BF16/FP8 加速计算 + FP32 保护 SNN 状态变量 | 速度↑ 显存↓ |
| **FP8 实验性加速** | NVIDIA H100/RTX 4090 的 8 位浮点格式（E4M3/E5M2） | 显存减少 50% |
| **冷知识归档** | INT4 量化压缩失活超边至 NVMe | 无限学习容量 |
| **脉冲预算控制** | PI 控制器动态调节阈值/抑制强度 | 目标能耗 |
| **动态输出扩容** | 增量分类任务自动增加输出神经元 | 无需重训练 |

---

## 系统架构

```text
输入编码层 → 海马体快速学习层 → 皮层 LSHN 核心网络层 → 输出解码层
                 ↑                                    ↓
             生成式回放 ←──────────────────────  冷知识归档器
                 ↓
         全局神经调节器 (ACh/NE/DA)
                 ↓
         变分自由能引擎 (VFE 监控)
```

### 四层前向通路

#### 1. 输入编码层 (MODWTEncoder)

- **多尺度重叠离散小波变换**（MODWT）
- 支持 Haar/DB4 小波基，3 层分解
- 泊松编码生成脉冲序列

#### 2. 海马体快速学习层 (SpikingAutoEncoder)

- 脉冲自编码器架构
- 高可塑性 LIF 神经元
- 生成在线/离线回放信号

#### 3. 皮层核心网络层 (CorticalLayer)

- **50 万液态门控元胞**（可选树突非线性）
- **双势阱超图突触**（5 万条超边）
- **隐式 MoE**（100 功能分区 + 侧向抑制）
- 轴突延迟学习 + 三因素可塑性
- 稳态可塑性控制 + 凋亡生发机制

#### 4. 输出解码层 (DynamicExpansionHead)

- 脉冲计数解码
- 动态类别扩容
- 支持多任务头

### 五大核心引擎

| 引擎 | 职责 | 关键功能 |
|:---:|:---|:---|
| `ClockSyncEngine` | 多时间尺度时钟同步 | 1ms/100ms/1000ms 事件触发 |
| `FreeEnergyEngine` | 变分自由能计算 | VFE = F + λ_E · E[events] 分解监控 |
| `GlobalNeuromodulator` | 神经调质计算 | ACh(精度)/NE(温度)/DA(第三因子) |
| `SpikeBudgetController` | 能量预算反馈 | PI 控制维持目标脉冲数 |
| `KnowledgeArchiver` | 冷知识归档 | INT4 量化压缩失活超边 |

---

## 安装指南

### 环境要求

| 组件 | 最低版本 | 推荐配置 |
|:---:|:---:|:---|
| **Python** | 3.12+ | 3.12+ |
| **PyTorch** | 2.2.0+ | 2.2.0+（支持 `torch.autocast` 混合精度） |
| **CUDA** | 12.1+ | RTX 4090/A100/H100（大规模仿真） |
| **显存** | 24G | 96G（支持 50 万神经元 + 5 万超边） |

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/2841649220/LSHNN.git
cd LSHNN

# 安装依赖
pip install -r requirements.txt

# （可选）开发模式安装
pip install -e .
```

### 依赖清单

```text
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.24.0
snntorch>=0.7.0
pyyaml>=6.0
typing-extensions>=4.5.0
safetensors>=0.4.0
```

---

## 使用示例

### 基础训练循环

```python
import torch
from lshn import LSHNModel

# 初始化模型
model = LSHNModel(
    input_dim=128,
    num_neurons=100000,  # 10 万神经元（单卡验证规模）
    num_groups=50,
    max_edges=10000,
    device='cuda'
)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(100):
    for batch_x, batch_target in dataloader:
        batch_x = batch_x.cuda()
        batch_target = batch_target.cuda()

        # 每个样本运行 20 个快时间步（20ms）
        for step in range(20):
            output_dict = model.forward_step(batch_x, batch_target)

        # 计算损失（使用输出解码）
        loss = torch.mean((output_dict['output'] - batch_target) ** 2)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 获取监控报告
    report = model.get_monitoring_report()
    print(f"Epoch {epoch}: VFE={report['vfe_recent_mean']:.4f}, "
          f"ACh={report['modulator_ACh']:.3f}, "
          f"存活超边={report['alive_edges_ratio']:.2%}")
```

### 增量学习任务（类别扩容）

```python
# 任务 1：类别 0-1
model = LSHNModel(initial_classes=2)
train(task_1_data)

# 任务 2：新增类别 2-3
new_params = model.expand_classes(2)  # 扩容 2 个新类别
optimizer.add_param_group({'params': new_params})
train(task_2_data)

# 任务 3：再新增类别 4-5
new_params = model.expand_classes(2)
optimizer.add_param_group({'params': new_params})
train(task_3_data)

# 验证所有旧任务（无灾难性遗忘）
for task in [task_1_data, task_2_data, task_3_data]:
    acc = evaluate(task)
    print(f"Task accuracy: {acc:.2%}")
```

### 可解释监控仪表盘

```python
report = model.get_monitoring_report()

# VFE 分解
print(f"VFE 总量：{report['vfe_recent_mean']:.4f}")
print(f"预测误差：{report['accuracy_trend']:.4f}")
print(f"结构复杂度：{report['complexity_trend']:.4f}")
print(f"能量代价：{report['energy_trend']:.4f}")

# 神经调质状态
print(f"ACh(精度): {report['modulator_ACh']:.3f}")
print(f"NE(温度): {report['modulator_NE']:.3f}")
print(f"DA(第三因子): {report['modulator_DA']:.3f}")

# 结构演化状态
print(f"存活超边比例：{report['alive_edges_ratio']:.2%}")
print(f"存活神经元比例：{report['alive_neurons_ratio']:.2%}")
print(f"平均发放率：{report['mean_firing_rate']:.2%}")

# 轴突延迟统计
print(f"平均延迟：{report['delay_mean']:.2f}ms")
print(f"延迟熵：{report['delay_entropy']:.3f}")
```

### 冷知识归档配置

```python
# 启用 INT4 冷归档（NVMe 路径）
model = LSHNModel(
    archive_dir="/mnt/nvme/cold_archive",  # 建议 NVMe SSD
    cold_threshold=0.05,                   # s_e < 0.05 判定为冷边
)

# 超慢时钟（每 1000ms）自动触发：
# 1. 检测冷边：~edge_mask OR s_e < 0.05
# 2. INT4 量化压缩
# 3. 导出至 archive_dir
# 4. 重置冷槽供重新生长
```

---

## 配置系统

### 默认配置 (configs/default.yaml)

```yaml
# 模型架构
model:
  input_dim: 128
  hidden_dim: 1024
  num_neurons: 500000      # 50 万皮层神经元
  num_groups: 100          # 100 个功能分区
  max_edges: 50000         # 5 万条超边
  enable_dendrites: true   # 树突非线性

# 多时间尺度时钟
clocks:
  fast_ms: 1               # 快时钟 (脉冲传播)
  slow_ms: 100             # 慢时钟 (突触/调质)
  ultra_slow_ms: 1000      # 超慢时钟 (结构演化)

# 混合精度
precision:
  mixed_precision: true
  autocast_dtype: "bfloat16"  # 可选: "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"

# FP8 实验性配置（需 PyTorch 2.1+ 和 H100/RTX 4090）
fp8:
  enabled: false           # 默认关闭，设为 true 启用实验性 FP8
  format: "e4m3"           # 可选: "e4m3" (E4M3) 或 "e5m2" (E5M2)
  protected_states:        # 以下状态强制使用 FP32
    - "neuron_v"          # 神经元膜电位
    - "neuron_s"          # 神经元发放状态
    - "synapse_trace"     # 资格迹

# 冷知识归档
archiver:
  enabled: true
  archive_dir: "./cold_archive"
  cold_threshold: 0.05
```

### CLI 参数覆盖

```bash
# 使用 10 万神经元规模训练
python scripts/train.py --num_neurons 100000 --hidden_dim 512

# 禁用混合精度（调试用）
python scripts/train.py --mixed_precision false

# 启用 FP8 实验性加速（需 H100/RTX 4090）
python scripts/train.py --fp8_enabled true --fp8_format e4m3

# 更改脉冲预算目标
python scripts/train.py --target_spikes_per_step 2500
```

---

## 混合精度训练与 FP8 加速

### BF16 混合精度（默认）

LSHN 默认使用 BF16（Brain Floating Point 16）混合精度训练：

- **计算加速**：BF16 提供与 FP32 相似的数值范围，适合深度学习训练
- **显存优化**：相比 FP32 可减少约 50% 显存占用
- **状态保护**：SNN 状态变量（膜电位、发放状态、资格迹）强制使用 FP32

### FP8 实验性加速（PyTorch 2.1+）

FP8 是 NVIDIA H100 和 RTX 4090 引入的 8 位浮点格式，提供两种变体：

| 格式 | 指数位 | 尾数位 | 数值范围 | 适用场景 |
|:---:|:---:|:---:|:---:|:---|
| **E4M3** | 4 位 | 3 位 | ±448 | 前向激活、权重 |
| **E5M2** | 5 位 | 2 位 | ±57344 | 梯度计算 |

**优势**：
- 相比 BF16 可进一步减少 50% 显存占用
- 在支持 TensorFloat-32 的硬件上可获得额外加速

**限制**：
- 需要 PyTorch 2.1 或更高版本
- 仅支持 NVIDIA H100、RTX 4090 及更新架构
- SNN 状态变量仍需 FP32 保护（膜电位动态范围敏感）

**启用方式**：

```python
# 方式 1：配置文件中启用
precision:
  autocast_dtype: "float8_e4m3fn"  # 或 "float8_e5m2"

# 方式 2：CLI 参数启用
python scripts/train.py --autocast_dtype float8_e4m3fn
```

---

## 项目结构

```text
LSHN_Project/
├── lshn/                          # 核心库
│   ├── model.py                   # 端到端模型 (LSHNModel)
│   ├── layers/
│   │   ├── io/                    # 输入输出层
│   │   │   ├── modwt_encoder.py          # MODWT 小波编码
│   │   │   └── dynamic_expansion_head.py # 动态扩容头
│   │   ├── hippocampus/           # 海马体层
│   │   │   ├── spiking_ae.py             # 脉冲自编码器
│   │   │   └── replay_generator.py       # 回放生成器
│   │   └── cortex/                # 皮层层
│   │       ├── cortical_layer.py         # 皮层核心
│   │       └── implicit_moe.py           # 隐式 MoE
│   ├── core/
│   │   ├── cells/                 # 神经元元胞
│   │   │   └── liquid_cell.py            # 液态门控元胞
│   │   ├── synapses/              # 突触模块
│   │   │   ├── bistable_hypergraph.py    # 双势阱超图
│   │   │   └── axonal_delay.py           # 轴突延迟
│   │   ├── plasticity/            # 可塑性模块
│   │   │   ├── three_factor.py           # 三因素学习
│   │   │   └── homeostatic.py            # 稳态控制
│   │   └── evolution/             # 结构演化
│   │       └── prune_growth.py           # 凋亡生发
│   └── engine/                    # 全局引擎
│       ├── clock_sync.py                 # 时钟同步
│       ├── free_energy.py                # VFE 计算
│       ├── global_modulator.py           # 神经调质
│       ├── budget_control.py             # 脉冲预算
│       └── knowledge_archiver.py         # 知识归档
├── scripts/
│   ├── train.py                   # 训练脚本
│   └── eval.py                    # 评估脚本
├── configs/
│   └── default.yaml               # 默认配置
├── tests/                         # 单元测试 (120 项)
├── docs/                          # 文档
│   ├── architecture.md            # 形式化架构
│   └── api_reference.md           # API 参考
└── requirements.txt
```

---

## 文档导航

| 文档 | 描述 | 链接 |
|:---:|:---|:---|
| [技术白皮书](docs/LSHN_Technical_Whitepaper.md) | 完整理论框架与技术细节 | [查看](https://github.com/2841649220/LSHNN/blob/main/docs/LSHN_Technical_Whitepaper.md) |
| [架构文档](docs/architecture.md) | 形式化数学描述与符号定义 | [查看](https://github.com/2841649220/LSHNN/blob/main/docs/architecture.md) |
| [API 参考](docs/api_reference.md) | 核心模块张量规格与动力学方程 | [查看](https://github.com/2841649220/LSHNN/blob/main/docs/api_reference.md) |
| [配置文件](configs/default.yaml) | 全部超参数说明 | [查看](https://github.com/2841649220/LSHNN/blob/main/configs/default.yaml) |

### 快速链接

- **项目首页**: https://github.com/2841649220/LSHN
- **技术白皮书**: https://github.com/2841649220/LSHN/blob/main/docs/LSHN_Technical_Whitepaper.md
- **架构文档**: https://github.com/2841649220/LSHN/blob/main/docs/architecture.md
- **API 参考**: https://github.com/2841649220/LSHN/blob/main/docs/api_reference.md

---

## 硬件建议

### 推荐配置

| 规模 | 神经元数 | 超边数 | 推荐硬件 | 显存占用 |
|:---:|:---:|:---:|:---:|:---:|
| **入门** | 1 万 | 1000 | RTX 3090 (24G) | ~8G |
| **验证** | 10 万 | 1 万 | RTX 4090 (24G) / A100 (40G) | ~20G |
| **全规模** | 50 万 | 5 万 | A100 (80G) / H100 (80G) | ~60G |
| **极限** | 100 万 | 10 万 | A100×2 / H100×2 | ~120G |

### 优化技巧

1. **启用混合精度**：`--mixed_precision true`（默认开启）
2. **启用 FP8 加速**（实验性）：`--fp8_enabled true`（需 H100/RTX 4090，可减少 50% 显存）
3. **降低脉冲预算**：`--target_spikes_per_step 2500`（降低 50% 能耗）
4. **禁用树突非线性**：`--enable_dendrites false`（节省 20% 显存）
5. **减少功能分区**：`--num_groups 50`（降低侧抑制计算）

---

## 常见问题

### Q: LSHN 与传统 SNN 的区别？

**A**: LSHN 的核心创新在于：

1. **双势阱突触**：分离快权重 $\hat{w}$ 与慢结构 $s_e$，实现"结构固化、权重微调"
2. **变分自由能统一**：所有模块优化同一目标函数 $\mathcal{J}=\mathcal{F}+\lambda_E\cdot E[\text{events}]$
3. **神经调质闭环**：ACh/NE/DA 动态调节精度/温度/第三因子
4. **冷知识归档**：INT4 压缩失活超边，实现无限学习容量

### Q: 如何验证灾难性遗忘缓解效果？

**A**: 运行增量学习基准测试：

```bash
python scripts/train.py --config configs/default.yaml \
  --dataset split_mnist --tasks 5
python scripts/eval.py --compute_forgetting_metric
```

遗忘率 $F \approx 0$ 表示无灾难性遗忘。

### Q: 冷知识归档如何工作？

**A**: 超慢时钟（每 1000ms）自动触发：

1. 检测冷边：`~edge_mask OR s_e < 0.05`
2. INT4 通道级量化：$\hat{w}_{\text{int4}} = \text{round}\left(\frac{w-\mu}{\sigma}\cdot 7\right)\odot\text{sign}(w)$
3. 导出至 NVMe（元数据 + 量化参数 + 拓扑）
4. 重置冷槽：$w_\text{hat}=0, s_e=0.5$ 供重新生长

### Q: FP8 实验性加速的兼容性如何？

**A**: FP8 支持情况：

- **硬件要求**：NVIDIA H100、RTX 4090 及更新架构
- **软件要求**：PyTorch 2.1 或更高版本
- **格式选择**：
  - E4M3（推荐）：适合前向激活和权重，数值范围 ±448
  - E5M2：适合梯度计算，数值范围 ±57344
- **保护机制**：SNN 状态变量（膜电位、发放状态、资格迹）强制使用 FP32，避免数值不稳定

### Q: 如何适配 MindSpore/昇腾 NPU？

**A**: 当前版本基于 PyTorch 实现。MindSpore 迁移计划：

- **阶段 1**（1-6 月）：PyTorch 机制验证
- **阶段 2**（7-12 月）：MindSpore 框架迁移
- **阶段 3**（13-18 月）：昇腾 NPU 性能优化

---

## 许可证

本项目采用 **MIT 许可证**。详见 [LICENSE](LICENSE) 文件。

---

## 引用

如使用本代码进行研究，请引用：

```bibtex
@misc{lshn2026,
  title={Liquid Spiking Hypergraph Network: A Neuromorphic System for Catastrophe-Free Continual Learning},
  author={Apocalypse},
  year={2026},
  url={https://github.com/2841649220/LSHN}
}
```

---

## 联系方式

- **技术问题**: 提交 GitHub Issue
- **合作意向**: 发送邮件至 2023365722@cmu.edu.cn

---

<p align="center">
  <strong>最后更新</strong>: 2026 年 3 月 |
  <strong>版本</strong>: v0.9 Beta |
  <strong>作者</strong>: Apocalypse
</p>
