# Liquid Spiking Hypergraph Network (LSHN)

液态脉冲超图网络 (LSHN) 是一种受脑启发的下一代持续学习系统，旨在解决深度学习中的灾难性遗忘问题。

## 核心架构

LSHN 结合了以下前沿技术：
- **变分自由能 (VFE) 最小化**: 基于主动推理 (Active Inference) 的学习目标函数。
- **脉冲神经网络 (SNN)**: 模拟生物神经元的高效、异步计算。
- **双势阱超图拓扑**: 将突触表示为可演化的超边，支持高阶特征关联。
- **三因素可塑性**: 结合前/后突触活动与全局神经调节信号 (DA/ACh/NE)。
- **多时间尺度动力学**: 区分快 (1ms)、慢 (100ms) 和超慢 (1000ms) 过程，实现知识的固化。

## 项目结构

- `lshn/core/`: 核心组件 (神经元元胞、双势阱突触、结构演化等)。
- `lshn/layers/`: 架构分层 (输入/输出、海马体快速学习层、皮层核心层)。
- `lshn/engine/`: 运行引擎 (时钟同步、VFE计算、脉冲预算控制、全局调节)。
- `tests/`: 单元测试与端到端测试。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
python -m pytest tests/
```

### 运行示例

```bash
python experiments/stage1_dynamics/run_single_module.py
```

## 状态与路线图

- [x] 核心液态门控元胞 (LiquidGatedCell) 实现
- [x] 双势阱超图突触 (BistableHypergraphSynapse) 与 STDP 迹
- [x] 多时间尺度时钟同步引擎
- [x] 端到端模型架构集成
- [x] 修复实验脚本字典键名不一致 (run_single_module.py)
- [x] 修复 ReplayGenerator 状态管理 (使用 None 初始化避免设备不一致)
- [x] 修复 DynamicExpansionHead 扩容优化器兼容性 (in-place data 更新)
- [x] 修复 model.reset() 遗漏 budget_ctrl 重置
- [x] 修复 train.py 动态模块注册 (使用 add_module)
- [ ] 修复 PyG 超图卷积的维度兼容性问题
- [ ] 验证跨任务持续学习的抗遗忘能力
