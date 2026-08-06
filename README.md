# LSHN — 液态脉冲超图网络

面向持续学习的类脑脉冲网络系统:以液态门控细胞、双稳态超图突触与变分自由能最小化为核心,事件驱动、低功耗、可解释。

> 完整文档入口见 [docs/README.md](docs/README.md)(架构、API、白皮书、开发计划均在其中导航)。

## 核心特性

- **多时间尺度时钟**:1 ms 快 / 100 ms 慢 / 1000 ms 超慢三级时钟解耦脉冲传播、突触可塑性与结构演化
- **液态门控细胞 + 双稳态超图突触**:树突区室非线性、STDP 与三因素学习规则,事件驱动稀疏计算
- **海马体–皮层双系统**:自编码器快速编码 + 慢时钟生成式回放,实现类脑系统巩固
- **隐式 MoE + 脉冲预算控制器**:侧抑制稀疏激活,自适应阈值/抑制维持目标发放率
- **变分自由能引擎**:VFE 分解监控与自适应能量正则化,全程可解释

## 快速开始

```bash
pip install -r requirements.txt   # 或 pip install -e .
python scripts/train.py --synthetic    # 合成数据最小训练 (每个样本 20 快步)
python scripts/eval.py --synthetic     # 评估
```

```python
from lshn import LSHNModel
model = LSHNModel(input_dim=128, hidden_dim=256, num_neurons=1000, num_groups=10)
for x, target in dataloader:
    for _ in range(20):                # 每样本 20 个快时间步 (20 ms)
        output = model.forward_step(x, target)
    report = model.get_monitoring_report()   # VFE/神经调质/脉冲预算监控
```

## 文档导航

| 文档 | 说明 |
| --- | --- |
| [docs/README.md](docs/README.md) | 完整文档入口(推荐从这里开始) |
| [docs/architecture.md](docs/architecture.md) | 系统架构与模块设计 |
| [docs/api_reference.md](docs/api_reference.md) | API 参考 |
| [docs/LSHN_Technical_Whitepaper.md](docs/LSHN_Technical_Whitepaper.md) | 技术白皮书 |
| [docs/plan_260301.md](docs/plan_260301.md) | 开发计划 |

## 信息

- 版本:v0.1.1
- 许可证:MIT
- GitHub:https://github.com/2841649220/LSHN
