# LSHN v0.1.1 修复与优化计划 (2026-08-06)

基于 7 个并行审查代理的发现综合整理。严重级别: **critical** = 训练/评估结果错误, **major** = 行为缺陷。

## 一、Critical 修复(训练正确性)

| # | 问题 | 位置 | 修复方案 |
|---|------|------|----------|
| C1 | 三因素误差符号反了:Δw=+η·e·(pred−target) 是正反馈(放大误差) | three_factor.py:63 | `error = target - pred`,补符号方向单测 |
| C2 | 脉冲计数单位错配 100×:窗口累加和 vs 每步目标 | cortical_layer.py:287-296 | `get_spike_count_and_reset` 按窗口步数归一化(新增 `_window_steps` 计数) |
| C3 | 超图拓扑不持久化+随机重建:eval/续训用全新随机拓扑 | model.py:164-195 | 注册 persistent buffer(进 state_dict)+固定种子 2026+过滤结果缓存(超慢时钟后置脏) |

## 二、Major 修复(行为/数值缺陷)

| # | 问题 | 位置 | 修复方案 |
|---|------|------|----------|
| M1 | 多跳资格迹非压缩(因子≤1.5>1),e_trace 可指数发散 | bistable_hypergraph.py:273-279 | 多跳项 ×(1−λ_e)=0.1;e_trace clamp ±10 保险;监控跟踪 max\|e\| |
| M2 | s_e 双势阱噪声 12× 势垒深度,双稳性失效 | bistable_hypergraph.py:323-326 | σ=√(2·dt·T_eff·ΔU),ΔU=α/4,T_eff=clamp(T,1e-6,0.5) |
| M3 | 热路径每步 2-3 次设备同步(bool 判断 0 维 CUDA 张量 + .item()) | liquid_cell.py:267-269,335 | torch.where GPU 侧 EMA;GPU 索引写 delta_window |
| M4 | eval 期间 _spike_acc/发放率 EMA 无界累积污染 | cortical_layer.py:195-207 | 累积按 self.training 门控;_on_slow_clock eval 分支消费清零 |
| M5 | 样本级状态泄漏:v/延迟缓冲跨 batch 继承(破坏"每样本 20 步"语义) | train.py:218-231 | 新增 model.reset_sample_state()(清 v/dendrite/延迟缓冲/prev_spk),train.py 每 batch 调用 |
| M6 | 隐式 MoE/STDP 用均值场 prev_spk,批内逐样本竞争缺失 | cortical_layer.py:161,185 | prev_spk 改 (batch, N) per-sample buffer(batch 变化重建);STDP 迹保持均值场并注释 |
| M7 | 扩容后优化器全量重建:旧权重 Adam 矩清零,与"旧权重冻结"相悖;expand() 返回 None 契约断裂 | dynamic_expansion_head.py | expand() 返回新增参数;调用方改 add_param_group;守卫/init std 0.01 |
| M8 | MODWT 基线发放率 40-50% 失控(与预算语义冲突) | modwt_encoder.py | sigmoid 前减可学习阈值偏置(init 1.0);STE dtype 统一;合并 GEMM |
| M9 | 回放系统失效:无模式吸引项、R_replay 恒 0.498 信息量为零、伪脉冲被丢弃 | replay_generator.py + model.py | 注入 W_dec^T·spk_hippo 吸引项;R_replay=sigmoid(state).mean();超慢时钟离线回放走皮层 |
| M10 | 海马体解码器从不训练(重构通路缺失) | spiking_ae.py + model.py + train.py | forward_step 输出 recon_loss(MSE(dec(spk_hippo), spk_encoded)),train.py 并入 loss(权重可配) |
| M11 | 剪枝判定过早(单样本历史即剪)+生发空转(复活神经元无存活边)+ 无最低边存活保护 | prune_growth.py | 历史满 10 槽才剪;min_alive_edge_ratio 保护;生发联动复活关联超边 |
| M12 | 死神经元关联边被突触缩放放大到 ±1,复活时爆发 | cortical_layer.py:265-268 | 死神经元映射的边 scale→1.0 |
| M13 | DA 闭环开环(reward 无来源)+curiosity 符号疑似反 | global_modulator.py | curiosity=max(0,surprise−0.1);model 传 reward=1−mean\|err\| |
| M14 | τ 时间常数单位混用(ms vs 慢时钟步),tau_ca=5000 门控冻结 | configs + global_modulator | 配置注释统一"慢时钟步";tau_ca 默认 500 |
| M15 | ACh ×5 与文档不符,precision 语义漂移 | global_modulator.py:144 | 去 ×5,clamp [0.1,2.0] |
| M16 | 轴突延迟学习用错迹(stdp_delta 白算)+ 缓冲 keep 逻辑错误 | cortical_layer.py:271 + axonal_delay.py | timing_error 改用延迟模块自身迹;keep=min(旧批,新批);离散化移慢时钟 |
| M17 | 复杂度项用存活比例而非 E[S] 发放率(文档不符) | free_energy.py + model.py | 传皮层平均发放率 |
| M18 | enable_active_inference 死参数静默失效 | model.py:58 | 存储标志+启用时告警 |
| M19 | 配置 ~20 个死 key(clock/axonal_delay/three_factor.f_max/cell.threshold 等) | configs/default.yaml | 消费关键 key(clocks/axonal_delay/f_max/threshold);删除或标注其余 |
| M20 | 监控报告在 model.reset() 后打印(全 0) | train.py | 报告时机移到训练后、评估前 |

## 三、Minor/优化

- 时钟周期从配置读取 + steps_since_slow 跟踪(clock_sync.py)
- 星形胶质 W_error/W_activity/bias 改 buffer(死参数);活跃比例阈值
- pre/post decay 常数注册 buffer;多跳项改组和 scatter;延迟 gather 混合切片索引
- 输入归一化 EMA 仅训练模式;PoissonErrorEncoder 内部 no_grad
- w_hat 在 optimizer.step 后 clamp(双通道安全网)
- 海马体 delta_window 补误差驱动项
- 报告新增 e_trace_abs_max、alive_edges_mask_ratio
- train.py: seed、batch_size 配置、one-hot 移出循环、删 numpy
- eval.py: 构造即 device、fast_steps 统一、dataloader kwargs、单检查点跳过遗忘度
- setup.py/requirements 同步;根 README.md 恢复
- 测试:新增 7 个模块测试文件 + 修复 e2e 恒真断言 + eval 模式/GPU 慢时钟覆盖

## 四、明确不做的(记录原因)

- **Triton kernel**:本机无 triton(生产 Linux 服务器可启用),本轮以纯 PyTorch 优化为主(kernel 启动/同步消除已覆盖大部分收益),docs 标注 Triton 路径为后续选项
- **真实 MODWT 小波分解**:需要时间维输入,当前管线为单帧特征,保持"多尺度线性+泊松"并更新文档
- **active_inference 完整接线**:引擎公式需重写(info_gain 语义错误),超出本轮范围,保留独立可用
- **解码时间平均**:训练动态已验证(合成数据 100%),单步解码保持,文档说明
- **w_hat 双通道合并**:保持"在线可塑性+BPTT"混合(已文档化),加 clamp 安全网
- **神经元凋亡 90% 阈值**:默认拓扑每条边 8 成员,每神经元约 4 边,"0 存活边"≈">90% 死亡",等价,保留现状

## 五、实施波次

- **Wave 1**(并行 3 代理): A=突触/细胞/延迟; B=可塑性/稳态/演化/引擎; C=IO/海马体/扩容
- **Wave 2**(并行 2 代理): D=cortical_layer.py+model.py(集成); E=脚本
- **Wave 3**: 配置/打包/README
- **Wave 4**: 测试补充与修复
- **Wave 5**(主线程): 全量 pytest + GPU 多任务训练冒烟
- **Wave 6**: git 历史清理(整理至 main 单提交,删除 rust-acceleration 分支痕迹)
- **Wave 7**: docs 全面更新
