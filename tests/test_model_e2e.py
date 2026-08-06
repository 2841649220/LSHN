"""
测试: LSHNModel 端到端集成测试
覆盖: 白皮书 §4.1 四层架构, 三时钟同步, VFE引擎, 神经调节, 预算控制

这些测试验证整个模型的数据流、时钟触发、动态扩容、
监控报告和持续学习指标等端到端功能。
"""
import math
import torch
import pytest
from lshn.model import LSHNModel
from lshn.utils.metrics import ContinualLearningMetrics


# 降低模型规模以加速测试
SMALL_CFG = dict(
    input_dim=32,
    hidden_dim=64,
    num_neurons=128,
    num_groups=4,
    max_edges=64,
    initial_classes=2,
    enable_dendrites=False,
    enable_active_inference=False,
    target_spikes_per_step=20,
)

# 更小规模: 用于训练循环 / 扩容回归 / 梯度活跃验收 (保证单测 < 30s)
EXPAND_CFG = dict(
    input_dim=16,
    hidden_dim=32,
    num_neurons=64,
    num_groups=4,
    max_edges=32,
    initial_classes=2,
    enable_dendrites=False,
    enable_active_inference=False,
    target_spikes_per_step=20,
)


def _device():
    """GPU 可用时用 GPU, 否则 CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _test_devices():
    """参数化设备列表: CPU 必测, CUDA 可用时追加"""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def _seed(seed: int = 0):
    """固定 CPU + GPU 随机种子 (GPU 测试需要单独 seed CUDA RNG)"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TestLSHNModelInit:
    
    def test_basic_init(self):
        """测试模型基本初始化"""
        model = LSHNModel(**SMALL_CFG)
        assert model.input_dim == 32
        assert model.hidden_dim == 64
        assert model.num_neurons == 128
    
    def test_submodule_existence(self):
        """测试所有子模块存在"""
        model = LSHNModel(**SMALL_CFG)
        
        # 四层
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'hippocampus')
        assert hasattr(model, 'replay_generator')
        assert hasattr(model, 'cortex')
        assert hasattr(model, 'decoder')
        
        # 引擎
        assert hasattr(model, 'clock')
        assert hasattr(model, 'vfe_engine')
        assert hasattr(model, 'budget_ctrl')
        assert hasattr(model, 'neuromodulator')
    
    def test_with_dendrites(self):
        """测试启用树突的模型"""
        cfg = dict(SMALL_CFG)
        cfg['enable_dendrites'] = True
        model = LSHNModel(**cfg)
        assert model.cortex.cell.enable_dendrites


class TestForwardStep:
    
    @pytest.fixture
    def model(self):
        return LSHNModel(**SMALL_CFG)
    
    def test_single_step_no_target(self, model):
        """测试单步前向（无目标）"""
        x = torch.randn(4, 32)  # batch=4
        result = model.forward_step(x)
        
        assert "output" in result
        assert "spk_cortex" in result
        assert "spk_hippo" in result
        assert "spk_encoded" in result

        # 解码层 F.linear(spk_cortex, weight, bias) 保留 batch 维:
        # SMALL_CFG 下 batch=4, initial_classes=2 → 精确 (4, 2)
        assert result["output"].shape == (4, 2)
    
    def test_single_step_with_target(self, model):
        """测试单步前向（带目标，触发三因素可塑性）"""
        x = torch.randn(4, 32)
        target = torch.zeros(4, 2)
        target[:, 0] = 1.0  # one-hot
        
        result = model.forward_step(x, target=target)
        assert "output" in result
    
    def test_multiple_fast_steps(self, model):
        """测试连续多个快时钟步"""
        x = torch.randn(4, 32)
        
        for step in range(10):
            result = model.forward_step(x)
        
        assert model.clock.fast_steps == 10
    
    def test_spk_cortex_is_binary(self, model):
        """测试皮层脉冲为二值"""
        x = torch.randn(4, 32)
        result = model.forward_step(x)
        
        spk = result["spk_cortex"]
        assert torch.all((spk == 0.0) | (spk == 1.0))


class TestClockTriggers:
    
    def test_slow_clock_at_100(self):
        """测试第100步触发慢时钟"""
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)
        
        for step in range(100):
            model.forward_step(x)
        
        assert model.clock.slow_steps == 1
        # VFE 应被计算
        assert model._last_vfe is not None
        # 调制信号应被更新
        assert model._last_modulation is not None
    
    def test_ultra_slow_clock_at_1000(self):
        """测试第1000步触发超慢时钟"""
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)
        
        for step in range(1000):
            model.forward_step(x)
        
        assert model.clock.ultra_slow_steps == 1
        assert model.clock.slow_steps == 10
    
    def test_modulation_dict_fields(self):
        """测试慢时钟后调制信号包含所有字段"""
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)
        
        for step in range(100):
            model.forward_step(x)
        
        mod = model._last_modulation
        assert "ACh" in mod
        assert "NE" in mod
        assert "DA" in mod
        assert "plasticity_gate" in mod


class TestDynamicExpansion:
    
    def test_expand_classes(self):
        """测试动态类别扩容"""
        model = LSHNModel(**SMALL_CFG)
        
        # 初始 2 类
        x = torch.randn(2, 32)
        r1 = model.forward_step(x)
        initial_out_dim = r1["output"].shape[-1]
        assert initial_out_dim == 2
        
        # 扩容到 4 类
        model.expand_classes(2)
        r2 = model.forward_step(x)
        new_out_dim = r2["output"].shape[-1]
        assert new_out_dim == 4
    
    def test_expand_classes_multiple(self):
        """测试多次扩容"""
        model = LSHNModel(**SMALL_CFG)
        model.expand_classes(3)  # 2 → 5
        model.expand_classes(2)  # 5 → 7
        
        x = torch.randn(2, 32)
        result = model.forward_step(x)
        assert result["output"].shape[-1] == 7


class TestMonitoring:
    
    def test_monitoring_report_empty(self):
        """测试无慢时钟更新时的监控报告"""
        model = LSHNModel(**SMALL_CFG)
        report = model.get_monitoring_report()
        
        # 即使没有慢时钟更新，结构统计应存在
        assert "alive_edges_ratio" in report
        assert "alive_neurons_ratio" in report
        assert "mean_firing_rate" in report
    
    def test_monitoring_report_after_slow_clock(self):
        """测试慢时钟后的完整监控报告"""
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)
        
        for _ in range(100):
            model.forward_step(x)
        
        report = model.get_monitoring_report()
        
        # VFE 分解
        assert "vfe_recent_mean" in report
        assert "J_recent_mean" in report
        
        # 调质
        assert "modulator_ACh" in report
        assert "modulator_NE" in report
        assert "modulator_DA" in report
        
        # 预算 (PI 控制真实输出键: theta_adj / budget_error)
        assert "budget_theta_adj" in report
        assert "budget_error" in report
        
        # 结构
        assert "alive_edges_ratio" in report
        assert "delay_mean" in report
    
    def test_monitoring_values_finite(self):
        """测试监控报告中所有值有限"""
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)
        
        for _ in range(100):
            model.forward_step(x)
        
        report = model.get_monitoring_report()
        for key, val in report.items():
            assert isinstance(val, (int, float)), f"{key} type is {type(val)}"
            assert math.isfinite(val), f"{key} 非有限值: {val}"


class TestReset:
    
    def test_reset_clears_state(self):
        """测试 reset 清除所有状态"""
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)
        
        for _ in range(50):
            model.forward_step(x)
        
        model.reset()

        assert model.clock.fast_steps == 0
        assert model._last_modulation is None
        assert model._last_vfe is None
        # 脉冲累加器与窗口步数清零 (训练期间已累加过, reset 必须归零)
        assert model.cortex._spike_acc.item() == 0
        assert model.cortex._window_steps == 0


class TestContinualLearningMetrics:
    
    def test_accuracy_tracking(self):
        """测试准确率矩阵更新"""
        metrics = ContinualLearningMetrics(num_tasks=5)
        
        metrics.update_accuracy(0, 0, 0.95)
        metrics.update_accuracy(1, 0, 0.80)
        metrics.update_accuracy(1, 1, 0.90)
        
        assert metrics.R[0, 0].item() == pytest.approx(0.95)
        assert metrics.R[1, 0].item() == pytest.approx(0.80)
        assert metrics.R[1, 1].item() == pytest.approx(0.90)
    
    def test_average_accuracy(self):
        """测试平均准确率计算"""
        metrics = ContinualLearningMetrics(num_tasks=3)
        
        metrics.update_accuracy(0, 0, 0.90)
        assert metrics.average_accuracy(0) == pytest.approx(0.90)
        
        metrics.update_accuracy(1, 0, 0.80)
        metrics.update_accuracy(1, 1, 0.85)
        avg = metrics.average_accuracy(1)
        assert avg == pytest.approx((0.80 + 0.85) / 2)
    
    def test_forgetting_measure(self):
        """测试遗忘率计算"""
        metrics = ContinualLearningMetrics(num_tasks=3)
        
        # 任务0后对0的准确率
        metrics.update_accuracy(0, 0, 0.95)
        # 任务0的遗忘 = 0 (只有一个任务)
        assert metrics.forgetting_measure(0) == 0.0
        
        # 任务1后对0的准确率下降了
        metrics.update_accuracy(1, 0, 0.70)
        metrics.update_accuracy(1, 1, 0.90)
        
        # 遗忘 = (max_past_R[*,0] - R[1,0]) / 1 = (0.95 - 0.70) / 1 = 0.25
        assert metrics.forgetting_measure(1) == pytest.approx(0.25)
    
    def test_spike_sparsity(self):
        """测试脉冲稀疏度记录 (sparsity = 1 - 发放密度)"""
        metrics = ContinualLearningMetrics(num_tasks=3)

        # (batch, num_neurons) 0/1 脉冲张量: 5% 元素为 1 → 密度 0.05, 稀疏度 0.95
        spk = torch.zeros(10, 100)
        spk[:, :5] = 1.0  # 5% 活跃

        metrics.record_spike_sparsity(spk)
        assert metrics.get_average_sparsity() == pytest.approx(0.95)
    
    def test_report(self):
        """测试完整报告"""
        metrics = ContinualLearningMetrics(num_tasks=3)
        metrics.update_accuracy(0, 0, 0.90)
        
        report = metrics.report(0)
        assert "avg_accuracy" in report
        assert "forgetting" in report
        assert "avg_sparsity" in report


class TestGradientFlow:

    def test_output_has_grad(self):
        """测试模型输出支持梯度"""
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32, requires_grad=True)

        result = model.forward_step(x)
        loss = result["output"].sum()
        loss.backward()

        assert x.grad is not None


class TestExpandEndToEnd:
    """
    扩容后继续训练 端到端回归测试 (bug 验收):
    旧实现 DynamicExpansionHead.expand 用 `.data` 重赋 Parameter,
    扩容后第一次 backward 抛 AddmmBackward0 invalid gradient。
    """

    @staticmethod
    def _train_batch(model, optimizer, device):
        """单次完整训练步: forward + loss + backward + optimizer.step"""
        x = torch.randn(4, EXPAND_CFG["input_dim"], device=device)
        target = torch.zeros(4, model.decoder.num_classes, device=device)
        target[:, 0] = 1.0  # one-hot 指向第一个类别

        optimizer.zero_grad()
        result = model.forward_step(x, target=target)
        loss = result["output"].mean()
        loss.backward()
        optimizer.step()
        return loss

    def test_expand_then_continue_training(self):
        """扩容 + 重建优化器后继续训练: 不抛异常且 loss 有限"""
        device = _device()
        _seed(0)
        model = LSHNModel(**EXPAND_CFG).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randn(4, EXPAND_CFG["input_dim"], device=device)

        # 预热: 膜电位积分 ~10 步后网络才开始发放 (与真实训练一致)
        for _ in range(15):
            model.forward_step(x)

        # 训练几个 batch
        for _ in range(3):
            loss = self._train_batch(model, optimizer, device)
            assert torch.isfinite(loss).item()

        # 动态扩容 2 个新类别
        model.expand_classes(2)
        assert model.decoder.num_classes == EXPAND_CFG["initial_classes"] + 2

        # 重建优化器 (与 scripts/train.py 扩容后一致: 新 Parameter 旧 Adam 态失效)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 扩容后继续训练 — 旧代码此处 backward 崩溃
        for _ in range(3):
            loss = self._train_batch(model, optimizer, device)
            assert torch.isfinite(loss).item()


class TestGradientActivity:
    """
    梯度活跃验收测试 (bug 验收):
    网络初始化静默曾导致梯度死亡 — 关键参数梯度恒为 0, 训练 loss 恒为 log2。
    """

    def test_gradients_alive_after_training(self):
        """训练 5 个 batch 后关键参数梯度非零, 网络非静默"""
        device = _device()
        _seed(0)  # 固定 CPU + CUDA RNG: 只 seed CPU 时 GPU 采样流随进程内
        # 之前测试的 CUDA 采样量漂移, 导致同一测试多次运行结果不同
        model = LSHNModel(**EXPAND_CFG).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randn(4, EXPAND_CFG["input_dim"], device=device)
        target = torch.zeros(4, model.decoder.num_classes, device=device)
        target[:, 0] = 1.0

        # 预热: 膜电位需积分 ~10 步才产生首个脉冲 (非 bug, 是神经元动力学)
        for _ in range(15):
            model.forward_step(x, target=target)

        max_decoder_grad = 0.0
        for _ in range(5):
            optimizer.zero_grad()
            result = model.forward_step(x, target=target)
            result["output"].mean().backward()
            # 记录该 batch 的解码器权重梯度 (逐 batch 零梯度前的瞬时值)
            max_decoder_grad = max(
                max_decoder_grad,
                model.decoder.weight.grad.abs().sum().item(),
            )
            optimizer.step()

        # 网络非静默: 有脉冲发放 (梯度死亡时 spk_window 全零)
        firing_rate = model.cortex.cell.get_firing_rate()
        assert firing_rate.mean().item() > 0.0

        # 关键参数梯度非零 (梯度死亡时这些恒为 0)。
        # 注意: 解码器权重梯度由该 batch 的二进制脉冲值决定, 极小模型 +
        # 稀疏发放时某一步恰好无皮层脉冲会使其为 0 (采样伪影, 非梯度死亡);
        # 故断言 5 个 batch 中最大的单 batch 解码器梯度非零 — 任一发放
        # batch 即满足, 对脉冲脱落鲁棒, 且梯度死亡时仍恒为 0。
        assert max_decoder_grad > 0.0

        # w_hat 经 STE 通路稳定非零; 全局用聚合梯度断言 (跨参数求和,
        # 对小样本噪声鲁棒)。
        w_hat_grad = model.cortex.synapse.w_hat.grad
        assert w_hat_grad is not None
        assert w_hat_grad.abs().sum().item() > 0.0

        total_grad = sum(
            p.grad.abs().sum().item()
            for p in model.parameters()
            if p.grad is not None
        )
        assert total_grad > 0.0


class TestVFEEngine:
    
    def test_vfe_after_training_steps(self):
        """测试训练步后 VFE 被正确计算"""
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)
        target = torch.zeros(2, 2)
        target[:, 0] = 1.0
        
        for _ in range(100):
            model.forward_step(x, target=target)
        
        vfe = model._last_vfe
        assert vfe is not None
        assert "vfe_total" in vfe
        assert "J_total" in vfe
        assert "accuracy_loss" in vfe
        assert "complexity_loss" in vfe
        assert "energy_cost" in vfe
    
    def test_vfe_history_accumulated(self):
        """测试 VFE 历史累积"""
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)

        for _ in range(300):
            model.forward_step(x)

        # 3 个慢时钟周期
        assert model.clock.slow_steps == 3
        assert len(model.vfe_engine.history["vfe_total"]) == 3

    def test_vfe_activity_as_4th_positional_param(self):
        """compute_vfe 第 4 位置参数 activity 计入复杂度项"""
        fe = LSHNModel(**SMALL_CFG).vfe_engine
        pe = torch.zeros(4)
        s_e = torch.full((8,), 0.5)

        r0 = fe.compute_vfe(pe, s_e, activity=0.0, synaptic_events=0)
        r1 = fe.compute_vfe(pe, s_e, activity=0.25, synaptic_events=0)
        # 复杂度 = structure_kl + activity; activity 增大 → 复杂度/VFE 增大
        assert r1["complexity_loss"] > r0["complexity_loss"]
        assert r1["vfe_total"] > r0["vfe_total"]
        assert r1["complexity_loss"] == pytest.approx(
            r1["structure_kl"] + 0.25
        )


class TestOnlineReplay:

    def test_replay_runs_at_slow_clock(self):
        """测试在线回放在慢时钟触发 (真实断言)"""
        _seed(0)
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)

        # 跑100步触发慢时钟（含在线回放）
        for _ in range(100):
            model.forward_step(x)

        # 回放状态已初始化: (1, hidden_dim)
        assert model.replay_generator._state is not None
        assert model.replay_generator._state.shape == (1, model.hidden_dim)

        # R_replay ∈ [0, 1] (伪脉冲平均活动度)
        R_replay = model._run_online_replay(0.1)
        assert 0.0 <= R_replay <= 1.0

        # 两次调用之间回放状态发生变化 (动力学推进, 非静止)
        s1 = model.replay_generator._state.clone()
        model._run_online_replay(0.1)
        assert not torch.allclose(model.replay_generator._state, s1)


class TestEvalMode:
    """eval 阶段引擎状态冻结回归测试 (修复后行为: 推理不污染引擎/权重)"""

    def test_eval_does_not_update_engines(self):
        """eval 模式跑 100 步 (过慢时钟边界): 引擎状态全部保持"""
        _seed(0)
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)
        s_e_before = model.cortex.synapse.s_e.clone()
        w_before = model.cortex.synapse.w_hat.clone()
        ach_before = model.neuromodulator.ACh.clone()

        model.eval()
        for _ in range(100):
            model.forward_step(x)

        # 慢时钟被触发但引擎不推进
        assert model.clock.slow_steps == 1
        assert model._last_vfe is None
        assert model._last_modulation is None
        assert model._budget_theta_adj == 0.0
        assert model._budget_inh_adj == 0.0
        # 结构/权重/调质不变
        assert torch.allclose(model.cortex.synapse.s_e, s_e_before)
        assert torch.allclose(model.cortex.synapse.w_hat, w_before)
        assert torch.allclose(model.neuromodulator.ACh, ach_before)
        assert model.neuromodulator.ACh.item() == pytest.approx(1.0)
        # eval 不累加脉冲计数
        assert model.cortex._spike_acc.item() == 0
        assert model.cortex._window_steps == 0

    def test_eval_no_recon_loss_key(self):
        """eval + target 时 forward_step 不输出 recon_loss 键 (仅训练可微)"""
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)
        target = torch.zeros(2, 2)
        target[:, 0] = 1.0

        # 训练模式: 有 recon_loss
        result = model.forward_step(x, target=target)
        assert "recon_loss" in result
        assert torch.isfinite(result["recon_loss"])

        # eval 模式: 无 recon_loss 键
        model.eval()
        result = model.forward_step(x, target=target)
        assert "recon_loss" not in result


class TestSlowClockOnDevices:
    """慢时钟跨设备回归: CPU 必测, CUDA 可用时附加 GPU 测试"""

    @pytest.mark.parametrize("device", _test_devices())
    def test_slow_clock_across_device(self, device):
        """设备上跑 105 步 (过慢时钟边界): VFE 计算、预算调整有限、无异常"""
        if device.type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")
        _seed(0)
        model = LSHNModel(**SMALL_CFG).to(device)
        x = torch.randn(4, SMALL_CFG["input_dim"], device=device)

        for _ in range(105):
            model.forward_step(x)

        # 慢时钟已触发且 VFE 已计算
        assert model.clock.slow_steps == 1
        assert model._last_vfe is not None
        assert model._last_vfe["vfe_total"] is not None
        # 预算调整量有限
        assert math.isfinite(model._budget_theta_adj)
        assert math.isfinite(model._budget_inh_adj)
        # 监控报告全有限
        report = model.get_monitoring_report()
        for key, val in report.items():
            assert math.isfinite(val), f"{key} 非有限值: {val}"


class TestBehaviorRegression:
    """行为变更回归断言 (超图持久化 / 样本级状态 / 脉冲计数语义 / 时钟)"""

    def test_hyperedge_index_persistent_buffer(self):
        """超图拓扑为持久 buffer, 固定种子 2026 生成, 可复现"""
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)
        model.forward_step(x)  # 惰性生成

        assert model._hyperedge_index.shape[0] == 2
        assert model._hyperedge_index.numel() > 0
        # 持久 buffer: 进入 state_dict
        assert "_hyperedge_index" in model.state_dict()
        # 固定种子 → 跨实例可复现
        model2 = LSHNModel(**SMALL_CFG)
        model2.forward_step(x)
        assert torch.equal(model._hyperedge_index, model2._hyperedge_index)

    def test_hyperedge_index_state_dict_roundtrip(self):
        """检查点拓扑严格加载 (惰性空 buffer → 检查点形状兼容)"""
        model = LSHNModel(**SMALL_CFG)
        model.forward_step(torch.randn(2, 32))
        sd = model.state_dict()

        model2 = LSHNModel(**SMALL_CFG)
        model2.load_state_dict(sd, strict=True)  # 修复前 size mismatch
        assert torch.equal(model2._hyperedge_index, model._hyperedge_index)

    def test_reset_sample_state(self):
        """reset_sample_state 清样本级状态, 不动引擎/权重"""
        _seed(0)
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)
        for _ in range(10):
            model.forward_step(x)
        assert model.cortex.cell.v.shape == (2, model.num_neurons)

        model.reset_sample_state()
        assert torch.all(model.cortex.cell.v == 0.0)
        assert torch.all(model.cortex.prev_spk == 0.0)
        assert torch.all(model.cortex.axonal_delay.spike_buffer == 0.0)
        # 引擎与权重保留
        assert model.clock.fast_steps == 10
        assert model.cortex.synapse.w_hat.abs().sum().item() > 0.0

    def test_spike_count_normalized_per_step(self):
        """脉冲计数语义: 返回每样本每快步平均值, 非窗口累计值"""
        _seed(0)
        model = LSHNModel(**SMALL_CFG)
        x = torch.randn(2, 32)

        for _ in range(10):
            model.forward_step(x)

        count = model.cortex.get_spike_count_and_reset()
        # 窗口均值 ≤ 单样本总脉冲上限 (num_neurons), 非负
        assert 0.0 <= count <= model.num_neurons
        # 重置后累加器/窗口步数清零
        assert model.cortex._spike_acc.item() == 0
        assert model.cortex._window_steps == 0
        assert model.cortex.get_spike_count_and_reset() == 0.0

    def test_clock_sync_properties(self):
        """ClockSyncEngine: steps_per_slow / steps_since_slow 语义"""
        model = LSHNModel(**SMALL_CFG)
        assert model.clock.steps_per_slow == 100

        for _ in range(50):
            model.forward_step(torch.randn(2, 32))
        assert model.clock.steps_since_slow == 50

        for _ in range(50):
            model.forward_step(torch.randn(2, 32))
        # 慢时钟触发后 steps_since_slow 归零
        assert model.clock.steps_since_slow == 0
        assert model.clock.slow_steps == 1


class TestEndToEndContinualLearning:
    
    def test_two_task_sequence(self):
        """测试两任务连续学习序列"""
        model = LSHNModel(**SMALL_CFG)
        metrics = ContinualLearningMetrics(num_tasks=2)
        
        # Task 0: 训练
        x0 = torch.randn(4, 32)
        t0 = torch.zeros(4, 2)
        t0[:, 0] = 1.0
        
        for _ in range(50):
            model.forward_step(x0, target=t0)
        
        # 评估 Task 0
        with torch.no_grad():
            r0 = model.forward_step(x0)
        metrics.update_accuracy(0, 0, 0.8)  # 模拟准确率
        
        # Task 1: 扩容并训练
        model.expand_classes(2)  # 2 → 4
        x1 = torch.randn(4, 32)
        t1 = torch.zeros(4, 4)
        t1[:, 2] = 1.0
        
        for _ in range(50):
            model.forward_step(x1, target=t1)
        
        metrics.update_accuracy(1, 0, 0.6)
        metrics.update_accuracy(1, 1, 0.75)
        
        report = metrics.report(1)
        assert report["avg_accuracy"] > 0.0
        assert "forgetting" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
