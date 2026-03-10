"""
Stage 1 实验: 单模块动力学验证
================================
逐一验证各核心模块的时间动力学行为:
- 液态门控细胞 (快/慢变量分离)
- 双稳态超图突触 (STDP + 资格迹 + 双势阱)
- 轴突延迟学习
- 全局神经调节器 (ACh/NE/DA + 星形胶质门控)
- 变分自由能引擎 + 预算控制
- 三时钟同步
- 皮层集成层

用法:
    python experiments/stage1_dynamics/run_single_module.py
"""
import sys
from pathlib import Path
import torch

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from lshn.core.cells import LiquidGatedCell
from lshn.core.synapses import BistableHypergraphSynapse, AxonalDelayModule
from lshn.core.plasticity import HomeostaticController
from lshn.engine import (
    ClockSyncEngine, FreeEnergyEngine, SpikeBudgetController,
    GlobalNeuromodulator,
)
from lshn.layers.cortex import CorticalLayer


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ───────────── 1. 液态门控细胞 ─────────────

def test_liquid_cell():
    sep("1. 液态门控细胞 — 快/慢变量分离")

    N = 20
    cell = LiquidGatedCell(num_neurons=N)
    cell.reset_hidden()

    firing_rates = []
    for t in range(200):
        # I_syn: (N,) 突触电流
        I_syn = (torch.randn(N) > 0.5).float() * 2.0
        spk, v = cell.step_fast(I_syn)
        firing_rates.append(spk.sum().item())

        if (t + 1) % 100 == 0:
            # 慢时钟: 更新 g_slow
            g_slow_signal = torch.randn(N) * 0.1
            cell.step_slow(g_slow_signal)
            modulation = cell.get_plasticity_modulation()
            avg_rate = cell.get_firing_rate().mean().item()
            print(f"  t={t+1}ms | 脉冲数={sum(firing_rates[-100:]):.0f}/100步 | "
                  f"avg_rate={avg_rate:.3f} | g_slow调制范围=[{modulation.min():.3f}, {modulation.max():.3f}]")

    print(f"  总脉冲: {sum(firing_rates):.0f} / {200*N} 可能 (稀疏度: {1 - sum(firing_rates)/(200*N):.3f})")


# ───────────── 2. 双稳态超图突触 ─────────────

def test_bistable_synapse():
    sep("2. 双稳态超图突触 — STDP + 双势阱")

    N = 10  # 神经元数
    syn = BistableHypergraphSynapse(num_neurons=N, out_channels=1)

    # 构建超图索引: 随机连接
    E = 8  # 超边数
    node_ids = torch.randint(0, N, (E,))
    edge_ids = torch.arange(E)
    hyperedge_index = torch.stack([node_ids, edge_ids])

    # 模拟 STDP 相互作用
    for t in range(50):
        x_in = (torch.randn(N, N) > 0.5).float()
        post_spk = (torch.rand(N) > 0.7).float()
        out = syn.step_fast(x_in, hyperedge_index, post_spk=post_spk)

    # 检查 STDP 迹
    print(f"  pre_trace  范围: [{syn.pre_trace.min():.4f}, {syn.pre_trace.max():.4f}]")
    print(f"  post_trace 范围: [{syn.post_trace.min():.4f}, {syn.post_trace.max():.4f}]")

    # 慢时钟: 双势阱结构更新
    syn.step_slow_structure(M_global=0.5, R_replay=0.1, T_temp=0.3)

    s_e = syn.s_e.detach()
    alive = syn.get_alive_mask()
    print(f"  s_e 均值: {s_e.mean():.4f} | 存活超边: {alive.sum()}/{syn.max_edges}")
    print(f"  有效权重范围: [{syn.get_effective_weights().min():.4f}, {syn.get_effective_weights().max():.4f}]")


# ───────────── 3. 轴突延迟 ─────────────

def test_axonal_delay():
    sep("3. 轴突延迟学习 — 环形缓冲区 + 延迟敏感STDP")

    E = 20  # 超边数/连接数
    delay_mod = AxonalDelayModule(max_edges=E, max_delay=15)
    delay_mod.reset()

    for t in range(100):
        pre_spk = (torch.rand(E) > 0.8).float()
        post_spk = (torch.rand(E) > 0.8).float()
        delayed_spk, stdp_delta = delay_mod.step_fast(pre_spk, post_spk)

        if t >= 90:
            print(f"  t={t}: 前脉冲={int(pre_spk.sum())} | "
                  f"后脉冲={int(post_spk.sum())} | "
                  f"延迟输出={int(delayed_spk.sum())} | "
                  f"STDP delta范围=[{stdp_delta.min():.4f}, {stdp_delta.max():.4f}]")

    # 慢时钟: 更新延迟值
    e_trace = torch.rand(E) * 0.5
    timing_error = torch.randn(E) * 0.1
    delay_mod.update_delays(e_trace, timing_error)

    stats = delay_mod.get_delay_stats()
    print(f"  延迟统计: mean={stats['delay_mean']:.2f}, "
          f"std={stats['delay_std']:.2f}, "
          f"max={stats['delay_max']:.2f}")


# ───────────── 4. 全局神经调节器 ─────────────

def test_neuromodulator():
    sep("4. 全局神经调节器 — ACh/NE/DA + 星形胶质门控")

    N = 50
    mod = GlobalNeuromodulator(num_neurons=N)

    for t in range(5):
        pred_err = 0.8 - t * 0.15  # 预测误差逐步减小
        firing_rate = 0.05 + t * 0.02

        result = mod.step_slow(
            prediction_error=pred_err,
            firing_rate=firing_rate,
        )

        lr_scale = mod.get_learning_rate_scale()
        temp = mod.get_temperature()
        third = mod.get_third_factor()

        print(f"  慢步{t}: pred_err={pred_err:.2f} | "
              f"ACh={result['ACh'].item():.4f} | "
              f"NE={result['NE'].item():.4f} | "
              f"DA={result['DA'].item():.4f} | "
              f"plasticity_gate={result['plasticity_gate'].mean():.4f} | "
              f"lr_scale={lr_scale.item():.4f}")


# ───────────── 5. VFE + 预算控制 ─────────────

def test_vfe_and_budget():
    sep("5. 变分自由能 + 脉冲预算控制")

    vfe = FreeEnergyEngine(kl_weight=0.01, energy_lambda=0.001)
    budget = SpikeBudgetController(target_spikes_per_step=30)
    budget.reset()

    for t in range(5):
        pred_error = torch.randn(10) * (0.8 - t * 0.1)
        s_e = torch.sigmoid(torch.randn(20))
        spike_count = 20 + t * 10  # 脉冲数逐步增加

        vfe_dict = vfe.compute_vfe(
            prediction_error=pred_error,
            s_e_tensor=s_e,
            active_neurons_ratio=0.6,
            synaptic_events=spike_count,
            precision=1.0,
        )

        budget_result = budget.step_control(spike_count)

        print(f"  步{t}: VFE={vfe_dict['vfe_total']:.4f} | "
              f"acc_loss={vfe_dict['accuracy_loss']:.4f} | "
              f"structure_kl={vfe_dict['structure_kl']:.4f} | "
              f"energy={vfe_dict['energy_cost']:.4f} | "
              f"spikes={spike_count} | "
              f"budget_theta_adj={budget_result['theta_adj']:.4f}")

    # VFE 分解报告
    report = vfe.get_decomposition_report()
    print(f"\n  VFE 分解报告:")
    for k, v in report.items():
        print(f"    {k}: {v:.6f}")


# ───────────── 6. 三时钟同步 ─────────────

def test_clock_sync():
    sep("6. 三时钟同步引擎")

    clock = ClockSyncEngine()

    slow_count = 0
    ultra_count = 0

    for t in range(2000):
        trigger_slow, trigger_ultra = clock.tick()
        if trigger_slow:
            slow_count += 1
        if trigger_ultra:
            ultra_count += 1

    print(f"  2000 快步中:")
    print(f"    慢时钟触发: {slow_count} 次 (预期 20)")
    print(f"    超慢时钟触发: {ultra_count} 次 (预期 2)")


# ───────────── 7. 皮层集成层 ─────────────

def test_cortical_layer():
    sep("7. 皮层集成层 — 全模块联动")

    N, E = 30, 15
    cortex = CorticalLayer(in_channels=N, num_neurons=N, num_groups=3, max_edges=E)
    neuromod = GlobalNeuromodulator(num_neurons=N)

    hyperedge_index = torch.stack([
        torch.randint(0, N, (E,)),
        torch.arange(E),
    ])

    total_spikes = 0
    for t in range(200):
        x = (torch.randn(N, N) > 0.5).float()
        spk = cortex.step_fast(x, hyperedge_index)
        total_spikes += int(spk.sum().item())

        if (t + 1) % 100 == 0:
            # 慢时钟
            mod = neuromod.step_slow(prediction_error=0.5, firing_rate=total_spikes / (100 * N))
            global_e = mod["DA"].expand(N)
            cortex.step_slow(global_e, M_global=0.5, R_replay=0.1, T_temp=mod["NE"].item())

            spike_count = cortex.get_spike_count_and_reset()
            print(f"  t={t+1}ms | 窗口脉冲={spike_count} | "
                  f"ACh={mod['ACh'].item():.3f} | NE={mod['NE'].item():.3f}")
            total_spikes = 0

    # 超慢时钟: 结构演化
    cortex.step_ultra_slow(VFE_full=0.5, VFE_masked_dict={})
    print(f"  超慢时钟结构演化完成")


# ───────────── 主入口 ─────────────

def main():
    print("LSHN Stage 1: 单模块动力学验证")
    print("=" * 60)

    test_liquid_cell()
    test_bistable_synapse()
    test_axonal_delay()
    test_neuromodulator()
    test_vfe_and_budget()
    test_clock_sync()
    test_cortical_layer()

    sep("所有模块验证完成")


if __name__ == "__main__":
    main()
