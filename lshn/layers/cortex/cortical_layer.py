"""
皮层 LSHN 核心网络层
白皮书 §4.1, §4.2

整合了:
- 多尺度液态门控元胞 (含树突非线性)
- 双势阱超图突触 (含多跳资格迹)
- 轴突延迟模块
- 隐式MoE (横向抑制)
- 三因素可塑性
- 稳态可塑性控制
- 凋亡生发机制
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from lshn.core.cells.liquid_cell import LiquidGatedCell
from lshn.core.synapses.bistable_hypergraph import BistableHypergraphSynapse
from lshn.core.synapses.axonal_delay import AxonalDelayModule
from lshn.core.plasticity.three_factor import ThreeFactorPlasticity
from lshn.core.plasticity.homeostatic import HomeostaticController
from lshn.layers.cortex.implicit_moe import ImplicitMoE
from lshn.core.evolution.prune_growth import PruneGrowthModule


class CorticalLayer(nn.Module):
    """
    皮层 LSHN 核心网络层
    
    由微观/中观/宏观三层液态门控元胞组成，通过动态超边连接，
    搭配局部抑制竞争的隐式MoE机制与结构演化，
    实现特征的分区表征与长期知识存储。
    """
    def __init__(self, in_channels: int, num_neurons: int, num_groups: int,
                 max_edges: int, enable_dendrites: bool = False,
                 inhibition_strength: float = 0.5, input_gain: float = 1.0,
                 device=None, dtype=None, cfg: Optional[dict] = None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        cfg = cfg or {}
        cell_cfg = cfg.get("cell", {})
        syn_cfg = cfg.get("synapse", {})
        three_cfg = cfg.get("three_factor", {})
        homeo_cfg = cfg.get("homeostatic", {})
        moe_cfg = cfg.get("implicit_moe", {})
        evol_cfg = cfg.get("evolution", {})
        if moe_cfg.get("inhibition_strength") is not None:
            inhibition_strength = moe_cfg["inhibition_strength"]
        cortex_cfg = cfg.get("cortex", {})
        input_gain = cortex_cfg.get("input_gain", input_gain)

        self.num_neurons = num_neurons
        self.max_edges = max_edges

        # === 核心模块 ===
        self.cell = LiquidGatedCell(
            num_neurons=num_neurons,
            tau_v=cell_cfg.get("tau_fast", 10.0),
            tau_g_slow=cell_cfg.get("tau_slow", 200.0),
            theta_0=cell_cfg.get("threshold", 1.0),
            enable_dendrites=enable_dendrites,
            num_branches=cell_cfg.get("dendrite_branches", 4),
            dendrite_threshold=cell_cfg.get("dendrite_threshold", 0.3),
            noise_std=cell_cfg.get("noise_std", 0.01),
            input_gain=input_gain,
            device=device, dtype=dtype
        )

        self.synapse = BistableHypergraphSynapse(
            in_channels=in_channels,
            out_channels=max_edges,
            tau_pre=syn_cfg.get("tau_pre", 20.0),
            tau_post=syn_cfg.get("tau_post", 20.0),
            device=device, dtype=dtype
        )

        # 轴突延迟模块 (延迟范围/学习率从配置读取)
        axd_cfg = cfg.get("axonal_delay", {})
        self.axonal_delay = AxonalDelayModule(
            max_edges=max_edges,
            max_delay=axd_cfg.get("max_delay", 20),
            delay_lr=axd_cfg.get("delay_lr", 0.001),
            min_delay=axd_cfg.get("min_delay", 1),
            device=device, dtype=dtype
        )

        # 超边特征 → 神经元输入电流的映射
        self.edge_to_neuron = nn.Linear(max_edges, num_neurons, bias=False, **factory_kwargs)

        # === 可塑性模块 ===
        self.plasticity = ThreeFactorPlasticity(
            learning_rate=three_cfg.get("lr", 0.01),
            importance_lambda=three_cfg.get("importance_lambda", 0.0),
        )
        # 注: 泊松误差编码器由模型层持有 (model.error_encoder), 皮层层不重复创建
        self.homeostatic = HomeostaticController(
            num_neurons=num_neurons,
            target_rate=homeo_cfg.get("target_rate", 0.05),
            scaling_strength=homeo_cfg.get("scaling_strength", 0.1),
            ie_lr=homeo_cfg.get("ie_lr", 0.001),
            tau_rate=homeo_cfg.get("tau_rate", 100.0),
            device=device, dtype=dtype
        )

        # === 结构模块 ===
        self.implicit_moe = ImplicitMoE(num_neurons, num_groups, inhibition_strength,
                                        **factory_kwargs)
        self.prune_growth = PruneGrowthModule(
            max_neurons=num_neurons, max_edges=max_edges,
            prune_threshold=evol_cfg.get("prune_threshold", 0.0),
            min_alive_ratio=evol_cfg.get("min_alive_ratio", 0.3),
            growth_probability=evol_cfg.get("growth_probability", 0.05),
            device=device, dtype=dtype
        )

        # 多跳资格迹: 超边按功能组划分的块对角邻接 (组内超边互相传播,
        # 白皮书 §3.5.3)。组内归一化保证多跳项有界。
        self._build_edge_group_adjacency()

        # 上一步的脉冲状态 — 逐样本持有 (batch, num_neurons):
        # 侧向抑制 I_inh = implicit_moe(prev_spk) 变为逐样本 (修复批内
        # 竞争缺失: 原实现存 batch 均值, 平衡批次下类别差异被均值抹平);
        # STDP 迹聚合 (aggregate_spikes_to_edges) 内部对 batch 取均值,
        # 与旧实现数值一致, 迹保持均值场语义。
        # 初始为空 (0, num_neurons), 首次 forward 按 batch 重建
        # (参考 LiquidGatedCell.v 的均值场重建模式)。
        # persistent=False: 瞬态状态, 不进 state_dict。
        self.register_buffer(
            "prev_spk", torch.zeros(0, num_neurons, **factory_kwargs),
            persistent=False,
        )

        # 脉冲计数累加器 (GPU 张量, 慢时钟边界才 .item() 同步一次;
        # 记录"每样本每快步"平均脉冲数: 累计 per-sample 均值 ÷ 累计步数,
        # 与预算控制器目标 (target_spikes_per_step) 同单位)。
        # _window_steps 为 Python 计数: 窗口内累计的快步步数
        # (仅 training 累加, 与 _spike_acc 同步递增)。
        self.register_buffer(
            "_spike_acc", torch.zeros((), **factory_kwargs)
        )
        self._window_steps = 0

    def _build_edge_group_adjacency(self):
        """构造超边局部组邻接矩阵 (块对角, 组内互相连接)"""
        adj = torch.zeros(
            self.max_edges, self.max_edges,
            device=self.prune_growth.edge_mask.device,
            dtype=torch.float32,
        )
        num_groups = self.implicit_moe.num_groups
        group_size = (self.max_edges + num_groups - 1) // num_groups
        for g in range(num_groups):
            lo = g * group_size
            hi = min((g + 1) * group_size, self.max_edges)
            if hi > lo:
                adj[lo:hi, lo:hi] = 1.0 / max(1.0, float(hi - lo))
        self.synapse.set_local_group_adjacency(adj)

    def step_fast(self, x_in: torch.Tensor, hyperedge_index: torch.Tensor,
                  theta_ie: Optional[torch.Tensor] = None,
                  inh_scale: float = 1.0
                  ) -> torch.Tensor:
        """
        1ms 前向步

        Args:
            x_in: (batch, in_channels) 前一层脉冲
            hyperedge_index: (2, N_edges) 超图拓扑
            theta_ie: (num_neurons,) 稳态可塑性的阈值调整量
            inh_scale: 侧向抑制强度缩放 (预算控制器调节)
        """
        # 0. prev_spk 逐样本持有: 批量变化时重建 (空则清零, 否则保留
        # 旧状态的批量均值首行 — 与 LiquidGatedCell.v 的均值场重建模式一致)
        batch_size = x_in.shape[0] if x_in.dim() > 1 else 1
        if self.prev_spk.shape[0] != batch_size:
            new_prev = self.prev_spk.new_empty(batch_size, self.num_neurons)
            if self.prev_spk.shape[0] > 0:
                new_prev.copy_(self.prev_spk.mean(dim=0, keepdim=True))
            else:
                new_prev.zero_()
            self.prev_spk = new_prev

        # 1. 突触聚合输入电流 (传入 post_spk 和 g_slow 用于完整STDP和多跳迹)
        g_slow = self.cell.get_plasticity_modulation()

        # 超边级前/后突触脉冲 (按超图拓扑聚合)
        # 输出通路: 逐样本求和聚合 (默认拓扑每条边固定 K 个成员,
        # sum = mean×K, 驱动强度 8 倍, 缓解稀疏脉冲通路的多重小尺度衰减;
        # 必须保留 batch 维, 否则平衡批次下类别信息被批量均值抹平);
        # STDP 迹/共发放保持 (max_edges,) 批量均值语义。
        pre_edge = self.synapse.aggregate_spikes_to_edges_batch(
            x_in, hyperedge_index, reduction="sum"
        )  # (batch, max_edges)
        # post_edge: prev_spk 为逐样本 (batch, num_neurons), 聚合函数内部
        # 对 batch 取均值 — 与旧实现 (prev_spk 存 batch 均值) 数值一致,
        # STDP 迹保持均值场语义。
        post_edge = self.synapse.aggregate_spikes_to_edges(self.prev_spk, hyperedge_index)

        # 轴突延迟: 延迟后的前突触活动参与输出通路 (白皮书 §1.3.4/§3.5.4)
        delayed_pre, _stdp_delta = self.axonal_delay.step_fast(pre_edge, post_edge)
        # STE 直通: 延迟读值来自无梯度的缓冲状态, 以当前步聚合值直通梯度,
        # 保持 loss → x_in (编码器/海马体) 的可微路径
        # (时域延迟的梯度近似, SNN 延迟缓冲的标准做法)
        delayed_pre = delayed_pre + (pre_edge - pre_edge.detach())

        syn_out = self.synapse.step_fast(
            x_in, hyperedge_index,
            post_spk=post_edge,
            g_slow=g_slow,
            delayed_pre=delayed_pre,
        )

        # 2. 从超边特征映射到每个神经元的输入电流
        # syn_out shape: (batch, max_edges)
        I_syn = self.edge_to_neuron(syn_out)

        # 应用掩码 (凋亡神经元电流强制为0)
        I_syn = I_syn * self.prune_growth.neuron_mask.float()

        # 3. 隐式 MoE 侧向抑制电流 — 逐样本: prev_spk 保留 batch 维,
        # 组内抑制竞争按样本独立计算 (修复批内竞争缺失)
        I_inh = self.implicit_moe(self.prev_spk) * inh_scale

        # 4. 元胞动力学更新 (含树突非线性和STE)
        spk_out, mem_out = self.cell.step_fast(I_syn, I_inh=I_inh, theta_ie=theta_ie)

        # 5. 死神经元完全静默: 输出处掩码 (防止死神经元经膜电位残留/噪声
        # 偶发放电泄漏进 decoder、STDP 迹与稳态可塑性)
        neuron_mask = self.prune_growth.neuron_mask.float()
        spk_out = spk_out * neuron_mask

        # 6. 更新稳态可塑性 (仅训练: 评估阶段不污染发放率 EMA 统计)
        # 取 batch 平均记录
        if self.training:
            spk_for_homeo = spk_out.detach()
            if spk_for_homeo.dim() > 1:
                spk_for_homeo = spk_for_homeo.mean(dim=0)
            self.homeostatic.step_fast(spk_for_homeo)

            # 7. 更新脉冲计数 (GPU 张量累加, 慢时钟边界才 .item() 同步,
            #    消除热路径设备同步; 窗口步数同步递增 — 仅 training,
            #    评估/回放外步不计入预算统计)
            if spk_out.dim() > 1:
                self._spike_acc.data.add_(spk_out.detach().mean(dim=0).sum())
            else:
                self._spike_acc.data.add_(spk_out.detach().sum())
            self._window_steps += 1

        # 更新状态 (用于下一步 STDP / 侧向抑制) — 无条件更新:
        # 评估阶段同样需要逐样本 prev_spk 驱动抑制与 STDP 输入。
        # 逐样本持有 (batch, num_neurons), 不再取 batch 均值。
        prev_val = spk_out.detach()
        if prev_val.dim() > 1:
            self.prev_spk.data.copy_(prev_val)
        else:
            self.prev_spk.data.copy_(prev_val.unsqueeze(0))

        return spk_out
    
    def apply_plasticity(self, error_spk: torch.Tensor,
                         error_neuron: Optional[torch.Tensor] = None,
                         neuromodulator: Optional[torch.Tensor] = None,
                         plasticity_gate: float = 1.0,
                         importance: Optional[torch.Tensor] = None):
        """
        应用三因素可塑性规则更新快权重

        Args:
            error_spk: (max_edges,) 超边级误差信号 (按超图拓扑聚合)
            error_neuron: (num_neurons,) 神经元级误差信号 (用于 delta_window,
                保持与 neuron 维度的对齐; 缺省时退化为对 error_spk 的截断/填充)
            neuromodulator: DA 第三因子
            plasticity_gate: 星形胶质全局可塑性门控 [0, 1]
            importance: (max_edges,) 归一化边重要性 (EWC-lite, 可选)
        """
        self.plasticity(
            self.synapse.w_hat,
            self.synapse.e_trace,
            error_spk,
            neuromodulator=neuromodulator,
            plasticity_gate=plasticity_gate,
            importance=importance,
        )
        # 记录神经元级误差到细胞窗口 (驱动 g_slow 慢门控)
        if error_neuron is None:
            if error_spk.shape[0] > self.num_neurons:
                error_neuron = error_spk[:self.num_neurons]
            else:
                error_neuron = torch.nn.functional.pad(
                    error_spk, (0, self.num_neurons - error_spk.shape[0])
                )
        self.cell.update_delta_window(error_neuron.detach())
        
    def step_slow(self, global_e: torch.Tensor, M_global: float, R_replay: float, T_temp: float):
        """
        100ms 更新慢变量与结构双势阱
        """
        # 细胞慢门控更新
        self.cell.step_slow(global_e)
        
        # 结构双势阱更新
        self.synapse.step_slow_structure(M_global, R_replay, T_temp)
        
        # 稳态可塑性慢更新
        homeo_result = self.homeostatic.step_slow()

        # 突触缩放应用到快权重。
        # 超边→神经元映射: 每条超边归入其输出权重最大的神经元,
        # 使 per-neuron 缩放真正按神经元生效 (而非全局标量)。
        # alive_neuron_mask: 死神经元关联边的缩放因子强制 1.0,
        # 防止死边被 scale=2.0 持续放大到 ±1 饱和、复活时爆发
        # (稳态约束与凋亡生发机制协同, 白皮书 §4.1)。
        neuron_to_edge_map = self.edge_to_neuron.weight.detach().abs().argmax(dim=0)
        self.synapse.w_hat.data.copy_(
            self.homeostatic.apply_to_weights(
                self.synapse.w_hat, neuron_to_edge_map,
                alive_neuron_mask=self.prune_growth.neuron_mask,
            )
        )

        # 轴突延迟学习: 时序误差使用延迟模块自身的迹
        # (pre_trace_delayed 已含延迟后的前活动, post_trace 为后活动),
        # 与延迟学习规则内部的迹一致 — 而非突触模块的 STDP 迹
        timing_error = self.axonal_delay.pre_trace_delayed - self.axonal_delay.post_trace
        self.axonal_delay.update_delays(self.synapse.e_trace, timing_error)

        return homeo_result
    
    def step_ultra_slow(self, VFE_full: float, VFE_masked_dict: dict,
                        task_importance_mask: Optional[torch.Tensor] = None,
                        hyperedge_index: Optional[torch.Tensor] = None):
        """
        1000ms 超慢时钟: 凋亡生发
        """
        self.prune_growth.step_ultra_slow_evolution(
            VFE_full, VFE_masked_dict, task_importance_mask,
            hyperedge_index=hyperedge_index,
        )
    
    def get_spike_count_and_reset(self) -> float:
        """获取并重置脉冲计数 (用于能量预算)

        返回值为"每样本每快步平均脉冲数": 慢时钟窗口内累计的
        per-sample 均值除以窗口内累计的快步步数, 与预算控制器目标
        (target_spikes_per_step, per-sample 每步语义) 同单位,
        无需再按 batch/步数归一化。
        """
        count = float(self._spike_acc.item()) / max(self._window_steps, 1)
        self._spike_acc.zero_()
        self._window_steps = 0
        return count
