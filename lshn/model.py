"""
端到端 LSHN 模型 (LSHNModel)
白皮书 §4.1: 分层解耦的四层架构

    输入编码层 → 海马体快速学习层 → 皮层LSHN核心网络层 → 输出解码层

所有模块通过标准化接口通信，支持单独验证、替换与扩展。

集成:
- 多时间尺度时钟同步 (快1ms / 慢100ms / 超慢1000ms)
- 全局神经调节器 (ACh/NE/DA + 星形胶质门控)
- 变分自由能引擎 (VFE + 能量正则化)
- 脉冲预算控制器 (PI控制)
- 在线回放 (每100快步触发, 白皮书 §3.6.2)
- 三因素可塑性 + 泊松误差编码
- BF16 混合精度 (autocast, 仅包裹前向计算; SNN 状态变量始终 FP32)
- 冷知识 INT4 归档 (KnowledgeArchiver, 每超慢时钟触发)
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from lshn.layers.io.modwt_encoder import MODWTEncoder
from lshn.layers.hippocampus.spiking_ae import SpikingAutoEncoder
from lshn.layers.hippocampus.replay_generator import ReplayGenerator
from lshn.layers.cortex.cortical_layer import CorticalLayer
from lshn.layers.io.dynamic_expansion_head import DynamicExpansionHead
from lshn.core.plasticity.three_factor import PoissonErrorEncoder
from lshn.engine.clock_sync import ClockSyncEngine
from lshn.engine.free_energy import FreeEnergyEngine
from lshn.engine.budget_control import SpikeBudgetController
from lshn.engine.global_modulator import GlobalNeuromodulator
from lshn.engine.knowledge_archiver import KnowledgeArchiver


class LSHNModel(nn.Module):
    """
    液态脉冲超图网络 端到端模型
    
    四层管道:
    1. 输入编码层 (MODWTEncoder): 连续信号 → 多尺度脉冲序列
    2. 海马体快速学习层 (SpikingAutoEncoder + ReplayGenerator): 快速编码 + 回放
    3. 皮层核心网络层 (CorticalLayer): 分区表征 + 长期知识存储
    4. 输出解码层 (DynamicExpansionHead): 脉冲 → 任务输出
    
    引擎:
    - ClockSyncEngine: 多时间尺度时钟
    - FreeEnergyEngine: VFE + J = F + λ_E * E[events]
    - SpikeBudgetController: PI 脉冲预算控制
    - GlobalNeuromodulator: ACh/NE/DA + 星形胶质门控
    """
    
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dim: int = 1024,
                 num_neurons: int = 500000,
                 num_groups: int = 100,
                 max_edges: int = 50000,
                 initial_classes: int = 2,
                 enable_dendrites: bool = False,
                 enable_active_inference: bool = False,
                 target_spikes_per_step: int = 5000,
                 # ── 混合精度 ──
                 mixed_precision: bool = True,
                 autocast_dtype: torch.dtype = torch.bfloat16,
                 # ── 冷归档 ──
                 archive_dir: str = "./cold_archive",
                 archive_group_size: int = 64,
                 cold_threshold: float = 0.05,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neurons = num_neurons

        # ── 混合精度设置 ──
        self._mixed_precision = mixed_precision
        self._autocast_dtype = autocast_dtype
        # cold_threshold: s_e < 此值且已被剪枝的超边判定为冷边
        self._cold_threshold = cold_threshold
        
        # ========== 四层架构 ==========
        
        # 1. 输入编码层
        self.encoder = MODWTEncoder(input_dim, hidden_dim, num_scales=3)
        
        # 2. 海马体快速学习层
        self.hippocampus = SpikingAutoEncoder(
            input_dim=hidden_dim, hidden_dim=hidden_dim, **factory_kwargs
        )
        self.replay_generator = ReplayGenerator(hidden_dim=hidden_dim)
        
        # 海马体 → 皮层的投射
        self.hippo_to_cortex = nn.Linear(hidden_dim, num_neurons, bias=False, **factory_kwargs)
        
        # 3. 皮层核心网络层
        self.cortex = CorticalLayer(
            in_channels=num_neurons,
            num_neurons=num_neurons,
            num_groups=num_groups,
            max_edges=max_edges,
            enable_dendrites=enable_dendrites,
            **factory_kwargs
        )
        
        # 4. 输出解码层
        self.decoder = DynamicExpansionHead(in_features=num_neurons, initial_classes=initial_classes)
        
        # 泊松误差编码器
        self.error_encoder = PoissonErrorEncoder(f_max=1.0)
        
        # ========== 引擎 ==========
        
        # 多时间尺度时钟
        self.clock = ClockSyncEngine()
        
        # 变分自由能引擎
        self.vfe_engine = FreeEnergyEngine(kl_weight=0.01, energy_lambda=0.001)
        
        # 脉冲预算控制器
        self.budget_ctrl = SpikeBudgetController(target_spikes_per_step=target_spikes_per_step)
        
        # 全局神经调节器
        self.neuromodulator = GlobalNeuromodulator(num_neurons=num_neurons, **factory_kwargs)
        
        # ========== 状态 ==========
        
        # 默认超图拓扑 (在初始化时创建简单拓扑，后续由结构演化更新)
        self._default_hyperedge_index = None
        
        # 最近一次的调制信号缓存
        self._last_modulation = None
        self._last_vfe = None
        self._last_budget = None
        self._last_output = None  # 缓存最近一步的输出，用于 VFE 预测误差计算
        
        # 累计脉冲数 (用于能量预算)
        self._step_spike_count = 0

        # ── 冷知识归档器 ──
        self.archiver = KnowledgeArchiver(
            archive_dir=archive_dir,
            group_size=archive_group_size,
        )
        
    def _get_hyperedge_index(self, device) -> torch.Tensor:
        """
        获取当前超图拓扑 (如果没有则创建默认拓扑)
        默认: 每个组内全连接的简单超边
        """
        if self._default_hyperedge_index is not None:
            # 确保 num_neurons 匹配 (可能因为演化改变了，虽然目前是固定的)
            return self._default_hyperedge_index.to(device)
        
        # 创建简单拓扑: 随机稀疏连接
        # 重要: 这里的 src 必须被限制在 [0, self.num_neurons) 范围内
        num_edges = min(self.cortex.max_edges, self.num_neurons * 2)
        src = torch.randint(0, self.num_neurons, (num_edges,), device=device)
        # 超图中，每条超边连接 src 到 edge_id
        edge_ids = torch.arange(num_edges, device=device) % self.cortex.max_edges
        self._default_hyperedge_index = torch.stack([src, edge_ids], dim=0)
        return self._default_hyperedge_index
        
    def forward_step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None
                     ) -> Dict[str, torch.Tensor]:
        """
        单步前向传播 (1ms 快时钟)
        
        完整数据流:
        x → Encoder → Hippocampus → Cortex → Decoder → output
        
        精度策略:
        - 整体计算包裹在 torch.autocast(BF16) 中，加速 nn.Linear / HypergraphConv
        - SNN 状态变量 (v, g_fast, g_slow, a, theta, traces) 在各自模块内
          通过 .float() 强制回 FP32，不受 autocast 影响
        
        Args:
            x: (batch, input_dim) 原始输入
            target: (batch, num_classes) 目标 (用于误差驱动学习)
            
        Returns:
            dict with 'output', 'spk_cortex', 'spk_hippo', etc.
        """
        device = x.device
        device_type = "cuda" if device.type == "cuda" else "cpu"

        # ── BF16 autocast 包裹前向计算 ──────────────────────────────
        # SNN 内部的 FP32 精度围栏 (.float() cast) 保护所有状态变量,
        # 不需要在此处手动排除任何子模块.
        with torch.autocast(device_type=device_type,
                            dtype=self._autocast_dtype,
                            enabled=self._mixed_precision):

            # 1. 输入编码: 连续信号 → 脉冲
            spk_encoded = self.encoder(x)  # (batch, hidden_dim)

            # 2. 海马体快速编码
            spk_hippo = self.hippocampus.step_fast(spk_encoded)  # (hidden_dim,)

            # 3. 海马体 → 皮层投射
            I_hippo = self.hippo_to_cortex(spk_hippo)
            # I_hippo: (batch, num_neurons)

            # 扩展为供皮层核心使用的输入。
            # CorticalLayer.step_fast 现在可以直接处理 batch 维度的输入
            x_cortex = I_hippo

            # 4. 皮层前向
            hyperedge_index = self._get_hyperedge_index(device)

            # 获取稳态可塑性的阈值调整
            theta_ie = None
            if hasattr(self.cortex, 'homeostatic'):
                theta_ie = self.cortex.homeostatic.ie_plasticity.theta_ie

            spk_cortex = self.cortex.step_fast(x_cortex, hyperedge_index, theta_ie=theta_ie)

            # 5. 输出解码
            output = self.decoder(spk_cortex)

        # ── autocast 范围结束: 以下在 FP32 下运行 ──────────────────

        # 6. 累计脉冲计数
        self._step_spike_count += int(spk_cortex.detach().sum().item())
        
        # 缓存最近一步输出 (用于慢时钟 VFE 计算)
        self._last_output = output.detach()
        
        # 7. 如果有目标，生成误差脉冲并应用可塑性
        if target is not None:
            # 泊松编码的误差脉冲
            precision = 1.0
            if self._last_modulation is not None:
                precision = self._last_modulation.get("ACh", torch.tensor(1.0)).item()
            
            error_spk = self.error_encoder(output, target, precision=precision)
            
            # 三因素可塑性 (DA 作为第三因子)
            da_signal = None
            if self._last_modulation is not None:
                da_signal = self._last_modulation.get("DA", None)
            
            # 缩放误差脉冲到 max_edges 维度
            if error_spk.dim() > 1:
                error_spk_flat = error_spk.mean(dim=0)
            else:
                error_spk_flat = error_spk
                
            # 适配维度
            max_edges = self.cortex.max_edges
            if error_spk_flat.shape[0] < max_edges:
                error_spk_flat = torch.nn.functional.pad(
                    error_spk_flat, (0, max_edges - error_spk_flat.shape[0])
                )
            elif error_spk_flat.shape[0] > max_edges:
                error_spk_flat = error_spk_flat[:max_edges]
            
            self.cortex.apply_plasticity(error_spk_flat, neuromodulator=da_signal)
        
        # 8. 时钟推进 + 慢时钟/超慢时钟事件
        trigger_slow, trigger_ultra = self.clock.tick()
        
        if trigger_slow:
            self._on_slow_clock(target)
            
        if trigger_ultra:
            self._on_ultra_slow_clock()
        
        return {
            "output": output,
            "spk_cortex": spk_cortex.detach(),
            "spk_hippo": spk_hippo.detach() if isinstance(spk_hippo, torch.Tensor) else spk_hippo,
            "spk_encoded": spk_encoded.detach(),
        }
    
    def _on_slow_clock(self, target: Optional[torch.Tensor] = None):
        """
        慢时钟事件 (每100快步, 即100ms)
        
        1. 计算VFE + 能量预算
        2. 更新神经调节器 (ACh/NE/DA)
        3. 更新皮层慢变量 (g_slow, s_e 双势阱, 稳态可塑性, 轴突延迟)
        4. 在线回放 (白皮书 §3.6.2: 每100快步回放一次)
        5. 预算PI控制
        """
        device = next(self.parameters()).device
        
        # 1. VFE 计算 — 使用真正的预测误差
        if target is not None and self._last_output is not None:
            # 真正的预测误差: output - target (逐元素差)
            prediction_error = (self._last_output.float() - target.float()).detach()
            # 如果是 batch，取 batch 平均得到 (num_classes,) 向量
            if prediction_error.dim() > 1:
                prediction_error = prediction_error.mean(dim=0)
        else:
            # 无 target 时使用零向量 (不注入随机噪声)
            num_out = self._last_output.shape[-1] if self._last_output is not None else 10
            prediction_error = torch.zeros(num_out, device=device)
        
        spike_count = self.cortex.get_spike_count_and_reset()
        
        vfe_dict = self.vfe_engine.compute_vfe(
            prediction_error=prediction_error,
            s_e_tensor=self.cortex.synapse.s_e.detach(),
            active_neurons_ratio=float(self.cortex.prune_growth.neuron_mask.float().mean()),
            synaptic_events=spike_count,
            precision=self._last_modulation["ACh"].item() if self._last_modulation else 1.0
        )
        self._last_vfe = vfe_dict
        
        # 2. 神经调节器更新
        modulation = self.neuromodulator.step_slow(
            prediction_error=vfe_dict["accuracy_loss"],
            firing_rate=float(self.cortex.cell.get_firing_rate().mean()),
        )
        self._last_modulation = modulation
        
        # 3. 皮层慢变量更新
        # global_e = DA 作为全局探索/利用信号
        global_e = modulation["DA"].expand(self.num_neurons)
        M_global = vfe_dict["accuracy_loss"]  # 预测误差作为全局调制
        T_temp = modulation["NE"].item()  # NE 作为温度
        
        # 在线回放: 使用回放生成器生成伪样本的共发放强度
        R_replay = self._run_online_replay()
        
        homeo_result = self.cortex.step_slow(global_e, M_global, R_replay, T_temp)
        
        # 4. 海马体慢时钟更新
        self.hippocampus.cell.step_slow(global_e[:self.hidden_dim] if self.num_neurons >= self.hidden_dim 
                                         else global_e)
        
        # 5. 预算PI控制
        budget_result = self.budget_ctrl.step_control(spike_count)
        self._last_budget = budget_result
        
        # 自适应调整 λ_E
        self.vfe_engine.compute_energy_regularization_gradient(
            spike_count, self.budget_ctrl.target_budget
        )
    
    def _run_online_replay(self) -> float:
        """
        在线回放 (白皮书 §3.6.2)
        每100个快时间步执行一次，回放最近的输入模式。
        
        Returns:
            R_replay: 回放信号强度 (用于双势阱的检索项)
        """
        device = next(self.parameters()).device
        
        # 初始化回放状态
        self.replay_generator.init_state(batch_size=1, device=device)
        
        # 生成一步回放
        pseudo_spk = self.replay_generator.generate_step(self.hippocampus.decoder_linear)
        
        # 回放信号强度 = 伪样本的平均活动度
        R_replay = pseudo_spk.mean().item()
        
        return R_replay
    
    def _on_ultra_slow_clock(self):
        """
        超慢时钟事件 (每1000快步, 即1s)

        1. 凋亡/生发 (结构演化)
        2. 冷超边检测: ~edge_mask OR s_e < cold_threshold
        3. 将冷超边 INT4 压缩归档到 NVMe
        4. 重置冷槽: w_hat=0, s_e=0.5, edge_mask=True (可重新生长)
        """
        # 1. 结构演化 (简化: 使用 VFE 历史)
        if self._last_vfe is not None:
            VFE_full = self._last_vfe["vfe_total"]
            VFE_masked_dict = {}
            self.cortex.step_ultra_slow(VFE_full, VFE_masked_dict)

        # 2. 冷超边检测
        # edge_mask: (max_edges,) bool — True=活跃槽, False=已剪枝
        # 冷边 = 已被剪枝 OR s_e 极低 (结构性休眠)
        try:
            edge_mask = self.cortex.prune_growth.edge_mask  # (max_edges,) bool
            s_e = self.cortex.synapse.s_e.detach().float()  # (max_edges,) FP32

            # 冷边掩码: 剪枝掉的边 且/或 s_e 极低
            pruned_mask = ~edge_mask                          # 已剪枝
            dormant_mask = s_e < self._cold_threshold         # s_e 极低
            cold_mask = pruned_mask | dormant_mask            # 联合冷边

            cold_indices = cold_mask.nonzero(as_tuple=False).squeeze(1)  # (N_cold,)

            if cold_indices.numel() > 0:
                # 3. 提取冷边数据
                w_hat_cold = self.cortex.synapse.w_hat.detach().float()[cold_indices]
                s_e_cold = s_e[cold_indices]

                # 构建冷边的 COO 拓扑: 从全局 hyperedge_index 筛选
                device = cold_indices.device
                heidx = self._get_hyperedge_index(device)  # (2, E_total)

                # edge_ids 列对应 [0, max_edges) 中的 slot index
                # 在 heidx 中找 edge_id ∈ cold_indices 的条目
                edge_ids_in_heidx = heidx[1]  # (E_total,)
                # 建立 cold_indices 集合查找 (bool mask)
                cold_set = torch.zeros(
                    self.cortex.max_edges, dtype=torch.bool, device=device
                )
                cold_set[cold_indices] = True
                edge_in_cold = cold_set[edge_ids_in_heidx]  # (E_total,) bool

                if edge_in_cold.any():
                    heidx_cold = heidx[:, edge_in_cold]  # (2, N_cold_edges)
                else:
                    # 没有匹配的拓扑条目: 构造自环占位
                    heidx_cold = torch.stack([
                        cold_indices,
                        cold_indices % self.cortex.max_edges,
                    ], dim=0)

                archive_id = self.archiver.archive_cold_edges(
                    cold_indices=cold_indices,
                    w_hat_cold=w_hat_cold,
                    s_e_cold=s_e_cold,
                    hyperedge_index_cold=heidx_cold,
                    num_nodes=self.num_neurons,
                )

                # 4. 重置冷槽 (释放回活跃池供生长使用)
                with torch.no_grad():
                    self.cortex.synapse.w_hat.data[cold_indices] = 0.0
                    self.cortex.synapse.s_e.data[cold_indices] = 0.5
                    # 重新开放 edge_mask (允许 PruneGrowthModule 重新生长)
                    self.cortex.prune_growth.edge_mask[cold_indices] = True

        except AttributeError:
            # 若模块结构不完整 (如单元测试环境), 跳过归档
            pass
    
    def expand_classes(self, num_new_classes: int):
        """
        动态扩容输出类别
        
        Returns:
            new_params: 新增的参数列表，调用方可选择:
                1. optimizer.add_param_group({'params': new_params})  (增量更新，更高效)
                2. 重建整个 optimizer (简单但会丢失动量状态)
        """
        return self.decoder.expand(num_new_classes)
    
    def get_monitoring_report(self) -> Dict[str, float]:
        """
        返回可解释监控报告 (白皮书 §3.1.2 硬性交付)
        """
        report = {}
        
        # VFE 分解
        report.update(self.vfe_engine.get_decomposition_report())
        
        # 调质状态
        if self._last_modulation:
            for k, v in self._last_modulation.items():
                if isinstance(v, torch.Tensor):
                    report[f"modulator_{k}"] = v.item()
        
        # 预算状态
        if self._last_budget:
            report.update({f"budget_{k}": v for k, v in self._last_budget.items()})
        
        # 结构统计
        report["alive_edges_ratio"] = float(self.cortex.synapse.get_alive_mask().float().mean())
        report["alive_neurons_ratio"] = float(self.cortex.prune_growth.neuron_mask.float().mean())
        report["mean_firing_rate"] = float(self.cortex.cell.get_firing_rate().mean())
        
        # 轴突延迟统计
        report.update(self.cortex.axonal_delay.get_delay_stats())
        
        return report
    
    def reset(self):
        """重置所有状态"""
        self.clock.reset()
        self.cortex.cell.reset_hidden()
        self.hippocampus.cell.reset_hidden()
        self._last_modulation = None
        self._last_vfe = None
        self._last_budget = None
        self._step_spike_count = 0
        self.cortex.spike_count = 0
