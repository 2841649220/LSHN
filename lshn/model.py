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
"""
import warnings

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
                 hidden_dim: int = 256,
                 num_neurons: int = 1000,
                 num_groups: int = 10,
                 max_edges: int = 500,
                 initial_classes: int = 2,
                 enable_dendrites: bool = False,
                 enable_active_inference: bool = False,
                 target_spikes_per_step: int = 50,
                 device=None, dtype=None, cfg: Optional[dict] = None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        cfg = cfg or {}
        enc_cfg = cfg.get("encoder", {})
        fe_cfg = cfg.get("free_energy", {})
        budget_cfg = cfg.get("budget", {})
        nm_cfg = cfg.get("neuromodulator", {})
        astro_cfg = cfg.get("astrocyte", {})

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neurons = num_neurons

        # 主动推理标志 (预留接口: ActiveInferenceEngine 尚未接入主流程,
        # 该标志当前无效果, 仅保留配置接线)
        self.enable_active_inference = enable_active_inference
        if enable_active_inference:
            warnings.warn(
                "enable_active_inference=True 但 ActiveInferenceEngine 尚未接入"
                "主流程, 该标志当前无效果(预留接口)",
                UserWarning,
            )

        # ========== 四层架构 ==========

        # 1. 输入编码层
        self.encoder = MODWTEncoder(input_dim, hidden_dim,
                                    num_scales=enc_cfg.get("num_scales", 3),
                                    **factory_kwargs)

        # 2. 海马体快速学习层
        hippo_gain = cfg.get("hippocampus", {}).get("input_gain", 2.0)
        self.hippocampus = SpikingAutoEncoder(
            input_dim=hidden_dim, hidden_dim=hidden_dim,
            input_gain=hippo_gain, **factory_kwargs
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
            cfg=cfg,
            **factory_kwargs
        )

        # 4. 输出解码层
        self.decoder = DynamicExpansionHead(in_features=num_neurons, initial_classes=initial_classes,
                                            **factory_kwargs)

        # 泊松误差编码器 (误差符号: error = target - pred, 负反馈环;
        # 最大发放率从配置 three_factor.f_max 读取)
        self.error_encoder = PoissonErrorEncoder(
            f_max=cfg.get("three_factor", {}).get("f_max", 1.0)
        )

        # ========== 引擎 ==========

        # 多时间尺度时钟 (周期从配置 clocks.fast_ms/slow_ms/ultra_slow_ms 读取)
        clock_cfg = cfg.get("clocks", {})
        self.clock = ClockSyncEngine(
            fast_ms=clock_cfg.get("fast_ms", 1),
            slow_ms=clock_cfg.get("slow_ms", 100),
            ultra_slow_ms=clock_cfg.get("ultra_slow_ms", 1000),
        )

        # 变分自由能引擎
        self.vfe_engine = FreeEnergyEngine(
            kl_weight=fe_cfg.get("kl_weight", 0.01),
            energy_lambda=fe_cfg.get("energy_lambda", 0.001),
            energy_lambda_lr=fe_cfg.get("energy_lambda_lr", 0.0001),
        )

        # 脉冲预算控制器
        self.budget_ctrl = SpikeBudgetController(
            target_spikes_per_step=target_spikes_per_step,
            kp=budget_cfg.get("kp", 0.01),
            ki=budget_cfg.get("ki", 0.001),
            max_integral=budget_cfg.get("integral_max", 100.0),
            theta_adj_scale=budget_cfg.get("threshold_adj_scale", 0.1),
            inh_adj_scale=budget_cfg.get("inhibition_adj_scale", 0.05),
        )

        # 全局神经调节器
        self.neuromodulator = GlobalNeuromodulator(
            num_neurons=num_neurons,
            tau_ach=nm_cfg.get("tau_ach", 200.0),
            tau_ne=nm_cfg.get("tau_ne", 100.0),
            tau_da=nm_cfg.get("tau_da", 150.0),
            tau_ca=astro_cfg.get("tau_ca", 500.0),
            **factory_kwargs
        )

        # ========== 状态 ==========

        # 超图拓扑持久化: 注册为持久 buffer (进 state_dict → 检查点携带
        # 拓扑, 训练/评估连接一致)。首次调用时以固定种子 2026 生成,
        # 保证跨运行可复现; 按 edge_mask 过滤的结果缓存于
        # _hyperedge_cache (普通属性, 不进 state_dict; 结构演化
        # 即超慢时钟后由 _on_ultra_slow_clock 置空失效)。
        self.register_buffer(
            "_hyperedge_index", torch.empty(2, 0, dtype=torch.long, device=device),
            persistent=True,
        )
        self._hyperedge_cache = None

        # 边重要性 (EWC-lite, 白皮书 §3.4 结构版 EWC): 每个慢时钟按
        # 共发放活跃度 × 有效权重累积, 高重要性边 (旧任务已固化结构)
        # 的三因素可塑性被衰减, 保护旧任务不被新任务覆盖。
        # persistent=True: 随检查点持久化, 跨任务累积 (不重置)。
        self.register_buffer(
            "_edge_importance", torch.zeros(max_edges, **factory_kwargs),
            persistent=True,
        )

        # 最近海马体隐层脉冲 (在线回放模式吸引项源, 白皮书 §3.6.2;
        # persistent=False: 瞬态, 不进 state_dict)
        self.register_buffer(
            "_last_spk_hippo", torch.zeros(hidden_dim, **factory_kwargs),
            persistent=False,
        )

        # 最近一次的调制信号缓存
        self._last_modulation = None
        self._last_vfe = None
        self._last_budget = None

        # 预算控制器最近一次输出的调整量 (慢时钟更新, 快时钟应用)
        self._budget_theta_adj = 0.0
        self._budget_inh_adj = 0.0

        # ACh 精度 / 星形胶质门控缓存 (慢时钟更新为 Python float,
        # 快时钟直接使用, 消除每步 .item() 设备同步)
        self._ach_precision = 1.0
        self._plasticity_gate = 1.0

        # 累计脉冲数 (已弃用: 由 cortex._spike_acc 承担; 保留属性以兼容测试)
        self._step_spike_count = 0
        
    def _get_hyperedge_index(self, device) -> torch.Tensor:
        """
        获取当前超图拓扑 (首次调用时生成默认拓扑, 随检查点持久化)

        默认拓扑: 每条超边连接 K=8 个随机源神经元 (可重叠),
        构成真正的"超边" (多成员高阶关联)。
        (原实现每条边只连 1 个源节点, 超图退化为随机置换。)
        返回前按存活超边掩码过滤连接。

        持久化/确定性/缓存:
        - 拓扑注册为持久 buffer (2, N_conn) int64, 随 state_dict
          保存/加载 — 检查点携带拓扑, 训练/评估连接一致;
        - 首次调用以固定种子 2026 生成, 保证跨运行可复现
          (torch.manual_seed 同时重置 CPU 与 CUDA 默认生成器);
        - 按 edge_mask 过滤的结果缓存于 _hyperedge_cache, 仅在
          结构演化 (超慢时钟) 后失效重建, 快时钟热路径零重复开销。
        """
        if self._hyperedge_index.numel() == 0:
            # 首次调用: 固定种子生成默认拓扑并写入持久 buffer。
            # 生成逻辑保持: num_edges = min(max_edges, num_neurons),
            # k=8, src ∈ [0, num_neurons)。直接创建在目标 device 上,
            # buffer 机制负责随 .to(device) 迁移。
            torch.manual_seed(2026)
            num_edges = min(self.cortex.max_edges, self.num_neurons)
            k = 8
            src = torch.randint(0, self.num_neurons, (k, num_edges), device=device)
            edge_ids = (
                torch.arange(num_edges, device=device)
                .unsqueeze(0).expand(k, -1).reshape(-1)
            )
            self._hyperedge_index = torch.stack([src.reshape(-1), edge_ids], dim=0)

        if self._hyperedge_cache is not None:
            return self._hyperedge_cache

        # 按结构演化的超边存活掩码过滤连接
        idx = self._hyperedge_index.to(device)
        edge_mask = self.cortex.prune_growth.edge_mask
        if edge_mask is not None and not edge_mask.all():
            alive = edge_mask[idx[1].clamp(0, self.cortex.max_edges - 1)]
            filtered = idx[:, alive]
            # 全灭保护: 若过滤后无连接, 回退到未过滤拓扑
            if filtered.shape[1] > 0:
                idx = filtered
        self._hyperedge_cache = idx
        return idx

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """拓扑 buffer 惰性生成与检查点形状兼容。

        新建模型的 _hyperedge_index 为空 buffer (2, 0) (惰性生成),
        检查点携带已生成的拓扑 (2, N_conn) — 两者形状不同, strict 加载
        默认报 size mismatch。此处直接接管检查点张量 (重新注册 buffer),
        保持惰性生成语义, 使严格加载在 训练/评估/恢复 全链路可用;
        接管后缓存置空 (以检查点拓扑重建过滤缓存)。
        """
        key = prefix + "_hyperedge_index"
        if key in state_dict:
            ckpt_tensor = state_dict[key]
            if self._hyperedge_index.shape != ckpt_tensor.shape:
                # 空 buffer (2, 0) 与检查点拓扑 (2, N_conn) 形状不同:
                # 直接替换 buffer 值 (保持注册与持久化标志), 使常规
                # 加载路径形状匹配, 不产生 size mismatch / missing /
                # unexpected 键错误。注意: 不能 pop 后重赋 (module
                # __setattr__ 只注册已在 _buffers 中的名字)。
                self._buffers["_hyperedge_index"] = ckpt_tensor.detach().clone().to(
                    self._hyperedge_index.device
                )
                self._hyperedge_cache = None
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        
    def forward_step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None
                     ) -> Dict[str, torch.Tensor]:
        """
        单步前向传播 (1ms 快时钟)
        
        完整数据流:
        x → Encoder → Hippocampus → Cortex → Decoder → output
        
        Args:
            x: (batch, input_dim) 原始输入
            target: (batch, num_classes) 目标 (用于误差驱动学习)
            
        Returns:
            dict with 'output', 'spk_cortex', 'spk_hippo', etc.
        """
        device = x.device
        
        # 1. 输入编码: 连续信号 → 脉冲
        spk_encoded = self.encoder(x)  # (batch, hidden_dim)
        
        # 2. 海马体快速编码
        spk_hippo = self.hippocampus.step_fast(spk_encoded)  # (batch, hidden_dim)

        # 记录最近海马体隐层脉冲 (在线回放模式吸引项源, 白皮书 §3.6.2)
        if spk_hippo.dim() > 1:
            self._last_spk_hippo.data.copy_(spk_hippo.detach().mean(dim=0))
        else:
            self._last_spk_hippo.data.copy_(spk_hippo.detach())

        # 3. 海马体 → 皮层投射
        I_hippo = self.hippo_to_cortex(spk_hippo)
        # I_hippo: (batch, num_neurons)
        
        # 扩展为供皮层核心使用的输入。
        # CorticalLayer.step_fast 现在可以直接处理 batch 维度的输入
        x_cortex = I_hippo
        
        # 4. 皮层前向
        hyperedge_index = self._get_hyperedge_index(device)

        # 获取稳态可塑性的阈值调整, 叠加预算控制器的阈值调整量
        theta_ie = None
        if hasattr(self.cortex, 'homeostatic'):
            theta_ie = self.cortex.homeostatic.ie_plasticity.theta_ie
        if theta_ie is not None and self._budget_theta_adj != 0.0:
            theta_ie = theta_ie + self._budget_theta_adj

        spk_cortex = self.cortex.step_fast(
            x_cortex, hyperedge_index, theta_ie=theta_ie,
            inh_scale=1.0 + self._budget_inh_adj,
        )
        
        # 5. 输出解码
        output = self.decoder(spk_cortex)

        # 6. 如果有目标，生成误差脉冲并应用可塑性
        if target is not None:
            # 泊松编码的误差脉冲 (精度由 ACh 调制, 慢时钟缓存为 Python float)
            error_spk = self.error_encoder(output, target, precision=self._ach_precision)

            # 三因素可塑性 (DA 作为第三因子, 星形胶质门控缩放学习率)
            da_signal = None
            if self._last_modulation is not None:
                da_signal = self._last_modulation.get("DA", None)

            # 误差按超图拓扑回传 (拓扑感知路由):
            # 类别级误差 → 经解码器权重投影回神经元空间 → 按超边聚合。
            # (原实现按索引 pad/截断, 类别误差只落到前 2-4 条超边,
            #  496/500 条超边无误差信号; delta_window 也随之为 edge 级错位。)
            error_mean = error_spk.detach().mean(dim=0)  # (num_classes,)
            error_neuron = self.decoder.weight.detach().t() @ error_mean  # (num_neurons,)
            error_edges = self.cortex.synapse.aggregate_spikes_to_edges(
                error_neuron.unsqueeze(0), hyperedge_index
            ).squeeze(0)  # (max_edges,)

            # EWC-lite: 归一化边重要性 (EWC-lite, 白皮书 §3.4 结构版 EWC)。
            # 高重要性边 (旧任务固化结构) 的三因素更新被衰减, 新任务学习
            # 主要发生在低重要性边上; 重要性跨任务累积 (不重置)。
            imp_norm = None
            if self.cortex.plasticity.importance_lambda > 0.0:
                imp_max = self._edge_importance.max().clamp(min=1e-6)
                imp_norm = (self._edge_importance / imp_max).detach()

            self.cortex.apply_plasticity(
                error_edges,
                error_neuron=error_neuron,
                neuromodulator=da_signal,
                plasticity_gate=self._plasticity_gate,
                importance=imp_norm,
            )

            # 海马体 delta_window 修复: 皮层神经元级误差沿投射权重回投到
            # 海马体隐层 (原实现从未喂给海马体, delta_window 恒零 →
            # 海马体慢门控 g_slow 只被脉冲驱动, 误差驱动的适应缺失)
            error_hippo = self.hippo_to_cortex.weight.detach().t() @ error_neuron
            self.hippocampus.cell.update_delta_window(error_hippo)

        # 8. 时钟推进 + 慢时钟/超慢时钟事件
        trigger_slow, trigger_ultra = self.clock.tick()
        
        if trigger_slow:
            self._on_slow_clock(target, output)
            
        if trigger_ultra:
            self._on_ultra_slow_clock()
        
        result = {
            "output": output,
            "spk_cortex": spk_cortex.detach(),
            "spk_hippo": spk_hippo.detach() if isinstance(spk_hippo, torch.Tensor) else spk_hippo,
            "spk_encoded": spk_encoded.detach(),
        }
        if self.training and target is not None:
            # 海马体重构损失 (可微, 供 train.py 并入总损失; 仅训练且有
            # target 时出现该键, 调用方需用 .get 容错)
            result["recon_loss"] = self.hippocampus.reconstruction_loss(
                spk_hippo, spk_encoded
            )
        return result
    
    def _on_slow_clock(self, target: Optional[torch.Tensor] = None,
                       output: Optional[torch.Tensor] = None):
        """
        慢时钟事件 (每100快步, 即100ms)
        
        1. 计算VFE + 能量预算
        2. 更新神经调节器 (ACh/NE/DA)
        3. 更新皮层慢变量 (g_slow, s_e 双势阱, 稳态可塑性, 轴突延迟)
        4. 在线回放 (白皮书 §3.6.2: 每100快步回放一次)
        5. 预算PI控制
        """
        # eval 阶段不推进引擎状态: 推理时无真实误差信号 (prediction_error=0),
        # 若照常运行预算 PI 控制/调制器/结构演化, 会以零误差污染阈值、
        # 权重与 EMA, 扭曲评估指标并跨任务残留。
        if not self.training:
            # 防御 eval 期间无 reset 的用法: 消费清零脉冲累加器
            # (eval 快步不再累加, 此处仅清残留计数)
            self.cortex.get_spike_count_and_reset()
            return

        device = next(self.parameters()).device

        # 1. VFE 计算 — 使用真实预测误差 (target 与 output 恒同传)
        if target is not None and output is not None:
            prediction_error = (output.detach() - target.detach()).reshape(-1)
        else:
            # 无目标步 (如纯推理预热): 零误差占位
            prediction_error = torch.zeros(10, device=device)

        # 每样本每快步平均脉冲数 (窗口内已按累计步数归一化, 与预算目标同单位)
        spike_count = self.cortex.get_spike_count_and_reset()

        vfe_dict = self.vfe_engine.compute_vfe(
            prediction_error=prediction_error,
            s_e_tensor=self.cortex.synapse.s_e.detach(),
            # 复杂度项 = E[S] 平均发放率语义 (与白皮书一致)
            activity=float(self.cortex.cell.get_firing_rate().mean()),
            synaptic_events=spike_count,
            precision=self._ach_precision
        )
        self._last_vfe = vfe_dict

        # 2. 神经调节器更新
        # DA 闭环激活: 奖赏 = 1 − 平均绝对误差 (由真实预测误差构造的
        # 带符号奖赏信号: 误差小 → 奖赏高 → DA 上调三因素可塑性)
        reward_signal = float(
            torch.clamp(1.0 - prediction_error.detach().mean().abs(),
                        min=-1.0, max=1.0)
        )
        modulation = self.neuromodulator.step_slow(
            prediction_error=vfe_dict["accuracy_loss"],
            firing_rate=float(self.cortex.cell.get_firing_rate().mean()),
            reward_signal=reward_signal,
        )
        self._last_modulation = modulation
        # 缓存快时钟使用的标量 (慢时钟同步一次即可)
        self._ach_precision = float(modulation["ACh"].item())
        self._plasticity_gate = float(modulation["plasticity_gate"].item())

        # 3. 皮层慢变量更新
        # global_e = DA 作为全局探索/利用信号
        # (unsqueeze 后再 expand, 兼容 0-dim 标量的旧 torch 版本)
        global_e = modulation["DA"].unsqueeze(0).expand(self.num_neurons)
        M_global = vfe_dict["accuracy_loss"]  # 预测误差作为全局调制
        T_temp = modulation["NE"].item()  # NE 作为温度

        # 在线回放: 使用回放生成器生成伪样本的共发放强度 (模式吸引)
        R_replay = self._run_online_replay(T_temp)

        homeo_result = self.cortex.step_slow(global_e, M_global, R_replay, T_temp)

        # EWC-lite 重要性累积 (仅保护机制启用时): 共发放活跃度 × 有效权重
        # 绝对值作为"边对当前任务的因果贡献"代理 (与 _on_ultra_slow_clock
        # 的贡献度代理同构), 跨任务累积 —— 旧任务固化的边重要性高,
        # 后续任务训练时其可塑性被 1/(1+λ·imp) 衰减 (白皮书 §3.4)。
        if self.cortex.plasticity.importance_lambda > 0.0:
            with torch.no_grad():
                coact_now = self.cortex.synapse.coact_window.mean(dim=0)
                eff_w_now = self.cortex.synapse.get_effective_weights()
                self._edge_importance.data.add_(
                    (coact_now * eff_w_now.abs()).clamp(min=0.0)
                )

        # 4. 海马体慢时钟更新
        if self.num_neurons >= self.hidden_dim:
            hippo_input = global_e[:self.hidden_dim]
        else:
            hippo_input = torch.nn.functional.pad(global_e, (0, self.hidden_dim - self.num_neurons))
        self.hippocampus.cell.step_slow(hippo_input)

        # 5. 预算PI控制
        # spike_count 已是"每样本每步平均脉冲数", 直接与
        # target_spikes_per_step (per-sample 语义) 比较
        budget_result = self.budget_ctrl.step_control(spike_count)
        self._budget_theta_adj = budget_result["theta_adj"]
        self._budget_inh_adj = budget_result["inh_adj"]
        self._last_budget = budget_result

        # 自适应调整 λ_E (与预算控制器同单位: 每样本每步脉冲数)
        self.vfe_engine.compute_energy_regularization_gradient(
            spike_count, self.budget_ctrl.target_budget
        )
    
    def _run_online_replay(self, T_temp: float) -> float:
        """
        在线回放 (白皮书 §3.6.2)
        每100个快时间步执行一次，回放最近的输入模式。

        模式吸引 (修复回放退化): 回放状态被牵引到最近编码模式附近 —
        attractor = W_dec^T · S_hippo (最近海马体隐层脉冲经解码权重回投),
        以 inject_rate=0.3 混合进泄漏-动量二阶动力学, 使回放与近期记忆
        相关而非纯噪声漂移。

        Args:
            T_temp: 回放温度 (NE 探索信号, 传入 generate_step)

        Returns:
            R_replay: 回放信号强度 (用于双势阱的检索项)
        """
        device = next(self.parameters()).device

        # 初始化回放状态 (dtype 与模型一致, 防止 float64/混合精度下不匹配)
        self.replay_generator.init_state(
            batch_size=1, device=device, dtype=next(self.parameters()).dtype
        )

        # 吸引项: pattern = W_dec^T · S_hippo, 形状 (hidden_dim,)
        # (decoder_linear: Linear(hidden_dim, hidden_dim), W^T ∈ (hidden_dim, hidden_dim))
        pattern = self.hippocampus.decoder_linear.weight.detach().t() @ self._last_spk_hippo
        self.replay_generator.inject_pattern(pattern.unsqueeze(0), inject_rate=0.3)

        # 生成一步回放 (温度 = NE 探索信号)
        pseudo_spk = self.replay_generator.generate_step(
            self.hippocampus.decoder_linear, temperature=T_temp
        )

        # 回放信号强度 = 伪样本的平均活动度
        R_replay = float(pseudo_spk.mean().item())

        return R_replay
    
    def _on_ultra_slow_clock(self):
        """
        超慢时钟事件 (每1000快步, 即1s)

        1. 离线回放 + 皮层巩固
        2. 凋亡/生发 (结构演化)
        3. 全局精度/温度更新
        """
        # eval 阶段不推进结构演化 (避免评估污染结构状态)
        if not self.training:
            return

        # 结构演化 (使用真实逐边贡献度代理):
        # Contribution_e = VFE(移除边 e) - VFE(full) ≈ 共发放活跃度 × |有效权重|
        # —— 对"边的重要性"的一阶代理: 活跃且权重显著的边贡献大,
        # 长期不共发放且权重趋零的边视为可剪枝。
        if self._last_vfe is not None:
            VFE_full = self._last_vfe["vfe_total"]
            coact = self.cortex.synapse.coact_window.mean(dim=0).detach()
            eff_w = self.cortex.synapse.get_effective_weights().detach()
            importance = (coact * eff_w.abs()).tolist()
            # 使 prune_growth 内的 contribution_e = vfe_val - VFE_full = importance
            VFE_masked_dict = {
                e: VFE_full + imp for e, imp in enumerate(importance)
            }
            hyperedge_index = self._get_hyperedge_index(next(self.parameters()).device)
            self.cortex.step_ultra_slow(VFE_full, VFE_masked_dict,
                                        hyperedge_index=hyperedge_index)

        # 结构演化可能改变 edge_mask → 拓扑过滤缓存失效 (下次调用重建)
        self._hyperedge_cache = None

        # 超慢时钟离线回放: K=5 步伪脉冲经皮层传播巩固
        # (白皮书 §3.6.2 离线回放: 回放共发放进入 coact_window, 驱动
        #  双势阱检索项, 巩固旧知识对应超边结构)
        with torch.no_grad():
            device = next(self.parameters()).device
            if (self.replay_generator._state is None
                    or self.replay_generator._state.numel() == 0):
                # 防御冷启动: 从未经过在线回放 (慢时钟) 时的状态初始化
                self.replay_generator.init_state(
                    batch_size=1, device=device,
                    dtype=next(self.parameters()).dtype,
                )
            idx = self._get_hyperedge_index(device)
            for _ in range(5):
                pseudo = self.replay_generator.generate_step(
                    self.hippocampus.decoder_linear, temperature=0.1
                )
                I_replay = self.hippo_to_cortex(pseudo)
                self.cortex.step_fast(I_replay, idx)
        # 消费回放脉冲: 不进入预算控制器统计
        self.cortex.get_spike_count_and_reset()

    def expand_classes(self, num_new_classes: int):
        """动态扩容输出类别"""
        self.decoder.expand(num_new_classes)
    
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
        
        # 预算状态 (step_control 返回的 "budget_error" 键已带 budget_ 前缀,
        # 避免双重前缀 "budget_budget_error")
        if self._last_budget:
            report.update({
                k if k.startswith("budget_") else f"budget_{k}": v
                for k, v in self._last_budget.items()
            })
        
        # 结构统计
        report["alive_edges_ratio"] = float(self.cortex.synapse.get_alive_mask().float().mean())
        # s_e 语义 (双势阱) 的 alive_edges_mask_ratio 区分于结构演化 edge_mask
        report["alive_edges_mask_ratio"] = float(self.cortex.prune_growth.edge_mask.float().mean())
        report["alive_neurons_ratio"] = float(self.cortex.prune_growth.neuron_mask.float().mean())
        report["mean_firing_rate"] = float(self.cortex.cell.get_firing_rate().mean())
        report["e_trace_abs_max"] = float(self.cortex.synapse.e_trace.detach().abs().max())

        # 时钟步数
        report["clock_fast_steps"] = float(self.clock.fast_steps)
        report["clock_slow_steps"] = float(self.clock.slow_steps)

        # 轴突延迟统计
        report.update(self.cortex.axonal_delay.get_delay_stats())

        return report
    
    def reset(self):
        """重置所有状态 (保留学习到的权重/结构知识)"""
        self.clock.reset()
        self.cortex.cell.reset_hidden()
        self.hippocampus.cell.reset_hidden()
        self.cortex.axonal_delay.reset()
        self.cortex.synapse.reset()
        self.cortex.homeostatic.reset()
        self.cortex.prev_spk.zero_()
        self.cortex._spike_acc.zero_()
        self.cortex._window_steps = 0
        self.vfe_engine.reset()
        self.neuromodulator.reset()
        self.budget_ctrl.reset()
        self.replay_generator._state = None
        self.replay_generator._velocity = None
        self._budget_theta_adj = 0.0
        self._budget_inh_adj = 0.0
        self._last_modulation = None
        self._last_vfe = None
        self._last_budget = None
        self._ach_precision = 1.0
        self._plasticity_gate = 1.0
        self._step_spike_count = 0
        # 拓扑未变, 缓存置空无害 (下次调用按当前 edge_mask 重建)
        self._hyperedge_cache = None
        self._last_spk_hippo.zero_()

    def reset_sample_state(self):
        """清理样本级瞬态状态 (膜电位/延迟缓冲/prev_spk), 保留权重、迹与引擎状态。
        训练脚本在每个 batch 起点调用, 保证"每样本 20 快步"语义独立。"""
        self.cortex.cell.v.zero_()
        if self.cortex.cell.enable_dendrites:
            self.cortex.cell.dendrite.branch_potential.zero_()
        self.cortex.axonal_delay.reset()  # 清延迟环形缓冲 (迹一并清, 延迟学习从零开始累积;
                                          # delay_continuous 学习值保留, reset() 只清缓冲与迹)
        self.cortex.prev_spk.zero_()
        self._last_spk_hippo.zero_()
