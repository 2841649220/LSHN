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
from lshn.core.plasticity.three_factor import ThreeFactorPlasticity, PoissonErrorEncoder
from lshn.core.plasticity.homeostatic import HomeostaticController
from lshn.layers.cortex.implicit_moe import ImplicitMoE
from lshn.core.evolution.prune_growth import PruneGrowthModule


class CorticalLayer(nn.Module):
    """
    皮层 LSHN 核心网络层
    
    由微观/中观/宏观三层液态门控元胞组成，通过动态超边连接，
    搭配局部抑制竞争的隐式MoE机制与结构演化，
    实现特征的分区表征与长期知识存储。
    
    修复:
    - 添加维度对齐工具函数
    - MoE 侧向抑制与预算控制联动
    - 脉冲计数使用滑动窗口
    """
    def __init__(self, in_channels: int, num_neurons: int, num_groups: int, 
                 max_edges: int, enable_dendrites: bool = False,
                 inhibition_strength: float = 0.5,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_neurons = num_neurons
        self.max_edges = max_edges
        
        # === 核心模块 ===
        self.cell = LiquidGatedCell(
            num_neurons=num_neurons, 
            enable_dendrites=enable_dendrites,
            device=device, dtype=dtype
        )
        
        self.synapse = BistableHypergraphSynapse(
            num_neurons=num_neurons, 
            out_channels=1,
            max_edges=max_edges,
            device=device, dtype=dtype
        )
        
        # 轴突延迟模块
        self.axonal_delay = AxonalDelayModule(
            max_edges=max_edges, device=device, dtype=dtype
        )
        
        # 超边特征 → 神经元输入电流的映射
        # HypergraphConv 输出形状是 (num_neurons, out_channels)
        self.edge_to_neuron = nn.Linear(num_neurons, num_neurons, bias=False, **factory_kwargs)
        
        # === 可塑性模块 ===
        self.plasticity = ThreeFactorPlasticity(learning_rate=0.01)
        self.error_encoder = PoissonErrorEncoder(f_max=1.0)
        self.homeostatic = HomeostaticController(
            num_neurons=num_neurons, device=device, dtype=dtype
        )
        
        # === 结构模块 ===
        self.implicit_moe = ImplicitMoE(num_neurons, num_groups, inhibition_strength)
        self.prune_growth = PruneGrowthModule(
            max_neurons=num_neurons, max_edges=max_edges, device=device, dtype=dtype
        )
        
        # 上一步的脉冲状态
        self.register_buffer("prev_spk", torch.zeros(num_neurons, **factory_kwargs))
        
        # 修复: 脉冲计数使用滑动窗口 (100步)
        self.register_buffer("spike_count_window", torch.zeros(100, dtype=torch.long, device=device))
        self.register_buffer("spike_window_idx", torch.tensor(0, dtype=torch.long, device=device))

    def _align_tensor(self, tensor: torch.Tensor, target_size: int, 
                      mode: str = 'pad') -> torch.Tensor:
        """
        维度对齐工具函数
        
        Args:
            tensor: 输入张量
            target_size: 目标大小
            mode: 'pad' 填充 或 'truncate' 截取
        """
        current_size = tensor.shape[0]
        if current_size == target_size:
            return tensor
        if current_size < target_size:
            if mode == 'pad':
                return torch.nn.functional.pad(tensor, (0, target_size - current_size))
            return tensor
        return tensor[:target_size]

    def step_fast(self, x_in: torch.Tensor, hyperedge_index: torch.Tensor,
                  theta_ie: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
        """
        1ms 前向步
        
        Args:
            x_in: (batch, in_channels) 前一层脉冲
            hyperedge_index: (2, N_edges) 超图拓扑
            theta_ie: (num_neurons,) 稳态可塑性的阈值调整量
        """
        # 1. 轴突延迟处理: 使用维度对齐工具函数
        pre_spk_for_delay = x_in.detach()
        if pre_spk_for_delay.dim() > 1:
            pre_spk_for_delay = pre_spk_for_delay.mean(dim=0)
        pre_spk_for_delay = self._align_tensor(pre_spk_for_delay, self.max_edges)
        
        prev_spk_aligned = self._align_tensor(self.prev_spk, self.max_edges)
        
        delayed_spk, stdp_delta = self.axonal_delay.step_fast(
            pre_spk_for_delay, prev_spk_aligned
        )
        
        # 2. 突触聚合输入电流 (传入 post_spk 和 g_slow 用于完整STDP和多跳迹)
        g_slow = self.cell.get_plasticity_modulation()
        
        # 确保 x_in 在 batch 维度映射
        syn_out = self.synapse.step_fast(
            x_in, hyperedge_index, 
            post_spk=self.prev_spk,
            g_slow=g_slow
        )
        
        # 3. 从超边特征映射到每个神经元的输入电流
        # syn_out shape: (batch, num_neurons, out_channels) -> squeeze to (batch, num_neurons)
        if syn_out.dim() == 3:
            syn_out = syn_out.squeeze(-1)
        I_syn = self.edge_to_neuron(syn_out) 
        
        # 将延迟 STDP 信号注入突触电流 (作为时序调制)
        stdp_modulation = self._align_tensor(stdp_delta, self.num_neurons)
        if I_syn.dim() > 1:
            I_syn = I_syn + 0.1 * stdp_modulation.unsqueeze(0)
        else:
            I_syn = I_syn + 0.1 * stdp_modulation
        
        # 应用掩码 (凋亡神经元电流强制为0)
        I_syn = I_syn * self.prune_growth.neuron_mask.float()
        
        # 3. 隐式 MoE 侧向抑制电流 (传入 batch 会逐个计算)
        I_inh = self.implicit_moe(self.prev_spk)
        
        # 4. 元胞动力学更新 (含树突非线性和STE)
        spk_out, mem_out = self.cell.step_fast(I_syn, I_inh=I_inh, theta_ie=theta_ie)
        
        # 5. 更新稳态可塑性 (快时钟: 记录发放率)
        # 取 batch 平均记录
        spk_for_homeo = spk_out.detach()
        if spk_for_homeo.dim() > 1:
            spk_for_homeo = spk_for_homeo.mean(dim=0)
        self.homeostatic.step_fast(spk_for_homeo)
        
        # 6. 修复: 使用滑动窗口记录脉冲计数
        spk_count = int(spk_out.detach().sum().item())
        idx = self.spike_window_idx % 100
        self.spike_count_window[idx] = spk_count
        self.spike_window_idx.add_(1)
        
        # 更新状态 (用于下一步 STDP)
        # 同样取平均以简化状态
        if spk_out.dim() > 1:
            self.prev_spk = spk_out.detach().mean(dim=0)
        else:
            self.prev_spk = spk_out.detach()
        
        return spk_out
    
    def apply_plasticity(self, error_spk: torch.Tensor, neuromodulator: Optional[torch.Tensor] = None):
        """
        应用三因素可塑性规则更新快权重
        
        Args:
            error_spk: (max_edges,) 泊松编码的误差脉冲
            neuromodulator: DA 第三因子
        """
        # 确保 error_spk 维度与 e_trace 匹配
        e_trace = self.synapse.e_trace
        if error_spk.shape[0] != e_trace.shape[0]:
            if error_spk.shape[0] < e_trace.shape[0]:
                error_spk = torch.nn.functional.pad(error_spk, (0, e_trace.shape[0] - error_spk.shape[0]))
            else:
                error_spk = error_spk[:e_trace.shape[0]]
        
        self.plasticity(
            self.synapse.w_hat,
            e_trace,
            error_spk,
            neuromodulator=neuromodulator
        )
        # 更新后记录 delta 到细胞窗口
        self.cell.update_delta_window(error_spk[:self.num_neurons] if error_spk.shape[0] > self.num_neurons else 
                                       torch.nn.functional.pad(error_spk, (0, self.num_neurons - error_spk.shape[0])))
        
    def step_slow(self, global_e: torch.Tensor, M_global: float, R_replay: float, T_temp: float,
                  inh_adj: float = 0.0):
        """
        100ms 更新慢变量与结构双势阱
        
        修复: 添加 inh_adj 参数，与脉冲预算控制器联动
        """
        # 修复: MoE 侧向抑制与预算控制联动
        if inh_adj != 0.0:
            self.implicit_moe.adjust_inhibition(inh_adj)
        
        # 细胞慢门控更新
        self.cell.step_slow(global_e)
        
        # 结构双势阱更新
        self.synapse.step_slow_structure(M_global, R_replay, T_temp)
        
        # 稳态可塑性慢更新
        homeo_result = self.homeostatic.step_slow()
        
        # 突触缩放应用到快权重
        self.synapse.w_hat.data.copy_(
            self.homeostatic.apply_to_weights(self.synapse.w_hat)
        )
        
        # 轴突延迟学习 (使用资格迹和时序误差)
        # 确保维度匹配: 只使用 axonal_delay 支持的边数
        timing_error = self.synapse.pre_trace - self.synapse.post_trace  # 简化的时序误差
        max_delay_edges = self.axonal_delay.max_edges
        e_trace_subset = self.synapse.e_trace[:max_delay_edges]
        timing_error_subset = timing_error[:max_delay_edges]
        self.axonal_delay.update_delays(e_trace_subset, timing_error_subset)
        
        return homeo_result
    
    def step_ultra_slow(self, VFE_full: float, VFE_masked_dict: dict,
                        hyperedge_index: Optional[torch.Tensor] = None,
                        task_importance_mask: Optional[torch.Tensor] = None):
        """
        1000ms 超慢时钟: 凋亡生发
        """
        self.prune_growth.step_ultra_slow_evolution(
            VFE_full, VFE_masked_dict, hyperedge_index, task_importance_mask
        )
    
    def get_spike_count_and_reset(self) -> int:
        """获取滑动窗口内的脉冲计数总和 (用于能量预算)"""
        count = int(self.spike_count_window.sum().item())
        return count
    
    def reset_spike_count_window(self):
        """重置脉冲计数窗口 (在慢时钟周期结束时调用)"""
        self.spike_count_window.zero_()
        self.spike_window_idx.zero_()
    
    def reset(self):
        """重置所有状态"""
        self.cell.reset_hidden()
        self.prev_spk.zero_()
        self.spike_count_window.zero_()
        self.spike_window_idx.zero_()
        if hasattr(self.synapse, 'reset'):
            self.synapse.reset()
        if hasattr(self.axonal_delay, 'reset'):
            self.axonal_delay.reset()
        if hasattr(self.homeostatic, 'reset'):
            self.homeostatic.reset()
        if hasattr(self.prune_growth, 'reset'):
            self.prune_growth.reset()
