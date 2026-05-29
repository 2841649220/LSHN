"""
双势阱脉冲超图突触 (Bistable Hypergraph Synapse)
白皮书 §3.3, §3.5.3, §3.5.4

修复:
- 完整的 STDP 迹更新 (pre * post 共发放)
- 多跳资格迹 (高阶传播，g_slow 门控)
- record_coact 在 step_fast 中自动调用
- 轴突延迟集成接口
- 使用手动 scatter 消息传递替代 HypergraphConv，修复维度兼容性
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class BistableHypergraphSynapse(nn.Module):
    """
    双势阱脉冲超图突触
    
    分离快权重(w_hat)与慢结构(s_e):
    - w_hat: 快变权重 ∈ [-1,1], 对应突触传递效率, 由三因素规则快速更新
    - s_e: 结构变量 ∈ [0,1], 对应超边存在的后验概率, 双势阱慢速演化
    - e_trace: 资格迹, 记录超边的活动历史
    
    有效权重: w_e = w_max * s_e * w_hat
    
    多跳资格迹 (白皮书 §3.5.3):
        e_e(t+1) = λ_e * e_e(t) + y_pre(t) * y_post(t) 
                   + σ(g_post^slow) * Σ_{e' ∈ local_group} w_{e'} * e_{e'}(t)
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 w_max: float = 1.0, alpha: float = 0.1, beta: float = 0.05,
                 trace_decay: float = 0.9,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_max = w_max
        self.alpha = alpha
        self.beta = beta
        self.trace_decay = trace_decay
        
        self.max_edges = out_channels
        
        # 快权重 w_hat (三因素规则更新)
        self.w_hat = nn.Parameter(torch.randn(out_channels, **factory_kwargs) * 0.1)
        # 结构变量 s_e (双势阱慢演化, 不参与梯度)
        self.s_e = nn.Parameter(torch.ones(out_channels, **factory_kwargs) * 0.5, requires_grad=False)
        
        # === 资格迹 (完整实现) ===
        self.register_buffer("e_trace", torch.zeros(out_channels, **factory_kwargs))
        
        # STDP 迹 (前/后突触)
        self.register_buffer("pre_trace", torch.zeros(out_channels, **factory_kwargs))
        self.register_buffer("post_trace", torch.zeros(out_channels, **factory_kwargs))
        
        # 共发放率历史 (10个慢时间步)
        self.register_buffer("coact_window", torch.zeros(10, out_channels, **factory_kwargs))
        self.register_buffer("window_idx", torch.tensor(0, dtype=torch.long, device=device))
        
        # 多跳传播的局部组连接矩阵
        self.local_group_adj = None
        
        # STDP 时间常数
        self.tau_pre = 20.0
        self.tau_post = 20.0

    def set_local_group_adjacency(self, adj: torch.Tensor):
        """
        设置局部超边组的邻接矩阵 (用于多跳资格迹传播)
        adj: (max_edges, max_edges) 稀疏或密集矩阵
        """
        self.local_group_adj = adj

    def _hypergraph_message_passing(self, x_in: torch.Tensor, 
                                     hyperedge_index: torch.Tensor,
                                     effective_w: torch.Tensor) -> torch.Tensor:
        """
        手动超图消息传递，替代 HypergraphConv。
        
        对每条超边，聚合其连接的源节点特征，乘以有效权重。
        
        Args:
            x_in: (batch, num_nodes) 节点特征
            hyperedge_index: (2, N_connections) 超图拓扑 [src_nodes, edge_ids]
            effective_w: (max_edges,) 每条超边的有效权重
            
        Returns:
            edge_out: (batch, max_edges) 超边级聚合特征
        """
        src, edge_ids = hyperedge_index
        batch_size, num_nodes = x_in.shape
        
        edge_out = torch.zeros(batch_size, self.max_edges, 
                               device=x_in.device, dtype=x_in.dtype)
        
        if src.numel() == 0:
            return edge_out
        
        valid_mask = edge_ids < self.max_edges
        src_valid = src[valid_mask]
        edge_ids_valid = edge_ids[valid_mask]
        
        if src_valid.numel() == 0:
            return edge_out
        
        src_clamped = src_valid.clamp(0, num_nodes - 1)
        src_features = x_in[:, src_clamped]
        
        edge_ids_expanded = edge_ids_valid.unsqueeze(0).expand(batch_size, -1)
        edge_out.scatter_add_(1, edge_ids_expanded, src_features)
        
        edge_out = edge_out * effective_w.unsqueeze(0)
        
        return edge_out

    def step_fast(self, x_in: torch.Tensor, hyperedge_index: torch.Tensor, 
                  post_spk: Optional[torch.Tensor] = None,
                  g_slow: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
        """
        快时钟 (1ms) 前向更新
        
        Args:
            x_in: (batch, in_channels) 或 (in_channels,) 前突触脉冲
            hyperedge_index: (2, N_edges)
            post_spk: (max_edges,) 或 (batch, max_edges) 后突触脉冲
            g_slow: (num_neurons,) 后突触神经元的慢门控
        """
        is_batched = x_in.dim() > 1
        if not is_batched:
            x_in = x_in.unsqueeze(0)
            
        # 1. 有效权重
        effective_w = self.w_max * self.s_e * self.w_hat
        
        # 2. === 完整的 STDP 迹更新 ===
        pre_decay = torch.exp(torch.tensor(-1.0 / self.tau_pre, device=x_in.device))
        self.pre_trace.data.mul_(pre_decay)
        
        pre_spk_per_edge = x_in.mean(dim=0)
        
        if pre_spk_per_edge.shape[0] > self.max_edges:
            pre_spk_per_edge = pre_spk_per_edge[:self.max_edges]
        elif pre_spk_per_edge.shape[0] < self.max_edges:
            pre_spk_per_edge = torch.nn.functional.pad(
                pre_spk_per_edge, (0, self.max_edges - pre_spk_per_edge.shape[0])
            )
        
        self.pre_trace.data.add_(pre_spk_per_edge)
        
        # 后突触迹
        post_decay = torch.exp(torch.tensor(-1.0 / self.tau_post, device=x_in.device))
        self.post_trace.data.mul_(post_decay)
        
        if post_spk is not None:
            if post_spk.dim() > 1:
                post_spk_edge = post_spk.mean(dim=0)
            else:
                post_spk_edge = post_spk
                
            if post_spk_edge.shape[0] > self.max_edges:
                post_spk_edge = post_spk_edge[:self.max_edges]
            elif post_spk_edge.shape[0] < self.max_edges:
                post_spk_edge = torch.nn.functional.pad(
                    post_spk_edge, (0, self.max_edges - post_spk_edge.shape[0])
                )
            self.post_trace.data.add_(post_spk_edge)
        
        # 3. === 多跳资格迹更新 ===
        coact = pre_spk_per_edge * (self.post_trace + 0.1)
        
        multihop_term = torch.zeros_like(self.e_trace)
        if self.local_group_adj is not None and g_slow is not None:
            g_slow_gate = torch.sigmoid(g_slow.mean())
            weighted_traces = effective_w * self.e_trace
            multihop_term = g_slow_gate * (self.local_group_adj @ weighted_traces)
        
        self.e_trace.data.copy_(self.trace_decay * self.e_trace + coact + multihop_term)
        
        # 4. 记录共发放
        self._auto_record_coact(coact)
        
        # 5. 超图消息传递
        out = self._hypergraph_message_passing(x_in, hyperedge_index, effective_w)
        
        if not is_batched:
            out = out.squeeze(0)
        
        return out
    
    def _auto_record_coact(self, coact_val: torch.Tensor):
        """自动记录共发放到慢时间窗口"""
        idx = self.window_idx % 10
        self.coact_window[idx] = coact_val.detach()
        self.window_idx += 1

    def step_slow_structure(self, M_global: float, R_replay: float, T_temp: float, dt_slow: float = 0.1):
        """
        慢时钟 (100ms) 结构双势阱更新
        
        势能: U(s_e) = 0.25α * s_e^4 - 0.5α * s_e^2
        dU/ds = α * s_e^3 - α * s_e = α * s_e * (s_e^2 - 1)
        
        更新规则:
            s_e(t+1) = clip(s_e + dt * (-dU/ds + β*C_e + γ*M + δ*R) + noise, 0, 1)
        """
        C_e = self.coact_window.mean(dim=0)
        
        dU_ds = self.alpha * self.s_e * (self.s_e ** 2 - 1.0)
        
        noise = torch.randn_like(self.s_e) * torch.sqrt(
            torch.tensor(2 * dt_slow * max(T_temp, 1e-6), device=self.s_e.device)
        )
        
        gamma, delta = 0.02, 0.05
        ds_e = -dU_ds + self.beta * C_e + gamma * M_global + delta * R_replay
        
        self.s_e.data.add_(ds_e * dt_slow + noise)
        self.s_e.data.clamp_(0.0, 1.0)

    def record_coact(self, coact_val: torch.Tensor):
        """手动记录共发放 (兼容旧接口)"""
        self._auto_record_coact(coact_val)
    
    def get_effective_weights(self) -> torch.Tensor:
        """返回有效权重 w_e = w_max * s_e * w_hat"""
        return self.w_max * self.s_e * self.w_hat
    
    def get_alive_mask(self, threshold: float = 0.05) -> torch.Tensor:
        """返回存活超边的掩码 (s_e > threshold)"""
        return (self.s_e > threshold).detach()
