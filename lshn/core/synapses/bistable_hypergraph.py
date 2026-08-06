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
import math
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
                 trace_decay: float = 0.9, tau_pre: float = 20.0, tau_post: float = 20.0,
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
        # 初始化 std=0.5: 与阈值 θ=1.0、s_e≈0.5 配合,保证初始化即有足够
        # 输入电流驱动发放。原 std=0.1 使有效权重 ~±0.05,皮层输入电流量级
        # ~0.02 ≪ 阈值 1.0,网络初始化即完全静默 (STE 梯度死亡)。
        self.w_hat = nn.Parameter(torch.randn(out_channels, **factory_kwargs) * 0.5)
        # 结构变量 s_e (双势阱慢演化, 不参与梯度)
        self.s_e = nn.Parameter(torch.ones(out_channels, **factory_kwargs) * 0.5, requires_grad=False)
        
        # === 资格迹 (完整实现) ===
        self.register_buffer("e_trace", torch.zeros(out_channels, **factory_kwargs))
        
        # STDP 迹 (前/后突触)
        self.register_buffer("pre_trace", torch.zeros(out_channels, **factory_kwargs))
        self.register_buffer("post_trace", torch.zeros(out_channels, **factory_kwargs))

        # 前/后突触迹衰减常数 (预计算, 避免热路径每步建张量)
        self.register_buffer(
            "pre_decay", torch.tensor(math.exp(-1.0 / tau_pre), **factory_kwargs)
        )
        self.register_buffer(
            "post_decay", torch.tensor(math.exp(-1.0 / tau_post), **factory_kwargs)
        )
        
        # 共发放率历史 (10个慢时间步)
        self.register_buffer("coact_window", torch.zeros(10, out_channels, **factory_kwargs))
        self.register_buffer("window_idx", torch.tensor(0, dtype=torch.long, device=device))
        
        # 多跳传播的局部组连接矩阵 (空 buffer 占位, 由 set_local_group_adjacency
        # 填充; 注册为 buffer 以随 .to(device) 迁移)
        self.register_buffer("local_group_adj", torch.empty(0, 0, **factory_kwargs))
        # 每条超边所属局部组的代表边 id (块对角邻接每行 argmax 即组代表)。
        # 多跳项用 scatter 组均值替代稠密 (E,E) matmul (见 set_local_group_adjacency)。
        self.register_buffer("edge_group_ids", torch.empty(0, dtype=torch.long, device=device))
        
        # STDP 时间常数
        self.tau_pre = tau_pre
        self.tau_post = tau_post

    def set_local_group_adjacency(self, adj: torch.Tensor):
        """
        设置局部超边组的邻接矩阵 (用于多跳资格迹传播)
        adj: (max_edges, max_edges) 稀疏或密集矩阵

        注册为 buffer: 随 .to(device) 迁移, 并进入 state_dict。

        额外从块对角均匀邻接 (A[e,e'] = 1/组大小) 推导每条超边所属组
        (每行 argmax 即组代表边), 供 step_fast 用 scatter 组均值代替
        稠密 (E,E) matmul (两者数学等价: 组内均值再广播)。
        """
        adj = adj.to(device=self.e_trace.device, dtype=self.e_trace.dtype)
        self.register_buffer("local_group_adj", adj)
        self.register_buffer(
            "edge_group_ids",
            torch.argmax(adj, dim=1).to(dtype=torch.long),
        )

    def aggregate_spikes_to_edges(self, spk: torch.Tensor,
                                  hyperedge_index: torch.Tensor,
                                  reduction: str = "mean") -> torch.Tensor:
        """
        按超图拓扑把节点级脉冲聚合到超边级:
            pre_edge[e] = mean/sum_{batch, (src, e) ∈ index} spk[src]

        Args:
            spk: (batch, num_nodes) 或 (num_nodes,)
            hyperedge_index: (2, N_connections) 超图拓扑 [src_nodes, edge_ids]
            reduction: "mean" (STDP 迹/共发放, 均值语义) 或 "sum" (输出通路,
                对固定成员数 K 的默认拓扑, sum = mean×K 常数因子,
                驱动强度为 8 倍, 避免有效权重与线性层初始化三重衰减)

        Returns:
            (max_edges,) 每条超边的前突触脉冲聚合
        """
        if spk.dim() == 1:
            spk = spk.unsqueeze(0)
        batch_size, num_nodes = spk.shape

        src, edge_ids = hyperedge_index
        if src.numel() == 0:
            return torch.zeros(self.max_edges, device=spk.device, dtype=spk.dtype)

        valid = edge_ids < self.max_edges
        src_v = src[valid].clamp(0, num_nodes - 1)
        edge_v = edge_ids[valid]

        flat_edge = edge_v.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        acc = torch.zeros(self.max_edges, device=spk.device, dtype=spk.dtype)
        acc.scatter_add_(0, flat_edge, spk[:, src_v].reshape(-1))
        if reduction == "mean":
            cnt = torch.zeros(self.max_edges, device=spk.device, dtype=torch.long)
            cnt.scatter_add_(0, flat_edge, torch.ones_like(flat_edge))
            return acc / cnt.clamp(min=1).float()
        return acc

    def aggregate_spikes_to_edges_batch(self, spk: torch.Tensor,
                                        hyperedge_index: torch.Tensor,
                                        reduction: str = "sum") -> torch.Tensor:
        """
        按超图拓扑把节点级脉冲聚合到超边级, 保留 batch 维度 (输出通路用):
            pre_edge[b, e] = sum/mean_{connections of e} spk[b, src]

        Args:
            spk: (batch, num_nodes) 或 (num_nodes,)
            hyperedge_index: (2, N_connections) 超图拓扑 [src_nodes, edge_ids]
            reduction: "sum" (默认, 输出通路驱动) 或 "mean"

        Returns:
            (batch, max_edges) 逐样本超边聚合
        """
        if spk.dim() == 1:
            spk = spk.unsqueeze(0)
        batch_size, num_nodes = spk.shape

        src, edge_ids = hyperedge_index
        if src.numel() == 0:
            return torch.zeros(batch_size, self.max_edges, device=spk.device, dtype=spk.dtype)

        valid = edge_ids < self.max_edges
        src_v = src[valid].clamp(0, num_nodes - 1)
        edge_v = edge_ids[valid]

        out = torch.zeros(batch_size, self.max_edges, device=spk.device, dtype=spk.dtype)
        out.scatter_add_(1, edge_v.unsqueeze(0).expand(batch_size, -1), spk[:, src_v])
        if reduction == "mean":
            cnt = torch.zeros(self.max_edges, device=spk.device, dtype=torch.long)
            cnt.scatter_add_(0, edge_v, torch.ones_like(edge_v))
            out = out / cnt.clamp(min=1).unsqueeze(0)
        return out

    def reset(self):
        """重置所有迹与共发放窗口 (保留权重/结构知识 s_e, w_hat)"""
        self.pre_trace.zero_()
        self.post_trace.zero_()
        self.e_trace.zero_()
        self.coact_window.zero_()
        self.window_idx.zero_()

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

        # 按连接数归一化 (与 aggregate_spikes_to_edges 的均值语义保持一致:
        # 超边规模不应线性放大输出电流)
        cnt = torch.zeros(self.max_edges, device=x_in.device, dtype=torch.long)
        cnt.scatter_add_(0, edge_ids_valid, torch.ones_like(edge_ids_valid))
        edge_out = edge_out / cnt.clamp(min=1).unsqueeze(0)

        edge_out = edge_out * effective_w.unsqueeze(0)

        return edge_out

    def step_fast(self, x_in: torch.Tensor, hyperedge_index: torch.Tensor,
                  post_spk: Optional[torch.Tensor] = None,
                  g_slow: Optional[torch.Tensor] = None,
                  delayed_pre: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
        """
        快时钟 (1ms) 前向更新

        Args:
            x_in: (batch, in_channels) 或 (in_channels,) 前突触脉冲
            hyperedge_index: (2, N_edges)
            post_spk: (max_edges,) 或 (batch, max_edges) 后突触脉冲
            g_slow: (num_neurons,) 后突触神经元的慢门控
            delayed_pre: (max_edges,) 轴突延迟后的前突触活动 (可选)
                若提供,输出通路直接使用延迟活动 × 有效权重,
                使延迟学习真正参与消息传递 (白皮书 §1.3.4/§3.5.4)。
        """
        is_batched = x_in.dim() > 1
        if not is_batched:
            x_in = x_in.unsqueeze(0)
            
        # 1. 有效权重
        effective_w = self.w_max * self.s_e * self.w_hat
        
        # 2. === 完整的 STDP 迹更新 ===
        # 衰减常数在 __init__ 预计算 (热路径不重复建张量)
        self.pre_trace.data.mul_(self.pre_decay)

        # 按超图拓扑聚合前突触脉冲到超边级 (替代按通道截断/填充)
        pre_spk_per_edge = self.aggregate_spikes_to_edges(x_in, hyperedge_index)

        self.pre_trace.data.add_(pre_spk_per_edge)
        
        # 后突触迹
        self.post_trace.data.mul_(self.post_decay)
        
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
        # 纯共发放迹 (与白皮书公式一致: e = λ·e + y_pre·y_post + 多跳项)。
        # 注意: 不再使用 +0.1 基线 —— 基线会使迹退化为"前突触活跃度",
        # 前突触活跃即更新,稀释共发放信用分配。
        # 压缩因子有界: |e_{t+1}| ≤ λ·|e_t| + |coact| + σ(ḡ)·(1−λ)·mean|w⊙e_t|
        # ≤ (λ + (1−λ)·w_max·σ(ḡ))·max|e_t| + |coact| ≤ 1·max|e_t| + |coact|
        # (w_max≤1 时系数非扩张, 无指数发散; 原实现 0.9 + 0.6·1 = 1.5 > 1 可发散)。
        coact = pre_spk_per_edge * self.post_trace

        multihop_term = torch.zeros_like(self.e_trace)
        if self.edge_group_ids.numel() > 0 and g_slow is not None:
            g_slow_gate = torch.sigmoid(g_slow.mean())
            weighted_traces = effective_w * self.e_trace
            # 块对角均匀邻接 A[e,e'] = 1/组大小 ⇒ (A@w)[e] = 组内均值。
            # scatter 组均值替代稠密 (E,E) matmul: 每步 O(E) 而非 O(E²)。
            group_sum = torch.zeros(self.max_edges, device=self.e_trace.device,
                                    dtype=self.e_trace.dtype)
            group_sum.scatter_add_(0, self.edge_group_ids, weighted_traces)
            group_cnt = torch.zeros(self.max_edges, device=self.e_trace.device,
                                    dtype=self.e_trace.dtype)
            group_cnt.scatter_add_(0, self.edge_group_ids,
                                   torch.ones_like(weighted_traces))
            group_mean = (group_sum / group_cnt.clamp(min=1.0))[self.edge_group_ids]
            multihop_term = g_slow_gate * (1.0 - self.trace_decay) * group_mean

        self.e_trace.data.copy_(self.trace_decay * self.e_trace + coact + multihop_term)
        # 保险 clamp: 防止任何数值异常导致迹发散 (不影响正常收敛轨迹)
        self.e_trace.data.clamp_(-10.0, 10.0)

        # 4. 记录共发放
        self._auto_record_coact(coact)

        # 5. 输出通路: 优先使用轴突延迟后的前突触活动 (延迟参与消息传递)
        if delayed_pre is not None:
            # delayed_pre: (batch, max_edges) 或 (max_edges,) (单样本)
            if delayed_pre.dim() == 1:
                delayed_pre = delayed_pre.unsqueeze(0)
            out = delayed_pre * effective_w.unsqueeze(0)
            if not is_batched:
                out = out.squeeze(0)
            return out
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

        # 噪声按势垒深度标定: U(s) = ¼αs⁴ − ½αs² 的势阱在 s=±1, 势垒顶在
        # s=0, 势阱深度 ΔU = U(0) − U(±1) = α/4, σ = √(2·dt·T_eff·ΔU)。
        # 原实现 σ = √(2·dt·T) 在 NE∈[0.01,2]
        # 时可达 0.316,是势垒深度 (α=0.1 → ΔU=0.025) 的 ~12 倍,
        # 静默期 s_e 纯随机游走直接越过势垒, 双稳性被噪声淹没;
        # 现最大 σ ≈ √(2·0.1·0.5·0.025) = 0.05, 与 ΔU 同量级 (~2×),
        # 双稳结构得以保持, 温度仍可在 T_eff ∈ [1e-6, 0.5] 内调节探索强度。
        T_eff = max(min(T_temp, 0.5), 1e-6)
        noise = torch.randn_like(self.s_e) * torch.sqrt(
            torch.tensor(2 * dt_slow * T_eff * self.alpha / 4.0, device=self.s_e.device)
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
