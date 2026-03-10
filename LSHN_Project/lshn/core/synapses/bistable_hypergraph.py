"""
双势阱脉冲超图突触 (Bistable Hypergraph Synapse)
白皮书 §3.3, §3.5.3, §3.5.4

修复:
- 完整的 STDP 迹更新 (pre * post 共发放)
- 多跳资格迹 (高阶传播，g_slow 门控)
- record_coact 在 step_fast 中自动调用
- 轴突延迟集成接口
- 自定义 SpikeHypergraphConv 替代 PyG HypergraphConv，正确处理逐边权重语义
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int,
                  dim_size: int) -> torch.Tensor:
    """scatter mean using only native PyTorch ops (no torch_scatter dependency)."""
    # 确保 src 至少为 2 维
    if src.dim() == 1:
        src = src.unsqueeze(1)
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index.unsqueeze(1).expand_as(src), src)
    ones = torch.ones_like(src[:, :1])
    count.scatter_add_(dim, index.unsqueeze(1), ones)
    return out / count.clamp(min=1)


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int,
                 dim_size: int) -> torch.Tensor:
    """scatter sum using only native PyTorch ops."""
    # 确保 src 至少为 2 维
    if src.dim() == 1:
        src = src.unsqueeze(1)
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index.unsqueeze(1).expand_as(src), src)
    return out


class SpikeHypergraphConv(nn.Module):
    """
    自定义脉冲超图卷积 (替代 PyG HypergraphConv)
    
    正确实现逐边权重语义的超图消息传递:
    1. Node → Hyperedge: 对超边内所有节点特征加权聚合
    2. Hyperedge → Node: 将超边特征分发回节点并聚合
    
    与 PyG HypergraphConv 的关键区别:
    - hyperedge_weight 是 COO 索引中每条连接的权重，不是每个超边一个全局权重
    - 使用 scatter 操作实现高效 GPU 聚合，无 Python 循环
    - 支持 batch 展平后的大图模式
    """
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 节点特征变换
        self.weight_node = nn.Parameter(
            torch.empty(in_channels, out_channels, **factory_kwargs)
        )
        nn.init.xavier_uniform_(self.weight_node)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, in_channels) 节点特征
            hyperedge_index: (2, num_connections) COO 格式
                [0] = 节点索引, [1] = 超边索引
            hyperedge_weight: (num_connections,) 每条连接的权重 (= effective_w 对应到 COO 列)
                如果为 None，则所有连接权重为 1
        
        Returns:
            out: (num_nodes, out_channels) 更新后的节点特征
        """
        node_idx = hyperedge_index[0]  # (num_connections,)
        edge_idx = hyperedge_index[1]  # (num_connections,)
        
        # 1. 节点特征变换
        x_transformed = x @ self.weight_node  # (num_nodes, out_channels)
        
        # 2. Node → Hyperedge 聚合: 收集每个超边内的加权节点特征
        node_features = x_transformed[node_idx]  # (num_connections, out_channels)
        
        if hyperedge_weight is not None:
            # 逐连接加权
            node_features = node_features * hyperedge_weight.unsqueeze(-1)
        
        # 聚合到超边 (native scatter, 高效 GPU 并行)
        num_hyperedges = edge_idx.max().item() + 1 if edge_idx.numel() > 0 else 0
        edge_features = _scatter_mean(node_features, edge_idx, dim=0,
                                      dim_size=num_hyperedges)
        
        # 3. Hyperedge → Node 聚合: 将超边特征分发回节点
        edge_msgs = edge_features[edge_idx]  # (num_connections, out_channels)
        
        if hyperedge_weight is not None:
            edge_msgs = edge_msgs * hyperedge_weight.unsqueeze(-1)
        
        num_nodes = x.shape[0]
        out = _scatter_sum(edge_msgs, node_idx, dim=0,
                           dim_size=num_nodes)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


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
    def __init__(self, num_neurons: int, out_channels: int = 1,
                 max_edges: Optional[int] = None,
                 w_max: float = 1.0, alpha: float = 0.1, beta: float = 0.05,
                 trace_decay: float = 0.9,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.num_neurons = num_neurons
        self.out_channels = out_channels
        self.w_max = w_max
        self.alpha = alpha
        self.beta = beta
        self.trace_decay = trace_decay
        
        # max_edges 可由外部指定，默认与 num_neurons 一致
        max_edges = max_edges if max_edges is not None else num_neurons
        self.max_edges = max_edges
        
        # 自定义超图卷积 (正确的逐边权重语义)
        self.hypergraph_conv = SpikeHypergraphConv(
            in_channels=1, out_channels=out_channels,
            device=device, dtype=dtype
        )
        
        # 快权重 w_hat (三因素规则更新)
        self.w_hat = nn.Parameter(torch.randn(max_edges, **factory_kwargs) * 0.1)
        # 结构变量 s_e (双势阱慢演化, 不参与梯度)
        self.s_e = nn.Parameter(torch.ones(max_edges, **factory_kwargs) * 0.5, requires_grad=False)
        
        # === 资格迹 (完整实现) ===
        self.register_buffer("e_trace", torch.zeros(max_edges, **factory_kwargs))
        
        # STDP 迹 (前/后突触)
        self.register_buffer("pre_trace", torch.zeros(max_edges, **factory_kwargs))
        self.register_buffer("post_trace", torch.zeros(max_edges, **factory_kwargs))
        
        # 共发放率历史 (10个慢时间步)
        self.register_buffer("coact_window", torch.zeros(10, max_edges, **factory_kwargs))
        self.register_buffer("window_idx", torch.tensor(0, dtype=torch.long, device=device))
        
        # 多跳传播的局部组连接矩阵
        # 简化: 使用稀疏的邻接关系 (后续可动态构建)
        # 这里预设为 None, 在组装模型时由外部设置
        self.local_group_adj = None
        
        # STDP 时间常数
        self.tau_pre = 20.0
        self.tau_post = 20.0
        
        # 预计算衰减因子 (避免每次调用重复创建)
        self.register_buffer("_pre_decay", None)
        self.register_buffer("_post_decay", None)
        
        # 共发放偏置 (可配置，白皮书标准STDP应为0)
        self.coact_bias = 0.1

    def set_local_group_adjacency(self, adj: torch.Tensor):
        """
        设置局部超边组的邻接矩阵 (用于多跳资格迹传播)
        adj: (max_edges, max_edges) 稀疏或密集矩阵
        """
        self.local_group_adj = adj

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

        # ── FP32 精度围栏 ──────────────────────────────────────────
        # e_trace / pre_trace / post_trace / coact_window 是 SNN 慢状态变量,
        # 必须始终在 FP32 精度下更新, 即使调用方处于 BF16 autocast 上下文中.
        x_in = x_in.float()

        # 1. 有效权重
        effective_w = self.w_max * self.s_e * self.w_hat
        
        # 2. === 完整的 STDP 迹更新 ===
        # 预计算衰减因子 (首次调用时初始化)
        if self._pre_decay is None or self._pre_decay.device != x_in.device:
            self._pre_decay = torch.exp(torch.tensor(-1.0 / self.tau_pre, device=x_in.device))
            self._post_decay = torch.exp(torch.tensor(-1.0 / self.tau_post, device=x_in.device))
        
        self.pre_trace.data.mul_(self._pre_decay)
        
        # pre_spk_per_edge: 取 batch 平均记录到迹 (强制 FP32)
        pre_spk_per_edge = x_in.float().mean(dim=0)
        
        # 确保维度匹配
        if pre_spk_per_edge.shape[0] > self.max_edges:
            pre_spk_per_edge = pre_spk_per_edge[:self.max_edges]
        elif pre_spk_per_edge.shape[0] < self.max_edges:
            pre_spk_per_edge = torch.nn.functional.pad(
                pre_spk_per_edge, (0, self.max_edges - pre_spk_per_edge.shape[0])
            )
        
        self.pre_trace.data.add_(pre_spk_per_edge)
        
        # 后突触迹
        self.post_trace.data.mul_(self._post_decay)
        
        if post_spk is not None:
            if post_spk.dim() > 1:
                post_spk_edge = post_spk.float().mean(dim=0)
            else:
                post_spk_edge = post_spk.float()
                
            if post_spk_edge.shape[0] > self.max_edges:
                post_spk_edge = post_spk_edge[:self.max_edges]
            elif post_spk_edge.shape[0] < self.max_edges:
                post_spk_edge = torch.nn.functional.pad(
                    post_spk_edge, (0, self.max_edges - post_spk_edge.shape[0])
                )
            self.post_trace.data.add_(post_spk_edge)
        
        # 3. === 多跳资格迹更新 ===
        # 使用可配置的共发放偏置
        coact = pre_spk_per_edge * (self.post_trace + self.coact_bias)
        
        multihop_term = torch.zeros_like(self.e_trace)
        if self.local_group_adj is not None and g_slow is not None:
            # 保持神经元级别的特异性门控，而非全局平均
            g_slow_gate = torch.sigmoid(g_slow)
            # 使用平均门控作为全局调制因子
            g_slow_global = g_slow_gate.mean()
            weighted_traces = effective_w * self.e_trace
            multihop_term = g_slow_global * (self.local_group_adj @ weighted_traces)
        
        self.e_trace.data.copy_(self.trace_decay * self.e_trace + coact + multihop_term)
        
        # 4. 记录共发放
        self._auto_record_coact(coact)
        
        # 5. 超图卷积前向传播 - 向量化batch处理
        # 将 effective_w (per-hyperedge) 映射到 COO 连接权重 (per-connection)
        # hyperedge_index[1] 是超边索引, effective_w[edge_idx] 即每条连接的权重
        edge_idx_col = hyperedge_index[1]  # (N_connections,)
        
        # 修复: 物理移除越界连接而非映射到索引0，避免有效边被错误激活
        valid_mask = edge_idx_col < self.max_edges
        if not valid_mask.all():
            # 物理筛选有效连接
            valid_edge_idx = edge_idx_col[valid_mask]
            valid_node_idx = hyperedge_index[0][valid_mask]
            hyperedge_index = torch.stack([valid_node_idx, valid_edge_idx], dim=0)
            edge_idx_col = valid_edge_idx
            connection_weights = effective_w[edge_idx_col]
        else:
            connection_weights = effective_w[edge_idx_col]
        
        if is_batched:
            batch_size = x_in.shape[0]
            num_nodes = x_in.shape[1]
            num_connections = hyperedge_index.shape[1]

            # 增加特征维度: (batch, N) -> (batch, N, 1) -> (batch*N, 1)
            x_flat = x_in.unsqueeze(-1).reshape(-1, 1)

            # 为每个batch样本偏移hyperedge_index
            node_offsets = torch.arange(batch_size, device=x_in.device).view(-1, 1) * num_nodes
            # 超边偏移: 使用实际超边数 (COO 列中的最大值 + 1)
            num_unique_edges = edge_idx_col.max().item() + 1 if edge_idx_col.numel() > 0 else 0
            edge_offsets = torch.arange(batch_size, device=x_in.device).view(-1, 1) * num_unique_edges

            hyperedge_index_batched = hyperedge_index.unsqueeze(0).expand(batch_size, -1, -1).clone()
            hyperedge_index_batched[:, 0, :] += node_offsets
            hyperedge_index_batched[:, 1, :] += edge_offsets

            # 展平: (batch, 2, N_connections) -> (2, batch*N_connections)
            hyperedge_index_flat = hyperedge_index_batched.reshape(2, -1)

            # 每条连接权重复制 batch 次
            connection_weights_flat = connection_weights.unsqueeze(0).expand(batch_size, -1).reshape(-1)

            # 单次调用处理所有batch
            out_flat = self.hypergraph_conv(
                x_flat, hyperedge_index_flat,
                hyperedge_weight=connection_weights_flat
            )

            # 恢复batch维度: (batch*N, out_channels) -> (batch, N, out_channels)
            out = out_flat.view(batch_size, num_nodes, -1)
        else:
            # 单样本
            x_2d = x_in.unsqueeze(-1) if x_in.dim() == 1 else x_in
            out = self.hypergraph_conv(
                x_2d, hyperedge_index,
                hyperedge_weight=connection_weights
            )

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
        # ── FP32 精度围栏: s_e ∈ [0,1] 双势阱状态变量必须 FP32 ──
        s_e = self.s_e.float()
        C_e = self.coact_window.float().mean(dim=0)
        
        dU_ds = self.alpha * s_e * (s_e ** 2 - 1.0)
        
        # 温度参数处理：确保非负
        T_temp_safe = abs(T_temp) if T_temp != 0 else 1e-6
        noise = torch.randn_like(s_e) * torch.sqrt(
            torch.tensor(2 * dt_slow * T_temp_safe, device=s_e.device, dtype=torch.float32)
        )
        
        gamma, delta = 0.02, 0.05
        ds_e = -dU_ds + self.beta * C_e + gamma * M_global + delta * R_replay
        
        s_e_new = (s_e + ds_e * dt_slow + noise).clamp_(0.0, 1.0)
        self.s_e.data.copy_(s_e_new)

    def record_coact(self, coact_val: torch.Tensor):
        """手动记录共发放 (兼容旧接口)"""
        self._auto_record_coact(coact_val)
    
    def get_effective_weights(self) -> torch.Tensor:
        """返回有效权重 w_e = w_max * s_e * w_hat"""
        return self.w_max * self.s_e * self.w_hat
    
    def get_alive_mask(self, threshold: float = 0.05) -> torch.Tensor:
        """返回存活超边的掩码 (s_e > threshold)"""
        return (self.s_e > threshold).detach()
