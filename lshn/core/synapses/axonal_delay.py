"""
轴突延迟学习模块 (Axonal Delay Learning)
白皮书 §1.3.4, §3.5.4: 可学习传导延迟 × STDP/eligibility 的时序信用分配
把可学习传导延迟纳入可塑性闭环，增强序列、语音、事件流任务的时间表征能力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AxonalDelayModule(nn.Module):
    """
    轴突传导延迟模块
    
    为每条超边维护一个可学习的传导延迟 d_e ∈ [d_min, d_max]（离散化为整数步），
    实现方式为环形缓冲区（delay buffer），延迟后的脉冲再参与 STDP 和权重更新。
    
    延迟学习规则：
        Δd_e = η_d * ∂L/∂d_e ≈ η_d * e_trace * (pre_spike_delayed - post_spike) 
    利用资格迹和前后脉冲时差的梯度近似来调整延迟。
    """
    
    def __init__(self, max_edges: int, max_delay: int = 20, min_delay: int = 1,
                 delay_lr: float = 0.001, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.max_edges = max_edges
        self.max_delay = max_delay
        self.min_delay = min_delay
        self.delay_lr = delay_lr
        
        # 可学习的连续延迟值（手动更新，非梯度优化，故使用 buffer 而非 Parameter）
        # 初始化为 min_delay (而非中间值): 延迟模块已接入前向通路,
        # 若初始延迟 ~10 步, 静态输入样本在 20 步窗口内到达胞体的信号
        # 会被大幅衰减; 从无延迟起步, 由延迟学习规则按需增大。
        init_delay = torch.ones(max_edges, **factory_kwargs) * float(min_delay)
        self.register_buffer("delay_continuous", init_delay)
        
        # 离散化后的延迟索引
        self.register_buffer(
            "delay_discrete",
            torch.ones(max_edges, dtype=torch.long, device=device) * min_delay
        )
        
        # 环形缓冲区：存储最近 max_delay 步的前突触脉冲
        # shape: (batch, max_delay, max_edges) — 按样本独立持有
        # (输出通路必须保留 batch 维, 否则平衡批次下类别信息在
        #  延迟缓冲处被批量均值抹平)。初始 batch=1, 运行时按需扩展。
        # persistent=False: 瞬态状态, 不进 state_dict。
        self.register_buffer(
            "spike_buffer",
            torch.zeros(1, max_delay, max_edges, **factory_kwargs),
            persistent=False,
        )
        self.register_buffer(
            "buffer_ptr",
            torch.tensor(0, dtype=torch.long, device=device)
        )
        
        # 延迟相关的STDP迹
        self.register_buffer(
            "pre_trace_delayed",
            torch.zeros(max_edges, **factory_kwargs)
        )
        self.register_buffer(
            "post_trace",
            torch.zeros(max_edges, **factory_kwargs)
        )
        
        self.trace_decay = 0.95
        
    def reset(self):
        """重置所有状态"""
        self.spike_buffer.zero_()
        self.buffer_ptr.zero_()
        self.pre_trace_delayed.zero_()
        self.post_trace.zero_()
        # 重新同步离散化 (reset 可能发生在 update_delays 之后, 保证
        # step_fast 读取的 delay_discrete 与 delay_continuous 一致)
        self._discretize_delays()
        
    def _discretize_delays(self):
        """将连续延迟离散化为整数步"""
        self.delay_discrete.data.copy_(
            torch.clamp(
                torch.round(self.delay_continuous).long(),
                self.min_delay,
                self.max_delay - 1
            )
        )
    
    def step_fast(self, pre_spk: torch.Tensor, post_spk: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        快时钟 (1ms) 前向
        
        Args:
            pre_spk: (max_edges,) 或 (batch, max_edges) 当前步的前突触脉冲
            post_spk: (max_edges,) 或 (batch, max_edges) 后突触脉冲
            
        Returns:
            delayed_spk: (max_edges,) 经过延迟的前突触脉冲
            stdp_delta: (max_edges,) 延迟相关的STDP信号
        """
        # 处理 batch 维度: 保留 batch 维 (输出通路需要逐样本延迟值);
        # post_spk 仅用于延迟学习迹 (max_edges,), 取批量均值即可
        if pre_spk.dim() == 1:
            pre_spk = pre_spk.unsqueeze(0)  # (1, max_edges)
        if post_spk.dim() > 1:
            post_spk = post_spk.mean(dim=0)

        batch_size = pre_spk.shape[0]

        # 1. 批量维度变化时扩展缓冲 (保留已有历史)
        # keep 只受 batch 维度约束: max_delay 是时间维 (每样本的时间历史),
        # 与 batch 无关。原实现 min(max_delay, ...) 在 batch 64→128 时只复制
        # 20 行 (max_delay=20), 时间历史被截断, 且批次缩小 (128→16) 时
        # 超出新 batch 的切片赋值 shape 不匹配 —— 两者都以 min(旧batch, 新batch)
        # 为界。batch 切换后前 max_delay 步读到零填充 (延迟历史冷启动),
        # 属预期行为, 由 STDP 迹与延迟学习逐步恢复。
        if self.spike_buffer.shape[0] != batch_size:
            new_buf = self.spike_buffer.new_zeros(
                batch_size, self.max_delay, self.max_edges
            )
            keep = min(self.spike_buffer.shape[0], batch_size)
            new_buf[:keep] = self.spike_buffer[:keep]
            self.spike_buffer = new_buf

        # 2. 将当前脉冲存入环形缓冲区
        # 必须 detach: 直接写入带梯度的张量会使 buffer 产生 CopySlices
        # 图节点, 缓冲链入计算图并在跨 batch 后触发
        # "backward through the graph a second time" 崩溃
        ptr = self.buffer_ptr % self.max_delay
        self.spike_buffer[:, ptr] = pre_spk.detach()

        # 3. 从缓冲区读取延迟后的脉冲 (逐样本)
        # 延迟只在慢时钟 update_delays 中变化, 离散化也只在彼处执行
        # (round→long→clamp 3 kernel 不进热路径); __init__ 后 delay_discrete
        # 初始值已是 min_delay, reset() 末尾重新同步, 此处直接使用。
        # 读取位置 = (当前指针 - 延迟) mod max_delay
        read_idx = (self.buffer_ptr - self.delay_discrete) % self.max_delay
        # delayed[b, e] = buf[b, read_idx[e], e]
        # gather 沿 permute 后的时间维 (dim 2): 只需 read_idx 一个 (b,E) 索引
        # (expand 视图), 免去原高级索引的 arange(batch) 与 arange(max_edges)
        # 两个索引张量分配。
        delayed_spk = torch.gather(
            self.spike_buffer.permute(0, 2, 1),  # (b, E, T) 视图, 无拷贝
            2,
            read_idx.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1),
        ).squeeze(-1)  # (batch, max_edges)

        # 4. 更新延迟相关的STDP迹 (批量均值)
        self.pre_trace_delayed.data.mul_(self.trace_decay).add_(delayed_spk.mean(dim=0))
        self.post_trace.data.mul_(self.trace_decay).add_(post_spk)

        # 5. 计算延迟敏感的STDP信号
        stdp_delta = delayed_spk.mean(dim=0) * self.post_trace - post_spk * self.pre_trace_delayed

        # 6. 前进指针
        self.buffer_ptr.data.add_(1)

        if batch_size == 1 and self.spike_buffer.shape[0] == 1:
            return delayed_spk.squeeze(0), stdp_delta
        return delayed_spk, stdp_delta
    
    def update_delays(self, e_trace: torch.Tensor, timing_error: torch.Tensor):
        """
        慢时钟 (100ms) 延迟学习
        
        根据资格迹和时序误差调整延迟值:
            Δd = η_d * e_trace * timing_error
        
        timing_error > 0: 脉冲到达太早 → 增大延迟
        timing_error < 0: 脉冲到达太晚 → 减小延迟
        
        Args:
            e_trace: (max_edges,) 资格迹
            timing_error: (max_edges,) 时序误差信号
        """
        delta_d = self.delay_lr * e_trace * timing_error
        self.delay_continuous.data.add_(delta_d).clamp_(float(self.min_delay), float(self.max_delay - 1))
        # 慢时钟离散化: 延迟仅在此变化, step_fast (快时钟) 不再重复离散化
        self._discretize_delays()
    
    def get_delay_stats(self) -> dict:
        """返回延迟分布的统计信息（用于可解释监控）"""
        d = self.delay_continuous.detach()
        return {
            "delay_mean": d.mean().item(),
            "delay_std": d.std().item(),
            "delay_min": d.min().item(),
            "delay_max": d.max().item(),
            "delay_entropy": -torch.sum(
                F.softmax(d, dim=0) * F.log_softmax(d, dim=0)
            ).item()
        }
