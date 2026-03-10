"""
多尺度液态门控元胞 (Multi-Scale Liquid Gated Cell)
白皮书 §3.2, §1.3.2, §4.2.1

包含:
- 快门控 (1ms, 调制离子通道) 
- 慢门控 (100ms, 调制全局可塑性和发放阈值)
- 树突非线性选项 (局部阈值、Ca尖峰样事件、分支独立积分)
- STE (直通估计器) 使替代梯度可流通
- g_slow 实际用于调制可塑性和膜电位噪声

参考: [R3] Dendrify, [R4] Temporal dendritic heterogeneity
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class DendriteCompartment(nn.Module):
    """
    树突亚室非线性模块 (Dendrite Compartment)
    白皮书 §1.3.2: 在点神经元之外引入轻量的"树突分支/亚室"非线性,
    局部阈值、Ca尖峰样事件、分支独立积分,提升单元表达能力。
    
    每个神经元拥有 num_branches 个独立的树突分支,
    每个分支进行独立的非线性积分,然后汇聚到胞体。
    """
    
    def __init__(self, num_neurons: int, num_branches: int = 4,
                 dendrite_threshold: float = 0.3,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.num_neurons = num_neurons
        self.num_branches = num_branches
        self.dendrite_threshold = dendrite_threshold
        
        # 每个分支的独立权重 (用于将输入电流分配到不同树突分支)
        self.branch_weights = nn.Parameter(
            torch.randn(num_branches, num_neurons, **factory_kwargs) * 0.1
        )
        
        # 每个分支的局部膜电位 (Ca尖峰)
        self.register_buffer(
            "branch_potential",
            torch.zeros(num_branches, num_neurons, **factory_kwargs)
        )
        
        # 分支衰减常数
        self.branch_decay = 0.8
        
    def reset(self):
        self.branch_potential.zero_()
    
    def forward(self, I_syn: torch.Tensor) -> torch.Tensor:
        """
        对输入电流进行树突非线性处理 (支持batch向量化)

        Args:
            I_syn: (num_neurons,) 或 (batch, num_neurons) 突触输入电流

        Returns:
            I_dendrite: (num_neurons,) 或 (batch, num_neurons) 树突处理后的电流 (汇聚到胞体)
        """
        is_batched = I_syn.dim() > 1

        if is_batched:
            batch_size = I_syn.shape[0]

            # 1. 将输入分配到各分支 (分支加权) - 向量化
            # I_syn: (batch, num_neurons) -> (batch, 1, num_neurons)
            # branch_weights: (num_branches, num_neurons) -> (1, num_branches, num_neurons)
            # branch_input: (batch, num_branches, num_neurons)
            branch_input = self.branch_weights.unsqueeze(0) * I_syn.unsqueeze(1)

            # 2. 分支独立积分 - 扩展buffer以匹配batch维度
            # branch_potential: (num_branches, num_neurons) -> (batch, num_branches, num_neurons)
            branch_potential = self.branch_potential.unsqueeze(0).expand(batch_size, -1, -1).clone()

            branch_potential = self.branch_decay * branch_potential + branch_input

            # 3. 分支局部非线性: Ca尖峰样事件 - 向量化
            above_threshold = (branch_potential > self.dendrite_threshold).float()
            ca_spike = above_threshold * torch.relu(branch_potential - self.dendrite_threshold) * 2.0

            # 未超过阈值的分支保持线性传递
            linear_pass = (1.0 - above_threshold) * branch_potential

            branch_output = linear_pass + ca_spike

            # 4. 树突分支汇聚到胞体 (求和) - (batch, num_neurons)
            I_dendrite = branch_output.sum(dim=1)

            # 5. 分支重置 (Ca尖峰后重置该分支)
            branch_potential_after_reset = branch_potential * (1.0 - above_threshold * 0.5)

            # 修复: 使用 detach() 避免计算图问题，确保梯度流稳定
            self.branch_potential.data.copy_(branch_potential_after_reset.detach().mean(dim=0))

            return I_dendrite
        else:
            # 单样本情况 - 保持原有实现
            # 1. 将输入分配到各分支 (分支加权)
            # branch_input: (num_branches, num_neurons)
            branch_input = self.branch_weights * I_syn.unsqueeze(0)

            # 2. 分支独立积分 (in-place 保护 buffer 注册)
            self.branch_potential.data.copy_(self.branch_decay * self.branch_potential + branch_input)

            # 3. 分支局部非线性: Ca尖峰样事件
            # 超过局部阈值的分支产生非线性放大 (模拟树突钙尖峰)
            above_threshold = (self.branch_potential > self.dendrite_threshold).float()
            ca_spike = above_threshold * torch.relu(self.branch_potential - self.dendrite_threshold) * 2.0

            # 未超过阈值的分支保持线性传递
            linear_pass = (1.0 - above_threshold) * self.branch_potential

            branch_output = linear_pass + ca_spike

            # 4. 树突分支汇聚到胞体 (求和)
            I_dendrite = branch_output.sum(dim=0)

            # 5. 分支重置 (Ca尖峰后重置该分支, in-place 保护 buffer 注册)
            self.branch_potential.data.mul_(1.0 - above_threshold * 0.5)

            return I_dendrite


class LiquidGatedCell(nn.Module):
    """
    多尺度液态门控元胞 (Liquid Gated Cell)
    
    核心状态变量:
    - v: 膜电位 (快, ms级)
    - g_fast: 快门控 (快, ms级, 调制离子通道)  
    - g_slow: 慢门控 (慢, 100ms级, 调制可塑性和噪声)
    - a: 适应变量 (慢, 秒级, 模拟慢速钾电流)
    
    新增功能:
    - 树突非线性 (可选)
    - STE (直通估计器) 使梯度可流通
    - g_slow 实际调制膜电位噪声强度和可塑性窗口
    """
    def __init__(self, num_neurons: int, tau_v: float = 10.0, tau_g_fast: float = 5.0, 
                 tau_g_slow: float = 200.0, tau_a: float = 100.0, theta_0: float = 1.0,
                 enable_dendrites: bool = False, num_branches: int = 4,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_neurons = num_neurons
        
        self.tau_v = tau_v
        self.tau_g_fast = tau_g_fast
        self.tau_g_slow = tau_g_slow
        self.tau_a = tau_a
        self.a_inc = 0.05
        self.theta_0 = theta_0
        self.enable_dendrites = enable_dendrites
        
        # 树突非线性模块 (可选)
        if enable_dendrites:
            self.dendrite = DendriteCompartment(
                num_neurons, num_branches, **factory_kwargs
            )
        
        # 门控线性层参数 (element-wise, 低开销)
        self.W_f = nn.Parameter(torch.randn(num_neurons, **factory_kwargs) * 0.1)
        self.U_f = nn.Parameter(torch.randn(num_neurons, **factory_kwargs) * 0.1)
        self.bias_f = nn.Parameter(torch.zeros(num_neurons, **factory_kwargs))
        
        self.W_s = nn.Parameter(torch.randn(num_neurons, **factory_kwargs) * 0.1)
        self.U_s = nn.Parameter(torch.randn(num_neurons, **factory_kwargs) * 0.1)
        self.Z_s = nn.Parameter(torch.randn(num_neurons, **factory_kwargs) * 0.1)
        self.bias_s = nn.Parameter(torch.zeros(num_neurons, **factory_kwargs))
        
        # 快变量状态
        self.register_buffer("v", torch.zeros(num_neurons, **factory_kwargs))
        self.register_buffer("g_fast", torch.zeros(num_neurons, **factory_kwargs))
        
        # 慢变量状态
        self.register_buffer("g_slow", torch.ones(num_neurons, **factory_kwargs) * 0.5)
        self.register_buffer("a", torch.zeros(num_neurons, **factory_kwargs))
        self.register_buffer("theta", torch.ones(num_neurons, **factory_kwargs) * theta_0)
        
        # 滑动窗口记录 (100步)
        self.register_buffer("spk_window", torch.zeros(100, num_neurons, **factory_kwargs))
        self.register_buffer("delta_window", torch.zeros(100, num_neurons, **factory_kwargs))
        
        # 当前环形缓冲区的索引
        self.register_buffer("window_idx", torch.tensor(0, dtype=torch.long, device=device))
        
    @staticmethod
    def _fast_sigmoid(x, alpha=25.0):
        """替代梯度函数"""
        return torch.sigmoid(alpha * x)

    def reset_hidden(self):
        """重置所有状态"""
        self.v.zero_()
        self.g_fast.zero_()
        self.g_slow.fill_(0.5)
        self.a.zero_()
        self.theta.fill_(self.theta_0)
        self.spk_window.zero_()
        self.delta_window.zero_()
        self.window_idx.zero_()
        if self.enable_dendrites:
            self.dendrite.reset()

    def step_fast(self, I_syn: torch.Tensor, I_ext: Optional[torch.Tensor] = None, 
                  I_inh: Optional[torch.Tensor] = None,
                  theta_ie: Optional[torch.Tensor] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        快时钟 (1ms) 前向更新
        
        Eq: v(t+1) = τ_v^{-1} * v(t) + (1-τ_v^{-1}) * (I_syn + I_ext - I_inh) 
                      + σ(g_fast) * η(t)
        
        Args:
            I_syn: 突触输入电流 (batch, num_neurons) 或 (num_neurons,)
            I_ext: 外部输入电流 (可选)
            I_inh: 侧向抑制电流 (可选)
            theta_ie: 稳态可塑性的阈值调整量 (可选)
        """
        # 确保 I_syn 有 batch 维度以便一致处理
        original_is_batched = I_syn.dim() > 1
        if not original_is_batched:
            I_syn = I_syn.unsqueeze(0)

        # ── FP32 精度围栏 ──────────────────────────────────────────
        # SNN 状态变量 (v, g_fast, g_slow, a, theta) 必须始终为 FP32.
        # 当 autocast (BF16) 激活时, I_syn 可能被降级为 BF16 / FP16;
        # 在此强制转回 FP32, 确保所有状态更新在 FP32 精度下执行.
        I_syn = I_syn.float()

        batch_size = I_syn.shape[0]

        if I_ext is None:
            I_ext = 0.0
        if I_inh is None:
            I_inh = 0.0
        
        # 树突非线性处理 (如果启用) - 向量化batch处理
        if self.enable_dendrites:
            I_syn = self.dendrite(I_syn)
            
        decay = 1.0 - (1.0 / self.tau_v)
        
        # 膜电位 v 扩展到 batch 维度
        # 使用 expand 而非检查 shape[0] == batch_size (因为 shape[0] == num_neurons)
        v_current = self.v.unsqueeze(0).expand(batch_size, -1)
        g_fast_current = self.g_fast.unsqueeze(0).expand(batch_size, -1)
        a_current = self.a.unsqueeze(0).expand(batch_size, -1)

        # g_slow 调制噪声强度 - 使用 sqrt 而非线性缩放，避免噪声过大
        noise_scale = 0.01 * torch.sqrt(1.0 + self.g_slow).unsqueeze(0)
        noise = torch.randn_like(v_current) * noise_scale
        
        # 膜电位更新
        v_next = decay * v_current + (1.0 - decay) * (I_syn + I_ext - I_inh) + g_fast_current * noise
        
        # 有效阈值
        effective_theta = self.theta.unsqueeze(0)
        if theta_ie is not None:
            effective_theta = effective_theta + theta_ie.unsqueeze(0)
        
        # === STE 脉冲发放 ===
        spk_soft = self._fast_sigmoid(v_next - effective_theta)
        spk_hard = (v_next >= effective_theta).float()
        spk = spk_hard + (spk_soft - spk_soft.detach())
        
        # 软重置并更新状态 (取 batch 平均回写 buffer，或保留 batch 状态)
        # 这里的实现选择: 状态 buffer 始终保留单样本维度，batch 运行时临时扩展
        v_post_reset = v_next - spk_hard * effective_theta
        self.v.data.copy_(v_post_reset.mean(dim=0))
        
        # 快门控更新
        decay_f = 1.0 - (1.0 / self.tau_g_fast)
        g_fast_target = torch.sigmoid(self.W_f.unsqueeze(0) * v_next + self.U_f.unsqueeze(0) * a_current + self.bias_f.unsqueeze(0))
        g_fast_next = decay_f * g_fast_current + (1.0 - decay_f) * g_fast_target
        self.g_fast.data.copy_(g_fast_next.mean(dim=0))
        
        # 更新滑动窗口 - 使用 in-place 操作保护 buffer 状态
        with torch.no_grad():
            idx = self.window_idx % 100
            self.spk_window[idx] = spk_hard.mean(dim=0)
            self.window_idx.add_(1)
        
        # 如果原始输入不是 batch，则返回时剥离 batch 维度
        if not original_is_batched:
            return spk.squeeze(0), v_post_reset.squeeze(0)
        return spk, v_post_reset

    def step_slow(self, global_e: torch.Tensor):
        """
        慢时钟 (100ms) 更新
        
        更新适应变量 a, 慢门控 g_slow，并同步调整发放阈值 theta。
        g_slow 的输出将被其他模块读取用于:
        - 调制可塑性学习率
        - 调制多跳资格迹的传播
        - 控制噪声注入强度
        """
        # ── FP32 精度围栏 ──
        global_e = global_e.float()
        mean_spk = self.spk_window.float().mean(dim=0)
        mean_delta = self.delta_window.float().mean(dim=0)
        
        # 适应变量更新 (in-place 保护 buffer 注册)
        self.a.data.mul_(1.0 - 1.0 / self.tau_a).add_(self.a_inc * mean_spk)
        self.theta.data.copy_(self.theta_0 + self.a)
        
        # 慢门控更新
        # g_slow 同时接收全局星形胶质细胞调控信号 (global_e)
        decay_s = 1.0 - (1.0 / self.tau_g_slow)
        g_slow_target = torch.sigmoid(
            self.W_s * mean_spk + self.U_s * mean_delta + self.Z_s * global_e + self.bias_s
        )
        self.g_slow.data.copy_(decay_s * self.g_slow + (1.0 - decay_s) * g_slow_target)

    def update_delta_window(self, delta_spk: torch.Tensor):
        """记录反向传递的误差脉冲到窗口中，供慢时钟更新使用。"""
        idx = (self.window_idx - 1) % 100
        self.delta_window[idx] = delta_spk
    
    def get_plasticity_modulation(self) -> torch.Tensor:
        """
        返回 g_slow 作为可塑性调制因子
        供 三因素可塑性、资格迹传播等模块使用
        """
        return self.g_slow
    
    def get_firing_rate(self) -> torch.Tensor:
        """返回当前滑动窗口内的平均发放率"""
        return self.spk_window.mean(dim=0)
