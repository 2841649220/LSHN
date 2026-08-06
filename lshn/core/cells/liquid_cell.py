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
        对输入电流进行树突非线性处理

        Args:
            I_syn: (num_neurons,) 或 (batch, num_neurons) 突触输入电流

        Returns:
            I_dendrite: 与 I_syn 相同形状 树突处理后的电流 (汇聚到胞体)
        """
        is_batched = I_syn.dim() > 1
        if not is_batched:
            I_syn = I_syn.unsqueeze(0)

        batch_size = I_syn.shape[0]

        # branch_input: (num_branches, batch, num_neurons)
        branch_input = self.branch_weights.unsqueeze(1) * I_syn.unsqueeze(0)

        # 分支电位更新: 保留 batch 维度 (每个样本独立积分),输出使用可微路径
        # (梯度可流向 branch_weights 与 I_syn),buffer 仅保存 detached 状态副本
        # (取 batch 平均回写,与 LiquidGatedCell 的均值场状态策略一致)
        updated_potential = self.branch_decay * self.branch_potential.unsqueeze(1) + branch_input
        self.branch_potential.data.copy_(updated_potential.detach().mean(dim=1))

        above_threshold = (updated_potential > self.dendrite_threshold).float()
        ca_spike = above_threshold * torch.relu(updated_potential - self.dendrite_threshold) * 2.0
        linear_pass = (1.0 - above_threshold) * updated_potential
        branch_output = linear_pass + ca_spike

        I_dendrite = branch_output.sum(dim=0)  # (batch, num_neurons)

        self.branch_potential.data.mul_(1.0 - above_threshold.detach().mean(dim=1) * 0.5)

        if not is_batched:
            I_dendrite = I_dendrite.squeeze(0)

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
                 dendrite_threshold: float = 0.3, noise_std: float = 0.01,
                 input_gain: float = 1.0,
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
        self.noise_std = noise_std
        # 输入电流增益: 自适应归一化后的目标驱动量级 (≈ 有效输入标准差)。
        # 配合 _input_std_ema 使发放率对输入分布/拓扑演化鲁棒
        # (初始化"唤醒"关键旋钮, 由皮层/海马体配置按层级设置)
        self.input_gain = input_gain
        # 输入电流标准差的 EMA (用于自适应归一化)。
        # 首次观测非零输入时直接赋值 (避免慢收敛), 静默期冻结不衰减。
        self.register_buffer(
            "_input_std_ema", torch.ones((), **factory_kwargs) * 1e-4,
            persistent=False,
        )
        self.register_buffer(
            "_norm_seen", torch.zeros((), dtype=torch.bool, device=device),
            persistent=False,
        )

        # 树突非线性模块 (可选)
        if enable_dendrites:
            self.dendrite = DendriteCompartment(
                num_neurons, num_branches, dendrite_threshold, **factory_kwargs
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
        # v (膜电位) 按样本持有 (batch, num_neurons): 膜电位是快变量, 样本间
        # 必须独立积分。原实现每步写回 batch 均值 → 零均值输入下 E_batch[v]≈0,
        # buffer 永久塌缩, 膜电位从不积分 (每步 v≈0.1·input), 网络初始化
        # 即静默, 这是"网络学不到东西"的第一性根因。
        # 初始为空 (0, num_neurons), 首次 forward 时按 batch 重建。
        # persistent=False: 膜电位是瞬态状态, 不进 state_dict
        # (检查点更小, 加载时无形状不匹配告警)。
        self.register_buffer("v", torch.zeros(0, num_neurons, **factory_kwargs),
                             persistent=False)
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
    def _fast_sigmoid(x, alpha=10.0):
        """替代梯度函数 (STE 代理梯度)

        alpha=25 时 sigmoid'(25·(v−θ)) 在 |v−θ|>0.15 处即衰减到 <0.01,
        阈值附近的梯度带过窄, 初始化稍有偏差梯度即数值死亡;
        alpha=10 使梯度带加宽 ~2.5 倍, 梯度更易流通。
        """
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
        is_batched = I_syn.dim() > 1
        if not is_batched:
            I_syn = I_syn.unsqueeze(0)
            
        batch_size = I_syn.shape[0]
        
        if I_ext is None:
            I_ext = 0.0
        if I_inh is None:
            I_inh = 0.0
        
        # 树突非线性处理 (如果启用)
        if self.enable_dendrites:
            # DendriteCompartment 也需要适配 batch
            I_syn = self.dendrite(I_syn)
            
        decay = 1.0 - (1.0 / self.tau_v)

        # 膜电位 v 按样本持有 (batch, num_neurons): 批量变化时重建状态
        # (保留旧状态的批量均值); g_fast/a 为慢门控/适应变量, 维持
        # 单样本语义并取 batch 平均回写 (慢变量均值场化可接受)。
        if self.v.shape[0] != batch_size:
            new_v = self.v.new_empty(batch_size, self.num_neurons)
            if self.v.shape[0] > 0:
                new_v.copy_(self.v.mean(dim=0, keepdim=True))
            else:
                new_v.zero_()
            self.v = new_v
        v_current = self.v
        g_fast_current = self.g_fast.unsqueeze(0).expand(batch_size, -1)
        a_current = self.a.unsqueeze(0).expand(batch_size, -1)

        # g_slow 调制噪声强度 (仅训练时注入)
        if self.training:
            noise_scale = self.noise_std * (1.0 + self.g_slow.unsqueeze(0))
            noise = torch.randn_like(v_current) * noise_scale
        else:
            noise = torch.zeros_like(v_current)
        
        # 输入电流自适应归一化: 除以其 EMA 标准差, 使有效驱动量级 ≈ input_gain,
        # 与输入绝对尺度无关 (对数据集分布变化/结构演化导致的驱动衰减鲁棒)。
        # 首次观测直接赋值; 静默期 (std=0) 冻结 ema 不衰减, 恢复输入时
        # 不会因陈旧小尺度而病理放大。分母下限仅防除零。
        # 语义与原实现逐分支一致: 仅当 cur_std>0 时更新 ema; 未见输入前直接
        # 赋值。原实现用 0 维 CUDA 张量的 Python bool 判断 (if cur_std > 0 /
        # if not self._norm_seen), 每快步 2 次全设备同步; 现用 torch.where /
        # logical_or_ 纯 GPU 侧表达, 语义完全等价且无同步。
        # EMA 仅在 training 模式更新: 评估/推理不污染归一化统计。
        if self.training:
            cur_std = I_syn.detach().std()
            new_ema = torch.where(self._norm_seen,
                                  self._input_std_ema * 0.99 + cur_std * 0.01,
                                  cur_std)
            self._input_std_ema.data.copy_(
                torch.where(cur_std > 0, new_ema, self._input_std_ema)
            )
            self._norm_seen.data.logical_or_(cur_std > 0)
        norm_scale = self._input_std_ema.clamp(min=1e-4)

        # 膜电位更新。两点与白皮书 §3.3.3 的偏差说明 (实现有意保留):
        # 1) 白皮书膜电位方程含 −a_t 适应项, 本实现将适应折入有效阈值
        #    (theta = theta_0 + a, 见 step_slow), 方程中无显式 −a_t;
        # 2) input_gain 是整项增益, 实际同样放大 I_syn 中携带的负向
        #    (抑制性) 分量, 并非"仅兴奋性" —— 保留以维持自适应归一化的
        #    净驱动尺度语义。
        v_next = decay * v_current + (1.0 - decay) * (
            I_syn * (self.input_gain / norm_scale) + I_ext - I_inh
        ) + g_fast_current * noise
        
        # 有效阈值
        effective_theta = self.theta.unsqueeze(0)
        if theta_ie is not None:
            effective_theta = effective_theta + theta_ie.unsqueeze(0)
        
        # === STE 脉冲发放 ===
        spk_soft = self._fast_sigmoid(v_next - effective_theta)
        spk_hard = (v_next >= effective_theta).float()
        spk = spk_hard + (spk_soft - spk_soft.detach())
        
        # 软重置并更新状态 (按样本写回, 保留 batch 维度的独立膜电位轨迹)
        v_post_reset = v_next - spk_hard * effective_theta
        self.v.data.copy_(v_post_reset.detach())
        
        # 快门控更新
        decay_f = 1.0 - (1.0 / self.tau_g_fast)
        g_fast_target = torch.sigmoid(self.W_f.unsqueeze(0) * v_next + self.U_f.unsqueeze(0) * a_current + self.bias_f.unsqueeze(0))
        g_fast_next = decay_f * g_fast_current + (1.0 - decay_f) * g_fast_target
        self.g_fast.data.copy_(g_fast_next.mean(dim=0))
        
        # 更新滑动窗口
        idx = self.window_idx % 100
        self.spk_window[idx] = spk_hard.detach().mean(dim=0)
        self.window_idx += 1
        
        # 如果原始输入不是 batch，则返回时剥离 batch 维度
        if not is_batched:
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
        mean_spk = self.spk_window.mean(dim=0)
        mean_delta = self.delta_window.mean(dim=0)
        
        self.a.data.mul_(1.0 - 1.0 / self.tau_a).add_(self.a_inc * mean_spk)
        self.theta.data.copy_(self.theta_0 + self.a)
        
        decay_s = 1.0 - (1.0 / self.tau_g_slow)
        g_slow_target = torch.sigmoid(
            self.W_s * mean_spk + self.U_s * mean_delta + self.Z_s * global_e + self.bias_s
        )
        self.g_slow.data.mul_(decay_s).add_((1.0 - decay_s) * g_slow_target)

    def update_delta_window(self, delta_spk: torch.Tensor):
        """记录反向传递的误差脉冲到窗口中，供慢时钟更新使用。"""
        # 0 维 CUDA 张量索引合法 (全 GPU 侧), 避免 .item() 每快步同步一次
        delta_spk = delta_spk.detach().to(self.delta_window.device, self.delta_window.dtype)
        self.delta_window[(self.window_idx - 1) % 100] = delta_spk
    
    def get_plasticity_modulation(self) -> torch.Tensor:
        """
        返回 g_slow 作为可塑性调制因子
        供 三因素可塑性、资格迹传播等模块使用
        """
        return self.g_slow
    
    def get_firing_rate(self) -> torch.Tensor:
        """返回当前滑动窗口内的平均发放率"""
        return self.spk_window.mean(dim=0)
