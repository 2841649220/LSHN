"""
测试: 轴突延迟学习模块 (AxonalDelayModule)
覆盖: 白皮书 §1.3.4, §3.5.4
"""
import torch
import pytest
from lshn.core.synapses.axonal_delay import AxonalDelayModule


class TestAxonalDelayModule:
    
    @pytest.fixture
    def delay_mod(self):
        """标准测试延迟模块"""
        return AxonalDelayModule(max_edges=16, max_delay=20, min_delay=1)
    
    def test_init_shapes(self, delay_mod):
        """测试初始化后参数和缓冲区形状"""
        assert delay_mod.delay_continuous.shape == (16,)
        assert delay_mod.delay_discrete.shape == (16,)
        # 缓冲按样本持有: (batch, max_delay, max_edges), 初始 batch=1 动态扩展
        assert delay_mod.spike_buffer.shape == (1, 20, 16)
        assert delay_mod.pre_trace_delayed.shape == (16,)
        assert delay_mod.post_trace.shape == (16,)
        assert delay_mod.buffer_ptr.item() == 0
    
    def test_init_delay_values(self, delay_mod):
        """测试初始延迟值 = min_delay (从无延迟起步, 由学习规则按需增大)"""
        expected = 1.0
        assert torch.allclose(delay_mod.delay_continuous,
                              torch.ones(16) * expected)
    
    def test_step_fast_output_shape(self, delay_mod):
        """测试快时钟返回 (delayed_spk, stdp_delta) 的形状"""
        pre_spk = (torch.rand(16) > 0.7).float()
        post_spk = (torch.rand(16) > 0.7).float()
        
        delayed_spk, stdp_delta = delay_mod.step_fast(pre_spk, post_spk)
        
        assert delayed_spk.shape == (16,)
        assert stdp_delta.shape == (16,)
    
    def test_step_fast_buffer_advances(self, delay_mod):
        """测试每步 buffer_ptr 递增"""
        pre_spk = torch.zeros(16)
        post_spk = torch.zeros(16)
        
        for i in range(5):
            delay_mod.step_fast(pre_spk, post_spk)
            assert delay_mod.buffer_ptr.item() == i + 1
    
    def test_delay_actually_delays(self):
        """测试脉冲经过延迟模块后确实被延迟"""
        mod = AxonalDelayModule(max_edges=4, max_delay=10, min_delay=3)
        
        # 设置所有延迟为3步
        mod.delay_continuous.data.fill_(3.0)
        mod._discretize_delays()
        
        post_spk = torch.zeros(4)
        
        # 第0步: 注入一个脉冲到edge 0
        pre_spk = torch.zeros(4)
        pre_spk[0] = 1.0
        d0, _ = mod.step_fast(pre_spk, post_spk)
        
        # 延迟3步，第0步应该读不到
        assert d0[0].item() == 0.0
        
        # 继续推进直到第3步
        for step in range(1, 3):
            d, _ = mod.step_fast(torch.zeros(4), post_spk)
            assert d[0].item() == 0.0  # 延迟尚未到达
        
        # 第3步: 应该读到延迟后的脉冲
        d3, _ = mod.step_fast(torch.zeros(4), post_spk)
        assert d3[0].item() == 1.0
    
    def test_stdp_signal_ltp_ltd(self):
        """测试 STDP 信号的 LTP/LTD 方向 (确定性输入)"""
        mod = AxonalDelayModule(max_edges=4, max_delay=5, min_delay=1)
        mod.delay_continuous.data.fill_(1.0)
        mod._discretize_delays()

        pre1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        post1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        zero4 = torch.zeros(4)

        # === 场景1: pre+post 连续共发放 → LTP (stdp_delta > 0) ===
        # stdp_delta = delayed_spk * post_trace - post_spk * pre_trace_delayed
        # Step 0: pre+post 同时发放 (延迟1步未到, 只积累 post_trace=1)
        _, _ = mod.step_fast(pre1, post1)

        # Step 1: Step0 的前脉冲延迟到达 (delayed[0]=1), post 再次发放:
        #   post_trace = 0.95*1+1 = 1.95 > pre_trace_delayed = 0+1 = 1 → LTP 正项占优
        delayed, stdp = mod.step_fast(zero4, post1)
        assert delayed[0].item() == 1.0      # 延迟后的前脉冲已到达
        assert stdp[0].item() > 0.0          # LTP

        # === 场景2: pre 连续发放后 post 才发放 → LTD (stdp_delta < 0) ===
        mod.reset()
        mod.delay_continuous.data.fill_(1.0)
        mod._discretize_delays()

        # Step 0-1: 前突触连续发放 (Step1 时 Step0 的 pre 延迟到达, pre_trace=1)
        _, _ = mod.step_fast(pre1, zero4)
        _, _ = mod.step_fast(pre1, zero4)

        # Step 2: Step1 的前脉冲延迟到达 (delayed[0]=1) 且 post 同时发放:
        #   pre_trace_delayed = 0.95*1+1 = 1.95 > post_trace = 0+1 = 1 → LTD 负项占优
        delayed2, stdp2 = mod.step_fast(zero4, post1)
        assert delayed2[0].item() == 1.0
        assert stdp2[0].item() < 0.0         # LTD

        # 其余超边无活动: 信号为 0 且有限
        assert stdp[1:].abs().sum().item() == 0.0
        assert torch.isfinite(stdp2).all().item()
    
    def test_update_delays(self, delay_mod):
        """测试慢时钟延迟学习"""
        old_delay = delay_mod.delay_continuous.clone()
        
        e_trace = torch.randn(16)
        timing_error = torch.randn(16) * 0.5
        
        delay_mod.update_delays(e_trace, timing_error)
        
        # 延迟应被更新
        assert not torch.allclose(delay_mod.delay_continuous, old_delay)
        # 延迟仍在合法范围内
        assert torch.all(delay_mod.delay_continuous >= float(delay_mod.min_delay))
        assert torch.all(delay_mod.delay_continuous <= float(delay_mod.max_delay - 1))
    
    def test_update_delays_direction(self):
        """测试延迟学习方向：正 timing_error → 延迟增大"""
        mod = AxonalDelayModule(max_edges=4, max_delay=20, min_delay=1, delay_lr=0.1)
        mod.delay_continuous.data.fill_(10.0)
        
        e_trace = torch.ones(4)
        timing_error = torch.ones(4)  # 到达太早 → 增大延迟
        
        old_d = mod.delay_continuous.clone()
        mod.update_delays(e_trace, timing_error)
        
        # Δd = η * e_trace * timing_error > 0 → delay 增大
        assert torch.all(mod.delay_continuous >= old_d)
    
    def test_get_delay_stats(self, delay_mod):
        """测试延迟分布统计"""
        stats = delay_mod.get_delay_stats()
        
        assert "delay_mean" in stats
        assert "delay_std" in stats
        assert "delay_min" in stats
        assert "delay_max" in stats
        assert "delay_entropy" in stats
        
        assert isinstance(stats["delay_mean"], float)
    
    def test_reset(self, delay_mod):
        """测试状态重置"""
        # 先推几步
        for _ in range(10):
            delay_mod.step_fast(torch.rand(16), torch.rand(16))
        
        delay_mod.reset()
        
        assert torch.all(delay_mod.spike_buffer == 0.0)
        assert delay_mod.buffer_ptr.item() == 0
        assert torch.all(delay_mod.pre_trace_delayed == 0.0)
        assert torch.all(delay_mod.post_trace == 0.0)
    
    def test_ring_buffer_wraps(self):
        """测试环形缓冲区正确回绕"""
        mod = AxonalDelayModule(max_edges=4, max_delay=5, min_delay=1)
        
        # 推超过 max_delay 步
        for i in range(20):
            pre_spk = torch.zeros(4)
            pre_spk[0] = float(i % 3 == 0)
            mod.step_fast(pre_spk, torch.zeros(4))
        
        # 不应崩溃，buffer_ptr 应持续增长
        assert mod.buffer_ptr.item() == 20
    
    def test_discretize_clamps(self):
        """测试离散化延迟的裁剪"""
        mod = AxonalDelayModule(max_edges=4, max_delay=10, min_delay=2)

        # 设置超出范围的值
        mod.delay_continuous.data[0] = 0.0   # 低于 min
        mod.delay_continuous.data[1] = 100.0  # 高于 max
        mod._discretize_delays()

        assert mod.delay_discrete[0].item() >= 2
        assert mod.delay_discrete[1].item() <= 9

    def test_batched_forward_shape(self):
        """① pre_spk (4, E) 前向形状: 延迟输出保留 batch 维"""
        mod = AxonalDelayModule(max_edges=16, max_delay=20, min_delay=1)
        pre_spk = (torch.rand(4, 16) > 0.5).float()
        post_spk = torch.zeros(16)

        delayed_spk, stdp_delta = mod.step_fast(pre_spk, post_spk)

        assert delayed_spk.shape == (4, 16)
        assert stdp_delta.shape == (16,)

    def test_batch_switch_preserves_history(self):
        """② batch 2→4→2 切换不抛错且历史保留 (detach 修复回归防护)"""
        mod = AxonalDelayModule(max_edges=4, max_delay=5, min_delay=1)
        mod.delay_continuous.data.fill_(1.0)
        mod._discretize_delays()

        # 步 1 (batch=2): 写入样本 0 的脉冲到环形缓冲槽 0
        pre2 = torch.zeros(2, 4)
        pre2[0, 0] = 1.0
        mod.step_fast(pre2, torch.zeros(4))
        assert mod.spike_buffer.shape == (2, 5, 4)
        assert mod.spike_buffer[0, 0, 0].item() == 1.0

        # 步 2 (batch=4): 缓冲扩展, 旧行保留; 延迟 1 步读到槽 0
        pre4 = torch.zeros(4, 4)
        d4, _ = mod.step_fast(pre4, torch.zeros(4))
        assert mod.spike_buffer.shape == (4, 5, 4)
        assert d4[0, 0].item() == 1.0  # 历史随 batch 扩展保留

        # 步 3 (batch=2): 缓冲收缩, 保留前 2 行历史
        pre2b = torch.zeros(2, 4)
        d2, _ = mod.step_fast(pre2b, torch.zeros(4))
        assert mod.spike_buffer.shape == (2, 5, 4)
        assert mod.spike_buffer[0, 0, 0].item() == 1.0  # 槽 0 历史未被清除

    def test_backward_across_batch_switch(self):
        """③ 带梯度输入跨 batch backward 不崩 (缓冲 detach 回归防护)"""
        torch.manual_seed(0)
        mod = AxonalDelayModule(max_edges=8, max_delay=4, min_delay=1)
        mod.delay_continuous.data.fill_(1.0)
        mod._discretize_delays()

        def ste_step(batch):
            """模拟皮层 STE 直通: 延迟读值无梯度, 外部以当前聚合值直通"""
            x = torch.randn(batch, 8, requires_grad=True)
            d, _ = mod.step_fast(x, torch.zeros(8))
            return x, d + (x - x.detach())

        x1, d1 = ste_step(2)
        x4, d4 = ste_step(4)   # batch 扩展
        x2, d2 = ste_step(2)   # batch 收缩

        # 修复前: 缓冲经 CopySlices 链入计算图 → backward 崩溃
        (d1.sum() + d4.sum() + d2.sum()).backward()
        assert x1.grad is not None
        assert x4.grad is not None
        assert x2.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
