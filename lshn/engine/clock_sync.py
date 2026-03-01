import time

# 全局时钟契约 Constants
T_FAST_MS = 1        # 快时间步 (膜电位, 脉冲发放, 快门控, 快权重)
T_SLOW_MS = 100      # 慢时间步 (慢门控, 适应变量, 双势阱结构概率 s_e, 能量控制器)
T_ULTRA_MS = 1000    # 超慢时间步 (因果贡献度计算, 神经元生发/凋亡, 海马体回放)

class ClockSyncEngine:
    """
    多时间尺度时钟同步器
    统筹 LSHN 中不同机制的更新频率，严格分离快、慢、超慢变量。
    """
    def __init__(self):
        self.fast_steps = 0
        self.slow_steps = 0
        self.ultra_slow_steps = 0

    def tick(self):
        """
        前进一步快时钟 (1ms)。并返回是否触发慢时钟和超慢时钟。
        """
        self.fast_steps += 1
        
        trigger_slow = (self.fast_steps % (T_SLOW_MS // T_FAST_MS)) == 0
        trigger_ultra_slow = (self.fast_steps % (T_ULTRA_MS // T_FAST_MS)) == 0
        
        if trigger_slow:
            self.slow_steps += 1
        if trigger_ultra_slow:
            self.ultra_slow_steps += 1
            
        return trigger_slow, trigger_ultra_slow
    
    def reset(self):
        self.fast_steps = 0
        self.slow_steps = 0
        self.ultra_slow_steps = 0
