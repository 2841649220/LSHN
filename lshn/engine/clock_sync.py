# 全局时钟契约 Constants (模块级默认值; 可通过 ClockSyncEngine 构造参数覆盖)
T_FAST_MS = 1        # 快时间步 (膜电位, 脉冲发放, 快门控, 快权重)
T_SLOW_MS = 100      # 慢时间步 (慢门控, 适应变量, 双势阱结构概率 s_e, 能量控制器)
T_ULTRA_MS = 1000    # 超慢时间步 (因果贡献度计算, 神经元生发/凋亡, 海马体回放)

class ClockSyncEngine:
    """
    多时间尺度时钟同步器
    统筹 LSHN 中不同机制的更新频率，严格分离快、慢、超慢变量。
    """
    def __init__(self, fast_ms: int = 1, slow_ms: int = 100, ultra_slow_ms: int = 1000):
        # 时间尺度 (毫秒): 默认与模块级常量一致, 可通过构造参数覆盖
        # (例如从配置读取自定义周期)
        self.fast_ms = fast_ms
        self.slow_ms = slow_ms
        self.ultra_slow_ms = ultra_slow_ms

        self.fast_steps = 0
        self.slow_steps = 0
        self.ultra_slow_steps = 0
        # 自上次慢触发以来的快步数 (供调用方判断慢窗口内推进了多久,
        # 触发慢时钟时归零)
        self.steps_since_slow = 0

    @property
    def steps_per_slow(self) -> int:
        """每个慢周期包含的快步数 (只读)"""
        return self.slow_ms // self.fast_ms

    def tick(self):
        """
        前进一步快时钟 (1ms)。并返回是否触发慢时钟和超慢时钟。
        """
        self.fast_steps += 1
        self.steps_since_slow += 1

        trigger_slow = (self.fast_steps % self.steps_per_slow) == 0
        trigger_ultra_slow = (self.fast_steps % (self.ultra_slow_ms // self.fast_ms)) == 0

        if trigger_slow:
            self.slow_steps += 1
            self.steps_since_slow = 0
        if trigger_ultra_slow:
            self.ultra_slow_steps += 1

        return trigger_slow, trigger_ultra_slow

    def reset(self):
        self.fast_steps = 0
        self.slow_steps = 0
        self.ultra_slow_steps = 0
        self.steps_since_slow = 0
