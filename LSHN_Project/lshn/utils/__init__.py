"""
lshn.utils — 工具函数 (稀疏核、持续学习指标)
"""

from lshn.utils.sparse_kernel import sparse_event_driven_matmul, masked_hyperedge_update
from lshn.utils.metrics import ContinualLearningMetrics

__all__ = [
    "sparse_event_driven_matmul",
    "masked_hyperedge_update",
    "ContinualLearningMetrics",
]
