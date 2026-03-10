"""
事件驱动稀疏矩阵乘法与超图拓扑更新核心算子
白皮书 §4.4:
  基于事件驱动仿真，仅当神经元发放脉冲时才提取相应权重进行计算。

优化:
  - 批量模式使用 torch.vmap 向量化，消除 Python for-loop
  - 保留 C++ 扩展路径 (lshn_csrc) 优先使用
  - 单样本路径使用稀疏索引提取，零开销
"""

import torch
import warnings
from typing import Optional

# 尝试导入编译好的 C++ 扩展，如果失败则回退到 Python 实现
CSRC_AVAILABLE = False
try:
    import lshn_csrc
    CSRC_AVAILABLE = True
except ImportError:
    pass
except Exception as e:
    warnings.warn(
        f"lshn_csrc C++ extension failed to load: {e}. "
        "Falling back to Python implementation.",
        stacklevel=2,
    )

if not CSRC_AVAILABLE:
    warnings.warn(
        "lshn_csrc C++ extension is not installed. "
        "Falling back to Python implementation. "
        "Run 'pip install -e .' in the project root to install.",
        stacklevel=2,
    )

# ------------------------------------------------------------------
#  检测 torch.vmap 可用性 (PyTorch ≥ 2.0 提供 torch.vmap)
# ------------------------------------------------------------------
_VMAP_AVAILABLE = hasattr(torch, "vmap")


# ==================================================================
#  单样本事件驱动稀疏矩阵乘法 (vmap 的被映射函数)
# ==================================================================

def _single_sparse_matmul(spk: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    单样本事件驱动稀疏矩阵乘法。
    spk:    (in_features,) 脉冲张量 {0, 1}
    weight: (out_features, in_features) 权重矩阵
    返回:   (out_features,)
    """
    active_indices = torch.nonzero(spk, as_tuple=False).squeeze(-1)
    if active_indices.numel() == 0:
        return torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    return weight[:, active_indices].sum(dim=1)


# ==================================================================
#  vmap 批量版本 (PyTorch ≥ 2.0)
# ==================================================================

def _batched_sparse_matmul_vmap(spk: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    使用 torch.vmap 向量化批量事件驱动稀疏矩阵乘法。

    spk:    (batch_size, in_features)
    weight: (out_features, in_features)
    返回:   (batch_size, out_features)

    注意: vmap 要求被映射函数的所有控制流对每个样本一致。
    由于不同样本的 active_indices 长度不同，直接 vmap _single_sparse_matmul
    无法工作。因此采用"掩码广播乘法"策略:
      out[b] = (spk[b].unsqueeze(0) * weight).sum(dim=1)
    这在 vmap 下编译为高效的批量掩码矩阵乘法。
    """
    def _masked_matmul(s: torch.Tensor) -> torch.Tensor:
        # s: (in_features,), weight: (out_features, in_features)
        # 逐元素乘法 + 求和 = 仅对 s=1 的列累加
        return (s.unsqueeze(0) * weight).sum(dim=1)

    return torch.vmap(_masked_matmul)(spk)


# ==================================================================
#  纯 PyTorch fallback (无 vmap 时的向量化实现)
# ==================================================================

def _batched_sparse_matmul_fallback(spk: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    不使用 vmap 的向量化批量版本。
    利用 PyTorch 广播: out = (spk @ weight.T) 等效于事件驱动乘法。
    由于 spk 是 {0,1} 二值张量，spk @ weight.T 天然只累加脉冲列。

    spk:    (batch_size, in_features)
    weight: (out_features, in_features)
    返回:   (batch_size, out_features)
    """
    # spk: (B, in) @ weight.T: (in, out) → (B, out)
    # 对于二值脉冲张量，这等效于仅累加 active 列
    return spk @ weight.t()


# ==================================================================
#  公共 API
# ==================================================================

def sparse_event_driven_matmul(spk: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    事件驱动稀疏矩阵乘法。
    仅当神经元发放脉冲 (spk=1) 时，才提取相应的权重进行计算，
    从而大幅降低计算开销，适用于 SNN 场景。

    参数:
        spk:    (batch_size, in_features) 或 (in_features,) 脉冲张量，元素为 0 或 1
        weight: (out_features, in_features) 权重矩阵

    返回:
        out: (batch_size, out_features) 或 (out_features,) 结果
    """
    # --- C++ 扩展优先 (CPU only) ---
    if CSRC_AVAILABLE:
        if spk.device.type == 'cpu' and weight.device.type == 'cpu':
            return lshn_csrc.sparse_event_driven_matmul(spk, weight)

    # --- 单样本 ---
    if spk.dim() == 1:
        return _single_sparse_matmul(spk, weight)

    # --- 批量: vmap > fallback ---
    if _VMAP_AVAILABLE:
        return _batched_sparse_matmul_vmap(spk, weight)
    else:
        return _batched_sparse_matmul_fallback(spk, weight)


def masked_hyperedge_update(hyperedge_index: torch.Tensor,
                            edge_mask: torch.Tensor) -> torch.Tensor:
    """
    根据超边的存活掩码，更新拓扑结构，返回存活的 hyperedge_index。

    参数:
        hyperedge_index: (2, num_connections) COO 格式超图拓扑
        edge_mask:       (max_edges,) bool 存活掩码

    返回:
        filtered_index: (2, num_surviving_connections) 过滤后的拓扑
    """
    if CSRC_AVAILABLE:
        if hyperedge_index.device.type == 'cpu' and edge_mask.device.type == 'cpu':
            return lshn_csrc.masked_hyperedge_update(hyperedge_index, edge_mask)

    if edge_mask.all():
        return hyperedge_index

    alive_edge_ids = torch.nonzero(edge_mask, as_tuple=False).squeeze(-1)
    valid_connections = torch.isin(hyperedge_index[1], alive_edge_ids)

    return hyperedge_index[:, valid_connections]
