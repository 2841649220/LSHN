import torch
import warnings

# 尝试导入编译好的 C++ 扩展，如果失败则回退到 Python 实现
try:
    import lshn_csrc
    CSRC_AVAILABLE = True
except ImportError:
    CSRC_AVAILABLE = False
    warnings.warn("lshn_csrc C++ extension is not installed. Falling back to slow Python implementation. Run 'pip install -e .' in the project root to install.")

def sparse_event_driven_matmul(spk: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    事件驱动稀疏矩阵乘法。
    仅当神经元发放脉冲 (spk=1) 时，才提取相应的权重进行计算，
    从而大幅降低计算开销，适用于 SNN 场景。
    
    参数:
    spk: (batch_size, in_features) 或 (in_features,) 脉冲张量，元素为 0 或 1
    weight: (out_features, in_features) 权重矩阵
    
    返回:
    out: (batch_size, out_features) 或 (out_features,) 结果
    """
    if CSRC_AVAILABLE:
        # 确保输入在 CPU 上（因为目前的 C++ 实现只支持 CPU），或者在 C++ 端增加 CUDA 支持
        # 注意: 这里的 csrc 默认 CPU。如果需要 CUDA 支持，C++ 端需要重写。
        if spk.device.type == 'cpu' and weight.device.type == 'cpu':
            return lshn_csrc.sparse_event_driven_matmul(spk, weight)
        
    # PyTorch 纯 Python 实现
    if spk.dim() == 1:
        active_indices = torch.nonzero(spk).squeeze(-1)
        if active_indices.numel() == 0:
            return torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
        return weight[:, active_indices].sum(dim=1)
    else:
        # 向量化实现: spk @ weight.T, 利用脉冲的二值性加速
        # 对于稀疏脉冲, 先收集活跃索引再聚合比 dense matmul 更高效
        active_mask = spk.bool()
        # 计算每行的活跃数, 判断是否值得用稀疏路径
        nnz_per_row = active_mask.sum(dim=1)
        avg_nnz = nnz_per_row.float().mean().item()
        total_cols = spk.shape[1]
        
        if avg_nnz < total_cols * 0.3:
            # 稀疏路径: 逐行收集活跃索引
            out = torch.zeros(spk.shape[0], weight.shape[0], device=weight.device, dtype=weight.dtype)
            for b in range(spk.shape[0]):
                active_indices = active_mask[b].nonzero(as_tuple=True)[0]
                if active_indices.numel() > 0:
                    out[b] = weight[:, active_indices].sum(dim=1)
            return out
        else:
            # 密集路径: 直接矩阵乘法
            return spk @ weight.T

def masked_hyperedge_update(hyperedge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
    """
    根据超边的存活掩码，更新拓扑结构，返回存活的 hyperedge_index。
    """
    if CSRC_AVAILABLE:
        if hyperedge_index.device.type == 'cpu' and edge_mask.device.type == 'cpu':
            return lshn_csrc.masked_hyperedge_update(hyperedge_index, edge_mask)
            
    if edge_mask.all():
        return hyperedge_index
        
    alive_edge_ids = torch.nonzero(edge_mask).squeeze(-1)
    valid_connections = torch.isin(hyperedge_index[1], alive_edge_ids)
    
    return hyperedge_index[:, valid_connections]
