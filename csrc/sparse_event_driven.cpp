#include <torch/extension.h>
#include <vector>

// 事件驱动的稀疏矩阵乘法 (CPU)
// 仅计算发放脉冲的前突触权重
// spk: (batch_size, in_features) 或者是 (in_features,) 的 boolean/float 脉冲张量
// weight: (out_features, in_features)
torch::Tensor sparse_event_driven_matmul_cpu(torch::Tensor spk, torch::Tensor weight) {
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    
    if (spk.dim() == 1) {
        // (in_features,) 向量情况
        auto active_indices = torch::nonzero(spk).squeeze(-1);
        if (active_indices.numel() == 0) {
            return torch::zeros({weight.size(0)}, weight.options());
        }
        
        // 提取被激活的列并求和
        auto selected_weights = weight.index_select(1, active_indices);
        return selected_weights.sum(/*dim=*/1);
    } else if (spk.dim() == 2) {
        // (batch_size, in_features) 批量情况
        int64_t batch_size = spk.size(0);
        int64_t out_features = weight.size(0);
        
        auto out = torch::zeros({batch_size, out_features}, weight.options());
        
        for (int64_t b = 0; b < batch_size; ++b) {
            auto b_spk = spk[b];
            auto active_indices = torch::nonzero(b_spk).squeeze(-1);
            if (active_indices.numel() > 0) {
                auto selected_weights = weight.index_select(1, active_indices);
                out[b] = selected_weights.sum(/*dim=*/1);
            }
        }
        return out;
    } else {
        TORCH_CHECK(false, "spk must be a 1D or 2D tensor");
        return torch::Tensor();
    }
}

// 基于掩码的超边更新
// hyperedge_index: (2, num_connections)
// edge_mask: (num_edges,) boolean/uint8 tensor
torch::Tensor masked_hyperedge_update_cpu(torch::Tensor hyperedge_index, torch::Tensor edge_mask) {
    TORCH_CHECK(hyperedge_index.dim() == 2 && hyperedge_index.size(0) == 2, 
                "hyperedge_index must be of shape (2, N)");
    TORCH_CHECK(edge_mask.dim() == 1, "edge_mask must be a 1D tensor");

    // 如果全部存活，直接返回
    if (edge_mask.all().item<bool>()) {
        return hyperedge_index;
    }

    // 找到存活的 edge_ids
    auto alive_edge_ids = torch::nonzero(edge_mask).squeeze(-1);

    // 检查 hyperedge_index 的第二行 (edge indices) 是否在 alive_edge_ids 中
    auto edge_indices = hyperedge_index[1];
    
    // torch::isin 需要 PyTorch 1.10+
    auto valid_connections_mask = torch::isin(edge_indices, alive_edge_ids);
    
    // 使用 mask 进行过滤
    auto valid_indices = torch::nonzero(valid_connections_mask).squeeze(-1);
    
    return hyperedge_index.index_select(1, valid_indices);
}

// 绑定到 PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_event_driven_matmul", &sparse_event_driven_matmul_cpu, 
          "Sparse Event-Driven Matrix Multiplication (CPU)");
    m.def("masked_hyperedge_update", &masked_hyperedge_update_cpu, 
          "Masked Hyperedge Update (CPU)");
}
