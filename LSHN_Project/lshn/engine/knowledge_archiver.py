"""
知识归档器 (Knowledge Archiver)
白皮书 §4.2.1 — 三层记忆存储架构 Tier-3 (NVMe 冷存档)

冷知识 INT4 压缩方案:
  - w_hat ∈ [-1, 1]:  NF4 非线性分块量化 (group_size=64) + BF16 scale
  - s_e   ∈ [ 0, 1]:  线性 INT4 分块量化  (group_size=64) + BF16 scale + BF16 zero
  - 两个 INT4 值 bit-pack 进一个 uint8 字节 (高4位/低4位)
  - hyperedge_index:  COO → CSR (int32)
  - 资格迹 (e_trace, pre_trace, post_trace): 丢弃 (冷边已衰减至 0)
  - 序列化: torch.save + JSON 索引

精度约定:
  - 训练状态 (w_hat, s_e 活跃部分): 始终 FP32 / BF16
  - 归档 (冷边):   INT4 bit-packed uint8，仅在 retrieve 时解压回 FP32
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# NF4 量化表 (16 级, 基于正态分布分位数)
# 覆盖 [-1, 1], 对称分布尾部有更密集的级别
# ─────────────────────────────────────────────────────────────
_NF4_TABLE = torch.tensor([
    -1.0,       -0.6961928, -0.5250730, -0.3949301,
    -0.2844677, -0.1847513, -0.0917715,  0.0,
     0.0797546,  0.1609459,  0.2461693,  0.3379146,
     0.4407282,  0.5626170,  0.7229568,  1.0,
], dtype=torch.float32)  # shape: (16,)


# ─────────────────────────────────────────────────────────────
# 低层工具函数
# ─────────────────────────────────────────────────────────────

def _nf4_quantize(x: torch.Tensor, group_size: int = 64
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NF4 非线性分块量化 (w_hat ∈ [-1, 1])

    每 group_size 个元素共享一个 BF16 scale (最大绝对值).
    量化级别: 0-15 (4-bit), 返回 int8 存储 (值域 0-15).

    Args:
        x:          (N,) FP32 张量, 值域应 ∈ [-1, 1]
        group_size: 分组大小

    Returns:
        codes:  (N,) torch.int8,  值 ∈ [0, 15]
        scales: (num_groups,) torch.bfloat16, 每组的最大绝对值
    """
    x = x.float()
    N = x.numel()
    num_groups = math.ceil(N / group_size)

    # 填充到 group_size 的整数倍
    pad = num_groups * group_size - N
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad))

    x_groups = x.view(num_groups, group_size)  # (G, gs)

    # 每组 scale = max(|x|)，防止除零
    scales = x_groups.abs().max(dim=1).values.clamp(min=1e-8)  # (G,)

    # 归一化到 [-1, 1]
    x_norm = x_groups / scales.unsqueeze(1)  # (G, gs)

    # 查找最近 NF4 级别
    nf4 = _NF4_TABLE.to(x.device)  # (16,)
    # 广播: (G, gs, 1) vs (1, 1, 16) → (G, gs, 16)
    diff = (x_norm.unsqueeze(-1) - nf4.view(1, 1, 16)).abs()
    codes = diff.argmin(dim=-1).to(torch.int8)  # (G, gs), values ∈ [0, 15]

    # 去掉 pad
    codes_flat = codes.view(-1)[:N]
    return codes_flat, scales.to(torch.bfloat16)


def _nf4_dequantize(codes: torch.Tensor, scales: torch.Tensor,
                    group_size: int = 64, original_N: int = -1) -> torch.Tensor:
    """
    NF4 反量化 → FP32

    Args:
        codes:      (N,) torch.int8,  值 ∈ [0, 15]
        scales:     (num_groups,) torch.bfloat16
        group_size: 分组大小
        original_N: 原始元素数 (去除 pad)

    Returns:
        x: (original_N,) torch.float32
    """
    nf4 = _NF4_TABLE.to(codes.device)
    N = codes.numel()
    num_groups = math.ceil(N / group_size)

    # 填充到整数倍
    pad = num_groups * group_size - N
    if pad > 0:
        codes = torch.nn.functional.pad(codes.long(), (0, pad)).to(torch.int8)

    codes_long = codes.long().view(num_groups, group_size)  # (G, gs)
    x_norm = nf4[codes_long]  # (G, gs)

    scales_f32 = scales.to(torch.float32).unsqueeze(1)  # (G, 1)
    x = (x_norm * scales_f32).view(-1)  # (G*gs,)

    if original_N > 0:
        x = x[:original_N]
    return x


def _int4_linear_quantize(x: torch.Tensor, group_size: int = 64
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    线性 INT4 分块量化 (s_e ∈ [0, 1])

    公式: x ≈ scale * q + zero,  q ∈ [0, 15]
    每 group_size 个元素共享 BF16 scale + BF16 zero-point.

    Args:
        x:          (N,) FP32, 值域 [0, 1]
        group_size: 分组大小

    Returns:
        codes:  (N,) torch.int8,  值 ∈ [0, 15]
        scales: (num_groups,) torch.bfloat16
        zeros:  (num_groups,) torch.bfloat16
    """
    x = x.float()
    N = x.numel()
    num_groups = math.ceil(N / group_size)
    pad = num_groups * group_size - N
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad))

    x_groups = x.view(num_groups, group_size)  # (G, gs)

    x_min = x_groups.min(dim=1).values   # (G,)
    x_max = x_groups.max(dim=1).values   # (G,)

    # scale = (max - min) / 15, zero = min
    scales = ((x_max - x_min) / 15.0).clamp(min=1e-8)   # (G,)
    zeros = x_min                                          # (G,)

    # 量化: q = round((x - zero) / scale), 截断至 [0, 15]
    q = ((x_groups - zeros.unsqueeze(1)) / scales.unsqueeze(1)).round().long()
    q = q.clamp(0, 15).to(torch.int8)  # (G, gs)

    codes_flat = q.view(-1)[:N]
    return codes_flat, scales.to(torch.bfloat16), zeros.to(torch.bfloat16)


def _int4_linear_dequantize(codes: torch.Tensor, scales: torch.Tensor,
                              zeros: torch.Tensor, group_size: int = 64,
                              original_N: int = -1) -> torch.Tensor:
    """
    线性 INT4 反量化 → FP32

    Returns:
        x: (original_N,) torch.float32
    """
    N = codes.numel()
    num_groups = math.ceil(N / group_size)
    pad = num_groups * group_size - N
    if pad > 0:
        codes = torch.nn.functional.pad(codes.long(), (0, pad)).to(torch.int8)

    q = codes.long().view(num_groups, group_size).float()   # (G, gs)
    scales_f32 = scales.to(torch.float32).unsqueeze(1)      # (G, 1)
    zeros_f32 = zeros.to(torch.float32).unsqueeze(1)        # (G, 1)

    x = (q * scales_f32 + zeros_f32).view(-1)
    if original_N > 0:
        x = x[:original_N]
    return x.clamp(0.0, 1.0)


def _pack_int4_to_uint8(codes: torch.Tensor) -> torch.Tensor:
    """
    将两个 INT4 值 (值域 [0, 15]) bit-pack 进一个 uint8 字节.
    高 4 位 = 偶数索引元素, 低 4 位 = 奇数索引元素.
    如果元素总数为奇数, 末尾填充 0.

    Args:
        codes: (N,) torch.int8, 值 ∈ [0, 15]

    Returns:
        packed: (ceil(N/2),) torch.uint8
    """
    codes_long = codes.long()
    N = codes_long.numel()
    if N % 2 != 0:
        codes_long = torch.nn.functional.pad(codes_long, (0, 1))
    hi = codes_long[0::2] & 0xF   # 偶数 → 高 4 位
    lo = codes_long[1::2] & 0xF   # 奇数 → 低 4 位
    packed = ((hi << 4) | lo).to(torch.uint8)
    return packed


def _unpack_uint8_to_int4(packed: torch.Tensor, original_N: int) -> torch.Tensor:
    """
    解包 uint8 → int8 (值 ∈ [0, 15])

    Args:
        packed:     (ceil(N/2),) torch.uint8
        original_N: 原始元素数

    Returns:
        codes: (original_N,) torch.int8
    """
    packed_long = packed.long()
    hi = (packed_long >> 4) & 0xF   # 偶数
    lo = packed_long & 0xF           # 奇数

    # 交织恢复
    codes = torch.zeros(packed.numel() * 2, dtype=torch.long, device=packed.device)
    codes[0::2] = hi
    codes[1::2] = lo
    return codes[:original_N].to(torch.int8)


def _coo_to_csr(hyperedge_index: torch.Tensor, num_nodes: int
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    COO 超图索引 → CSR 格式

    Args:
        hyperedge_index: (2, E) int64, [node_ids; edge_ids]
        num_nodes:       节点数

    Returns:
        indptr:  (num_nodes+1,) int32, 每个节点在 indices 中的起始位置
        indices: (E,) int32, 按节点排序的超边 id
    """
    node_ids = hyperedge_index[0].to(torch.int32)
    edge_ids = hyperedge_index[1].to(torch.int32)
    E = node_ids.numel()

    # 按 node_id 排序
    sort_order = node_ids.argsort()
    node_ids_sorted = node_ids[sort_order]
    edge_ids_sorted = edge_ids[sort_order]

    # 构建 indptr
    counts = torch.zeros(num_nodes, dtype=torch.int32, device=hyperedge_index.device)
    counts.scatter_add_(0, node_ids_sorted.long(), torch.ones(E, dtype=torch.int32,
                                                               device=hyperedge_index.device))
    indptr = torch.zeros(num_nodes + 1, dtype=torch.int32, device=hyperedge_index.device)
    indptr[1:] = counts.cumsum(0)

    return indptr, edge_ids_sorted


def _csr_to_coo(indptr: torch.Tensor, indices: torch.Tensor
                 ) -> torch.Tensor:
    """
    CSR → COO hyperedge_index (2, E)
    """
    num_nodes = indptr.numel() - 1
    node_ids = torch.repeat_interleave(
        torch.arange(num_nodes, dtype=torch.int32, device=indptr.device),
        (indptr[1:] - indptr[:-1]).long()
    )
    return torch.stack([node_ids.long(), indices.long()], dim=0)


# ─────────────────────────────────────────────────────────────
# KnowledgeArchiver
# ─────────────────────────────────────────────────────────────

class KnowledgeArchiver:
    """
    冷知识 INT4 归档器

    存储格式 (每次调用 archive_cold_edges() 产生一个 .pt 文件):
    {
      "w_hat_packed":   (ceil(N_cold/2),)   uint8   — NF4 bit-packed
      "w_hat_scales":   (num_groups_w,)     bfloat16
      "se_packed":      (ceil(N_cold/2),)   uint8   — 线性INT4 bit-packed
      "se_scales":      (num_groups_se,)    bfloat16
      "se_zeros":       (num_groups_se,)    bfloat16
      "csr_indptr":     (num_nodes+1,)      int32
      "csr_indices":    (N_cold,)           int32
      "cold_indices":   (N_cold,)           int64   — 原始 edge slot 索引
      "num_nodes":      int
      "N_cold":         int
      "group_size":     int
      "timestamp":      float
      "archive_id":     str
    }

    JSON 索引文件 (archive_dir/index.json) 记录所有归档条目.

    使用方法:
        archiver = KnowledgeArchiver(archive_dir="./cold_archive")
        archiver.archive_cold_edges(
            cold_indices, w_hat_cold, s_e_cold,
            hyperedge_index_cold, num_nodes=1000
        )
        result = archiver.retrieve_archived_edges(archive_id)
    """

    def __init__(self, archive_dir: str = "./cold_archive", group_size: int = 64):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.group_size = group_size
        self._index_path = self.archive_dir / "index.json"
        self._index: List[Dict] = self._load_index()

    # ────────────────── 索引管理 ──────────────────

    def _load_index(self) -> List[Dict]:
        if self._index_path.exists():
            try:
                temp_path = self._index_path.with_suffix('.json.tmp')
                if temp_path.exists():
                    return []
                with open(self._index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_index(self):
        temp_path = self._index_path.with_suffix('.json.tmp')
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self._index_path)
        except IOError:
            if temp_path.exists():
                os.remove(temp_path)
            raise

    # ────────────────── 归档 ──────────────────

    def archive_cold_edges(
        self,
        cold_indices: torch.Tensor,
        w_hat_cold: torch.Tensor,
        s_e_cold: torch.Tensor,
        hyperedge_index_cold: torch.Tensor,
        num_nodes: int,
    ) -> str:
        """
        将冷超边压缩归档到 NVMe 文件.

        Args:
            cold_indices:         (N_cold,) int64 — 在 w_hat / s_e 中的 slot 索引
            w_hat_cold:           (N_cold,) float32 — 冷边快权重
            s_e_cold:             (N_cold,) float32 — 冷边结构变量
            hyperedge_index_cold: (2, N_cold) int64 — 冷边的 COO 拓扑
            num_nodes:            int — 总节点数

        Returns:
            archive_id: str — 归档 ID (文件名 stem)
        """
        N_cold = cold_indices.numel()
        if N_cold == 0:
            return ""

        # 强制 CPU + FP32，避免 autocast 污染量化逻辑
        w_hat_cpu = w_hat_cold.detach().cpu().float()
        s_e_cpu = s_e_cold.detach().cpu().float()
        cidx_cpu = cold_indices.detach().cpu()
        heidx_cpu = hyperedge_index_cold.detach().cpu()

        # 1. NF4 量化 w_hat
        w_codes, w_scales = _nf4_quantize(w_hat_cpu, self.group_size)
        w_packed = _pack_int4_to_uint8(w_codes)

        # 2. 线性 INT4 量化 s_e
        se_codes, se_scales, se_zeros = _int4_linear_quantize(s_e_cpu, self.group_size)
        se_packed = _pack_int4_to_uint8(se_codes)

        # 3. COO → CSR 拓扑
        csr_indptr, csr_indices = _coo_to_csr(heidx_cpu, num_nodes)

        # 4. 生成归档 ID + 路径
        archive_id = f"cold_{int(time.time() * 1000)}_{N_cold}edges"
        file_path = self.archive_dir / f"{archive_id}.pt"

        # 5. 保存 (原子写入)
        payload = {
            "w_hat_packed": w_packed,
            "w_hat_scales": w_scales,
            "se_packed": se_packed,
            "se_scales": se_scales,
            "se_zeros": se_zeros,
            "csr_indptr": csr_indptr,
            "csr_indices": csr_indices,
            "cold_indices": cidx_cpu,
            "num_nodes": num_nodes,
            "N_cold": N_cold,
            "group_size": self.group_size,
            "timestamp": time.time(),
            "archive_id": archive_id,
        }
        temp_path = file_path.with_suffix('.pt.tmp')
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, file_path)
        except IOError:
            if temp_path.exists():
                os.remove(temp_path)
            raise

        # 6. 更新 JSON 索引
        entry = {
            "archive_id": archive_id,
            "file": str(file_path),
            "N_cold": N_cold,
            "num_nodes": num_nodes,
            "timestamp": payload["timestamp"],
        }
        self._index.append(entry)
        self._save_index()

        return archive_id

    # ────────────────── 检索 ──────────────────

    def retrieve_archived_edges(
        self, archive_id: str, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        从归档文件解压冷超边, 恢复为 FP32.

        Args:
            archive_id: str — 归档 ID
            device:     目标设备 (默认 CPU)

        Returns:
            dict:
              "cold_indices":   (N_cold,) int64
              "w_hat":          (N_cold,) float32  (解压)
              "s_e":            (N_cold,) float32  (解压)
              "hyperedge_index":(2, N_cold) int64   (COO)
              "num_nodes":      int
        """
        # 定位文件
        file_path = self.archive_dir / f"{archive_id}.pt"
        if not file_path.exists():
            raise FileNotFoundError(f"Archive not found: {file_path}")

        payload = torch.load(file_path, map_location="cpu", weights_only=True)

        N_cold = payload["N_cold"]
        gs = payload["group_size"]

        # 解压 w_hat (NF4)
        w_codes = _unpack_uint8_to_int4(payload["w_hat_packed"], N_cold)
        w_hat = _nf4_dequantize(w_codes, payload["w_hat_scales"], gs, N_cold)

        # 解压 s_e (线性INT4)
        se_codes = _unpack_uint8_to_int4(payload["se_packed"], N_cold)
        s_e = _int4_linear_dequantize(
            se_codes, payload["se_scales"], payload["se_zeros"], gs, N_cold
        )

        # CSR → COO
        hyperedge_index = _csr_to_coo(payload["csr_indptr"], payload["csr_indices"])

        result = {
            "cold_indices": payload["cold_indices"],
            "w_hat": w_hat,
            "s_e": s_e,
            "hyperedge_index": hyperedge_index,
            "num_nodes": payload["num_nodes"],
        }

        if device is not None:
            result = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                      for k, v in result.items()}

        return result

    # ────────────────── 工具接口 ──────────────────

    def list_archives(self) -> List[Dict]:
        """返回所有归档条目 (JSON 索引)"""
        return list(self._index)

    def total_cold_edges(self) -> int:
        """返回历史累计归档的冷超边总数"""
        return sum(e.get("N_cold", 0) for e in self._index)

    def delete_archive(self, archive_id: str) -> bool:
        """删除指定归档文件并从索引中移除"""
        file_path = self.archive_dir / f"{archive_id}.pt"
        removed = False
        if file_path.exists():
            os.remove(file_path)
            removed = True
        self._index = [e for e in self._index if e["archive_id"] != archive_id]
        self._save_index()
        return removed
