import torch
from typing import Dict, List

class ContinualLearningMetrics:
    """
    持续学习指标评估模块
    计算灾难性遗忘、前向迁移、激活稀疏度等核心指标。
    """
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        # R 矩阵: R[i, j] 表示在学习完任务 i 之后，在任务 j 上的准确率
        self.R = torch.zeros((num_tasks, num_tasks))
        
        self.spike_counts = []
        
    def update_accuracy(self, trained_task_idx: int, eval_task_idx: int, acc: float):
        assert 0 <= trained_task_idx < self.num_tasks, (
            f"trained_task_idx={trained_task_idx} 超出合法范围 [0, {self.num_tasks})"
        )
        assert 0 <= eval_task_idx < self.num_tasks, (
            f"eval_task_idx={eval_task_idx} 超出合法范围 [0, {self.num_tasks})"
        )
        self.R[trained_task_idx, eval_task_idx] = acc

    def record_spike_sparsity(self, spk: torch.Tensor):
        """
        记录一批脉冲张量的稀疏度。

        Args:
            spk: (batch_size, num_neurons) 脉冲张量, 元素为 0/1。
                 内部先计算发放密度 density = mean(spk),
                 再转换为稀疏度 sparsity = 1 - density 并记录。
        """
        # 输入为脉冲张量: 先算发放密度, 再转稀疏度
        density = spk.float().mean().item()
        sparsity = 1.0 - density
        self.spike_counts.append(sparsity)

    def record_sparsity_value(self, sparsity: float):
        """
        直接记录已计算好的稀疏度标量 (sparsity = 1 - 发放密度)。

        适用于调用方已通过其他途径 (如模型监控报告) 得到稀疏度、
        无需再传原始脉冲张量的场景。
        """
        self.spike_counts.append(float(sparsity))

    def average_accuracy(self, current_task_idx: int) -> float:
        """平均准确率"""
        return self.R[current_task_idx, :current_task_idx+1].mean().item()

    def forgetting_measure(self, current_task_idx: int) -> float:
        """平均遗忘率: F_k = \frac{1}{k-1} \sum_{j=1}^{k-1} \max_{l \in \{1,..,k-1\}} (R_{l,j} - R_{k,j})"""
        if current_task_idx == 0:
            return 0.0
            
        forgetting = 0.0
        for j in range(current_task_idx):
            max_past_acc = torch.max(self.R[:current_task_idx, j]).item()
            current_acc = self.R[current_task_idx, j].item()
            forgetting += (max_past_acc - current_acc)
            
        return forgetting / current_task_idx

    def get_average_sparsity(self) -> float:
        if not self.spike_counts:
            return 0.0
        return sum(self.spike_counts) / len(self.spike_counts)
        
    def report(self, current_task_idx: int) -> Dict[str, float]:
        return {
            "avg_accuracy": self.average_accuracy(current_task_idx),
            "forgetting": self.forgetting_measure(current_task_idx),
            "avg_sparsity": self.get_average_sparsity()
        }
