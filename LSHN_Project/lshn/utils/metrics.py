import torch
from typing import Dict, List, Optional

class ContinualLearningMetrics:
    """
    持续学习指标评估模块
    计算灾难性遗忘、前向迁移、激活稀疏度等核心指标。
    """
    def __init__(self, num_tasks: int, device: Optional[torch.device] = None):
        self.num_tasks = num_tasks
        self.device = device or torch.device('cpu')
        self.R = torch.zeros((num_tasks, num_tasks), device=self.device)
        self.spike_counts: List[float] = []
        
    def update_accuracy(self, trained_task_idx: int, eval_task_idx: int, acc: float):
        self.R[trained_task_idx, eval_task_idx] = acc
        
    def record_spike_sparsity(self, spk: torch.Tensor):
        sparsity = spk.float().mean().item()
        self.spike_counts.append(sparsity)

    def average_accuracy(self, current_task_idx: int) -> float:
        return self.R[current_task_idx, :current_task_idx+1].mean().item()

    def forgetting_measure(self, current_task_idx: int) -> float:
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
    
    def reset(self):
        """重置所有指标"""
        self.R.zero_()
        self.spike_counts.clear()
