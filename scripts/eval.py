"""
LSHN 评估脚本
=============
从检查点加载模型，在指定任务上执行评估，输出持续学习指标。

用法:
    python scripts/eval.py --checkpoint checkpoints/lshn_task4.pt
    python scripts/eval.py --checkpoint checkpoints/lshn_task4.pt --synthetic --verbose
"""
import os
import sys
import argparse
import logging
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from lshn.model import LSHNModel
from lshn.utils.metrics import ContinualLearningMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lshn.eval")


# ──────────────── 数据加载 (复用 train.py 逻辑) ────────────────

def _make_synthetic_tasks(num_tasks: int, classes_per_task: int, input_dim: int = 128):
    """合成数据任务序列"""
    samples_per_class = 200
    tasks = []
    for t in range(num_tasks):
        task_classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        xs, ys = [], []
        for c in task_classes:
            torch.manual_seed(c * 42)  # 保证可复现
            center = torch.randn(input_dim) * 2
            x = center.unsqueeze(0) + torch.randn(samples_per_class, input_dim) * 0.5
            y = torch.full((samples_per_class,), c, dtype=torch.long)
            xs.append(x)
            ys.append(y)
        X, Y = torch.cat(xs), torch.cat(ys)
        tasks.append({
            "test_loader": DataLoader(TensorDataset(X, Y), batch_size=128, shuffle=False),
            "classes": task_classes,
        })
    return tasks


def _make_split_mnist_test(data_dir: str, num_tasks: int, classes_per_task: int):
    """Split-MNIST 测试集"""
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        test_ds = datasets.MNIST(data_dir, train=False, download=False, transform=transform)
    except Exception as e:
        log.warning(f"无法加载 MNIST ({e})，使用合成数据")
        return _make_synthetic_tasks(num_tasks, classes_per_task)

    tasks = []
    for t in range(num_tasks):
        cls_start = t * classes_per_task
        task_classes = list(range(cls_start, cls_start + classes_per_task))
        idx = [i for i, (_, y) in enumerate(test_ds) if y in task_classes]
        tasks.append({
            "test_loader": DataLoader(Subset(test_ds, idx), batch_size=128, shuffle=False),
            "classes": task_classes,
        })
    return tasks


# ──────────────── 评估核心 ────────────────

@torch.no_grad()
def evaluate_single_task(
    model: LSHNModel,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    fast_steps: int = 5,
) -> dict:
    """
    评估单个任务，返回准确率、脉冲稀疏度、平均输出熵等。
    """
    model.eval()
    correct = 0
    total = 0
    total_spikes = 0
    total_neurons = 0
    output_entropies = []

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if x_batch.shape[-1] != model.input_dim:
            if hasattr(model, "_input_proj"):
                x_batch = model._input_proj(x_batch)
            else:
                x_batch = x_batch[:, :model.input_dim]

        # 多步推理
        result = None
        for _ in range(fast_steps):
            result = model.forward_step(x_batch)

        outputs = result["output"]
        spk_cortex = result["spk_cortex"]

        # 准确率
        _, predicted = outputs.max(1)
        y_clamped = y_batch.clamp(0, num_classes - 1)
        correct += predicted.eq(y_clamped).sum().item()
        total += y_batch.size(0)

        # 脉冲稀疏度
        total_spikes += spk_cortex.sum().item()
        total_neurons += spk_cortex.numel()

        # 输出熵 (softmax后的Shannon熵, 越低越自信)
        probs = torch.softmax(outputs, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        output_entropies.extend(entropy.cpu().tolist())

    model.reset()

    accuracy = 100.0 * correct / max(total, 1)
    sparsity = 1.0 - (total_spikes / max(total_neurons, 1))
    mean_entropy = sum(output_entropies) / max(len(output_entropies), 1)

    return {
        "accuracy": accuracy,
        "sparsity": sparsity,
        "mean_entropy": mean_entropy,
        "total_samples": total,
    }


def full_evaluation(
    model: LSHNModel,
    tasks: list,
    device: torch.device,
    num_classes: int,
    fast_steps: int = 5,
    verbose: bool = False,
) -> dict:
    """
    全量评估：对每个任务计算指标，汇总持续学习综合指标。
    """
    cl_metrics = ContinualLearningMetrics(num_tasks=len(tasks))
    task_results = []

    for t, task in enumerate(tasks):
        result = evaluate_single_task(
            model, task["test_loader"], device,
            num_classes=num_classes, fast_steps=fast_steps,
        )
        task_results.append(result)
        # trained_task_idx = 最后训练的任务, eval_task_idx = 当前评估的任务
        last_trained = len(tasks) - 1
        cl_metrics.update_accuracy(trained_task_idx=last_trained, eval_task_idx=t,
                                   acc=result["accuracy"] / 100.0)
        cl_metrics.record_spike_sparsity(torch.tensor(result["sparsity"]))

        if verbose:
            log.info(
                f"  任务 {t} (类别 {task['classes']}): "
                f"Acc={result['accuracy']:.1f}% | "
                f"Sparsity={result['sparsity']:.3f} | "
                f"Entropy={result['mean_entropy']:.3f}"
            )

    # 综合指标
    last_trained = len(tasks) - 1
    avg_acc = cl_metrics.average_accuracy(current_task_idx=last_trained)
    forgetting = cl_metrics.forgetting_measure(current_task_idx=last_trained)
    avg_sparsity = cl_metrics.get_average_sparsity()

    # LSHN 内部监控
    lshn_report = model.get_monitoring_report()

    summary = {
        "average_accuracy": avg_acc,
        "forgetting_measure": forgetting,
        "average_sparsity": avg_sparsity,
        "per_task": task_results,
        "lshn_report": lshn_report,
        "cl_report": cl_metrics.report(current_task_idx=last_trained),
    }
    return summary


# ──────────────── 主入口 ────────────────

def main():
    parser = argparse.ArgumentParser(description="LSHN 持续学习评估")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="检查点文件路径")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (默认从检查点读取)")
    parser.add_argument("--synthetic", action="store_true",
                        help="使用合成数据")
    parser.add_argument("--fast_steps", type=int, default=5,
                        help="推理时快时钟步数")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每个任务的详细结果")
    args = parser.parse_args()

    # 加载检查点
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log.error(f"检查点不存在: {ckpt_path}")
        sys.exit(1)

    log.info(f"加载检查点: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 配置
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    elif "config" in ckpt:
        cfg = ckpt["config"]
    else:
        config_path = _project_root / "configs" / "default.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"使用设备: {device}")

    # 恢复模型
    model_cfg = cfg["model"]
    num_classes = ckpt.get("num_classes", model_cfg.get("initial_classes", 2))

    model = LSHNModel(
        input_dim=model_cfg.get("input_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        num_neurons=model_cfg.get("num_neurons", 500000),
        num_groups=model_cfg.get("num_groups", 100),
        max_edges=model_cfg.get("max_edges", 50000),
        initial_classes=model_cfg.get("initial_classes", 2),
        enable_dendrites=model_cfg.get("enable_dendrites", False),
        enable_active_inference=model_cfg.get("enable_active_inference", False),
        target_spikes_per_step=cfg.get("budget", {}).get("target_spikes_per_step", 5000),
    )

    # 扩展头到检查点保存时的类别数
    current_classes = model_cfg.get("initial_classes", 2)
    if num_classes > current_classes:
        model.expand_classes(num_classes - current_classes)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    log.info(f"模型恢复完成 | 类别数: {num_classes} | 训练到任务: {ckpt.get('task_id', '?')}")

    # 数据
    cl_cfg = cfg.get("continual", {})
    num_tasks = cl_cfg.get("tasks", 5)
    classes_per_task = cl_cfg.get("classes_per_task", 2)

    if args.synthetic:
        tasks = _make_synthetic_tasks(num_tasks, classes_per_task,
                                       input_dim=model_cfg.get("input_dim", 128))
    else:
        data_dir = cfg.get("data", {}).get("data_dir", "./data")
        tasks = _make_split_mnist_test(str(_project_root / data_dir), num_tasks, classes_per_task)

    # 只评估到检查点所训练的任务数
    trained_tasks = ckpt.get("task_id", len(tasks) - 1) + 1
    tasks = tasks[:trained_tasks]
    log.info(f"评估 {len(tasks)} 个已训练任务")

    # 执行评估
    log.info(f"\n{'='*60}")
    log.info("开始评估")
    log.info(f"{'='*60}")

    summary = full_evaluation(
        model, tasks, device,
        num_classes=num_classes,
        fast_steps=args.fast_steps,
        verbose=args.verbose,
    )

    # 打印结果
    log.info(f"\n{'='*60}")
    log.info("评估结果汇总")
    log.info(f"{'='*60}")
    log.info(f"  平均准确率:     {summary['average_accuracy']:.4f}")
    log.info(f"  遗忘度:         {summary['forgetting_measure']:.4f}")
    log.info(f"  平均脉冲稀疏度: {summary['average_sparsity']:.4f}")

    log.info("\nLSHN 内部监控:")
    for k, v in summary["lshn_report"].items():
        log.info(f"  {k}: {v:.6f}")

    log.info("\n持续学习指标:")
    for k, v in summary["cl_report"].items():
        log.info(f"  {k}: {v}")

    # 保存评估结果
    eval_out = ckpt_path.parent / f"eval_{ckpt_path.stem}.yaml"
    eval_data = {
        "average_accuracy": float(summary["average_accuracy"]),
        "forgetting_measure": float(summary["forgetting_measure"]),
        "average_sparsity": float(summary["average_sparsity"]),
        "per_task_accuracy": [r["accuracy"] for r in summary["per_task"]],
        "per_task_sparsity": [r["sparsity"] for r in summary["per_task"]],
    }
    with open(eval_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(eval_data, f, default_flow_style=False)
    log.info(f"\n评估结果已保存: {eval_out}")


if __name__ == "__main__":
    main()
