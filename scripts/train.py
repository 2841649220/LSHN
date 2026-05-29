"""
LSHN 训练脚本
=============
使用 LSHNModel 进行持续学习训练。
支持 Split-MNIST / Permuted-MNIST 等增量任务场景。

用法:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --task_id 0 --epochs 20
"""
import os
import sys
import time
import argparse
import logging
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

# 使脚本可在项目根目录或 scripts/ 目录下运行
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from lshn.model import LSHNModel
from lshn.utils.metrics import ContinualLearningMetrics

# ──────────────── 日志 ────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lshn.train")


# ──────────────── 数据加载 ────────────────

def _make_split_mnist(data_dir: str, num_tasks: int, classes_per_task: int):
    """
    生成 Split-MNIST 任务序列。
    每个任务包含 classes_per_task 个类别的训练/测试子集。
    返回: list[dict] 每个 dict 含 'train_loader', 'test_loader', 'classes'
    """
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),  # 28x28 → 784
        ])
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    except Exception as e:
        log.warning(f"无法加载 MNIST ({e})，使用合成数据替代")
        return _make_synthetic_tasks(num_tasks, classes_per_task)

    tasks = []
    all_classes = list(range(10))
    train_targets = torch.tensor(train_ds.targets)
    test_targets = torch.tensor(test_ds.targets)
    for t in range(num_tasks):
        cls_start = t * classes_per_task
        task_classes = all_classes[cls_start: cls_start + classes_per_task]
        if len(task_classes) == 0:
            break

        task_cls_tensor = torch.tensor(task_classes)
        train_mask = torch.isin(train_targets, task_cls_tensor)
        test_mask = torch.isin(test_targets, task_cls_tensor)
        train_idx = train_mask.nonzero(as_tuple=True)[0][:2000].tolist()
        test_idx = test_mask.nonzero(as_tuple=True)[0][:500].tolist()

        tasks.append({
            "train_loader": DataLoader(Subset(train_ds, train_idx), batch_size=64, shuffle=True),
            "test_loader": DataLoader(Subset(test_ds, test_idx), batch_size=128, shuffle=False),
            "classes": task_classes,
        })
    return tasks


def _make_synthetic_tasks(num_tasks: int, classes_per_task: int):
    """合成高斯数据任务序列 (不依赖 torchvision)"""
    input_dim = 128
    samples_per_class = 500
    tasks = []

    for t in range(num_tasks):
        task_classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        xs, ys = [], []
        for c in task_classes:
            center = torch.randn(input_dim) * 2
            x = center.unsqueeze(0) + torch.randn(samples_per_class, input_dim) * 0.5
            y = torch.full((samples_per_class,), c, dtype=torch.long)
            xs.append(x)
            ys.append(y)

        X = torch.cat(xs)
        Y = torch.cat(ys)
        n_train = int(0.8 * len(X))
        perm = torch.randperm(len(X))
        X, Y = X[perm], Y[perm]

        train_ds = TensorDataset(X[:n_train], Y[:n_train])
        test_ds = TensorDataset(X[n_train:], Y[n_train:])

        tasks.append({
            "train_loader": DataLoader(train_ds, batch_size=64, shuffle=True),
            "test_loader": DataLoader(test_ds, batch_size=128, shuffle=False),
            "classes": task_classes,
        })
    return tasks


# ──────────────── 训练一个任务 ────────────────

def train_task(
    model: LSHNModel,
    task_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_id: int,
    num_classes_so_far: int,
    cfg: dict,
):
    """
    对单个任务进行训练。每个样本运行 fast_steps_per_sample 个快时钟步。
    """
    model.train()
    epochs = cfg["training"].get("num_epochs", 10)
    fast_steps = cfg["training"].get("fast_steps_per_sample", 20)
    log_interval = cfg["training"].get("log_interval", 50)
    grad_clip = cfg["training"].get("gradient_clip", 1.0)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        step_count = 0
        t0 = time.time()

        for batch_idx, (x_batch, y_batch) in enumerate(task_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # 适配输入维度
            if x_batch.shape[-1] != model.input_dim:
                if not hasattr(model, "_input_proj"):
                    proj = nn.Linear(
                        x_batch.shape[-1], model.input_dim, bias=False
                    ).to(device)
                    model.add_module("_input_proj", proj)
                    optimizer.add_param_group({"params": proj.parameters()})
                x_batch = model._input_proj(x_batch)

            optimizer.zero_grad()

            # 对每个样本运行多步快时钟 (取最后一步输出)
            outputs = None
            for t_step in range(fast_steps):
                # 构建 one-hot 目标
                target_onehot = torch.zeros(
                    x_batch.size(0), num_classes_so_far, device=device
                )
                # 确保标签在范围内
                y_clamped = y_batch.clamp(0, num_classes_so_far - 1)
                target_onehot.scatter_(1, y_clamped.unsqueeze(1), 1.0)

                result = model.forward_step(x_batch, target=target_onehot)
                outputs = result["output"]

            # 计算分类损失 (只用最后一步的输出)
            loss = criterion(outputs, y_batch.clamp(0, num_classes_so_far - 1))

            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch.clamp(0, num_classes_so_far - 1)).sum().item()
            total += y_batch.size(0)
            step_count += 1

            if step_count % log_interval == 0:
                log.info(
                    f"  Task {task_id} | Epoch {epoch+1}/{epochs} | "
                    f"Step {step_count} | Loss {loss.item():.4f} | "
                    f"Acc {100.*correct/total:.1f}%"
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(step_count, 1)
        acc = 100.0 * correct / max(total, 1)
        log.info(
            f"Task {task_id} | Epoch {epoch+1}/{epochs} 完成 | "
            f"AvgLoss {avg_loss:.4f} | Acc {acc:.1f}% | 耗时 {elapsed:.1f}s"
        )

    # 重置模型内部时钟/隐状态，准备下一个任务
    model.reset()
    return avg_loss, acc


# ──────────────── 评估 (单任务) ────────────────

@torch.no_grad()
def evaluate_task(
    model: LSHNModel,
    test_loader: DataLoader,
    device: torch.device,
    num_classes_so_far: int,
    fast_steps: int = 5,
) -> float:
    """在给定任务测试集上计算准确率"""
    model.eval()
    correct = 0
    total = 0

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if x_batch.shape[-1] != model.input_dim:
            if hasattr(model, "_input_proj"):
                x_batch = model._input_proj(x_batch)
            else:
                x_batch = x_batch[:, :model.input_dim]

        # 多步推理取最后输出
        for _ in range(fast_steps):
            result = model.forward_step(x_batch)
        outputs = result["output"]

        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch.clamp(0, num_classes_so_far - 1)).sum().item()
        total += y_batch.size(0)

    model.reset()
    return 100.0 * correct / max(total, 1)


# ──────────────── 主入口 ────────────────

def main():
    parser = argparse.ArgumentParser(description="LSHN 持续学习训练")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--task_id", type=int, default=None,
                        help="只训练指定任务 (默认训练所有)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖配置中的 epoch 数")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="模型保存目录")
    parser.add_argument("--synthetic", action="store_true",
                        help="使用合成数据 (不需要下载 MNIST)")
    args = parser.parse_args()

    # 加载配置
    config_path = _project_root / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs

    # 设备
    device = torch.device("cuda" if cfg.get("device", {}).get("cuda", True) 
                          and torch.cuda.is_available() else "cpu")
    log.info(f"使用设备: {device}")

    # 模型
    model_cfg = cfg["model"]
    model = LSHNModel(
        input_dim=model_cfg.get("input_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_neurons=model_cfg.get("num_neurons", 1000),
        num_groups=model_cfg.get("num_groups", 10),
        max_edges=model_cfg.get("max_edges", 500),
        initial_classes=model_cfg.get("initial_classes", 2),
        enable_dendrites=model_cfg.get("enable_dendrites", False),
        enable_active_inference=model_cfg.get("enable_active_inference", False),
        target_spikes_per_step=cfg.get("budget", {}).get("target_spikes_per_step", 50),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"模型参数量: {param_count:,}")

    # 优化器
    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    # 数据
    cl_cfg = cfg.get("continual", {})
    num_tasks = cl_cfg.get("tasks", 5)
    classes_per_task = cl_cfg.get("classes_per_task", 2)

    if args.synthetic:
        tasks = _make_synthetic_tasks(num_tasks, classes_per_task)
    else:
        data_dir = cfg.get("data", {}).get("data_dir", "./data")
        tasks = _make_split_mnist(str(_project_root / data_dir), num_tasks, classes_per_task)

    log.info(f"持续学习: {len(tasks)} 个任务, 每任务 {classes_per_task} 个类别")

    # 持续学习指标
    cl_metrics = ContinualLearningMetrics(num_tasks=len(tasks))

    # 保存目录
    save_dir = _project_root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # ──────── 持续学习循环 ────────
    num_classes_so_far = model_cfg.get("initial_classes", 2)

    task_range = range(len(tasks))
    if args.task_id is not None:
        task_range = [args.task_id]

    for t in task_range:
        task = tasks[t]
        log.info(f"\n{'='*60}")
        log.info(f"开始任务 {t}: 类别 {task['classes']}")
        log.info(f"{'='*60}")

        # 如果新任务的类别超出当前头大小，动态扩展
        max_class = max(task["classes"]) + 1
        if max_class > num_classes_so_far:
            expand_by = max_class - num_classes_so_far
            model.expand_classes(expand_by)
            num_classes_so_far = max_class
            log.info(f"输出头扩展至 {num_classes_so_far} 类")

            # 扩展后需要重新构建优化器 (新参数)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg.get("learning_rate", 0.001),
                weight_decay=train_cfg.get("weight_decay", 0.0001),
            )

        # 训练当前任务
        loss, acc = train_task(
            model, task["train_loader"], optimizer, device,
            task_id=t, num_classes_so_far=num_classes_so_far, cfg=cfg,
        )

        # 评估所有已见任务
        log.info("评估所有已见任务...")
        for prev_t in range(t + 1):
            prev_task = tasks[prev_t]
            prev_acc = evaluate_task(
                model, prev_task["test_loader"], device,
                num_classes_so_far=num_classes_so_far,
                fast_steps=cfg["training"].get("fast_steps_per_sample", 5),
            )
            cl_metrics.update_accuracy(trained_task_idx=t, eval_task_idx=prev_t,
                                       acc=prev_acc / 100.0)
            log.info(f"  任务 {prev_t} 准确率: {prev_acc:.1f}%")

        # 打印监控报告
        report = model.get_monitoring_report()
        log.info("LSHN 监控报告:")
        for k, v in report.items():
            log.info(f"  {k}: {v:.6f}")

        # 持续学习综合指标
        avg_acc = cl_metrics.average_accuracy(current_task_idx=t)
        forgetting = cl_metrics.forgetting_measure(current_task_idx=t)
        log.info(f"平均准确率: {avg_acc:.4f} | 遗忘度: {forgetting:.4f}")

        # 保存检查点
        ckpt_path = save_dir / f"lshn_task{t}.pt"
        torch.save({
            "task_id": t,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": num_classes_so_far,
            "metrics": cl_metrics.report(current_task_idx=t),
            "config": cfg,
        }, ckpt_path)
        log.info(f"检查点已保存: {ckpt_path}")

    # ──────── 最终报告 ────────
    log.info(f"\n{'='*60}")
    log.info("训练完成! 最终持续学习指标:")
    log.info(f"{'='*60}")
    last_task = max(task_range) if isinstance(task_range, list) else len(tasks) - 1
    final_report = cl_metrics.report(current_task_idx=last_task)
    for k, v in final_report.items():
        log.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
