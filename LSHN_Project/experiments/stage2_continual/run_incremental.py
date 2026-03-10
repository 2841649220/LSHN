"""
Stage 2 实验: 持续学习增量场景
================================
使用 LSHNModel 端到端运行持续学习:
- Split 合成数据 (多任务增量)
- 动态类别扩展
- 遗忘度 / 平均准确率 / 脉冲稀疏度跟踪
- LSHN 监控报告输出

用法:
    python experiments/stage2_continual/run_incremental.py
    python experiments/stage2_continual/run_incremental.py --num_tasks 5 --epochs 3
"""
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from lshn.model import LSHNModel
from lshn.utils.metrics import ContinualLearningMetrics


def make_task_data(task_id: int, classes_per_task: int, input_dim: int,
                   n_train: int = 200, n_test: int = 50):
    """为一个任务生成合成高斯聚类数据"""
    xs_train, ys_train = [], []
    xs_test, ys_test = [], []

    for c_offset in range(classes_per_task):
        c = task_id * classes_per_task + c_offset
        torch.manual_seed(c * 137 + 42)
        center = torch.randn(input_dim) * 3.0

        x_tr = center.unsqueeze(0) + torch.randn(n_train, input_dim) * 0.5
        x_te = center.unsqueeze(0) + torch.randn(n_test, input_dim) * 0.5
        xs_train.append(x_tr)
        ys_train.append(torch.full((n_train,), c, dtype=torch.long))
        xs_test.append(x_te)
        ys_test.append(torch.full((n_test,), c, dtype=torch.long))

    X_tr, Y_tr = torch.cat(xs_train), torch.cat(ys_train)
    X_te, Y_te = torch.cat(xs_test), torch.cat(ys_test)

    # 打乱
    perm_tr = torch.randperm(len(X_tr))
    perm_te = torch.randperm(len(X_te))
    return (
        DataLoader(TensorDataset(X_tr[perm_tr], Y_tr[perm_tr]), batch_size=32, shuffle=True),
        DataLoader(TensorDataset(X_te[perm_te], Y_te[perm_te]), batch_size=64, shuffle=False),
        list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task)),
    )


@torch.no_grad()
def evaluate(model: LSHNModel, test_loader: DataLoader, device: torch.device,
             num_classes: int, fast_steps: int = 5) -> float:
    """评估准确率"""
    model.eval()
    correct, total = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        for _ in range(fast_steps):
            result = model.forward_step(x)
        pred = result["output"].argmax(dim=1)
        correct += pred.eq(y.clamp(0, num_classes - 1)).sum().item()
        total += y.size(0)
    model.reset()
    return 100.0 * correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: 持续学习增量实验")
    parser.add_argument("--num_tasks", type=int, default=3)
    parser.add_argument("--classes_per_task", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fast_steps", type=int, default=10)
    parser.add_argument("--input_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_neurons", type=int, default=500000)
    parser.add_argument("--max_edges", type=int, default=None,
                        help="最大超边数 (默认 num_neurons // 10)")
    parser.add_argument("--num_groups", type=int, default=100)
    args = parser.parse_args()

    # 自动计算联动参数
    if args.max_edges is None:
        args.max_edges = args.num_neurons // 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 构建模型
    model = LSHNModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_neurons=args.num_neurons,
        num_groups=args.num_groups,
        max_edges=args.max_edges,
        initial_classes=args.classes_per_task,
        target_spikes_per_step=args.num_neurons // 100,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    metrics = ContinualLearningMetrics(num_tasks=args.num_tasks)

    num_classes_so_far = args.classes_per_task

    # 准备所有任务的数据
    task_data = []
    for t in range(args.num_tasks):
        train_dl, test_dl, classes = make_task_data(
            t, args.classes_per_task, args.input_dim
        )
        task_data.append((train_dl, test_dl, classes))

    # ──────── 持续学习主循环 ────────
    for t in range(args.num_tasks):
        train_dl, test_dl, classes = task_data[t]
        print(f"\n{'='*50}")
        print(f"任务 {t}: 类别 {classes}")
        print(f"{'='*50}")

        # 动态扩展
        max_class = max(classes) + 1
        if max_class > num_classes_so_far:
            expand_by = max_class - num_classes_so_far
            model.expand_classes(expand_by)
            num_classes_so_far = max_class
            print(f"  扩展至 {num_classes_so_far} 类")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        # 训练
        model.train()
        for epoch in range(args.epochs):
            total_loss, correct, total = 0.0, 0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                target_oh = torch.zeros(x.size(0), num_classes_so_far, device=device)
                y_c = y.clamp(0, num_classes_so_far - 1)
                target_oh.scatter_(1, y_c.unsqueeze(1), 1.0)

                for _ in range(args.fast_steps):
                    result = model.forward_step(x, target=target_oh)

                loss = criterion(result["output"], y_c)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                correct += result["output"].argmax(1).eq(y_c).sum().item()
                total += y.size(0)

            acc = 100.0 * correct / max(total, 1)
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={total_loss/max(total,1):.4f} acc={acc:.1f}%")
        model.reset()

        # 评估所有已见任务
        print("  评估:")
        for prev_t in range(t + 1):
            _, prev_test_dl, _ = task_data[prev_t]
            prev_acc = evaluate(model, prev_test_dl, device, num_classes_so_far, args.fast_steps)
            metrics.update_accuracy(trained_task_idx=t, eval_task_idx=prev_t, acc=prev_acc / 100.0)
            print(f"    任务 {prev_t}: {prev_acc:.1f}%")

        avg_acc = metrics.average_accuracy(current_task_idx=t)
        forgetting = metrics.forgetting_measure(current_task_idx=t)
        print(f"  → 平均准确率={avg_acc:.4f} | 遗忘度={forgetting:.4f}")

    # ──────── 最终报告 ────────
    print(f"\n{'='*50}")
    print("最终持续学习指标")
    print(f"{'='*50}")
    final = metrics.report(current_task_idx=args.num_tasks - 1)
    for k, v in final.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # LSHN 内部监控
    print("\nLSHN 监控报告:")
    report = model.get_monitoring_report()
    for k, v in report.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # 准确率矩阵
    print(f"\n准确率矩阵 R[trained, eval]:")
    R = metrics.R[:args.num_tasks, :args.num_tasks]
    for i in range(args.num_tasks):
        row = " | ".join(f"{R[i,j].item():.3f}" for j in range(args.num_tasks))
        print(f"  T{i}: [{row}]")


if __name__ == "__main__":
    main()
