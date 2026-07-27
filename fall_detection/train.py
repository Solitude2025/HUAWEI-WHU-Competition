"""
训练脚本
-----------
训练 Light-TCN + Motion Feature Encoder 跌倒检测模型。

用法:
    # 基础训练
    python train.py --data_dir data/train --epochs 100
    
    # 从配置训练
    python train.py --config configs/config.yaml
    
    # 恢复训练
    python train.py --resume checkpoints/best.pth
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau
)
import numpy as np
from tqdm import tqdm
import time
from typing import Dict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fall_detector import FallDetector, create_fall_detector
from data.dataset import FallDetectionDataset, create_dataloader
from utils.metrics import compute_metrics, compute_model_efficiency


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, T, 1) 预测概率
            target: (B, T) 真实标签
        """
        pred = pred.squeeze(-1)  # (B, T)
        
        bce = nn.functional.binary_cross_entropy(
            pred, target, reduction='none'
        )
        
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        
        return focal.mean()


class CombinedLoss(nn.Module):
    """组合损失: BCE + Focal"""
    
    def __init__(self, bce_weight: float = 1.0, focal_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce = nn.BCELoss()
        # alpha>0.5 提高正类（跌倒帧）权重，优先保 recall（漏报代价远高于误报）
        self.focal = FocalLoss(alpha=0.75)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(pred.squeeze(-1), target) +
            self.focal_weight * self.focal(pred, target)
        )


def train_epoch(
    model: FallDetector,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        keypoints = batch["keypoints"].to(device)  # (B, T, 17, 3)
        bboxes = batch["bboxes"].to(device)         # (B, T, 4)
        labels = batch["labels"].to(device)          # (B, T)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(keypoints, bboxes)
        tcn_prob = outputs["fall_prob"].squeeze(-1)  # (B, T)
        
        # 损失
        loss = criterion(tcn_prob, labels)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集预测
        all_preds.append(tcn_prob.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        
        # 更新进度条
        if batch_idx % log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # 计算指标
    all_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    all_labels = np.concatenate([l.reshape(-1) for l in all_labels])
    
    # 检查是否有正样本
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    
    metrics = compute_metrics(all_preds, all_labels)
    
    avg_loss = total_loss / len(dataloader)
    
    result = {"loss": avg_loss}
    if metrics:
        result.update(metrics)
    else:
        result.update({"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0})
    
    return result


@torch.no_grad()
def validate(
    model: FallDetector,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """验证"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="[Val]")
    
    for batch in pbar:
        keypoints = batch["keypoints"].to(device)
        bboxes = batch["bboxes"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(keypoints, bboxes)
        tcn_prob = outputs["fall_prob"].squeeze(-1)
        
        loss = criterion(tcn_prob, labels)
        total_loss += loss.item()
        
        all_preds.append(tcn_prob.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    all_labels = np.concatenate([l.reshape(-1) for l in all_labels])
    
    # 检查是否有正样本
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    
    metrics = compute_metrics(all_preds, all_labels)
    
    avg_loss = total_loss / len(dataloader)
    
    result = {"loss": avg_loss}
    if metrics:
        result.update(metrics)
    else:
        result.update({"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0})
    
    return result


def main():
    parser = argparse.ArgumentParser(description="训练跌倒检测模型")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default="data/train",
                       help="训练数据目录")
    parser.add_argument("--val_dir", type=str, default="data/val",
                       help="验证数据目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="模型保存目录")
    parser.add_argument("--version", type=str, default="standard",
                       choices=["standard", "light", "large", 
                                "transformer-tcn-standard", "transformer-tcn-efficient", "transformer-tcn-light"],
                       help="模型版本")
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 合并参数
    epochs = args.epochs or config.get("training", {}).get("epochs", 100)
    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 32)
    lr = args.lr or config.get("training", {}).get("learning_rate", 0.001)
    save_dir = args.save_dir or config.get("training", {}).get("save_dir", "checkpoints")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Train] 设备: {device}")
    
    # 创建数据加载器
    train_loader = create_dataloader(
        args.data_dir,
        sequence_length=config.get("training", {}).get("sequence_length", 32),
        batch_size=batch_size,
        stride=config.get("training", {}).get("stride", 8),
        mode="train",
        use_augmentation=True,
    )
    
    val_loader = None
    if args.val_dir and os.path.exists(args.val_dir):
        val_loader = create_dataloader(
            args.val_dir,
            sequence_length=config.get("training", {}).get("sequence_length", 32),
            batch_size=batch_size,
            stride=config.get("training", {}).get("stride", 8),
            mode="val",
            use_augmentation=False,
        )
    
    # 创建模型
    model = create_fall_detector(version=args.version)
    model = model.to(device)
    
    # 打印模型信息
    efficiency = compute_model_efficiency(model)
    print(f"[Train] 模型参数: {efficiency['total_params']:,}")
    print(f"[Train] 模型大小: {efficiency['model_size_mb_fp32']:.2f} MB (FP32)")
    
    # 赛题合规检查
    if efficiency['total_params'] > 20_000_000:
        print(f"[WARNING] 参数量 {efficiency['total_params']:,} 超过赛题限制 20M!")
    if efficiency['model_size_mb_fp32'] > 80:
        print(f"[WARNING] 模型大小 {efficiency['model_size_mb_fp32']:.2f}MB 超过赛题限制 80MB!")
    
    # 损失函数
    criterion = CombinedLoss(bce_weight=1.0, focal_weight=1.0)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.get("training", {}).get("weight_decay", 0.0001),
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # 恢复训练
    start_epoch = 0
    best_f1 = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint.get("best_f1", 0.0)
        print(f"[Train] 从 epoch {start_epoch} 恢复训练")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    print(f"\n{'='*60}")
    print(f"开始训练: {epochs} epochs, batch_size={batch_size}")
    print(f"{'='*60}\n")
    
    # 训练历史记录（用于可视化曲线）
    history = []
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, log_interval=10,
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 验证
        val_metrics = {}
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device)
        
        # 打印结果
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} 完成 ({epoch_time:.1f}s) | LR: {current_lr:.6f}")
        print(f"  Train: loss={train_metrics['loss']:.4f}, "
              f"acc={train_metrics.get('accuracy', 0):.4f}, "
              f"f1={train_metrics.get('f1_score', 0):.4f}")
        
        if val_metrics:
            print(f"  Val:   loss={val_metrics['loss']:.4f}, "
                  f"acc={val_metrics.get('accuracy', 0):.4f}, "
                  f"f1={val_metrics.get('f1_score', 0):.4f}, "
                  f"prec={val_metrics.get('precision', 0):.4f}, "
                  f"rec={val_metrics.get('recall', 0):.4f}")
        
        # 保存历史
        history_entry = {
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "accuracy": train_metrics.get("accuracy", 0),
            "precision": train_metrics.get("precision", 0),
            "recall": train_metrics.get("recall", 0),
            "f1_score": train_metrics.get("f1_score", 0),
            "lr": current_lr,
        }
        for k in ["loss", "accuracy", "precision", "recall", "f1_score"]:
            if k in val_metrics:
                history_entry[f"val_{k}"] = val_metrics[k]
        history.append(history_entry)
        
        # 保存检查点（无验证集时回退到训练集 F1）
        if val_metrics:
            val_f1 = val_metrics.get("f1_score", 0.0)
        else:
            val_f1 = train_metrics.get("f1_score", 0.0)
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "history": history,
            "best_f1": best_f1,
            "config": config,
        }
        
        torch.save(checkpoint, os.path.join(save_dir, "latest.pth"))
        
        if is_best:
            torch.save(checkpoint, os.path.join(save_dir, "best.pth"))
            print(f"  >>> 保存最佳模型 (F1={best_f1:.4f})")
    
    print(f"\n{'='*60}")
    print(f"训练完成! 最佳 F1: {best_f1:.4f}")
    print(f"模型保存到: {save_dir}")
    print(f"{'='*60}")

    # 保底：若整个训练过程 F1 未超过 0（从未保存 best.pth），用 latest 兜底
    best_path = os.path.join(save_dir, "best.pth")
    latest_path = os.path.join(save_dir, "latest.pth")
    if not os.path.exists(best_path) and os.path.exists(latest_path):
        import shutil
        shutil.copy(latest_path, best_path)
        print("[Train] 未产生最佳模型，已将 latest.pth 复制为 best.pth")
    
    # ── 最终测试集评估 ──
    test_dir = "data/test"
    if os.path.exists(test_dir):
        print(f"\n{'='*60}")
        print(f"最终测试集评估: {test_dir}")
        print(f"{'='*60}")
        
        # 加载最佳模型
        best_path = os.path.join(save_dir, "best.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[Test] 加载最佳模型 (epoch={checkpoint['epoch']}, F1={best_f1:.4f})")
        
        test_loader = create_dataloader(
            test_dir,
            sequence_length=config.get("training", {}).get("sequence_length", 32),
            batch_size=batch_size,
            stride=config.get("training", {}).get("stride", 8),
            mode="val",
            use_augmentation=False,
        )
        
        test_metrics = validate(model, test_loader, criterion, device)
        
        print(f"\n  测试集指标:")
        print(f"    Accuracy:  {test_metrics.get('accuracy', 0):.4f}")
        print(f"    Precision: {test_metrics.get('precision', 0):.4f}")
        print(f"    Recall:    {test_metrics.get('recall', 0):.4f}")
        print(f"    F1 Score:  {test_metrics.get('f1_score', 0):.4f}")
        print(f"    AUC-ROC:   {test_metrics.get('auc_roc', 0):.4f}")
        print(f"    FPR:       {test_metrics.get('false_positive_rate', 0):.4f}")
        print(f"    Miss Rate: {test_metrics.get('miss_rate', 0):.4f}")
        print(f"    TP:{test_metrics.get('tp', 0)} FP:{test_metrics.get('fp', 0)} "
              f"FN:{test_metrics.get('fn', 0)} TN:{test_metrics.get('tn', 0)}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
