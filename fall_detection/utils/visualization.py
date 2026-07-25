"""
可视化工具
--------------
跌倒检测结果可视化：
- 关键点绘制
- 运动特征曲线
- 概率时序曲线
- 混淆矩阵
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os


# COCO 关键点连接（骨架连线）
SKELETON_CONNECTIONS = [
    (5, 6),   # 左肩-右肩
    (5, 7), (7, 9),   # 左臂
    (6, 8), (8, 10),  # 右臂
    (5, 11), (6, 12), # 肩-髋
    (11, 12),          # 左髋-右髋
    (11, 13), (13, 15), # 左腿
    (12, 14), (14, 16), # 右腿
    (0, 1), (0, 2),   # 鼻-眼
    (1, 3), (2, 4),   # 眼-耳
]

SKELETON_COLORS = [
    (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (255, 0, 0), (255, 0, 0),
    (0, 255, 255), (0, 255, 255), (0, 255, 255),
    (255, 255, 0), (255, 255, 0),
    (255, 0, 255), (255, 0, 255),
    (0, 128, 255), (0, 128, 255),
]


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,     # (17, 3) [x, y, conf]
    bbox: Optional[np.ndarray] = None,  # (4,)
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制关键点和骨架
    
    Args:
        image: (H, W, 3) BGR/RGB
        keypoints: (17, 3) 归一化坐标
        bbox: (4,) 归一化边界框
    
    Returns:
        绘制后的图像
    """
    h, w = image.shape[:2]
    vis = image.copy()
    
    # 反归一化
    kp_pixel = keypoints[:, :2] * np.array([w, h])
    kp_pixel = kp_pixel.astype(int)
    conf = keypoints[:, 2]
    
    # 画骨架
    for conn_idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
        if conf[i] > 0.3 and conf[j] > 0.3:
            pt1 = tuple(kp_pixel[i])
            pt2 = tuple(kp_pixel[j])
            clr = SKELETON_COLORS[conn_idx]
            cv2_line(vis, pt1, pt2, clr, thickness)
    
    # 画关键点
    for i in range(17):
        if conf[i] > 0.3:
            px, py = kp_pixel[i]
            cv2_circle(vis, (px, py), 4, color, -1)
    
    # 画边界框
    if bbox is not None:
        x1, y1, x2, y2 = (bbox * np.array([w, h, w, h])).astype(int)
        cv2_rect(vis, (x1, y1), (x2, y2), color, thickness)
    
    return vis


def cv2_line(img, pt1, pt2, color, thickness):
    import cv2
    cv2.line(img, pt1, pt2, color, thickness)

def cv2_circle(img, center, radius, color, thickness):
    import cv2
    cv2.circle(img, center, radius, color, thickness)

def cv2_rect(img, pt1, pt2, color, thickness):
    import cv2
    cv2.rectangle(img, pt1, pt2, color, thickness)


def plot_probability_curve(
    probabilities: np.ndarray,      # (T,)
    labels: Optional[np.ndarray] = None,  # (T,)
    title: str = "Fall Probability",
    save_path: Optional[str] = None,
):
    """
    绘制跌倒概率时序曲线
    
    Args:
        probabilities: 每帧概率
        labels: 真实标签（可选，用于对比）
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    T = len(probabilities)
    frames = np.arange(T)
    
    ax.plot(frames, probabilities, 'b-', linewidth=1.5, label='Predicted Probability')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    
    if labels is not None:
        gt_mask = labels > 0.5
        ax.fill_between(frames, 0, 1, where=gt_mask,
                        alpha=0.2, color='red', label='Ground Truth Fall')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Fall Probability')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    history: List[Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
):
    """
    绘制训练曲线：loss、accuracy、precision、recall、F1
    
    Args:
        history: 每个 epoch 的指标字典列表
                 [{"loss":0.5, "accuracy":0.95, "precision":0.93, "recall":0.94, "f1_score":0.93}, ...]
        save_path: 保存路径
        title: 图表标题
    """
    epochs = range(1, len(history) + 1)

    # 分离训练和验证指标
    has_train_prefix = any(k.startswith("train_") for k in history[0].keys())
    has_val_prefix = any(k.startswith("val_") for k in history[0].keys())

    # 确定 key 映射
    def get_key(prefix, base):
        """统一获取指标 key"""
        if has_train_prefix:
            return f"{prefix}{base}" if prefix else base
        return base

    is_train_only = not has_val_prefix

    # ---- 图1: Loss 曲线 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Loss
    ax = axes[0, 0]
    train_loss = [h.get(get_key("", "loss"), 0) for h in history]
    ax.plot(epochs, train_loss, "b-o", label="Train Loss", markersize=3)
    if not is_train_only:
        val_loss = [h.get(get_key("val_", "loss"), None) for h in history]
        val_loss = [v for v in val_loss if v is not None]
        if val_loss:
            val_epochs = range(1, len(val_loss) + 1)
            ax.plot(val_epochs, val_loss, "r-^", label="Val Loss", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    acc = [h.get(get_key("", "accuracy"), 0) for h in history]
    ax.plot(epochs, acc, "g-o", label="Train Acc", markersize=3)
    if not is_train_only:
        val_acc = [h.get(get_key("val_", "accuracy"), None) for h in history]
        val_acc = [v for v in val_acc if v is not None]
        if val_acc:
            val_epochs = range(1, len(val_acc) + 1)
            ax.plot(val_epochs, val_acc, "r-^", label="Val Acc", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision / Recall
    ax = axes[1, 0]
    prec = [h.get(get_key("", "precision"), 0) for h in history]
    rec = [h.get(get_key("", "recall"), 0) for h in history]
    ax.plot(epochs, prec, "orange", linestyle="-", marker="o", label="Train Precision", markersize=3)
    ax.plot(epochs, rec, "purple", linestyle="-", marker="s", label="Train Recall", markersize=3)
    if not is_train_only:
        val_prec = [h.get(get_key("val_", "precision"), None) for h in history]
        val_rec = [h.get(get_key("val_", "recall"), None) for h in history]
        val_prec = [v for v in val_prec if v is not None]
        val_rec = [v for v in val_rec if v is not None]
        if val_prec:
            val_epochs = range(1, len(val_prec) + 1)
            ax.plot(val_epochs, val_prec, "orange", linestyle="--", marker="^", label="Val Precision", markersize=3)
            ax.plot(val_epochs, val_rec, "purple", linestyle="--", marker="^", label="Val Recall", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1 Score
    ax = axes[1, 1]
    f1 = [h.get(get_key("", "f1_score"), 0) for h in history]
    ax.plot(epochs, f1, "g-o", label="Train F1", markersize=3)
    if not is_train_only:
        val_f1 = [h.get(get_key("val_", "f1_score"), None) for h in history]
        val_f1 = [v for v in val_f1 if v is not None]
        if val_f1:
            val_epochs = range(1, len(val_f1) + 1)
            ax.plot(val_epochs, val_f1, "r-^", label="Val F1", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Visual] 训练曲线已保存: {save_path}")
    else:
        plt.show()


def plot_lr_schedule(
    history: List[Dict[str, float]],
    save_path: Optional[str] = None,
):
    """
    绘制学习率变化曲线
    """
    epochs = range(1, len(history) + 1)
    lrs = [h.get("lr", 0) for h in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lrs, "b-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    if max(lrs) > 0:
        ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Visual] LR 曲线已保存: {save_path}")
    else:
        plt.show()


def plot_metrics_summary(
    metrics: Dict[str, float],
    title: str = "Evaluation Metrics",
    save_path: Optional[str] = None,
):
    """
    绘制单次评估的指标横向柱状图
    
    直观展示 accuracy / precision / recall / f1 / fpr / fnr
    
    Args:
        metrics: {"accuracy":0.95, "precision":0.93, "recall":0.94, "f1_score":0.93, "false_positive_rate":0.02, "miss_rate":0.06}
        title: 标题
        save_path: 保存路径
    """
    # 只取百分比指标（0~1 范围）
    display_keys = ["accuracy", "precision", "recall", "f1_score"]
    display_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    display_values = [metrics.get(k, 0) for k in display_keys]

    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(display_labels, display_values, color=colors, height=0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # 在条上标注数值
    for bar, val in zip(bars, display_values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=11, fontweight="bold")

    # 显示额外指标
    extra = []
    if "false_positive_rate" in metrics:
        extra.append(f"FPR: {metrics['false_positive_rate']:.4f}")
    if "miss_rate" in metrics:
        extra.append(f"Miss Rate: {metrics['miss_rate']:.4f}")
    if "auc_roc" in metrics:
        extra.append(f"AUC: {metrics['auc_roc']:.4f}")
    if "average_precision" in metrics:
        extra.append(f"AP: {metrics['average_precision']:.4f}")

    if extra:
        ax.text(0.5, -0.12, " | ".join(extra), ha="center", va="top",
                transform=ax.transAxes, fontsize=10, color="gray",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Visual] 指标概要已保存: {save_path}")
    else:
        plt.show()


def generate_evaluation_report(
    checkpoint_path: str,
    output_dir: str = "eval_report",
):
    """
    从训练检查点生成完整的评估报告（包含所有图表）
    
    Args:
        checkpoint_path: 模型检查点路径（.pth）
        output_dir: 输出目录
    """
    import json
    import torch

    os.makedirs(output_dir, exist_ok=True)

    # 加载检查点
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"[Eval] 加载检查点: epoch={ckpt.get('epoch', '?')}")

    # ---- 1. 训练历史提取 ----
    # 优先使用 history（包含所有 epoch），否则回退到最后一步
    history = ckpt.get("history", [])
    train_metrics = ckpt.get("train_metrics", {})
    val_metrics = ckpt.get("val_metrics", {})

    if not history and (train_metrics or val_metrics):
        entry = {}
        for k, v in train_metrics.items():
            entry[k] = v
        for k, v in val_metrics.items():
            entry[f"val_{k}"] = v
        entry["lr"] = ckpt.get("lr", 0.001)
        history.append(entry)

    print(f"[Eval] 加载历史记录: {len(history)} 个 epoch")

    if history:
        # 训练曲线
        plot_training_curves(
            history,
            save_path=os.path.join(output_dir, "training_curves.png"),
        )

        # 学习率曲线
        plot_lr_schedule(
            history,
            save_path=os.path.join(output_dir, "lr_schedule.png"),
        )

    # ---- 2. 最佳指标摘要 ----
    best_f1 = ckpt.get("best_f1", 0)
    final_metrics = ckpt.get("val_metrics", ckpt.get("train_metrics", {}))
    if final_metrics:
        plot_metrics_summary(
            final_metrics,
            title=f"Final Metrics (Best F1={best_f1:.4f})",
            save_path=os.path.join(output_dir, "metrics_summary.png"),
        )

    # ---- 3. 模型信息 ----
    model_info = ckpt.get("config", {})
    info_lines = [
        f"Epoch: {ckpt.get('epoch', '?')}",
        f"Best F1: {best_f1:.4f}",
        f"Model Version: {model_info.get('model', {}).get('version', 'standard')}",
    ]

    # if val_metrics:
    #     info_lines.append(f"Val Loss: {val_metrics.get('loss', 0):.4f}")
    #     info_lines.append(f"Val Accuracy: {val_metrics.get('accuracy', 0):.4f}")
    #     info_lines.append(f"Val Precision: {val_metrics.get('precision', 0):.4f}")
    #     info_lines.append(f"Val Recall: {val_metrics.get('recall', 0):.4f}")
    #     info_lines.append(f"Val F1: {val_metrics.get('f1_score', 0):.4f}")

    # 保存为 JSON
    report_json = {
        "epoch": ckpt.get("epoch", -1),
        "best_f1": best_f1,
        "train_metrics": {k: float(v) if not isinstance(v, (int, float)) else v
                          for k, v in train_metrics.items()},
        "val_metrics": {k: float(v) if not isinstance(v, (int, float)) else v
                        for k, v in val_metrics.items()},
        "model_config": {
            "version": model_info.get("model", {}).get("version", "standard"),
            "params": ckpt.get("model_state_dict", {}).get("params", "unknown"),
        },
    }

    with open(os.path.join(output_dir, "report.json"), "w") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)

    print(f"\n[Eval] 评估报告已生成: {output_dir}/")
    print(f"  - training_curves.png   (训练曲线)")
    print(f"  - lr_schedule.png       (学习率变化)")
    print(f"  - metrics_summary.png   (指标概要)")
    print(f"  - report.json           (数据报告)")

    return report_json


def plot_motion_features(
    features: np.ndarray,          # (T, C)
    feature_names: List[str],
    highlight_indices: Optional[List[int]] = None,
    title: str = "Motion Features",
    save_path: Optional[str] = None,
):
    """
    绘制运动特征时序曲线
    
    Args:
        features: (T, C) 特征矩阵
        feature_names: C 个特征名
        highlight_indices: 需要高亮的帧索引
        title: 标题
        save_path: 保存路径
    """
    T, C = features.shape
    n_cols = min(4, C)
    n_rows = (C + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i in range(C):
        ax = axes[i]
        ax.plot(features[:, i], 'b-', linewidth=1)
        ax.set_title(feature_names[i], fontsize=8)
        ax.set_xlabel('Frame')
        ax.grid(True, alpha=0.3)
        
        if highlight_indices:
            for idx in highlight_indices:
                ax.axvline(x=idx, color='r', linestyle='--', alpha=0.5)
    
    # 隐藏多余的子图
    for i in range(C, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    tn: int, fp: int, fn: int, tp: int,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    """绘制混淆矩阵"""
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Fall'])
    ax.set_yticklabels(['Normal', 'Fall'])
    
    # 添加数值标注
    for i in range(2):
        for j in range(2):
            text = f"{cm[i, j]}"
            ax.text(j, i, text, ha='center', va='center',
                   fontsize=16, fontweight='bold',
                   color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    plt.colorbar(im)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison(
    methods: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
):
    """
    多方法对比柱状图
    
    Args:
        methods: {method_name: {metric_name: value}}
    """
    n_methods = len(methods)
    n_metrics = len(metrics)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_methods
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    
    for i, (method_name, scores) in enumerate(methods.items()):
        values = [scores.get(m, 0) for m in metrics]
        offset = (i - n_methods/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=method_name, color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
