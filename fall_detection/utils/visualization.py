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
