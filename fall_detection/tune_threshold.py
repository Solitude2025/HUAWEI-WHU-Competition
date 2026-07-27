"""
判定阈值调优脚本
======================
在验证集上扫描 TCN 帧级判定阈值，输出：
1. precision / recall / F1 随阈值变化曲线（eval_report/threshold_tuning.png）
2. 推荐阈值（eval_report/threshold_tuning.json）：
   - best_f1: F1 最大点
   - safe:    recall >= 0.90 前提下 precision 最高点（跌倒检测优先保 recall）

用法:
    python tune_threshold.py --checkpoint checkpoints/best.pth
    python tune_threshold.py --checkpoint checkpoints/best.pth --val_dir data/val
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fall_detector import create_fall_detector
from data.dataset import create_dataloader


def collect_probs(model, val_loader, device):
    """收集验证集所有帧的预测概率与真实标签"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            kp = batch["keypoints"].to(device)
            bb = batch["bboxes"].to(device)
            out = model(kp, bb)
            probs = out["fall_prob"] if isinstance(out, dict) else out
            all_probs.append(probs.squeeze(-1).cpu().numpy().ravel())
            all_labels.append(batch["labels"].numpy().ravel())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def scan_thresholds(probs, labels, n_points=81):
    """扫描阈值，返回每个阈值下的 (precision, recall, f1)"""
    rows = []
    for thr in np.linspace(0.05, 0.95, n_points):
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        rows.append({"threshold": round(float(thr), 3),
                     "precision": round(prec, 4),
                     "recall": round(rec, 4),
                     "f1": round(f1, 4)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="判定阈值调优")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--version", type=str, default="standard")
    parser.add_argument("--recall_target", type=float, default=0.90,
                        help="safe 阈值的 recall 下限（默认 0.90）")
    parser.add_argument("--output", type=str, default="eval_report")
    args = parser.parse_args()

    device = "cpu"

    # 加载模型
    model = create_fall_detector(version=args.version)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = {k: v for k, v in ckpt["model_state_dict"].items()
                  if not k.endswith(".total_ops") and not k.endswith(".total_params")}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print(f"[Tune] 加载检查点: epoch={ckpt.get('epoch')}")

    # 验证集（不增强）
    val_loader = create_dataloader(
        args.val_dir, batch_size=64, mode="val",
        use_augmentation=False, num_workers=4,
    )

    print("[Tune] 收集验证集预测概率...")
    probs, labels = collect_probs(model, val_loader, device)
    print(f"[Tune] 共 {len(probs)} 帧 (跌倒帧 {int(labels.sum())}, "
          f"占 {labels.mean()*100:.1f}%)")

    rows = scan_thresholds(probs, labels)

    # 推荐阈值
    best_f1 = max(rows, key=lambda r: r["f1"])
    safe_candidates = [r for r in rows if r["recall"] >= args.recall_target]
    safe = max(safe_candidates, key=lambda r: r["precision"]) if safe_candidates else None

    print(f"\n[结果] 当前默认阈值 0.50:")
    cur = min(rows, key=lambda r: abs(r["threshold"] - 0.50))
    print(f"  P={cur['precision']:.3f} R={cur['recall']:.3f} F1={cur['f1']:.3f}")
    print(f"[结果] 最佳 F1 阈值 {best_f1['threshold']}:")
    print(f"  P={best_f1['precision']:.3f} R={best_f1['recall']:.3f} F1={best_f1['f1']:.3f}")
    if safe:
        print(f"[结果] 保守阈值 {safe['threshold']} (recall>={args.recall_target}):")
        print(f"  P={safe['precision']:.3f} R={safe['recall']:.3f} F1={safe['f1']:.3f}")

    # 保存
    os.makedirs(args.output, exist_ok=True)
    result = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "n_frames": len(probs),
        "best_f1": best_f1,
        "safe": safe,
        "curve": rows,
    }
    json_path = os.path.join(args.output, "threshold_tuning.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 画图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    thrs = [r["threshold"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(thrs, [r["precision"] for r in rows], label="Precision")
    plt.plot(thrs, [r["recall"] for r in rows], label="Recall")
    plt.plot(thrs, [r["f1"] for r in rows], label="F1")
    plt.axvline(best_f1["threshold"], ls="--", c="gray",
                label=f"best F1 @ {best_f1['threshold']}")
    if safe:
        plt.axvline(safe["threshold"], ls=":", c="red",
                    label=f"safe @ {safe['threshold']}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning on Validation Set")
    plt.legend()
    plt.grid(alpha=0.3)
    png_path = os.path.join(args.output, "threshold_tuning.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")

    print(f"\n[Tune] 已保存: {json_path} 和 {png_path}")


if __name__ == "__main__":
    main()
