"""
模型评估报告生成器
===================
从训练检查点生成完整的可视化评估报告。

包含图表:
1. Training Curves (Loss / Accuracy / Precision / Recall / F1)
2. Learning Rate Schedule
3. Metrics Summary (横向柱状图)
4. Confusion Matrix

用法:
    # 从最新检查点生成报告
    python evaluate.py

    # 指定检查点
    python evaluate.py --checkpoint checkpoints/best.pth

    # 从多个检查点对比
    python evaluate.py --checkpoint checkpoints/best.pth --compare checkpoints/experiment1/latest.pth

    # 推理测试 + 生成评测指标
    python evaluate.py --infer --video test.mp4 --checkpoint checkpoints/best.pth
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.visualization import (
    plot_training_curves,
    plot_lr_schedule,
    plot_metrics_summary,
    plot_confusion_matrix,
    generate_evaluation_report,
)
from utils.metrics import compute_metrics, compute_model_efficiency


def merge_checkpoint_history(
    checkpoint_paths: list,
) -> list:
    """
    从多个检查点合并训练历史
    
    如果检查点是按 epoch 保存的，将它们按顺序合并
    """
    all_history = []

    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        entry = {}
        train_m = ckpt.get("train_metrics", {})
        val_m = ckpt.get("val_metrics", {})

        for k, v in train_m.items():
            if isinstance(v, (int, float)):
                entry[k] = v
        for k, v in val_m.items():
            if isinstance(v, (int, float)):
                entry[f"val_{k}"] = v

        entry["epoch"] = ckpt.get("epoch", 0)
        entry["lr"] = ckpt.get("lr", ckpt.get("scheduler_state_dict", {}).get("_last_lr", [0.001])[0] if ckpt.get("scheduler_state_dict") else 0.001)
        entry["checkpoint"] = os.path.basename(ckpt_path)

        all_history.append(entry)

    # 按 epoch 排序
    all_history.sort(key=lambda x: x.get("epoch", 0))
    return all_history


def inspect_model(checkpoint_path: str):
    """打印模型信息"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", {})

    total_params = sum(p.numel() for p in state.values())
    model_size = total_params * 4 / (1024 * 1024)

    print()
    print("=" * 50)
    print(f"  Model: {ckpt.get('config', {}).get('model', {}).get('name', 'FallDetector')}")
    print(f"  Version: {ckpt.get('config', {}).get('model', {}).get('version', 'standard')}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Best F1: {ckpt.get('best_f1', 0):.4f}")
    print(f"  Total Params: {total_params:,}")
    print(f"  Model Size (FP32): {model_size:.2f} MB")
    print(f"  Model Size (INT8): {model_size / 4:.2f} MB (estimated)")
    print("=" * 50)
    print()

    # 合规检查
    checks = []
    if total_params <= 20_000_000:
        checks.append("[PASS] Params <= 20M")
    else:
        checks.append("[FAIL] Params > 20M")
    if model_size <= 80:
        checks.append("[PASS] Model size <= 80MB (FP32)")
    else:
        checks.append("[FAIL] Model size > 80MB (FP32)")

    for c in checks:
        print(f"  {c}")
    print()


def run_inference_eval(
    checkpoint_path: str,
    video_path: str,
    device: str = "cpu",
):
    """运行推理并生成指标"""
    from pipeline.inference import FallDetectionPipeline
    from pipeline.detector import YOLOPoseDetector

    print(f"[Eval] 推理测试: {video_path}")

    # 加载模型
    from models.fall_detector import create_fall_detector
    model = create_fall_detector("standard")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 创建检测器
    try:
        detector = YOLOPoseDetector()
        detector.load_model()
    except:
        from pipeline.detector import YOLOPoseDetectorSim
        detector = YOLOPoseDetectorSim()

    # 创建管线
    pipeline = FallDetectionPipeline(
        detector=detector,
        fall_detector=model,
        save_video=True,
        output_dir="output",
    )

    # 处理视频
    result = pipeline.process_video(video_path)

    # 计算指标
    total_fall_events = len(result.fall_events)
    total_frames = result.total_frames
    fps = result.fps
    avg_latency = result.avg_latency_ms

    metrics = {
        "total_frames": total_frames,
        "fps": fps,
        "avg_latency_ms": avg_latency,
        "fall_events": total_fall_events,
        "total_time_s": result.total_time,
    }

    return metrics, result


def main():
    parser = argparse.ArgumentParser(description="模型评估报告生成器")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pth",
                       help="模型检查点路径")
    parser.add_argument("--output", type=str, default="eval_report",
                       help="报告输出目录")
    parser.add_argument("--compare", type=str, nargs="+", default=None,
                       help="对比的检查点路径（多个）")
    parser.add_argument("--infer", action="store_true",
                       help="运行推理测试")
    parser.add_argument("--video", type=str, default=None,
                       help="推理测试的视频路径")
    parser.add_argument("--inspect", action="store_true",
                       help="仅检查模型信息")
    parser.add_argument("--device", type=str, default="cpu",
                       help="推理设备")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[Error] 检查点不存在: {args.checkpoint}")
        print("请先训练模型: python train.py --epochs 50")
        return

    # ---- 仅检查模型信息 ----
    if args.inspect:
        inspect_model(args.checkpoint)
        return

    # ---- 生成评估报告 ----
    print(f"[Eval] 从 {args.checkpoint} 生成评估报告...")

    report = generate_evaluation_report(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
    )

    # ---- 对比多个检查点 ----
    if args.compare:
        print(f"\n[Eval] 对比: {args.checkpoint} vs {args.compare}")

        all_paths = [args.checkpoint] + args.compare
        history = merge_checkpoint_history(all_paths)

        if history:
            plot_training_curves(
                history,
                save_path=os.path.join(args.output, "comparison_curves.png"),
                title="Model Comparison",
            )

        # 对比指标柱状图
        from utils.visualization import plot_comparison
        methods = {}
        for entry in history:
            name = entry.get("checkpoint", f"epoch_{entry['epoch']}")
            methods[name] = {
                "accuracy": entry.get("accuracy", 0),
                "precision": entry.get("precision", 0),
                "recall": entry.get("recall", 0),
                "f1_score": entry.get("f1_score", 0),
            }

        if methods:
            plot_comparison(
                methods,
                title="Checkpoint Comparison",
                save_path=os.path.join(args.output, "comparison_bars.png"),
            )

    # ---- 推理测试 ----
    if args.infer and args.video:
        infer_metrics, result = run_inference_eval(
            args.checkpoint, args.video, args.device
        )

        print(f"\n[Eval] 推理结果:")
        for k, v in infer_metrics.items():
            print(f"  {k}: {v}")

        # 保存推理指标
        with open(os.path.join(args.output, "inference_results.json"), "w") as f:
            json.dump(infer_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[Eval] 完成！报告保存在: {args.output}/")
    print("  查看 training_curves.png 查看训练曲线")
    print("  查看 metrics_summary.png 查看指标概要")


if __name__ == "__main__":
    main()
