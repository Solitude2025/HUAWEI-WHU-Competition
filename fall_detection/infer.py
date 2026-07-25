"""
推理演示脚本
--------------
支持模式：
1. 视频文件推理：python infer.py --video input.mp4
2. 摄像头实时推理：python infer.py --webcam
3. 批量数据集推理：python infer.py --dir videos/

用法:
    # 视频推理
    python infer.py --video test.mp4 --checkpoint checkpoints/best.pth --save
    
    # 实时摄像头
    python infer.py --webcam --checkpoint checkpoints/best.pth
    
    # 批量评测
    python infer.py --dir test_videos/ --checkpoint checkpoints/best.pth --eval
"""

import os
import sys
import argparse
import yaml
import torch
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fall_detector import FallDetector, create_fall_detector
from pipeline.inference import FallDetectionPipeline
from pipeline.detector import YOLOPoseDetector


def main():
    parser = argparse.ArgumentParser(description="跌倒检测推理演示")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="模型检查点路径")
    parser.add_argument("--video", type=str, default=None,
                       help="输入视频路径")
    parser.add_argument("--webcam", action="store_true",
                       help="使用摄像头实时推理")
    parser.add_argument("--dir", type=str, default=None,
                       help="批量视频目录")
    parser.add_argument("--device", type=str, default="cpu",
                       help="推理设备 (cpu/cuda)")
    parser.add_argument("--save", action="store_true",
                       help="保存推理结果")
    parser.add_argument("--eval", action="store_true",
                       help="评测模式（计算指标）")
    parser.add_argument("--ir", action="store_true",
                       help="红外模式")
    parser.add_argument("--output", type=str, default="output",
                       help="输出目录")
    parser.add_argument("--version", type=str, default="standard",
                       choices=["standard", "light", "large"],
                       help="模型版本")
    parser.add_argument("--no_display", action="store_true",
                       help="不显示实时画面")
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 创建跌倒检测模型
    print(f"[Infer] 加载模型 (version={args.version})...")
    fall_detector = create_fall_detector(version=args.version)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        fall_detector.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Infer] 加载检查点: epoch={checkpoint['epoch']}, "
              f"F1={checkpoint.get('best_f1', 'N/A')}")
    
    fall_detector.eval()
    
    # 统计模型信息
    total_params = sum(p.numel() for p in fall_detector.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)
    print(f"[Infer] 模型参数: {total_params:,} | 大小: {model_size_mb:.2f} MB (FP32)")
    
    # 创建检测器
    try:
        detector = YOLOPoseDetector(
            input_size=tuple(config.get("detector", {}).get("input_size", [640, 384])),
        )
        detector.load_model()
        print("[Infer] YOLOv8n-Pose 加载成功")
    except Exception as e:
        print(f"[Infer] YOLO 加载失败 ({e})，使用模拟模式")
        from pipeline.detector import YOLOPoseDetectorSim
        detector = YOLOPoseDetectorSim()
    
    # 创建推理管线
    pipeline = FallDetectionPipeline(
        detector=detector,
        fall_detector=fall_detector,
        sequence_length=config.get("pipeline", {}).get("sequence_length", 32),
        detection_interval=config.get("pipeline", {}).get("detection_interval", 2),
        ir_mode=args.ir,
        save_video=args.save,
        output_dir=args.output,
    )
    
    # 打印管线信息
    stats = pipeline.get_statistics()
    print(f"[Infer] 管线配置: seq_len={stats['sequence_length']}, "
          f"det_interval={stats['detection_interval']}, "
          f"IR={stats['ir_mode']}")
    print()
    
    # --- 摄像头模式 ---
    if args.webcam:
        print("[Infer] 启动摄像头实时推理 (按 'q' 退出)...")
        for frame_result in pipeline.process_webcam(camera_id=0):
            # 显示结果已在管线内部处理
            print(f"\rFrame {frame_result.frame_idx} | "
                  f"Alarms: {sum(frame_result.alarms.values())}", end="")
    
    # --- 单视频模式 ---
    elif args.video:
        if not os.path.exists(args.video):
            print(f"[Error] 视频不存在: {args.video}")
            return
        
        print(f"[Infer] 处理视频: {args.video}")
        result = pipeline.process_video(args.video)
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"推理完成")
        print(f"  总帧数: {result.total_frames}")
        print(f"  总耗时: {result.total_time:.2f}s")
        print(f"  FPS: {result.fps:.2f}")
        print(f"  平均延迟: {result.avg_latency_ms:.2f}ms")
        print(f"  跌倒事件数: {len(result.fall_events)}")
        
        for i, event in enumerate(result.fall_events):
            print(f"  事件 {i+1}: Person {event.person_id}, "
                  f"帧 {event.start_frame}-{event.end_frame}, "
                  f"最大概率: {event.max_probability:.3f}, "
                  f"时间: {event.timestamp:.1f}s")
        print(f"{'='*60}")
        
        # 检查赛题合规
        if result.avg_latency_ms > 100:
            print(f"\n[WARNING] 平均延迟 {result.avg_latency_ms:.1f}ms 超过 100ms 限制!")
        else:
            print(f"\n[OK] 平均延迟 {result.avg_latency_ms:.1f}ms 满足要求!")
    
    # --- 批量模式 ---
    elif args.dir:
        import glob
        video_exts = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
        videos = []
        for ext in video_exts:
            videos.extend(glob.glob(os.path.join(args.dir, ext)))
        
        if not videos:
            print(f"[Error] 目录中无视频: {args.dir}")
            return
        
        print(f"[Infer] 批量处理 {len(videos)} 个视频...")
        
        metrics = pipeline.evaluate_dataset(videos)
        
        print(f"\n{'='*60}")
        print(f"批量评测完成")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")
    
    else:
        parser.print_help()
        print("\n示例:")
        print("  python infer.py --video test.mp4 --save")
        print("  python infer.py --webcam")
        print("  python infer.py --dir test_videos/ --eval")


if __name__ == "__main__":
    main()
