"""
数据下载与预处理脚本
======================
一键下载两个参考数据集，并用 YOLOv8n-Pose 提取关键点，
转为项目训练所需的 keypoints/bboxes/labels 格式。

支持的参考数据集:
1. OmniFall (HuggingFace) - 综合跌倒检测基准
2. Fall Video Dataset (Kaggle) - 多来源编译

用法:
    # 完整流程（下载两个数据集 + 提取关键点 + 训练/验证集划分）
    python prepare_data.py --all

    # 只处理 OmniFall（推荐，数据质量高）
    python prepare_data.py --omnifall

    # 只处理 Kaggle 数据集
    python prepare_data.py --kaggle

    # 仅从已有视频目录提取关键点
    python prepare_data.py --video_dir path/to/videos --labels path/to/labels.txt

    # 指定输出目录
    python prepare_data.py --all --output data/processed
"""

import os
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm
import subprocess
from pathlib import Path
import tempfile
import shutil
import zipfile

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """检查并安装所需依赖"""
    deps = {
        "datasets": "datasets",
        "torch": "torch",
        "ultralytics": "ultralytics",
        "opencv": "opencv-python",
        "kagglehub": "kagglehub",
    }

    missing = []
    for name, pkg in deps.items():
        try:
            __import__(name.replace("-", "_").split(".")[0])
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[DataPrep] 安装缺失依赖: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        print("[DataPrep] 依赖安装完成")


def load_omnifall_dataset(
    output_dir: str,
    config: str = "of-sta-cs",
    max_samples: int = None,
):
    """
    加载 OmniFall 数据集
    
    OmniFall 由 8 个公开跌倒检测数据集整合而成，
    支持 staged (实验室场景) 和 in-the-wild (野外场景)。
    
    Args:
        output_dir: 输出目录
        config: 数据集配置
            - "of-sta-cs": 仅实验室场景 (recommended for training)
            - "of-sta-to-all-cs": 实验室→全场景
            - "of-syn": 合成数据
        max_samples: 最大样本数（用于测试）
    
    数据格式:
        OmniFall 样本包含:
        - path: 视频/帧序列路径
        - label: 0=非跌倒, 1=跌倒
        - start/end: 事件时间戳
    """
    print(f"\n{'='*60}")
    print(f"[DataPrep] 加载 OmniFall 数据集 (config={config})")
    print(f"{'='*60}")

    from datasets import load_dataset

    # 加载数据集
    ds = load_dataset("simplexsigil2/omnifall", config, split="train")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    
    print(f"[DataPrep] OmniFall 加载完成: {len(ds)} 条记录")
    print(f"[DataPrep] 列: {ds.column_names}")
    print(f"[DataPrep] 标签分布: {ds.info.features['label'].names if hasattr(ds.info.features, 'label') else 'binary'}")

    # 统计标签分布
    labels = ds["label"]
    fall_count = sum(1 for l in labels if l == 1)
    normal_count = sum(1 for l in labels if l == 0)
    print(f"[DataPrep]   跌倒样本: {fall_count}")
    print(f"[DataPrep]   非跌倒: {normal_count}")

    # 保存元数据
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "omnifall_metadata.json")

    # 只保存必要字段
    metadata = []
    for i in range(len(ds)):
        metadata.append({
            "index": i,
            "path": ds[i].get("path", ""),
            "label": int(ds[i]["label"]),
            "start": ds[i].get("start", 0),
            "end": ds[i].get("end", 0),
        })

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DataPrep] 元数据保存到: {meta_path}")

    return ds


def load_omnifall_with_video(
    output_dir: str,
    config: str = "of-syn",
    max_samples: int = 100,
):
    """
    加载 OmniFall 合成数据集并获取视频路径
    
    of-syn 合成数据包含自动化生成的跌倒视频，
    可以直接下载视频文件。
    
    of-syn 约 9.1GB，下载可能需要一些时间。
    """
    print(f"\n{'='*60}")
    print(f"[DataPrep] 下载 OmniFall 合成视频 (config={config})")
    print(f"{'='*60}")

    try:
        import omnifall

        # 加载带视频的数据集
        ds = omnifall.load(config, video=True)

        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        # 获取视频路径
        video_paths = ds["video"]
        labels = ds["label"]

        print(f"[DataPrep] 获取到 {len(video_paths)} 个视频")
        print(f"[DataPrep]   跌倒: {sum(1 for l in labels if l == 1)}")
        print(f"[DataPrep]   非跌倒: {sum(1 for l in labels if l == 0)}")

        # 保存视频路径列表
        video_list = []
        for i, (vp, lb) in enumerate(zip(video_paths, labels)):
            if isinstance(vp, str) and os.path.exists(vp):
                video_list.append({
                    "video_path": vp,
                    "label": int(lb),
                })

        # 保存到文件
        os.makedirs(output_dir, exist_ok=True)
        list_path = os.path.join(output_dir, "omnifall_videos.json")
        with open(list_path, "w") as f:
            json.dump(video_list, f, indent=2)

        print(f"[DataPrep] 视频列表保存到: {list_path}")

        return video_list

    except ImportError:
        print("[DataPrep] omnifall 包未安装，回退到 datasets 模式")
        return None
    except Exception as e:
        print(f"[DataPrep] 加载视频失败: {e}")
        return None


def download_kaggle_dataset(output_dir: str):
    """
    下载 Kaggle Fall Video Dataset
    
    需要 Kaggle API key (kaggle.json)。
    如果没有，会提示用户手动下载。
    
    Kaggle 数据集由多个来源的跌倒检测视频编译而成。
    """
    print(f"\n{'='*60}")
    print(f"[DataPrep] 下载 Fall Video Dataset (Kaggle)")
    print(f"{'='*60}")

    kaggle_path = os.path.join(output_dir, "kaggle_fall")

    try:
        import kagglehub

        # 下载数据集
        print("[DataPrep] 使用 kagglehub 下载...")
        path = kagglehub.dataset_download("payutch/fall-video-dataset")
        print(f"[DataPrep] 下载到: {path}")

        # 复制到输出目录
        if os.path.isdir(path):
            if os.path.exists(kaggle_path):
                shutil.rmtree(kaggle_path)
            shutil.copytree(path, kaggle_path)

        return kaggle_path

    except ImportError:
        print("[DataPrep] kagglehub 未安装")
    except Exception as e:
        print(f"[DataPrep] kagglehub 下载失败: {e}")

    # 回退方案：手动下载
    print(f"""
[DataPrep] ======================== 手动下载指引 ========================
[DataPrep] 自动下载失败，请手动下载:
[DataPrep] 
[DataPrep] 1. 打开: https://www.kaggle.com/datasets/payutch/fall-video-dataset
[DataPrep] 2. 点击 "Download" 按钮
[DataPrep] 3. 将下载的 zip 解压到:
[DataPrep]    {kaggle_path}
[DataPrep] ==============================================================
""")

    return kaggle_path if os.path.exists(kaggle_path) else None


def process_video_to_keypoints(
    video_path: str,
    label: int,
    output_dir: str,
    detector=None,
    sequence_length: int = 32,
    stride: int = 16,
    video_id: str = None,
    max_frames: int = None,
):
    """
    处理单个视频: 用 YOLOv8n-Pose 提取关键点 → 保存
    
    Args:
        video_path: 视频文件路径
        label: 0=正常, 1=跌倒
        output_dir: 输出目录
        detector: YOLO detector 实例
        sequence_length: 时序窗口长度
        stride: 滑动步幅
        video_id: 视频唯一标识
        max_frames: 最多处理的帧数
    
    Returns:
        num_sequences: 生成的序列数
    """
    import cv2

    if video_id is None:
        video_id = os.path.splitext(os.path.basename(video_path))[0]

    # 创建输出目录
    person_dir = os.path.join(output_dir, video_id, "person_0")
    os.makedirs(person_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[DataPrep]   无法打开视频: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    print(f"[DataPrep]   视频: {os.path.basename(video_path)} | "
          f"{total_frames} 帧 | {fps:.1f} fps | "
          f"标签: {'跌倒' if label == 1 else '正常'}")

    if max_frames:
        total_frames = min(total_frames, max_frames)

    # 读取所有帧并检测
    all_keypoints = []
    all_bboxes = []
    all_labels = []

    frame_idx = 0
    while True:
        if max_frames and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 检测
        h, w = frame.shape[:2]
        detections = detector.detect(frame, (h, w))

        if detections:
            # 取置信度最高的人体
            best_det = max(detections, key=lambda d: d.confidence)
            kp = best_det.keypoints
            bb = best_det.bbox
        else:
            # 没检测到 → 用空数据
            kp = np.zeros((17, 3))
            bb = np.zeros(4)

        all_keypoints.append(kp)
        all_bboxes.append(bb)
        all_labels.append(label)

        frame_idx += 1

        if frame_idx % 500 == 0:
            print(f"[DataPrep]     已处理 {frame_idx}/{total_frames} 帧")

    cap.release()

    if len(all_keypoints) == 0:
        print(f"[DataPrep]   无有效帧")
        return 0

    # 转为 numpy 数组
    all_keypoints = np.stack(all_keypoints)  # (T, 17, 3)
    all_bboxes = np.stack(all_bboxes)        # (T, 4)
    all_labels = np.array(all_labels)         # (T,)

    # 保存完整的序列（用于训练时滑动窗口）
    np.save(os.path.join(person_dir, "keypoints.npy"), all_keypoints)
    np.save(os.path.join(person_dir, "bboxes.npy"), all_bboxes)
    np.save(os.path.join(person_dir, "labels.npy"), all_labels)

    # 计算能生成的序列数
    num_sequences = max(0, (len(all_keypoints) - sequence_length) // stride + 1)
    print(f"[DataPrep]   保存完成: {len(all_keypoints)} 帧, "
          f"可生成 {num_sequences} 个序列")

    return num_sequences


def process_video_list(
    video_list: list,
    output_dir: str,
    max_videos: int = None,
    detector=None,
):
    """
    批量处理视频列表
    
    Args:
        video_list: [(video_path, label), ...]
        output_dir: 输出目录
        max_videos: 最多处理的视频数
        detector: YOLO detector
    """
    total_sequences = 0
    processed = 0
    failed = 0

    if max_videos:
        video_list = video_list[:max_videos]

    print(f"\n{'='*60}")
    print(f"[DataPrep] 批量处理 {len(video_list)} 个视频")
    print(f"{'='*60}")

    old_stdout = None  # 抑制 ultralytics 过多输出

    for i, item in enumerate(tqdm(video_list, desc="处理视频")):
        if isinstance(item, dict):
            video_path = item.get("video_path", item.get("path", ""))
            label = item.get("label", 0)
        elif isinstance(item, (list, tuple)):
            video_path, label = item[0], item[1]
        else:
            video_path, label = item, 0

        if not os.path.exists(video_path):
            failed += 1
            continue

        try:
            n_seq = process_video_to_keypoints(
                video_path=video_path,
                label=label,
                output_dir=output_dir,
                detector=detector,
                video_id=f"video_{i:04d}",
            )
            total_sequences += n_seq
            processed += 1
        except Exception as e:
            print(f"[DataPrep]   处理失败 [{video_path}]: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"[DataPrep] 批量处理完成:")
    print(f"  成功: {processed}, 失败: {failed}")
    print(f"  生成序列: {total_sequences}")
    print(f"  数据保存: {output_dir}")
    print(f"{'='*60}")

    return total_sequences


def split_train_val(
    data_dir: str,
    val_ratio: float = 0.15,
):
    """
    将数据划分为训练集和验证集
    
    按视频维度划分（同视频的所有序列在同一 split 中），
    避免数据泄露。
    """
    print(f"\n[DataPrep] 划分训练/验证集 (val_ratio={val_ratio})")

    # 收集所有视频目录
    videos = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not videos:
        print("[DataPrep]   无视频数据，跳过划分")
        return

    # 随机打乱
    import random
    random.seed(42)
    random.shuffle(videos)

    # 划分
    n_val = max(1, int(len(videos) * val_ratio))
    val_videos = videos[:n_val]
    train_videos = videos[n_val:]

    print(f"[DataPrep]   训练集: {len(train_videos)} 视频")
    print(f"[DataPrep]   验证集: {len(val_videos)} 视频")

    # 创建软链接/复制到 train/val 目录
    train_dir = os.path.join(os.path.dirname(data_dir), "train")
    val_dir = os.path.join(os.path.dirname(data_dir), "val")

    # 清除旧的
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 复制
    def copy_videos(video_list, target_dir):
        for vid in video_list:
            src = os.path.join(data_dir, vid)
            dst = os.path.join(target_dir, vid)
            shutil.copytree(src, dst)

    copy_videos(train_videos, train_dir)
    copy_videos(val_videos, val_dir)

    print(f"[DataPrep]   训练数据: {train_dir} ({len(train_videos)} 视频)")
    print(f"[DataPrep]   验证数据: {val_dir} ({len(val_videos)} 视频)")


def main():
    parser = argparse.ArgumentParser(description="下载并预处理跌倒检测数据集")
    parser.add_argument("--all", action="store_true",
                       help="完整流程（推荐）")
    parser.add_argument("--omnifall", action="store_true",
                       help="处理 OmniFall 数据集")
    parser.add_argument("--kaggle", action="store_true",
                       help="处理 Kaggle Fall Video Dataset")
    parser.add_argument("--video_dir", type=str, default=None,
                       help="自定义视频目录")
    parser.add_argument("--labels", type=str, default=None,
                       help="自定义标签文件")
    parser.add_argument("--output", type=str, default=None,
                       help="输出目录（默认: data/raw）")
    parser.add_argument("--max_videos", type=int, default=None,
                       help="最大处理视频数（测试用）")
    parser.add_argument("--split", action="store_true", default=True,
                       help="划分训练/验证集")
    parser.add_argument("--yolo_model", type=str, default="yolov8n-pose.pt",
                       help="YOLO 模型路径")
    parser.add_argument("--sequence_length", type=int, default=32,
                       help="时序窗口长度")
    parser.add_argument("--stride", type=int, default=16,
                       help="滑动步幅")
    parser.add_argument("--device", type=str, default="cuda",
                       help="YOLO 推理设备")

    args = parser.parse_args()

    # 设置输出目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output or os.path.join(project_root, "data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    # 检查依赖
    check_dependencies()

    # 初始化 YOLO 检测器
    print(f"\n[DataPrep] 初始化 YOLOv8n-Pose 检测器...")
    try:
        from ultralytics import YOLO
        # 允许从本地加载或自动下载
        yolo_model_path = args.yolo_model
        if not os.path.exists(yolo_model_path):
            print(f"[DataPrep] 模型不存在，将自动下载: {yolo_model_path}")
        model = YOLO(yolo_model_path)
        print(f"[DataPrep] YOLO 加载完成")

        # 创建检测器包装类以供 process_video_to_keypoints 使用
        class DetectorWrapper:
            def __init__(self, model, conf=0.25, device="cpu"):
                self.model = model
                self.conf = conf
                self.device = device

            def detect(self, image, original_shape=None):
                h_orig, w_orig = image.shape[:2]
                results = self.model(
                    image, conf=self.conf, verbose=False, device=self.device
                )
                detections = []
                for result in results:
                    if result.keypoints is None:
                        continue
                    boxes = result.boxes
                    kp_data = result.keypoints
                    if boxes is None or len(boxes) == 0:
                        continue
                    for i in range(len(boxes)):
                        from pipeline.detector import PersonDetection
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        bbox_norm = np.array([
                            box[0] / w_orig, box[1] / h_orig,
                            box[2] / w_orig, box[3] / h_orig,
                        ])
                        kp = kp_data.xy[i].cpu().numpy()
                        kp_conf = kp_data.conf[i].cpu().numpy()
                        kp_full = np.concatenate([
                            kp / np.array([w_orig, h_orig]),
                            kp_conf[:, None],
                        ], axis=1)
                        det = PersonDetection(
                            bbox=bbox_norm, keypoints=kp_full, confidence=conf
                        )
                        detections.append(det)
                return detections

        detector = DetectorWrapper(model, conf=0.25, device=args.device)

    except Exception as e:
        print(f"[DataPrep] YOLO 初始化失败: {e}")
        print(f"[DataPrep] 将使用模拟检测器")
        from pipeline.detector import YOLOPoseDetectorSim
        detector = YOLOPoseDetectorSim()

    # ---- OmniFall 处理 ----
    if args.omnifall or args.all:
        omnifall_raw_dir = os.path.join(output_dir, "omnifall")
        os.makedirs(omnifall_raw_dir, exist_ok=True)

        # 第一步：加载元数据
        ds = load_omnifall_dataset(
            output_dir=omnifall_raw_dir,
            config="of-sta-cs",
            max_samples=args.max_videos,
        )

        # 第二步：尝试获取视频
        video_list = load_omnifall_with_video(
            output_dir=omnifall_raw_dir,
            config="of-syn",
            max_samples=args.max_videos or 500,
        )

        # 如果获取到视频，提取关键点
        if video_list and len(video_list) > 0:
            kp_output = os.path.join(omnifall_raw_dir, "keypoints")
            process_video_list(
                video_list,
                output_dir=kp_output,
                detector=detector,
            )

    # ---- Kaggle 处理 ----
    if args.kaggle or args.all:
        kaggle_path = download_kaggle_dataset(output_dir)
        if kaggle_path and os.path.isdir(kaggle_path):
            # 扫描视频文件
            video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
            video_files = []
            for root, _, files in os.walk(kaggle_path):
                for f in files:
                    if f.lower().endswith(video_exts):
                        video_files.append(os.path.join(root, f))

            print(f"[DataPrep] Kaggle 数据集中找到 {len(video_files)} 个视频")

            if video_files:
                kp_output = os.path.join(output_dir, "kaggle", "keypoints")
                video_list = [(vp, 1 if "fall" in os.path.basename(vp).lower() else 0)
                             for vp in video_files]
                process_video_list(video_list, output_dir=kp_output, detector=detector)

    # ---- 自定义视频目录 ----
    if args.video_dir and os.path.isdir(args.video_dir):
        print(f"\n[DataPrep] 处理自定义视频目录: {args.video_dir}")

        video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
        video_files = []
        for root, _, files in os.walk(args.video_dir):
            for f in files:
                if f.lower().endswith(video_exts):
                    video_files.append(os.path.join(root, f))

        if not video_files:
            print(f"[DataPrep]   目录中无视频文件")
        else:
            # 尝试读取标签
            label_map = {}
            if args.labels and os.path.exists(args.labels):
                with open(args.labels, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            label_map[parts[0]] = int(parts[1])

            custom_output = os.path.join(output_dir, "custom", "keypoints")
            video_list = [
                (vp, label_map.get(os.path.basename(vp), 0))
                for vp in video_files
            ]
            process_video_list(video_list, output_dir=custom_output, detector=detector)

    # ---- 划分训练/验证集 ----
    if args.split:
        # 在所有 keypoints 目录中查找
        kp_dirs = [
            os.path.join(output_dir, "omnifall", "keypoints"),
            os.path.join(output_dir, "kaggle", "keypoints"),
            os.path.join(output_dir, "custom", "keypoints"),
        ]
        for kp_dir in kp_dirs:
            if os.path.isdir(kp_dir):
                video_count = len([d for d in os.listdir(kp_dir)
                                   if os.path.isdir(os.path.join(kp_dir, d))])
                if video_count > 0:
                    print(f"\n[DataPrep] 在 {kp_dir} 中找到 {video_count} 个视频")
                    split_train_val(kp_dir)

    # ---- 创建 data/train 和 data/val 的快捷方式 ----
    # 如果项目中有 data/train 和 data/val 的合并需求
    train_sources = [
        os.path.join(output_dir, "omnifall", "keypoints", "train"),
        os.path.join(output_dir, "kaggle", "keypoints", "train"),
        os.path.join(output_dir, "custom", "keypoints", "train"),
    ]
    val_sources = [
        os.path.join(output_dir, "omnifall", "keypoints", "val"),
        os.path.join(output_dir, "kaggle", "keypoints", "val"),
        os.path.join(output_dir, "custom", "keypoints", "val"),
    ]

    final_train = os.path.join(project_root, "data", "train")
    final_val = os.path.join(project_root, "data", "val")

    for final_dir, sources in [(final_train, train_sources), (final_val, val_sources)]:
        os.makedirs(final_dir, exist_ok=True)
        dest_idx = 0
        for src in sources:
            if os.path.isdir(src):
                for v in os.listdir(src):
                    v_src = os.path.join(src, v)
                    if os.path.isdir(v_src):
                        v_dst = os.path.join(final_dir, f"{dest_idx:04d}")
                        shutil.copytree(v_src, v_dst)
                        dest_idx += 1

    print(f"\n{'='*60}")
    print(f"[DataPrep] 数据处理完成!")
    print(f"  训练数据: {final_train}")
    print(f"  验证数据: {final_val}")
    print(f"\n  现在可以运行: python train.py --data_dir data/train --val_dir data/val --epochs 100")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
