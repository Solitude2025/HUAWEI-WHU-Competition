"""
数据下载与预处理脚本
======================
下载两个参考数据集，用 YOLOv8n-Pose 提取关键点，
转为项目训练所需的 keypoints/bboxes/labels 格式。

用法:
    # 从 OmniFall（HuggingFace）下载，提取关键点
    python prepare_data.py --omnifall

    # 从 Kaggle 下载
    python prepare_data.py --kaggle

    # 生成演示用的合成数据（无网络也可用）
    python prepare_data.py --demo

    # 完整流程
    python prepare_data.py --all
"""

import os
import sys
import json
import numpy as np
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _download_file(url: str, save_path: str, desc: str = ""):
    """
    下载文件并显示进度条
    
    用 requests + tqdm，保证进度条可见。
    支持断点续传（如果本地已有部分文件）。
    """
    import requests
    from tqdm import tqdm as _tqdm

    # 检查已有部分下载
    headers = {}
    mode = "wb"
    if os.path.exists(save_path):
        existing_size = os.path.getsize(save_path)
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
    else:
        existing_size = 0

    # 先发 HEAD 请求获取总大小
    resp_head = requests.head(url, allow_redirects=True, timeout=10)
    total_size = int(resp_head.headers.get("content-length", 0))

    if existing_size >= total_size:
        print(f"   文件已存在 ({total_size / 1024**3:.2f} GB)")
        return

    # 下载
    resp = requests.get(url, stream=True, timeout=30, headers=headers)
    resp.raise_for_status()

    desc_text = desc or os.path.basename(save_path)
    unit = "GB" if total_size > 1024**3 else "MB"
    div = 1024**3 if unit == "GB" else 1024**2
    total_display = total_size / div

    with _tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"  {desc_text}",
        initial=existing_size,
        ascii=True,
        ncols=80,
    ) as pbar:
        with open(save_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    actual_size = os.path.getsize(save_path)
    print(f"   完成: {actual_size / 1024**3:.2f} GB")


def check_dependencies():
    """检查并安装所需依赖"""
    deps = {"datasets": "datasets", "ultralytics": "ultralytics", "requests": "requests"}
    missing = []
    for name, pkg in deps.items():
        try:
            __import__(name.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[DataPrep] 安装缺失依赖: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "-q"])


def setup_yolo():
    """初始化 YOLOv8n-Pose 检测器"""
    from ultralytics import YOLO
    model_path = "yolov8n-pose.pt"
    if not os.path.exists(model_path):
        print("[DataPrep] 下载 YOLOv8n-Pose 模型...")
    model = YOLO(model_path)
    print(f"[DataPrep] YOLO 加载完成")

    # 检测器包装类
    class Detector:
        def __init__(self, model, conf=0.25):
            self.model = model
            self.conf = conf

        def detect(self, image):
            from pipeline.detector import PersonDetection
            import torch
            h_orig, w_orig = image.shape[:2]
            results = self.model(image, conf=self.conf, verbose=False)
            detections = []
            for result in results:
                if result.keypoints is None:
                    continue
                boxes = result.boxes
                kp_data = result.keypoints
                if boxes is None or len(boxes) == 0:
                    continue
                for i in range(len(boxes)):
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
                    detections.append(PersonDetection(bbox=bbox_norm, keypoints=kp_full, confidence=conf))
            return detections

    return Detector(model)


def process_video(video_path, label, output_dir, detector, video_id=None, max_frames=300):
    """处理单个视频：提取关键点 → 保存 npy"""
    import cv2
    video_id = video_id or os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [!] 无法打开: {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    kps, bbs = [], []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        dets = detector.detect(frame)
        if dets:
            best = max(dets, key=lambda d: d.confidence)
            kps.append(best.keypoints)
            bbs.append(best.bbox)
        else:
            kps.append(np.zeros((17, 3)))
            bbs.append(np.zeros(4))
        if (i + 1) % 100 == 0:
            print(f"    帧 {i+1}/{total}")
    cap.release()

    if len(kps) == 0:
        return 0

    person_dir = os.path.join(output_dir, video_id, "person_0")
    os.makedirs(person_dir, exist_ok=True)
    np.save(os.path.join(person_dir, "keypoints.npy"), np.stack(kps))
    np.save(os.path.join(person_dir, "bboxes.npy"), np.stack(bbs))
    np.save(os.path.join(person_dir, "labels.npy"), np.full(len(kps), label))
    return (len(kps) - 32) // 16 + 1  # 可生成的序列数


def download_omnifall(output_dir, detector, max_videos=500):
    """
    下载 OmniFall 合成视频 (HuggingFace) + 提取关键点
    
    流程:
    1. 下载 labels/of-syn.csv（标签）
    2. 下载 data_files/omnifall-synthetic_av1.tar（视频存档, ~9.72GB）
    3. 解压 tar
    4. 用 YOLOv8n-Pose 逐帧提取关键点
    5. 保存为 .npy 格式
    
    视频存档包含 12,000 个合成跌倒/日常动作视频，
    覆盖年龄、体型、环境、视角等多维度变化。
    """
    print(f"\n{'='*60}")
    print(f"[DataPrep] 下载 OmniFall 合成视频 (9.72 GB)")
    print(f"{'='*60}")
    
    import tarfile, csv
    
    omni_dir = os.path.join(output_dir, "omnifall")
    os.makedirs(omni_dir, exist_ok=True)
    
    # ---- 1. 下载标签 ----
    labels_path = os.path.join(omni_dir, "of-syn.csv")
    if not os.path.exists(labels_path):
        print("[1/4] 下载标签文件...")
        _download_file(
            url="https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/labels/of-syn.csv",
            save_path=labels_path,
            desc="标签文件",
        )
    else:
        print(f"[1/4] 标签文件已存在: {labels_path}")
    
    # 读取标签
    label_map = {}  # video_filename -> is_fall (0/1)
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("path", "")
            label = int(row.get("label", 0))
            if path:
                fname = os.path.basename(path)
                # OmniFall 16类标签:
                #   0=background/stand/walk(正常)
                #   1=fall(正在跌倒) ← 跌倒正样本
                #   2=fallen(已倒地) ← 跌倒正样本
                #   3=sit_down(坐下)  4=sitting(已坐下)
                #   5=lie_down(躺下)  6=lying(躺着)
                #   7=stand_up(站起)  8=standing(站着)
                #   9=walking(行走)   10-15=其他
                is_fall = 1 if label in (1, 2) else 0
                label_map[fname] = is_fall
    
    print(f"   标签加载: {len(label_map)} 条")
    
    # ---- 2. 下载视频存档 ----
    tar_path = os.path.join(omni_dir, "omnifall-synthetic_av1.tar")
    if not os.path.exists(tar_path):
        print("[2/4] 下载视频存档 (~9.72 GB)...")
        _download_file(
            url="https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/data_files/omnifall-synthetic_av1.tar",
            save_path=tar_path,
            desc="omnifall-synthetic_av1.tar",
        )
        print(f"   下载完成: {os.path.getsize(tar_path) / 1024**3:.2f} GB")
    else:
        print(f"[2/4] 视频存档已存在: {os.path.getsize(tar_path) / 1024**3:.2f} GB")
    
    # ---- 3. 解压 ----
    video_extract_dir = os.path.join(omni_dir, "videos")
    
    # 检查用户是否已手动解压到其他目录
    alt_extract_dir = os.path.join(omni_dir, "omnifall-synthetic_av1")
    if os.path.isdir(alt_extract_dir):
        alt_count = 0
        for root, _, files in os.walk(alt_extract_dir):
            alt_count += sum(1 for f in files if f.endswith(".mp4"))
        if alt_count >= max_videos:
            print(f"[3/4] 检测到用户已手动解压: {alt_extract_dir} ({alt_count} 个视频)")
            video_extract_dir = alt_extract_dir
            existing_mp4 = alt_count
        else:
            existing_mp4 = 0
    else:
        # 检查现有视频数量
        existing_mp4 = 0
        if os.path.isdir(video_extract_dir):
            for root, _, files in os.walk(video_extract_dir):
                existing_mp4 += sum(1 for f in files if f.endswith(".mp4"))
    
    if existing_mp4 < max_videos:
        if existing_mp4 > 0:
            print(f"[3/4] 已有 {existing_mp4} 个视频，但需要 {max_videos} 个，重新解压...")
            import shutil
            shutil.rmtree(video_extract_dir)
        else:
            print("[3/4] 解压视频存档...")
        
        os.makedirs(video_extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            video_members = [m for m in members if m.name.endswith(".mp4")]
            print(f"   存档共 {len(video_members)} 个视频，提取前 {max_videos} 个")
            
            for i, m in enumerate(video_members[:max_videos]):
                tar.extract(m, path=video_extract_dir)
                if (i + 1) % 200 == 0:
                    print(f"   已解压 {i+1}/{min(max_videos, len(video_members))}")
        
        print(f"   解压完成 -> {video_extract_dir}")
    else:
        print(f"[3/4] 视频已存在: {existing_mp4} 个")
    
    # ---- 4. 扫描视频并提取关键点 ----
    kp_dir = os.path.join(omni_dir, "keypoints")
    os.makedirs(kp_dir, exist_ok=True)
    
    video_files = []
    for root, _, files in os.walk(video_extract_dir):
        for f in files:
            if f.endswith(".mp4"):
                label = label_map.get(f, 0)
                video_files.append((os.path.join(root, f), label))
    
    print(f"\n[4/4] 提取关键点 (共 {len(video_files)} 个视频)...")
    total_seqs = 0
    processed = 0
    
    for i, (vp, lb) in enumerate(video_files[:max_videos]):
        n = process_video(vp, lb, kp_dir, detector,
                          video_id=f"omnifall_{i:04d}", max_frames=300)
        if n > 0:
            total_seqs += n
            processed += 1
        if (i + 1) % 50 == 0:
            print(f"   [{i+1}/{min(max_videos, len(video_files))}] 已处理 {processed} 个有效视频")
    
    print(f"\n[DataPrep] OmniFall: 处理 {processed} 个视频, {total_seqs} 个序列")
    return processed


def download_kaggle(output_dir, detector, max_videos=50):
    """下载 Kaggle Fall Video Dataset"""
    print(f"\n{'='*60}")
    print(f"[DataPrep] 下载 Kaggle Fall Video Dataset")
    print(f"{'='*60}")

    kaggle_dir = os.path.join(output_dir, "kaggle_raw")
    os.makedirs(kaggle_dir, exist_ok=True)

    try:
        import kagglehub
        print("[DataPrep] 使用 kagglehub 下载...")
        path = kagglehub.dataset_download("payutch/fall-video-dataset")
        print(f"[DataPrep] 下载到: {path}")
        if os.path.isdir(path):
            if os.path.exists(kaggle_dir):
                shutil.rmtree(kaggle_dir)
            shutil.copytree(path, kaggle_dir)
    except Exception as e:
        print(f"[DataPrep] kagglehub 下载失败: {e}")
        print("[DataPrep] 请手动下载:")
        print("  1. 打开 https://www.kaggle.com/datasets/payutch/fall-video-dataset")
        print("  2. 点击 Download")
        print(f"  3. 解压到 {kaggle_dir}")
        if not os.path.isdir(kaggle_dir):
            return 0

    # 扫描视频
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    videos = []
    for root, _, files in os.walk(kaggle_dir):
        for f in files:
            if f.lower().endswith(exts):
                label = 1 if "fall" in f.lower() else 0
                videos.append((os.path.join(root, f), label))

    print(f"[DataPrep] Kaggle: 找到 {len(videos)} 个视频")

    if not videos:
        return 0

    kp_dir = os.path.join(output_dir, "kaggle", "keypoints")
    os.makedirs(kp_dir, exist_ok=True)
    total_seqs = 0

    for i, (vp, lb) in enumerate(videos[:max_videos]):
        n = process_video(vp, lb, kp_dir, detector, video_id=f"kaggle_{i:04d}")
        if n > 0:
            total_seqs += n
        print(f"  [{i+1}] {os.path.basename(vp)} -> {n} seqs")

    print(f"[DataPrep] Kaggle: 处理 {min(len(videos), max_videos)} 个视频, {total_seqs} 序列")
    return total_seqs


def generate_demo_data(output_dir, num_videos=120):
    """
    生成仿真关键点数据，模拟真实跌倒/正常行为
    
    相比简单版本，这个版本增加了:
    - 坐下、弯腰、蹲下等"干扰动作"（接近跌倒但非跌倒）
    - 关键点抖动（模拟 YOLO 检测噪声）
    - 关键点部分丢失（模拟遮挡）
    - 不同跌倒速度（快速跌倒/缓慢滑倒）
    - 摄像机视角变化
    - 部分关键点置信度波动
    """
    print(f"\n{'='*60}")
    print(f"[DataPrep] 生成增强仿真数据集 ({num_videos} 个视频)")
    print(f"{'='*60}")

    data_dir = os.path.join(output_dir, "demo", "keypoints")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.RandomState(42)

    # 动作类型比例：40% 正常行走站立, 30% 跌倒, 30% 干扰动作(坐下/弯腰/蹲)
    action_types = []
    n_fall = num_videos * 3 // 10
    n_confuse = num_videos * 3 // 10
    n_normal = num_videos - n_fall - n_confuse
    for _ in range(n_normal): action_types.append("normal")
    for _ in range(n_fall): action_types.append("fall")
    for _ in range(n_confuse): action_types.append("confuse")
    rng.shuffle(action_types)

    for v in range(num_videos):
        person_dir = os.path.join(data_dir, f"video_{v:04d}", "person_0")
        os.makedirs(person_dir, exist_ok=True)

        T = 150  # 更长序列
        kp = np.zeros((T, 17, 3))
        bb = np.zeros((T, 4))
        lbl = np.zeros(T)
        action = action_types[v]

        # 随机抖动幅度（模拟不同质量的 YOLO 检测）
        jitter = rng.uniform(0.005, 0.025)

        # 随机检测质量（影响置信度）
        det_quality = rng.uniform(0.5, 0.95)

        # 视角偏移（模拟不同摄像机角度）
        view_offset = rng.uniform(-0.1, 0.1)

        # 随机跌倒速度
        fall_speed = rng.uniform(0.015, 0.035)

        base_x = 0.5 + view_offset
        base_ymin = rng.uniform(0.03, 0.08)
        base_ymax = rng.uniform(0.92, 0.97)

        for t in range(T):
            # -- 基础站立位置 --
            y_vals = np.linspace(base_ymin, base_ymax, 17)
            x_vals = np.full(17, base_x)

            if action == "normal":
                # 正常行走: 轻微左右摆动
                walk_phase = np.sin(t * 0.05) * 0.02
                x_vals += walk_phase
                y_vals += np.sin(t * 0.08 + np.arange(17) * 0.5) * 0.005
                lbl[t] = 0

                # 偶尔蹲下捡东西然后站起
                if 60 <= t <= 90:
                    squat_progress = (t - 60) / 30
                    if squat_progress < 0.5:
                        # 蹲下过程：上半身快速下降
                        factor = squat_progress * 2 * 0.4
                        y_vals[:11] = y_vals[:11] * (1 - factor) + 0.7 * factor
                    else:
                        # 站起过程
                        factor = (squat_progress - 0.5) * 2 * 0.4
                        y_vals[:11] = y_vals[:11] * factor + y_vals[:11].mean() * (1 - factor)

            elif action == "confuse":
                # 干扰动作: 坐下、弯腰、躺下（与跌倒视觉相似）
                confuse_type = rng.choice(["sit", "bend", "lie"])

                if confuse_type == "sit":
                    # 坐下: 臀部快速下降，上半身轻微前倾
                    if 30 <= t < 70:
                        p = (t - 30) / 40
                        # 肩部略降，臀部大幅下降
                        y_vals[:5] += p * 0.05    # 头/眼微降
                        y_vals[5:7] += p * 0.15   # 肩部降
                        y_vals[11:13] += p * 0.4  # 臀部大幅降
                        y_vals[13:] += p * 0.1    # 腿微降
                        bb[t] = [base_x - 0.18, base_ymin + p * 0.2,
                                 base_x + 0.18, base_ymax - p * 0.15]
                    else:
                        bb[t] = [base_x - 0.18, base_ymin, base_x + 0.18, base_ymax]
                    lbl[t] = 0

                elif confuse_type == "bend":
                    # 弯腰: 头部下移，臀部上移
                    if 30 <= t < 55:
                        p = (t - 30) / 25
                        y_vals[0] += p * 0.3       # 鼻下降
                        y_vals[11:13] -= p * 0.1   # 臀上翘
                        y_vals[5:7] += p * 0.05
                    elif 55 <= t < 90:
                        p = (t - 55) / 35
                        y_vals[0] -= p * 0.3
                        y_vals[11:13] += p * 0.1
                    lbl[t] = 0
                    bb[t] = [base_x - 0.18, base_ymin, base_x + 0.18, base_ymax]

                else:  # lie
                    # 躺下: 所有关键点横置 + 缓慢躺平 (极易被误判为跌倒)
                    if 30 <= t < 80:
                        p = (t - 30) / 50
                        # Y 坐标集中到 0.7-0.8 范围
                        y_vals = y_vals * (1 - p) + np.full(17, 0.75) * p
                        # X 坐标分散开
                        x_vals = base_x + np.linspace(-0.2, 0.2, 17) * p
                        bb[t] = [base_x - 0.2 - p * 0.1, 0.4, base_x + 0.2 + p * 0.1, 0.85]
                    else:
                        bb[t] = [base_x - 0.3, 0.4, base_x + 0.3, 0.85]
                    lbl[t] = 0

            else:  # fall
                # 跌倒: 所有关键点同时快速下降 + 人体框变宽
                if 30 <= t < 50:
                    # 自由落体阶段
                    p = (t - 30) * fall_speed
                    y_vals = y_vals * (1 - p) + np.full(17, 0.8) * p
                    bb[t] = [base_x - 0.18, base_ymin * (1 - p),
                             base_x + 0.18, 0.95 - (0.95 - base_ymax) * (1 - p)]
                    lbl[t] = 1
                elif 50 <= t < 55:
                    # 撞击地面
                    lbl[t] = 1
                    bb[t] = [base_x - 0.18, 0.4, base_x + 0.18, 0.85]
                else:
                    # 倒地后静止
                    y_vals = np.full(17, 0.8) + rng.randn(17) * 0.01
                    x_vals = base_x + np.linspace(-0.15, 0.15, 17)
                    bb[t] = [base_x - 0.18, 0.35, base_x + 0.18, 0.85]
                    lbl[t] = 1

            # -- 添加真实干扰 --
            # 1. 关键点抖动（YOLO 检测噪声）
            kp[t, :, 0] = x_vals + rng.randn(17) * jitter
            kp[t, :, 1] = y_vals + rng.randn(17) * jitter

            # 2. 关键点部分丢失（模拟遮挡）
            kp[t, :, 2] = det_quality + rng.rand(17) * (1 - det_quality)
            if rng.random() < 0.1:  # 10% 概率随机遮挡一个关键点
                occluded = rng.randint(0, 17)
                kp[t, occluded, 2] = 0.0

            # 3. 边界框噪声
            if bb[t].sum() == 0:
                bb[t] = [base_x - 0.18, base_ymin, base_x + 0.18, base_ymax]
            bb[t] += rng.randn(4) * 0.01

        # 裁剪到有效范围
        kp[..., :2] = np.clip(kp[..., :2], 0, 1)
        kp[..., 2] = np.clip(kp[..., 2], 0, 1)
        bb = np.clip(bb, 0, 1)

        np.save(os.path.join(person_dir, "keypoints.npy"), kp)
        np.save(os.path.join(person_dir, "bboxes.npy"), bb)
        np.save(os.path.join(person_dir, "labels.npy"), lbl)

    # 统计
    fall_count = sum(1 for a in action_types if a == "fall")
    confuse_count = sum(1 for a in action_types if a == "confuse")
    normal_count = sum(1 for a in action_types if a == "normal")
    total_seq = sum(max(0, (150 - 32) // 16 + 1) for _ in range(num_videos))
    print(f"[DataPrep] 增强仿真数据: {num_videos} 视频, ~{total_seq} 序列")
    print(f"[DataPrep]   正常行走: {normal_count}, 跌倒: {fall_count}, 干扰动作: {confuse_count}")
    return num_videos


def split_and_copy(source_dir, project_root, val_ratio=0.15):
    """划分训练/验证集并复制到 data/train, data/val"""
    import random

    videos = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])
    if not videos:
        return False

    random.seed(42)
    random.shuffle(videos)
    n_val = max(1, int(len(videos) * val_ratio))
    val_videos = videos[:n_val]
    train_videos = videos[n_val:]

    train_dir = os.path.join(project_root, "data", "train")
    val_dir = os.path.join(project_root, "data", "val")

    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    def copy_list(vlist, target):
        for v in vlist:
            src = os.path.join(source_dir, v)
            dst = os.path.join(target, v)
            shutil.copytree(src, dst, dirs_exist_ok=True)

    copy_list(train_videos, train_dir)
    copy_list(val_videos, val_dir)
    print(f"[DataPrep] 划分: 训练 {len(train_videos)} 视频, 验证 {len(val_videos)} 视频")
    return True


def merge_data_sources(source_dirs, merge_dir):
    """合并多个数据源到同一个目录"""
    os.makedirs(merge_dir, exist_ok=True)
    idx = 0
    for src in source_dirs:
        if not os.path.isdir(src):
            continue
        for v in os.listdir(src):
            v_src = os.path.join(src, v)
            if os.path.isdir(v_src):
                v_dst = os.path.join(merge_dir, f"{idx:04d}")
                shutil.copytree(v_src, v_dst, dirs_exist_ok=True)
                idx += 1
    return idx


def main():
    import argparse
    parser = argparse.ArgumentParser(description="下载并预处理跌倒检测数据集")
    parser.add_argument("--all", action="store_true", help="完整流程(OmniFall+Kaggle+Demo)")
    parser.add_argument("--omnifall", action="store_true", help="下载 OmniFall")
    parser.add_argument("--kaggle", action="store_true", help="下载 Kaggle")
    parser.add_argument("--demo", action="store_true", help="生成仿真数据(推荐入门)")
    parser.add_argument("--max_videos", type=int, default=30, help="每个数据集最多处理视频数")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--no_split", action="store_true", help="不划分训练/验证集")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_dir = args.output or os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    check_dependencies()
    detector = setup_yolo()

    # 收集所有关键点目录
    kp_sources = []
    do_everything = args.all or not any([args.omnifall, args.kaggle, args.demo])

    # ---- 1. OmniFall ----
    if args.omnifall or do_everything:
        n = download_omnifall(raw_dir, detector, args.max_videos)
        if n > 0:
            kp_sources.append(os.path.join(raw_dir, "omnifall", "keypoints"))

    # ---- 2. Kaggle ----
    if args.kaggle or do_everything:
        n = download_kaggle(raw_dir, detector, args.max_videos)
        if n > 0:
            kp_sources.append(os.path.join(raw_dir, "kaggle", "keypoints"))

    # ---- 3. 仿真数据（兜底） ----
    if args.demo or do_everything:
        n = generate_demo_data(raw_dir, max(60, args.max_videos * 2))
        kp_sources.append(os.path.join(raw_dir, "demo", "keypoints"))

    # ---- 合并到 data/merged ----
    merged_dir = os.path.join(raw_dir, "merged")
    total = merge_data_sources(kp_sources, merged_dir)
    print(f"\n[DataPrep] 合计: {total} 个视频")

    if total == 0:
        print("[DataPrep] !!! 没有成功生成任何数据，请检查网络或使用 --demo")
        return

    # ---- 划分训练/验证 ----
    if not args.no_split:
        split_and_copy(merged_dir, project_root, val_ratio=0.15)

    # 检查最终结果
    train_dir = os.path.join(project_root, "data", "train")
    val_dir = os.path.join(project_root, "data", "val")
    train_count = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]) if os.path.isdir(train_dir) else 0
    val_count = len([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]) if os.path.isdir(val_dir) else 0

    print(f"\n{'='*60}")
    print(f"[DataPrep] 完成!")
    print(f"  训练集: {train_dir} ({train_count} 视频)")
    print(f"  验证集: {val_dir} ({val_count} 视频)")
    print(f"\n  用法: python train.py --data_dir data/train --val_dir data/val --epochs 100")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
