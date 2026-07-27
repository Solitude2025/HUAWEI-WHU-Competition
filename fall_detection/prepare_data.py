"""
数据下载与预处理脚本
======================
从 OmniFall (HuggingFace) 下载合成视频数据集，
用 YOLOv8n-Pose 提取关键点，按帧级标签修复，
最终按 80% 训练 / 10% 验证 / 10% 测试 分流。

用法:
    # 完整流程（下载→提取→分流）
    python prepare_data.py --all

    # 只下载视频（不断点续传），不提取
    python prepare_data.py --download

    # 已下载视频但未提取时，只提取关键点
    python prepare_data.py --extract

    # 已提取好关键点时，只重新分流
    python prepare_data.py --split
"""

import os
import sys
import csv
import tarfile
import shutil
import subprocess
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════

def _download_file(url: str, save_path: str, desc: str = ""):
    """
    下载文件并显示进度条，支持断点续传
    """
    import requests
    from tqdm import tqdm as _tqdm

    headers = {}
    mode = "wb"
    if os.path.exists(save_path):
        existing_size = os.path.getsize(save_path)
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
    else:
        existing_size = 0

    resp_head = requests.head(url, allow_redirects=True, timeout=10)
    total_size = int(resp_head.headers.get("content-length", 0))

    if existing_size >= total_size:
        print(f"   文件已存在 ({total_size / 1024**3:.2f} GB)")
        return

    resp = requests.get(url, stream=True, timeout=30, headers=headers)
    resp.raise_for_status()

    desc_text = desc or os.path.basename(save_path)
    with _tqdm(
        total=total_size, unit="B", unit_scale=True,
        desc=f"  {desc_text}", initial=existing_size,
        ascii=True, ncols=80,
    ) as pbar:
        with open(save_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"   完成: {os.path.getsize(save_path) / 1024**3:.2f} GB")


def check_dependencies():
    """检查并安装所需依赖"""
    deps = {
        "ultralytics": "ultralytics",
        "requests": "requests",
        "sklearn": "scikit-learn",
    }
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
                    detections.append(PersonDetection(
                        bbox=bbox_norm, keypoints=kp_full, confidence=conf
                    ))
            return detections

    return Detector(model)


# ═══════════════════════════════════════════════
#  数据下载
# ═══════════════════════════════════════════════

OMNI_HF_URL = "https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main"
CSV_URL = f"{OMNI_HF_URL}/labels/of-syn.csv"
TAR_URL = f"{OMNI_HF_URL}/data_files/omnifall-synthetic_av1.tar"


def download_omnifall(omnifall_dir: str):
    """
    下载 OmniFall 数据：
    1. of-syn.csv（标签, ~1MB）
    2. omnifall-synthetic_av1.tar（视频, ~9.72GB，支持断点续传）
    """
    print(f"\n{'='*60}")
    print(f"[DataPrep] 下载 OmniFall 数据")
    print(f"{'='*60}")

    os.makedirs(omnifall_dir, exist_ok=True)

    # ---- 1. 下载标签 CSV ----
    csv_path = os.path.join(omnifall_dir, "of-syn.csv")
    if not os.path.exists(csv_path):
        print("[1/2] 下载标签文件 (of-syn.csv)...")
        _download_file(url=CSV_URL, save_path=csv_path, desc="of-syn.csv")
    else:
        print(f"[1/2] 标签文件已存在: {csv_path}")

    n_lines = sum(1 for _ in open(csv_path)) - 1  # 减去表头
    print(f"       {n_lines:,} 条标签记录")

    # ---- 2. 下载视频存档 ----
    tar_path = os.path.join(omnifall_dir, "omnifall-synthetic_av1.tar")
    if not os.path.exists(tar_path):
        print("[2/2] 下载视频存档 (~9.72 GB)...")
        _download_file(url=TAR_URL, save_path=tar_path, desc="omnifall-synthetic_av1.tar")
    else:
        tar_gb = os.path.getsize(tar_path) / 1024**3
        print(f"[2/2] 视频存档已存在: {tar_gb:.2f} GB")

    return csv_path, tar_path


# ═══════════════════════════════════════════════
#  视频 → 关键点提取 + 帧级标签修复
# ═══════════════════════════════════════════════

def _iter_frames(video_path: str):
    """
    逐帧生成视频帧。优先用 cv2；若解码失败（如 AV1 编码，
    OpenCV 只能读元数据不能解码），回退到 imageio-ffmpeg 的
    ffmpeg 二进制走 rawvideo 管道解码。
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, frame = cap.read()
    if ok:
        yield frame
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()
        return
    cap.release()

    # ---- cv2 解码失败，回退 ffmpeg 管道 ----
    if w <= 0 or h <= 0:
        print(f"  [!] 无法获取分辨率: {video_path}")
        return
    import imageio_ffmpeg
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    frame_size = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def process_video_with_segments(
    video_path: str,
    segments_df: "pd.DataFrame",
    output_dir: str,
    detector,
    video_id: str,
):
    """
    处理单个 OmniFall 视频：
    1. 逐帧读取
    2. YOLO 提取关键点
    3. 根据 CSV 中的 start/end 时间戳为每帧分配标签
       (label 1=fall, 2=fallen → 跌倒; 其余 → 正常)
    """
    import cv2
    import pandas as pd

    # cv2 只用于读取元数据（fps），即使无法解码 AV1 也能拿到
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [!] 无法打开: {video_path}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = 30.0

    # 对当前视频的所有片段按 start 排序
    video_segments = segments_df.sort_values("start")

    all_kps, all_bbs, all_labels = [], [], []

    for frame_idx, frame in enumerate(_iter_frames(video_path)):
        frame_time = frame_idx / fps

        # 找到当前帧属于哪个片段
        frame_label = 0  # 默认非跌倒
        for _, seg in video_segments.iterrows():
            if seg["start"] <= frame_time < seg["end"]:
                # label 1=fall, 2=fallen → 跌到; 其余 → 正常
                raw_label = int(seg["label"])
                frame_label = 1 if raw_label in (1, 2) else 0
                break

        dets = detector.detect(frame)
        if dets:
            best = max(dets, key=lambda d: d.confidence)
            kp = best.keypoints
            bb = best.bbox
        else:
            kp = np.zeros((17, 3))
            bb = np.zeros(4)

        all_kps.append(kp)
        all_bbs.append(bb)
        all_labels.append(frame_label)

    if len(all_kps) == 0:
        return 0

    person_dir = os.path.join(output_dir, video_id, "person_0")
    os.makedirs(person_dir, exist_ok=True)

    np.save(os.path.join(person_dir, "keypoints.npy"), np.stack(all_kps))
    np.save(os.path.join(person_dir, "bboxes.npy"), np.stack(all_bbs))
    np.save(os.path.join(person_dir, "labels.npy"), np.array(all_labels))

    seq_len = 32
    stride = 8
    return max(0, (len(all_kps) - seq_len) // stride + 1)


def extract_keypoints(omnifall_dir: str, detector, max_videos: int = 500):
    """
    解压视频 → 逐帧提取关键点 → 修复标签

    流程:
    1. 读取 of-syn.csv（所有片段标签）
    2. 解压 tar 中的 mp4
    3. 对每个视频逐帧提取关键点 + 分配帧级标签
    """
    import pandas as pd

    print(f"\n{'='*60}")
    print(f"[DataPrep] 提取关键点 (最多 {max_videos} 个视频)")
    print(f"{'='*60}")

    csv_path = os.path.join(omnifall_dir, "of-syn.csv")
    tar_path = os.path.join(omnifall_dir, "omnifall-synthetic_av1.tar")

    if not os.path.exists(csv_path) or not os.path.exists(tar_path):
        print("[ERROR] 请先运行 --download 下载数据")
        return False

    # ---- 1. 读取标签 CSV ----
    df = pd.read_csv(csv_path)
    print(f"[1/4] 读取标签文件: {len(df):,} 行, {df['path'].nunique():,} 个视频")

    # ---- 2. 按类别分层抽样视频（保证跌倒比例 + 多视角覆盖）----
    # tar 内按类别目录字典序排列，直接取前 N 个会全是 fall 类，
    # 因此按一级目录（动作类别）配额抽样；随机抽样自然覆盖
    # camera_elevation / camera_azimuth / camera_distance 等视角维度。
    target = max_videos if max_videos > 0 else df["path"].nunique()
    df["_cls"] = df["path"].str.split("/").str[0]
    classes = sorted(df["_cls"].unique())
    quota = {"fall": max(1, target * 24 // 100),      # 跌倒类 24%
             "fallen": max(1, target * 16 // 100)}    # 倒地类 16%
    other_classes = [c for c in classes if c not in quota]
    per_other = max(1, (target - sum(quota.values())) // max(1, len(other_classes)))

    rng = np.random.RandomState(42)
    selected = []
    for cls in classes:
        pool = sorted(df.loc[df["_cls"] == cls, "path"].unique())
        pool = list(pool)
        rng.shuffle(pool)
        q = quota.get(cls, per_other)
        selected.extend(pool[:q])
    # 配额取整可能不足 target，从剩余池补齐
    if len(selected) < target:
        rest = sorted(set(df["path"].unique()) - set(selected))
        rng.shuffle(rest)
        selected.extend(rest[: target - len(selected)])
    selected = selected[:target]
    selected_set = set(selected)

    sel_df = df[df["path"].isin(selected_set)]
    n_fall_videos = sel_df[sel_df["label"].isin([1, 2])]["path"].nunique()
    print(f"[2/4] 分层抽样 {len(selected)} 个视频 (含跌倒 {n_fall_videos} 个):")
    print("   类别分布:", dict(sel_df.groupby("_cls")["path"].nunique()))
    cam_view = sel_df.drop_duplicates("path")
    print("   俯仰角:", dict(cam_view["camera_elevation"].value_counts()),
          "| 方位角:", dict(cam_view["camera_azimuth"].value_counts()))

    # ---- 3. 解压选中的视频 ----
    video_dir = os.path.join(omnifall_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # 检查已解压的视频覆盖了选中集的多少
    existing_stems = set()
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.endswith(".mp4"):
                rel = os.path.relpath(os.path.join(root, f), video_dir)
                existing_stems.add(os.path.splitext(rel)[0].replace("\\", "/"))

    missing = [p for p in selected if p not in existing_stems]
    if missing:
        print(f"[3/4] 解压 {len(missing)} 个选中视频 (已有 {len(selected) - len(missing)} 个)...")
        wanted = {f"{p}.mp4" for p in missing}
        with tarfile.open(tar_path, "r") as tar:
            n_done = 0
            for m in tar:
                if not m.name.endswith(".mp4"):
                    continue
                name = m.name[2:] if m.name.startswith("./") else m.name
                if name in wanted:
                    tar.extract(m, path=video_dir)
                    n_done += 1
                    if n_done % 50 == 0:
                        print(f"   已解压 {n_done}/{len(missing)}")
                    if n_done >= len(missing):
                        break
        print(f"   解压完成 -> {video_dir}")
    else:
        print(f"[3/4] 视频已就位: {len(existing_stems & selected_set)} 个")

    # ---- 4. 扫描选中的视频 ----
    video_files = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.endswith(".mp4"):
                # relative_path 如 "fall/fall_ch_001"
                rel = os.path.relpath(os.path.join(root, f), video_dir)
                path_stem = os.path.splitext(rel)[0].replace("\\", "/")
                if path_stem in selected_set:
                    video_files.append((os.path.join(root, f), path_stem))

    # ---- 4. 提取关键点 ----
    kp_dir = os.path.join(omnifall_dir, "extracted_keypoints")
    os.makedirs(kp_dir, exist_ok=True)

    processed = 0
    total_seqs = 0

    # 按 path 将 CSV 分组（每个视频可能有多个片段）
    grouped = df.groupby("path")
    target = len(video_files)

    print(f"[4/4] 提取关键点 ({target} 个视频)...")

    for i, (vp, path_stem) in enumerate(video_files):
        # 获取这个视频的所有片段
        if path_stem in grouped.groups:
            segments = grouped.get_group(path_stem)
        else:
            segments = df[df["path"] == path_stem]
            if len(segments) == 0:
                print(f"  [!] 未找到标签: {path_stem}")
                continue

        n = process_video_with_segments(
            vp, segments, kp_dir, detector,
            video_id=f"{i:04d}",
        )
        if n > 0:
            total_seqs += n
            processed += 1
        if (i + 1) % 100 == 0:
            print(f"   已处理 {i+1}/{target} ({processed} 个有效)")

    print(f"\n[DataPrep] 提取完成: {processed} 个视频, ~{total_seqs} 个序列")
    return True


# ═══════════════════════════════════════════════
#  80/10/10 分流
# ═══════════════════════════════════════════════

def split_dataset(kp_dir: str, project_root: str):
    """
    按 80% 训练 / 10% 验证 / 10% 测试 分流到 data/train, val, test
    """
    import random

    print(f"\n{'='*60}")
    print(f"[DataPrep] 数据集分流 (80/10/10)")
    print(f"{'='*60}")

    dirs = sorted([
        d for d in os.listdir(kp_dir)
        if os.path.isdir(os.path.join(kp_dir, d))
    ])
    if not dirs:
        print(f"[ERROR] 目录中无数据: {kp_dir}")
        return False

    print(f"  可用视频数: {len(dirs)}")

    # 统计每个视频是否含跌倒（用于分层采样）
    has_fall = []
    for d in dirs:
        lp = os.path.join(kp_dir, d, "person_0", "labels.npy")
        if os.path.exists(lp):
            labels = np.load(lp)
            has_fall.append(1 if np.sum(labels == 1) > 0 else 0)
        else:
            has_fall.append(0)

    fall_count = sum(has_fall)
    normal_count = len(has_fall) - fall_count
    print(f"  含跌倒: {fall_count} | 纯正常: {normal_count}")

    # 分层采样
    train_dirs, temp_dirs = train_test_split(
        dirs, test_size=0.20, random_state=42, stratify=has_fall,
    )
    temp_labels = []
    for d in temp_dirs:
        lp = os.path.join(kp_dir, d, "person_0", "labels.npy")
        if os.path.exists(lp):
            temp_labels.append(1 if np.sum(np.load(lp) == 1) > 0 else 0)
        else:
            temp_labels.append(0)
    val_dirs, test_dirs = train_test_split(
        temp_dirs, test_size=0.50, random_state=42, stratify=temp_labels,
    )

    print(f"\n  分流结果:")
    print(f"    训练集: {len(train_dirs)} 个 ({len(train_dirs)/len(dirs)*100:.0f}%)")
    print(f"    验证集: {len(val_dirs)} 个 ({len(val_dirs)/len(dirs)*100:.0f}%)")
    print(f"    测试集: {len(test_dirs)} 个 ({len(test_dirs)/len(dirs)*100:.0f}%)")

    # 清理旧目录并创建
    data_dir = os.path.join(project_root, "data")
    for sub in ["train", "val", "test"]:
        target = os.path.join(data_dir, sub)
        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(target)

    def copy_set(vlist, target_dir):
        for i, vd in enumerate(vlist):
            src = os.path.join(kp_dir, vd)
            dst = os.path.join(target_dir, f"{i:04d}")
            shutil.copytree(src, dst)

    copy_set(train_dirs, os.path.join(data_dir, "train"))
    copy_set(val_dirs, os.path.join(data_dir, "val"))
    copy_set(test_dirs, os.path.join(data_dir, "test"))

    # 验证
    for sub in ["train", "val", "test"]:
        target = os.path.join(data_dir, sub)
        subdirs = [d for d in os.listdir(target) if os.path.isdir(os.path.join(target, d))]
        total_f, fall_f = 0, 0
        for d in subdirs:
            lp = os.path.join(target, d, "person_0", "labels.npy")
            if os.path.exists(lp):
                lb = np.load(lp)
                total_f += len(lb)
                fall_f += int(np.sum(lb == 1))
        pct = fall_f / total_f * 100 if total_f > 0 else 0
        print(f"\n    data/{sub}/: {len(subdirs):>4} 个视频, "
              f"{total_f:>6} 帧, 跌倒 {fall_f:>6} 帧 ({pct:.1f}%)")

    return True


# ═══════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="下载并预处理跌倒检测数据集")
    parser.add_argument("--all", action="store_true", help="完整流程: 下载+提取+分流")
    parser.add_argument("--download", action="store_true", help="仅下载数据")
    parser.add_argument("--extract", action="store_true", help="仅提取关键点")
    parser.add_argument("--split", action="store_true", help="仅重新分流 (需已有 extracted_keypoints)")
    parser.add_argument("--max_videos", type=int, default=500,
                       help="最大处理视频数 (-1=全部, 默认500)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    omnifall_dir = os.path.join(project_root, "data", "raw", "omnifall")
    kp_dir = os.path.join(omnifall_dir, "extracted_keypoints")

    # ── 模式选择 ──
    do_all = args.all or (not args.download and not args.extract and not args.split)

    if do_all or args.download:
        check_dependencies()
        download_omnifall(omnifall_dir)

    if do_all or args.extract:
        check_dependencies()
        detector = setup_yolo()
        extract_keypoints(omnifall_dir, detector, args.max_videos)

    if do_all or args.split:
        if not os.path.isdir(kp_dir):
            print(f"[ERROR] 未找到提取好的关键点数据: {kp_dir}")
            print("请先运行: python prepare_data.py --extract")
            return
        split_dataset(kp_dir, project_root)

    # ── 最终提示 ──
    print(f"\n{'='*60}")
    print(f"[DataPrep] 完成!")
    print(f"\n  训练: python train.py --data_dir data/train --val_dir data/val --epochs 50")
    print(f"  推理: python infer.py --test")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
