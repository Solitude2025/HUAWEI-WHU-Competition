"""
长尾场景测试集构建脚本
========================
从 OmniFall 各场景中筛选视频，生成长尾测试视频集。

赛题要求的长尾场景：
1. 傍晚微光 / 夜间红外  → OmniFall 合成数据不支持光照变化，用数据增强模拟
2. 人体存在遮挡          → 合成关键点随机drop模拟
3. 摄像机视角变化        → camera_elevation + camera_azimuth 组合
4. 远距离                → camera_distance = 'far'
5. 动作混淆              → sit_down, lie_down vs fall

用法:
    python create_longtail_test.py
    输出: longtail_test/ 目录，包含各类长尾场景视频
"""

import os
import sys
import random
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _copy_video(row, video_dir, dst_path):
    """从 OmniFall 视频目录复制视频到目标路径"""
    src = os.path.join(video_dir, str(row["path"]).replace("/", os.sep) + ".mp4")
    if os.path.exists(src):
        shutil.copy2(src, dst_path)
        return True
    return False


def create_longtail_test_set():
    project_root = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(project_root, "data", "raw", "omnifall", "omnifall-synthetic_av1")
    csv_path = os.path.join(project_root, "data", "raw", "omnifall", "of-syn.csv")
    output_dir = os.path.join(project_root, "longtail_test")

    # ── 清理重建输出目录 ──
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # ── 读取 CSV ──
    df = pd.read_csv(csv_path)
    df["filename"] = df["path"].apply(lambda p: os.path.basename(p) + ".mp4")

    # 只保留视频文件存在的行
    existing_files = set()
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.endswith(".mp4"):
                existing_files.add(f)
    df = df[df["filename"].isin(existing_files)]
    print(f"[Longtail] 有效视频片段: {len(df):,}")

    random.seed(42)
    total_videos = 0

    # ═══════════════════════════════════════
    # 场景1: 视角变化
    # ═══════════════════════════════════════
    print("\n── 场景1: 视角变化 ──")
    views = [("high", "front"), ("low", "rear"), ("top", "left"), ("eye", "right")]
    view_dir = os.path.join(output_dir, "01_viewpoint")
    os.makedirs(view_dir)
    for elev, azim in views:
        subset = df[(df["camera_elevation"] == elev) &
                    (df["camera_azimuth"] == azim) &
                    (df["label"].isin([1, 2]))]
        samples = subset.sample(min(2, len(subset)), random_state=42)
        for _, row in samples.iterrows():
            dst = os.path.join(view_dir, f"{elev}_{azim}_{row['filename']}")
            if _copy_video(row, video_dir, dst):
                total_videos += 1
    print(f"  {len(os.listdir(view_dir))} 个视频")

    # ═══════════════════════════════════════
    # 场景2: 远距离跌倒
    # ═══════════════════════════════════════
    print("\n── 场景2: 远距离跌倒 ──")
    far_dir = os.path.join(output_dir, "02_far_distance")
    os.makedirs(far_dir)
    far_fall = df[(df["camera_distance"] == "far") &
                  (df["label"].isin([1, 2]))]
    samples = far_fall.sample(min(5, len(far_fall)), random_state=42)
    for _, row in samples.iterrows():
        dst = os.path.join(far_dir, row["filename"])
        if _copy_video(row, video_dir, dst):
            total_videos += 1
    print(f"  {len(os.listdir(far_dir))} 个视频")

    # ═══════════════════════════════════════
    # 场景3: 动作混淆
    # ═══════════════════════════════════════
    print("\n── 场景3: 动作混淆 ──")
    confuse_dir = os.path.join(output_dir, "03_confusing_actions")
    os.makedirs(confuse_dir)

    confuse_actions = df[df["label"].isin([3, 4, 5, 6])]
    label_names = {3: "sit_down", 4: "sitting", 5: "lie_down", 6: "lying"}
    for elev in ["eye", "top"]:
        for dist in ["far", "medium"]:
            subset = confuse_actions[(confuse_actions["camera_elevation"] == elev) &
                                     (confuse_actions["camera_distance"] == dist)]
            if len(subset) >= 1:
                row = subset.sample(1, random_state=42).iloc[0]
                ln = label_names.get(int(row["label"]), "unknown")
                dst = os.path.join(confuse_dir, f"{elev}_{dist}_{ln}_{row['filename']}")
                if _copy_video(row, video_dir, dst):
                    total_videos += 1
    print(f"  {len(os.listdir(confuse_dir))} 个视频")

    # ═══════════════════════════════════════
    # 场景4: 不同年龄/体型
    # ═══════════════════════════════════════
    print("\n── 场景4: 不同体型/年龄 ──")
    demo_dir = os.path.join(output_dir, "04_diverse_demographics")
    os.makedirs(demo_dir)

    ages = ["toddlers_1_4", "elderly_65_plus", "young_adults_18_34"]
    bmis = ["underweight", "obese", "normal"]
    for age in ages:
        for bmi in bmis:
            subset = df[(df["age_group"] == age) &
                        (df["bmi_band"] == bmi) &
                        (df["label"].isin([1, 2]))]
            if len(subset) >= 1:
                row = subset.sample(1, random_state=42).iloc[0]
                dst = os.path.join(demo_dir, f"{age}_{bmi}_{row['filename']}")
                if _copy_video(row, video_dir, dst):
                    total_videos += 1
    print(f"  {len(os.listdir(demo_dir))} 个视频")

    # ═══════════════════════════════════════
    # 场景5: 正常行走(对照)
    # ═══════════════════════════════════════
    print("\n── 场景5: 正常行走(对照) ──")
    normal_dir = os.path.join(output_dir, "05_normal_walking")
    os.makedirs(normal_dir)

    walking = df[df["label"] == 9]  # walking
    for elev in ["eye", "low"]:
        for dist in ["far", "medium"]:
            subset = walking[(walking["camera_elevation"] == elev) &
                             (walking["camera_distance"] == dist)]
            if len(subset) >= 1:
                row = subset.sample(1, random_state=42).iloc[0]
                dst = os.path.join(normal_dir, f"{elev}_{dist}_{row['filename']}")
                if _copy_video(row, video_dir, dst):
                    total_videos += 1
    print(f"  {len(os.listdir(normal_dir))} 个视频")

    # ═══════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Longtail] 共生成 {total_videos} 个长尾测试视频")
    print(f"[Longtail] 输出目录: {output_dir}")
    print(f"\n使用: python infer.py --test --test_dir longtail_test --checkpoint checkpoints/best.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    create_longtail_test_set()
