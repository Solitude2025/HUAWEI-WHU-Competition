"""
下载 OmniFall 合成视频存档
直接下载: data_files/omnifall-synthetic_av1.tar (9.72 GB)
"""
import os, sys, subprocess, tarfile, shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. 安装 huggingface_hub
subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])

from huggingface_hub import hf_hub_download, snapshot_download

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "omnifall")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("下载 OmniFall 合成视频存档 (~9.72 GB)")
print(f"保存到: {output_dir}")
print("=" * 60)

# 下载 tar 包
tar_path = hf_hub_download(
    repo_id="simplexsigil2/omnifall",
    repo_type="dataset",
    filename="data_files/omnifall-synthetic_av1.tar",
    local_dir=output_dir,
    local_dir_use_symlinks=False,
)

print(f"\n下载完成: {tar_path}")
print(f"大小: {os.path.getsize(tar_path) / 1024**3:.2f} GB")

# 解压
print("\n正在解压...")
extract_dir = os.path.join(output_dir, "videos_extracted")
os.makedirs(extract_dir, exist_ok=True)

with tarfile.open(tar_path, "r") as tar:
    tar.extractall(path=extract_dir)

print(f"解压完成 -> {extract_dir}")

# 统计
video_count = sum(1 for f in Path(extract_dir).rglob("*.mp4"))
print(f"视频数量: {video_count}")

# 获取标签元数据
print("\n下载标签数据...")
labels_path = hf_hub_download(
    repo_id="simplexsigil2/omnifall",
    repo_type="dataset",
    filename="labels/of-syn.csv",
    local_dir=output_dir,
    local_dir_use_symlinks=False,
)
print(f"标签文件: {labels_path}")

print(f"\n{'='*60}")
print(f"下载完成！")
print(f"视频目录: {extract_dir} ({video_count} 个视频)")
print(f"标签文件: {labels_path}")
print(f"接下来: python prepare_data.py --omnifall 提取关键点")
print(f"{'='*60}")
