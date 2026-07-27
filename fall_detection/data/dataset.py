"""
跌倒检测数据集加载器
-----------------------
支持多种开源数据集格式：
- OmniFall (HuggingFace)
- Fall Video Dataset (Kaggle)
- UR Fall Detection Dataset
- 自定义标注数据

输出格式：
- keypoints: (T, 17, 3) 关键点序列
- bboxes: (T, 4) 边界框序列
- labels: (T,) 跌倒/正常标签 (1/0)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List, Dict
import json
import random

from .augment import AugmentationPipeline


class FallDetectionDataset(Dataset):
    """
    跌倒检测数据集
    
    支持从关键点/边界框预提取数据直接训练 TCN 部分，
    也可以从原始视频端到端训练（需要 YOLO Pose 预提取）。
    
    数据格式（预提取模式）：
        data/
        ├── video_001/
        │   ├── person_0/
        │   │   ├── keypoints.npy    # (T, 17, 3)
        │   │   ├── bboxes.npy       # (T, 4)
        │   │   └── labels.npy       # (T,)
        │   └── person_1/
        │       └── ...
        └── video_002/
            └── ...
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 32,
        stride: int = 8,
        use_augmentation: bool = True,
        augment_pipeline: Optional[AugmentationPipeline] = None,
        mode: str = "train",
        balance_ratio: float = 0.3,    # 跌倒样本占比
        min_fall_ratio: float = 0.1,   # 序列中至少包含的跌倒帧比例
    ):
        """
        Args:
            data_dir: 数据目录
            sequence_length: 时序窗口长度
            stride: 滑动步幅
            use_augmentation: 是否使用数据增强
            augment_pipeline: 增强管线
            mode: "train" / "val" / "test"
            balance_ratio: 训练集中跌倒样本的比例
            min_fall_ratio: 序列中至少包含的跌倒帧比例（用于正样本）
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.mode = mode
        self.balance_ratio = balance_ratio
        self.min_fall_ratio = min_fall_ratio
        
        # 数据增强
        self.use_augmentation = use_augmentation and mode == "train"
        self.augment = augment_pipeline or AugmentationPipeline()
        
        # 加载数据索引
        self.sequences = self._load_sequences()
        
        # 如果没找到数据，回退到合成数据
        if len(self.sequences) == 0:
            print(f"[Dataset] 目录 '{data_dir}' 中未找到有效数据，回退到合成数据")
            self.sequences = self._generate_synthetic_sequences()
        
        # 平衡采样
        if mode == "train" and balance_ratio > 0:
            self.fall_indices = [i for i, (_, _, l) in enumerate(self.sequences) if l > 0.5]
            self.normal_indices = [i for i, (_, _, l) in enumerate(self.sequences) if l <= 0.5]
    
    def _load_sequences(self) -> List[Tuple[str, int, float]]:
        """
        加载所有序列索引
        
        Returns:
            List[(video_dir, start_frame, fall_ratio), ...]
        """
        sequences = []
        
        if not os.path.exists(self.data_dir):
            print(f"[Dataset] 数据目录不存在: {self.data_dir}")
            print("[Dataset] 将使用模拟数据")
            return self._generate_synthetic_sequences()
        
        for video_name in os.listdir(self.data_dir):
            video_dir = os.path.join(self.data_dir, video_name)
            if not os.path.isdir(video_dir):
                continue
            
            for person_name in os.listdir(video_dir):
                person_dir = os.path.join(video_dir, person_name)
                if not os.path.isdir(person_dir):
                    continue
                
                # 读取标签
                labels_path = os.path.join(person_dir, "labels.npy")
                if not os.path.exists(labels_path):
                    continue
                labels = np.load(labels_path)
                
                # 滑动窗口生成序列
                T = len(labels)
                for start in range(0, T - self.sequence_length + 1, self.stride):
                    end = start + self.sequence_length
                    seq_labels = labels[start:end]
                    fall_ratio = seq_labels.mean()
                    sequences.append((person_dir, start, fall_ratio))
        
        print(f"[Dataset] 加载 {len(sequences)} 个序列")
        
        # 难负样本加权：计算每个负样本序列的躯干角变化幅度，
        # 坐下/躺下等易混负样本的变化大，训练时提高其采样权重
        if self.mode == "train":
            self._neg_weights = self._compute_hard_negative_weights(
                sequences, self.sequence_length
            )
        
        return sequences
    
    @staticmethod
    def _compute_hard_negative_weights(sequences, sequence_length: int, alpha: float = 3.0) -> List[float]:
        """
        为负样本序列计算采样权重：weight = 1 + alpha * norm(score)
        score = 窗口内躯干角最大逐帧变化（坐下/躺下等易混动作得分高）
        躯干角 = 肩中心-髋中心连线与垂直方向夹角
        """
        # 先按 person_dir 聚合，躯干角每个视频只算一次
        video_angles: Dict[str, np.ndarray] = {}
        scores = np.zeros(len(sequences), dtype=np.float32)
        for i, (person_dir, start, fall_ratio) in enumerate(sequences):
            if fall_ratio > 0.5:
                continue  # 正样本不加权
            if person_dir not in video_angles:
                kp_path = os.path.join(person_dir, "keypoints.npy")
                if not os.path.exists(kp_path):
                    video_angles[person_dir] = np.zeros(0)
                    continue
                kp = np.load(kp_path)  # (T, 17, 3)
                shoulder = (kp[:, 5, :2] + kp[:, 6, :2]) / 2
                hip = (kp[:, 11, :2] + kp[:, 12, :2]) / 2
                vec = shoulder - hip
                video_angles[person_dir] = np.arctan2(
                    np.abs(vec[:, 0]), np.abs(vec[:, 1]) + 1e-6
                )
            ang = video_angles[person_dir]
            end = min(start + 1 + sequence_length, len(ang))
            if end - start > 1:
                scores[i] = np.abs(np.diff(ang[start:end])).max()
        
        # 归一化到 [0, 1] 后加权
        if scores.max() > 0:
            norm = scores / scores.max()
        else:
            norm = scores
        weights = (1.0 + alpha * norm).tolist()
        return weights
    
    def _generate_synthetic_sequences(self) -> List:
        """生成合成数据用于演示"""
        print("[Dataset] 生成 200 个合成训练序列...")
        sequences = []
        for i in range(200):
            # 50% 跌倒, 50% 正常
            fall_ratio = 0.8 if i < 100 else 0.0
            sequences.append((f"synthetic_{i}", 0, fall_ratio))
        return sequences
    
    def _load_person_data(
        self,
        person_dir: str,
        start_frame: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载单个人员的序列数据"""
        try:
            kp = np.load(os.path.join(person_dir, "keypoints.npy"))
            bb = np.load(os.path.join(person_dir, "bboxes.npy"))
            lbl = np.load(os.path.join(person_dir, "labels.npy"))
            
            end = min(start_frame + self.sequence_length, len(kp))
            
            return (
                kp[start_frame:end],
                bb[start_frame:end],
                lbl[start_frame:end],
            )
        except:
            return self._generate_synthetic_sample()
    
    def _generate_synthetic_sample(
        self,
        is_fall: bool = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成合成样本"""
        if is_fall is None:
            is_fall = np.random.random() < 0.5
        
        T = self.sequence_length
        kp = np.zeros((T, 17, 3))
        bb = np.zeros((T, 4))
        lbl = np.zeros(T)
        
        if is_fall:
            # 模拟跌倒过程
            fall_start = T // 3
            fall_duration = T // 3
            
            for t in range(T):
                if t < fall_start:
                    # 站立阶段
                    kp[t, :, 0] = 0.5 + np.random.randn(17) * 0.02
                    kp[t, :, 1] = np.linspace(0.1, 0.95, 17) + np.random.randn(17) * 0.01
                    kp[t, :, 2] = 0.8 + np.random.rand(17) * 0.2
                    bb[t] = [0.3, 0.05, 0.7, 0.95]
                    lbl[t] = 0
                elif t < fall_start + fall_duration:
                    # 跌倒阶段
                    progress = (t - fall_start) / fall_duration
                    # Y 坐标快速下降
                    kp[t, :, 1] = np.linspace(0.1, 0.95, 17) * (1 - progress * 0.7) + progress * 0.7
                    kp[t, :, 0] = 0.5 + np.random.randn(17) * 0.03
                    kp[t, :, 2] = 0.5 + np.random.rand(17) * 0.3
                    bb[t] = [0.2, 0.2, 0.8, 0.85]
                    lbl[t] = 1
                else:
                    # 倒地阶段
                    kp[t, :, 0] = 0.5 + np.random.randn(17) * 0.03
                    kp[t, :, 1] = 0.8 + np.random.randn(17) * 0.03
                    kp[t, :, 2] = 0.4 + np.random.rand(17) * 0.3
                    bb[t] = [0.15, 0.4, 0.85, 0.85]
                    lbl[t] = 1
        else:
            # 正常行走/站立
            for t in range(T):
                kp[t, :, 0] = 0.5 + np.random.randn(17) * 0.03
                kp[t, :, 1] = np.linspace(0.1, 0.95, 17) + np.random.randn(17) * 0.02
                kp[t, :, 2] = 0.7 + np.random.rand(17) * 0.3
                bb[t] = [0.3, 0.05, 0.7, 0.95]
                lbl[t] = 0
        
        return kp, bb, lbl
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict:
                - keypoints: (T, 17, 3)
                - bboxes: (T, 4)
                - labels: (T,)
        """
        # 平衡采样（训练模式）
        if self.mode == "train" and self.balance_ratio > 0:
            if np.random.random() < self.balance_ratio:
                # 采样跌倒样本
                if self.fall_indices:
                    idx = random.choice(self.fall_indices)
                    person_dir, start, _ = self.sequences[idx]
                else:
                    person_dir, start, _ = self.sequences[index]
            else:
                # 采样正常样本（难负样本按躯干角变化加权，坐下/躺下被采概率更高）
                if self.normal_indices:
                    if getattr(self, "_neg_weights", None):
                        w = [self._neg_weights[i] for i in self.normal_indices]
                        idx = random.choices(self.normal_indices, weights=w, k=1)[0]
                    else:
                        idx = random.choice(self.normal_indices)
                    person_dir, start, _ = self.sequences[idx]
                else:
                    person_dir, start, _ = self.sequences[index]
        else:
            person_dir, start, _ = self.sequences[index]
        
        # 加载数据
        kp, bb, lbl = self._load_person_data(person_dir, start)
        
        # 填充到固定长度
        if len(kp) < self.sequence_length:
            pad_len = self.sequence_length - len(kp)
            kp = np.pad(kp, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
            bb = np.pad(bb, ((0, pad_len), (0, 0)), mode='edge')
            lbl = np.pad(lbl, (0, pad_len), mode='edge')
        
        # 数据增强（截断增强仅用于负样本，避免把跌倒序列截成"不像跌倒"）
        if self.use_augmentation:
            kp = self.augment.augment_keypoints(
                kp, allow_truncation=(lbl.sum() == 0)
            )
        
        return {
            "keypoints": torch.from_numpy(kp).float(),
            "bboxes": torch.from_numpy(bb).float(),
            "labels": torch.from_numpy(lbl).float(),
        }


def create_dataloader(
    data_dir: str,
    sequence_length: int = 32,
    batch_size: int = 32,
    stride: int = 8,
    mode: str = "train",
    num_workers: int = 4,
    use_augmentation: bool = True,
) -> DataLoader:
    """
    创建数据加载器的工厂函数
    """
    dataset = FallDetectionDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        stride=stride,
        use_augmentation=use_augmentation,
        mode=mode,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == "train"),
    )
