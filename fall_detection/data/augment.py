"""
红外域适配与数据增强模块
---------------------------
针对赛题第三大痛点（场景长尾问题）设计的数据增强策略。

增强策略：
1. 红外风格迁移：RGB → Infrared-like（CycleGAN / 灰度+噪声）
2. 低光增强：Gamma校正、CLAHE
3. 随机遮挡：Cutout、Random Erasing、关键点遮挡
4. 视角变换：Perspective、Affine
5. 运动模糊：模拟快速运动场景

设计理念（来自赛题分析）：
- 不去修改 YOLO 模型架构
- 在训练数据层面提升模型泛化能力
- 使 YOLO 能同时适应可见光和红外场景
"""

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from typing import Tuple, Optional, List, Dict
import random


class IRAugmentation:
    """
    红外场景数据增强
    
    将 RGB 图像转换为红外风格，用于训练 YOLO Pose 检测器，
    使其适应红外场景。
    """
    
    @staticmethod
    def rgb_to_grayscale_ir(
        image: np.ndarray,
        noise_level: float = 0.05,
        contrast_boost: float = 1.5,
    ) -> np.ndarray:
        """
        RGB → 灰度红外风格转换
        
        简单方法：灰度化 + 噪声 + 对比度增强
        适合快速数据增强
        
        Args:
            image: (H, W, 3) RGB/BGR, uint8 [0-255]
            noise_level: 高斯噪声标准差
            contrast_boost: 对比度增强系数
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 灰度化
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 对比度增强
        gray = np.clip(gray.astype(np.float32) * contrast_boost, 0, 255)
        
        # 高斯噪声（模拟红外传感器噪声）
        noise = np.random.randn(*gray.shape) * noise_level * 255
        gray = np.clip(gray + noise, 0, 255).astype(np.uint8)
        
        # 三通道复制
        ir_image = np.stack([gray, gray, gray], axis=-1)
        
        return ir_image
    
    @staticmethod
    def clahe_enhance(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        CLAHE 对比度增强（红外/低光场景）
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(image)
        
        return enhanced
    
    @staticmethod
    def gamma_correction(
        image: np.ndarray,
        gamma: float = None,
    ) -> np.ndarray:
        """
        Gamma 校正（低光增强）
        
        gamma < 1: 提亮暗区
        gamma > 1: 压暗亮区
        """
        if gamma is None:
            gamma = random.uniform(0.5, 1.5)
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        table = (np.arange(256) / 255.0) ** gamma * 255.0
        table = np.clip(table, 0, 255).astype(np.uint8)
        
        return cv2.LUT(image, table)


class SpatialAugmentation:
    """
    空间域数据增强
    
    针对赛题长尾问题（遮挡、视角变化）的增强。
    """
    
    @staticmethod
    def random_erase(
        image: np.ndarray,
        scale: Tuple[float, float] = (0.02, 0.2),
        ratio: Tuple[float, float] = (0.3, 3.3),
        max_boxes: int = 3,
    ) -> np.ndarray:
        """
        Random Erasing（随机擦除）
        
        模拟人体部分遮挡场景。
        """
        h, w = image.shape[:2]
        aug = image.copy()
        
        for _ in range(random.randint(1, max_boxes)):
            area = h * w
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)
            
            erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if erase_h < h and erase_w < w:
                x = random.randint(0, w - erase_w)
                y = random.randint(0, h - erase_h)
                
                # 用随机值或均值填充
                aug[y:y+erase_h, x:x+erase_w] = random.randint(0, 255)
        
        return aug
    
    @staticmethod
    def random_perspective(
        image: np.ndarray,
        keypoints: Optional[np.ndarray] = None,
        scale: float = 0.05,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机透视变换（模拟不同摄像机视角）
        
        Args:
            image: (H, W, 3)
            keypoints: (N, 2) or (N, 3) 关键点坐标
            scale: 变换幅度
        """
        h, w = image.shape[:2]
        
        # 随机偏移四个角
        offset = h * scale
        src = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ])
        dst = np.float32([
            [random.uniform(0, offset), random.uniform(0, offset)],
            [w - random.uniform(0, offset), random.uniform(0, offset)],
            [w - random.uniform(0, offset), h - random.uniform(0, offset)],
            [random.uniform(0, offset), h - random.uniform(0, offset)],
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        if keypoints is not None:
            kp_homo = np.ones((keypoints.shape[0], 3))
            kp_homo[:, :2] = keypoints[:, :2]
            kp_warped = kp_homo @ M.T
            kp_warped = kp_warped[:, :2] / (kp_warped[:, 2:3] + 1e-6)
            if keypoints.shape[1] == 3:
                kp_warped = np.concatenate([kp_warped, keypoints[:, 2:3]], axis=1)
            return warped, kp_warped
        
        return warped, None
    
    @staticmethod
    def copy_paste_occlusion(
        image: np.ndarray,
        occlude_ratio: float = 0.1,
        max_patches: int = 3,
    ) -> np.ndarray:
        """
        Copy-Paste 遮挡（模拟他人、家具遮挡）
        
        从图像中随机取一块贴到人体区域。
        """
        h, w = image.shape[:2]
        aug = image.copy()
        
        for _ in range(random.randint(1, max_patches)):
            # 随机选择遮挡区域
            patch_h = random.randint(h // 8, h // 4)
            patch_w = random.randint(w // 8, w // 4)
            
            sx = random.randint(0, w - patch_w)
            sy = random.randint(0, h - patch_h)
            
            # 粘贴位置（偏向中心，模拟遮挡人体）
            tx = random.randint(w // 4, 3 * w // 4 - patch_w)
            ty = random.randint(h // 4, 3 * h // 4 - patch_h)
            
            aug[ty:ty+patch_h, tx:tx+patch_w] = image[sy:sy+patch_h, sx:sx+patch_w]
        
        return aug


class MotionAugmentation:
    """
    运动相关增强
    
    模拟快速运动场景中的模糊效果。
    """
    
    @staticmethod
    def motion_blur(
        image: np.ndarray,
        kernel_size: int = None,
        angle: float = None,
    ) -> np.ndarray:
        """
        运动模糊
        
        Args:
            image: (H, W, 3)
            kernel_size: 模糊核大小
            angle: 模糊方向角度（度）
        """
        if kernel_size is None:
            kernel_size = random.choice([5, 7, 9, 11, 15])
        if angle is None:
            angle = random.uniform(0, 360)
        
        # 创建运动模糊核
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # 在核中画一条线
        angle_rad = np.deg2rad(angle)
        for i in range(kernel_size):
            x = int(center + (i - center) * np.cos(angle_rad))
            y = int(center + (i - center) * np.sin(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        kernel /= kernel.sum()
        
        return cv2.filter2D(image, -1, kernel)


class KeypointAugmentation:
    """
    关键点级别数据增强
    
    在关键点层面进行增强，模拟 YOLO Pose 的输出变化。
    """
    
    @staticmethod
    def keypoint_jitter(
        keypoints: np.ndarray,     # (T, 17, 3)
        jitter_std: float = 0.01,
    ) -> np.ndarray:
        """
        关键点抖动（模拟检测噪声）
        
        Args:
            keypoints: (T, 17, 3) [x, y, conf]
            jitter_std: 抖动标准差（归一化坐标）
        """
        jittered = keypoints.copy()
        noise = np.random.randn(*keypoints.shape[:2], 2) * jitter_std
        jittered[..., :2] += noise
        jittered[..., :2] = np.clip(jittered[..., :2], 0, 1)
        return jittered
    
    @staticmethod
    def keypoint_dropout(
        keypoints: np.ndarray,     # (T, 17, 3)
        drop_prob: float = 0.15,
    ) -> np.ndarray:
        """
        随机丢弃关键点（模拟遮挡导致的检测失败）
        
        Args:
            keypoints: (T, 17, 3)
            drop_prob: 丢弃概率
        """
        dropped = keypoints.copy()
        mask = np.random.rand(*keypoints.shape[:2]) > drop_prob
        
        # 部分关键点置信度置零但不改坐标
        # 模拟检测器在某些帧未检测到某些关键点
        for t in range(keypoints.shape[0]):
            for kp in range(keypoints.shape[1]):
                if not mask[t, kp]:
                    dropped[t, kp, 2] = 0.0
        
        return dropped
    
    @staticmethod
    def keypoint_mixup(
        kp1: np.ndarray,      # (T, 17, 3)
        kp2: np.ndarray,      # (T, 17, 3)
        alpha: float = 0.2,
    ) -> np.ndarray:
        """
        关键点 MixUp（混合两个人/场景的关键点序列）
        
        增强 TCN 的鲁棒性。
        """
        lam = np.random.beta(alpha, alpha)
        mixed = lam * kp1 + (1 - lam) * kp2
        return mixed


class AugmentationPipeline:
    """
    完整数据增强管线
    
    将多种增强策略组合，模拟长尾场景。
    """
    
    def __init__(
        self,
        ir_prob: float = 0.3,           # 红外风格概率
        erase_prob: float = 0.3,         # 随机擦除概率
        perspective_prob: float = 0.2,   # 透视变换概率
        blur_prob: float = 0.2,          # 运动模糊概率
        lowlight_prob: float = 0.3,      # 低光增强概率
        jitter_std: float = 0.01,        # 关键点抖动
        drop_prob: float = 0.1,          # 关键点丢弃概率
    ):
        self.ir_prob = ir_prob
        self.erase_prob = erase_prob
        self.perspective_prob = perspective_prob
        self.blur_prob = blur_prob
        self.lowlight_prob = lowlight_prob
        self.jitter_std = jitter_std
        self.drop_prob = drop_prob
    
    def augment_image(
        self,
        image: np.ndarray,
        is_training: bool = True,
    ) -> np.ndarray:
        """图像级别增强"""
        if not is_training:
            return image
        
        aug = image.copy()
        
        # 低光增强
        if random.random() < self.lowlight_prob:
            if random.random() < 0.5:
                aug = IRAugmentation.gamma_correction(aug)
            else:
                aug = IRAugmentation.clahe_enhance(aug)
        
        # 红外风格
        if random.random() < self.ir_prob:
            aug = IRAugmentation.rgb_to_grayscale_ir(aug)
        
        # 随机擦除
        if random.random() < self.erase_prob:
            aug = SpatialAugmentation.random_erase(aug)
        
        # 透视变换
        if random.random() < self.perspective_prob:
            aug, _ = SpatialAugmentation.random_perspective(aug)
        
        # 运动模糊
        if random.random() < self.blur_prob:
            aug = MotionAugmentation.motion_blur(aug)
        
        return aug
    
    def augment_keypoints(
        self,
        keypoints: np.ndarray,      # (T, 17, 3)
        is_training: bool = True,
    ) -> np.ndarray:
        """关键点级别增强"""
        if not is_training:
            return keypoints
        
        kp = keypoints.copy()
        
        # 关键点抖动
        if self.jitter_std > 0:
            kp = KeypointAugmentation.keypoint_jitter(kp, self.jitter_std)
        
        # 关键点丢弃
        if self.drop_prob > 0:
            kp = KeypointAugmentation.keypoint_dropout(kp, self.drop_prob)
        
        return kp
    
    def augment(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,      # (T, 17, 3)
        bboxes: np.ndarray,          # (T, 4)
        is_training: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """完整增强"""
        image = self.augment_image(image, is_training)
        keypoints = self.augment_keypoints(keypoints, is_training)
        return image, keypoints, bboxes
