"""
FallDetector (完整跌倒检测模型)
----------------------------------
整合 MotionFeatureEncoder + LightTCN + RuleRefinement 的端到端跌倒检测模型。

这是一个完整的轻量级时空语义跌倒检测框架，核心创新围绕三个模块：
1. Motion Feature Encoder: 将关键点序列编码为运动语义特征
2. Light-TCN: 膨胀因果卷积实现轻量时序建模
3. Rule Refinement: 物理约束后处理降低误报

模型总参数: < 100K（不含 YOLO Pose 部分）
推理延迟: < 10ms（TCN+Rule部分，不含 YOLO）

部署友好: 全卷积 + 规则引擎，天然支持 NPU INT8 量化
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from .motion_encoder import MotionFeatureEncoder
from .light_tcn import LightTCN, LightTCN_V2
from .rule_refinement import RuleRefinement, RuleConfig
from .transformer_tcn import TransformerTCN, EfficientTransformerTCN, create_transformer_tcn


class FallDetector(nn.Module):
    """
    端到端跌倒检测模型
    
    输入: 关键点序列 + 边界框序列
    输出: 每帧跌倒概率 + 报警标志
    
    Usage:
        model = FallDetector()
        
        # 训练模式（批量）
        probs, alarms = model(keypoints, bboxes)
        loss = criterion(probs, labels)
        
        # 推理模式（批量）
        with torch.no_grad():
            probs, alarms = model(keypoints, bboxes)
        
        # 在线推理（逐帧，使用 RuleRefinementOnline）
        online_model = model.to_online()
        for frame in video:
            prob, alarm = online_model.update(kp, bbox)
    """
    
    def __init__(
        self,
        # Motion Encoder 参数
        motion_feature_dim: int = 48,
        # TCN 参数
        tcn_hidden_dim: int = 64,
        tcn_num_layers: int = 5,
        tcn_kernel_size: int = 3,
        tcn_dilations: Optional[list] = None,
        tcn_dropout: float = 0.1,
        tcn_version: str = "light-tcn-v1",
        # Rule Refinement 参数
        rule_config: Optional[RuleConfig] = None,
        # 通用参数
        sequence_length: int = 32,
    ):
        """
        Args:
            motion_feature_dim: Motion Encoder 输出特征维度
            tcn_hidden_dim: TCN 隐藏层维度
            tcn_num_layers: TCN 层数
            tcn_kernel_size: TCN 卷积核大小
            tcn_dilations: 膨胀系数列表
            tcn_dropout: TCN Dropout 比例
            tcn_version: TCN 版本 ("light-tcn-v1", "light-tcn-v2", "transformer-tcn-standard", "transformer-tcn-efficient", "transformer-tcn-light")
            rule_config: 规则校正配置
            sequence_length: 时序窗口长度
        """
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # Motion Feature Encoder
        self.motion_encoder = MotionFeatureEncoder(
            feature_dim=motion_feature_dim,
        )
        
        # 根据版本选择 TCN 模型
        if tcn_version == "light-tcn-v2":
            self.tcn = LightTCN_V2(
                input_dim=motion_feature_dim,
                hidden_dim=tcn_hidden_dim,
                num_layers=tcn_num_layers,
                kernel_size=tcn_kernel_size,
                dilations=tcn_dilations,
                dropout=tcn_dropout,
            )
        elif tcn_version.startswith("transformer-tcn"):
            # 使用 Transformer-TCN
            transformer_version = tcn_version.replace("transformer-tcn-", "")
            self.tcn = create_transformer_tcn(
                version=transformer_version,
                input_dim=motion_feature_dim,
                embed_dim=tcn_hidden_dim,
                num_layers=tcn_num_layers,
                tcn_hidden_dim=tcn_hidden_dim,
                kernel_size=tcn_kernel_size,
                dilations=tcn_dilations,
                dropout=tcn_dropout,
            )
        else:  # 默认使用 Light-TCN v1
            self.tcn = LightTCN(
                input_dim=motion_feature_dim,
                hidden_dim=tcn_hidden_dim,
                num_layers=tcn_num_layers,
                kernel_size=tcn_kernel_size,
                dilations=tcn_dilations,
                dropout=tcn_dropout,
            )
        
        # Rule Refinement
        self.rule_refinement = RuleRefinement(rule_config)
        
        # 参数统计
        self._log_total_params()
    
    def _log_total_params(self):
        """统计总参数量"""
        # Motion Encoder 无参数
        motion_params = sum(p.numel() for p in self.motion_encoder.parameters())
        tcn_params = sum(p.numel() for p in self.tcn.parameters())
        rule_params = sum(p.numel() for p in self.rule_refinement.parameters())
        total = motion_params + tcn_params + rule_params
        
        print(f"[FallDetector] 总参数: {total:,}")
        print(f"  - MotionEncoder: {motion_params:,}")
        print(f"  - LightTCN: {tcn_params:,}")
        print(f"  - RuleRefinement: {rule_params:,}")
        
        # 检查是否满足赛题要求（< 20M）
        assert total < 20_000_000, f"Total params {total:,} exceeds 20M limit!"
    
    def forward(
        self,
        keypoints: torch.Tensor,    # (B, T, 17, 3)
        bboxes: torch.Tensor,        # (B, T, 4)
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            keypoints: 人体关键点序列
            bboxes: 人体边界框序列
            return_features: 是否返回中间特征
        
        Returns:
            dict with:
                - 'fall_prob': (B, T, 1) 原始 TCN 概率
                - 'refined_prob': (B, T, 1) 规则校正后概率
                - 'alarm': (B, T, 1) 报警标志
                - 'features': (B, T, C) 运动特征（可选）
        """
        B, T = keypoints.shape[:2]
        
        # Step 1: Motion Feature Encoding
        motion_features = self.motion_encoder(keypoints, bboxes)  # (B, T, C)
        
        # Step 2: TCN 时序建模
        tcn_prob = self.tcn(motion_features)  # (B, T, 1)
        
        # Step 3: Rule Refinement
        refined_prob, alarm = self.rule_refinement(
            tcn_prob, motion_features
        )
        
        outputs = {
            "fall_prob": tcn_prob,
            "refined_prob": refined_prob,
            "alarm": alarm,
        }
        
        if return_features:
            outputs["features"] = motion_features
        
        return outputs
    
    def forward_features(
        self,
        keypoints: torch.Tensor,
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """仅提取运动特征（用于特征可视化或下游分析）"""
        return self.motion_encoder(keypoints, bboxes)
    
    def forward_tcn(
        self,
        keypoints: torch.Tensor,
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """仅 TCN 概率输出（不含规则校正）"""
        features = self.motion_encoder(keypoints, bboxes)
        return self.tcn(features)
    
    @torch.no_grad()
    def predict(
        self,
        keypoints: torch.Tensor,
        bboxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理模式：返回校正概率 + 报警标志
        """
        self.eval()
        outputs = self.forward(keypoints, bboxes)
        return outputs["refined_prob"], outputs["alarm"]
    
    def count_parameters(self) -> Dict[str, int]:
        """统计各模块参数量（MB）"""
        counts = {}
        for name, module in [
            ("motion_encoder", self.motion_encoder),
            ("tcn", self.tcn),
            ("rule_refinement", self.rule_refinement),
        ]:
            params = sum(p.numel() for p in module.parameters())
            size_mb = params * 4 / (1024 * 1024)  # fp32
            counts[name] = {
                "params": params,
                "size_mb": size_mb,
            }
        counts["total_params"] = sum(c["params"] for c in counts.values())
        counts["total_mb"] = sum(c["size_mb"] for c in counts.values())
        return counts
    
    def get_model_size_mb(self) -> float:
        """获取模型 FP32 存储大小 (MB)"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / (1024 * 1024)
    
    def to_onnx(
        self,
        filepath: str,
        input_shape: Tuple = (1, 32, 48),
    ):
        """
        导出 TCN 部分为 ONNX（用于 NPU 部署）
        
        Motion Encoder 和 Rule Refinement 作为独立算子部署
        """
        self.eval()
        dummy_input = torch.randn(*input_shape)
        
        # 仅导出 TCN（Motion Encoder 是无参数的纯计算）
        torch.onnx.export(
            self.tcn,
            dummy_input,
            filepath,
            input_names=["motion_features"],
            output_names=["fall_prob"],
            dynamic_axes={
                "motion_features": {0: "batch", 1: "sequence"},
                "fall_prob": {0: "batch", 1: "sequence"},
            },
            opset_version=11,
            do_constant_folding=True,
        )
        print(f"[FallDetector] ONNX 导出到: {filepath}")


def create_fall_detector(
    version: str = "standard",
    **kwargs,
) -> FallDetector:
    """
    创建跌倒检测模型的工厂函数
    
    Args:
        version: "standard" (标准 Light-TCN), "light" (极轻量 Light-TCN), 
                 "large" (高精度 Light-TCN), 
                 "transformer-tcn-standard" (标准 Transformer-TCN),
                 "transformer-tcn-efficient" (高效 Transformer-TCN),
                 "transformer-tcn-light" (极轻量 Transformer-TCN)
    
    Returns:
        FallDetector 实例
    """
    configs = {
        "standard": {
            "motion_feature_dim": 48,
            "tcn_hidden_dim": 64,
            "tcn_num_layers": 5,
            "tcn_kernel_size": 3,
            "tcn_version": "light-tcn-v1",
            "tcn_dropout": 0.1,
        },
        "light": {
            "motion_feature_dim": 48,
            "tcn_hidden_dim": 32,
            "tcn_num_layers": 4,
            "tcn_kernel_size": 3,
            "tcn_version": "light-tcn-v2",
            "tcn_dropout": 0.1,
        },
        "large": {
            "motion_feature_dim": 64,
            "tcn_hidden_dim": 128,
            "tcn_num_layers": 6,
            "tcn_kernel_size": 5,
            "tcn_version": "light-tcn-v1",
            "tcn_dropout": 0.2,
        },
        "transformer-tcn-standard": {
            "motion_feature_dim": 48,
            "tcn_hidden_dim": 64,
            "tcn_num_layers": 3,
            "tcn_kernel_size": 3,
            "tcn_version": "transformer-tcn-standard",
            "tcn_dropout": 0.1,
        },
        "transformer-tcn-efficient": {
            "motion_feature_dim": 48,
            "tcn_hidden_dim": 48,
            "tcn_num_layers": 2,
            "tcn_kernel_size": 3,
            "tcn_version": "transformer-tcn-efficient",
            "tcn_dropout": 0.1,
        },
        "transformer-tcn-light": {
            "motion_feature_dim": 32,
            "tcn_hidden_dim": 32,
            "tcn_num_layers": 2,
            "tcn_kernel_size": 3,
            "tcn_version": "transformer-tcn-light",
            "tcn_dropout": 0.1,
        },
    }
    
    cfg = configs.get(version, configs["standard"])
    cfg.update(kwargs)
    
    return FallDetector(**cfg)
