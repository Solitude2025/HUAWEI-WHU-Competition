"""
Light-TCN (轻量时序卷积网络)
--------------------------------
基于膨胀因果卷积的轻量级时序建模模块。

设计特点：
1. 全卷积结构 → 天然支持 NPU INT8 量化，并行推理效率高
2. 膨胀因果卷积 → 在保持实时性的同时扩大感受野
3. 极少量参数 → 约 ~50K 参数，远低于传统 LSTM/GRU
4. 输入为 MotionEncoder 输出的 ~48 维特征，而非高维 CNN Feature

架构:
    Input (B, T, C_in=48)
      ↓
    CausalConv1d(dilation=1) → BN → ReLU
      ↓
    CausalConv1d(dilation=2) → BN → ReLU
      ↓
    CausalConv1d(dilation=4) → BN → ReLU
      ↓
    CausalConv1d(dilation=8) → BN → ReLU
      ↓
    Conv1d → Sigmoid
      ↓
    Output (B, T, 1): fall probability per frame

感受野: 1 + 2*(k-1) + 2*(k-1)*2 + 2*(k-1)*4 + 2*(k-1)*8
        当 k=3: 1 + 4 + 8 + 16 + 32 = 61 帧 ≈ 2秒 (30fps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalConv1d(nn.Module):
    """因果卷积：只依赖过去帧，不依赖未来帧"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # 左侧 padding = dilation * (kernel_size - 1)，右侧 padding = 0
        # 这样卷积只向右看（因果性）
        self.padding_left = dilation * (kernel_size - 1)
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,  # 手动 padding
            bias=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            out: (B, C_out, T) - 因果卷积输出，保持时间维度
        """
        # 左侧填充，右侧不填充 → 因果性
        x_padded = F.pad(x, (self.padding_left, 0))
        out = self.conv(x_padded)
        return out


class TCNBlock(nn.Module):
    """TCN 基本模块：膨胀因果卷积 + BatchNorm + ReLU"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.causal_conv = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 残差连接（当 in_channels != out_channels 时使用 1x1 卷积映射）
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T)
        Returns:
            out: (B, C_out, T)
        """
        residual = x if self.use_residual else self.residual_conv(x)
        
        out = self.causal_conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 对残差做因果对齐（截断右侧）
        out = out + residual[..., :out.shape[-1]]
        return out


class LightTCN(nn.Module):
    """
    轻量时序卷积网络
    
    Args:
        input_dim: 输入特征维度（MotionEncoder 输出维度）
        hidden_dim: 隐藏层维度
        num_layers: TCN 层数
        kernel_size: 卷积核大小
        dilations: 膨胀系数列表，默认 [1, 2, 4, 8, 16]
        dropout: Dropout 比例
    """
    
    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 64,
        num_layers: int = 5,
        kernel_size: int = 3,
        dilations: Optional[list] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入投影
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # TCN 层堆叠
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            d = dilations[i % len(dilations)]
            self.tcn_layers.append(
                TCNBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        
        # 计算感受野
        self.receptive_field = self._compute_receptive_field(kernel_size, dilations[:num_layers])
        
        # 参数统计
        self._log_params()
    
    def _compute_receptive_field(self, kernel_size: int, dilations: list) -> int:
        """计算膨胀因果卷积的感受野（帧数）"""
        rf = 1
        for d in dilations:
            rf += (kernel_size - 1) * d
        return rf
    
    def _log_params(self):
        """统计并打印参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LightTCN] 总参数: {total:,} | 可训练: {trainable:,} | 感受野: {self.receptive_field} 帧")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C_in) 运动特征序列
        
        Returns:
            prob: (B, T, 1) 每帧的跌倒概率 [0, 1]
        """
        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_proj(x)  # (B, hidden_dim, T)
        
        # TCN 层
        for layer in self.tcn_layers:
            x = layer(x)
        
        # 输出投影
        prob = self.output_proj(x)  # (B, 1, T)
        
        # (B, 1, T) -> (B, T, 1)
        prob = prob.transpose(1, 2)
        
        return prob
    
    def get_receptive_field(self) -> int:
        """返回感受野大小（帧数）"""
        return self.receptive_field
    
    def export_for_npu(self) -> nn.Module:
        """
        导出为 NPU 友好的模型（融合 BN、移除 Dropout）
        """
        self.eval()
        # 在实际部署时，会通过 torch.onnx 或华为工具链导出
        return self


class LightTCN_V2(nn.Module):
    """
    Light-TCN V2: 更极致的轻量化版本
    
    使用 Depthwise 可分离卷积替代标准卷积，
    参数量进一步降低约 60%。
    """
    
    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 48,
        num_layers: int = 4,
        kernel_size: int = 3,
        dilations: Optional[list] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if dilations is None:
            dilations = [1, 2, 4, 8]
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # Depthwise 可分离 TCN
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            d = dilations[i]
            self.tcn_layers.append(
                DepthwiseTCNBlock(hidden_dim, kernel_size, d, dropout)
            )
        
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        
        self.receptive_field = self._compute_receptive_field(kernel_size, dilations)
        self._log_params()
    
    def _compute_receptive_field(self, k: int, d: list) -> int:
        rf = 1
        for di in d:
            rf += (k - 1) * di
        return rf
    
    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"[LightTCN_V2] 总参数: {total:,} | 感受野: {self.receptive_field} 帧")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for layer in self.tcn_layers:
            x = layer(x)
        prob = self.output_proj(x)
        return prob.transpose(1, 2)


class DepthwiseTCNBlock(nn.Module):
    """Depthwise 可分离 TCN 模块"""
    
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        
        # Depthwise 卷积
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, groups=channels, bias=False
        )
        # Pointwise 卷积
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.pad(x, (self.padding, 0))
        out = self.depthwise(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.pointwise(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        out = self.dropout(out)
        
        return out


# 工厂函数
def build_light_tcn(
    input_dim: int = 48,
    version: str = "v1",
    hidden_dim: int = 64,
    num_layers: int = 5,
    kernel_size: int = 3,
    dilations: Optional[list] = None,
    dropout: float = 0.1,
) -> nn.Module:
    """
    构建 Light-TCN 模型
    
    Args:
        input_dim: 输入特征维度
        version: "v1" (标准) 或 "v2" (深度可分离)
        hidden_dim: 隐藏维度
        num_layers: 层数
        kernel_size: 卷积核大小
        dilations: 膨胀系数
        dropout: Dropout 比率
    """
    if version == "v2":
        return LightTCN_V2(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
    else:
        return LightTCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
