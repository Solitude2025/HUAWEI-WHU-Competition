# 轻量级时空语义跌倒检测框架

> **华为专项赛道：面向低算力端侧平台基于视觉的实时跌倒检测**

---

## 方案概述

本方案不采用传统的"YOLOv8n + TCN"直接分类跌倒的方式，而是设计了一个**完整的轻量级时空语义跌倒检测框架**，核心创新围绕三个模块展开：

| 模块 | 作用 | 创新点 |
|------|------|--------|
| **Motion Feature Encoder** | 将人体关键点序列编码为运动语义特征（~48维） | 替代高维 CNN Feature（640-1024维），TCN 计算量暴跌 |
| **Light-TCN** | 膨胀因果卷积实现轻量时序建模 | 全卷积结构，天然支持 NPU INT8 量化，部署优势远超 GRU |
| **Rule Refinement** | 物理约束后处理，时空一致性校正 | 几乎零计算量，大幅降低"坐下/弯腰/躺卧"误报 |

### 完整链路

```
视频输入(RGB/红外)
    │
    ▼
图像预处理(Resize, CLAHE)
    │
    ▼
YOLOv8n-Pose (人体检测+17关键点)
    │
    ▼
ByteTrack (人体ID连续跟踪)
    │
    ▼
Motion Feature Encoder (创新模块)
├── 人体中心速度       ├── 关键点速度
├── 人体框宽高比       ├── 躯干倾角
├── 人体面积变化       └── 重心变化
    │
    ▼
Light-TCN (8-16帧时序建模)
    │
    ▼
Rule Refinement (创新模块)
├── 角度持续时间判断   ├── 速度峰值判断
├── 人体静止持续时间   └── 连续多帧投票
    │
    ▼
报警 / 视频记录 / 日志
```

---

## 赛题合规

| 指标 | 限制 | 本方案 | 状态 |
|------|------|--------|------|
| 模型总参数 | ≤ 20M | ~70K (不含 YOLO) | ✓ |
| FP32 大小 | ≤ 80MB | ~0.3MB | ✓ |
| 推理延迟 | ≤ 100ms | ~5ms (TCN+Ruler) | ✓ |
| 纯视觉方案 | 是 | ✓ | ✓ |
| 可见光+红外 | 支持 | ✓ | ✓ |
| NPU 部署 | 要求 | 全卷积，支持 INT8 | ✓ |

---

## 项目结构

```
fall_detection/
├── models/                     # 核心模型
│   ├── motion_encoder.py       # 运动特征编码器
│   ├── light_tcn.py            # 轻量时序卷积网络
│   ├── rule_refinement.py      # 时空一致性规则校正
│   └── fall_detector.py        # 完整跌倒检测模型
├── pipeline/                   # 推理管线
│   ├── detector.py             # YOLOv8n-Pose 封装
│   ├── tracker.py              # ByteTrack 封装
│   └── inference.py            # 完整推理管线
├── data/                       # 数据处理
│   ├── augment.py              # 数据增强 + 红外域适配
│   └── dataset.py              # 数据集加载器
├── utils/                      # 工具
│   ├── metrics.py              # 评估指标
│   └── visualization.py        # 可视化
├── configs/
│   └── config.yaml             # 配置文件
├── train.py                    # 训练脚本
├── infer.py                    # 推理脚本
├── export.py                   # 模型导出(NPU部署)
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 基础训练（使用合成数据演示）
python train.py --epochs 50 --batch_size 32 --version standard

# 使用真实数据训练
python train.py --data_dir data/train --val_dir data/val --epochs 100

# 使用配置文件
python train.py --config configs/config.yaml
```

### 3. 推理测试

```bash
# 视频文件推理
python infer.py --video test.mp4 --checkpoint checkpoints/best.pth --save

# 实时摄像头
python infer.py --webcam --checkpoint checkpoints/best.pth

# 批量评测
python infer.py --dir test_videos/ --checkpoint checkpoints/best.pth --eval

# 红外模式
python infer.py --video ir_test.mp4 --checkpoint checkpoints/best.pth --ir --save
```

### 4. NPU 部署导出

```bash
# 导出 ONNX
python export.py --checkpoint checkpoints/best.pth --format onnx

# 导出 TorchScript
python export.py --checkpoint checkpoints/best.pth --format torchscript

# 华为 Ascend NPU 部署
atc --model=fall_detector_standard.onnx \
    --framework=5 \
    --output=fall_detector \
    --soc_version=Ascend310 \
    --input_shape="motion_features:1,32,48" \
    --input_format=ND
```

---

## 三大痛点针对性优化

### 痛点 1：动作语义混淆

**问题**：快速坐下、弯腰、躺下与跌倒视觉表征相似，YOLO 直接分类误报频发。

**解决方案**：
- **Motion Feature Encoder**：不做 RGB → 跌倒，而是提取人体运动学特征
- **关键点轨迹分析**：跌倒时所有关键点同时快速下降，坐下只有臀部下降
- **Rule Refinement**：躯干角度 < 30° 持续判定 + 速度峰值检查 + 多帧投票

### 痛点 2：端侧算力瓶颈

**问题**：3D-CNN / Video Transformer 推理延迟高，无法满足实时性。

**解决方案**：
- **Motion Feature Encoder**：将 640-1024 维 CNN Feature 压缩到 ~48 维
- **Light-TCN**：全卷积结构，天然支持 NPU INT8 量化
- **隔帧检测优化**：YOLO 隔 2 帧运行，TCN 每帧运行
- **分辨率自适应**：1080P → 640×384 推理

### 痛点 3：长尾场景泛化差

**问题**：低光/红外/遮挡/视角变化场景下精度下降。

**解决方案**：
- **红外域适配**：RGB → 灰度红外风格迁移增强
- **低光增强**：Gamma 校正 + CLAHE
- **遮挡模拟**：Random Erasing + Copy-Paste 遮挡
- **视角增强**：随机透视变换 + 仿射变换

---

## 数据准备

### 参考数据集

- [OmniFall (HuggingFace)](https://huggingface.co/datasets/simplexsigil2/omnifall)
- [Fall Video Dataset (Kaggle)](https://www.kaggle.com/datasets/payutch/fall-video-dataset)

### 数据格式

```
data/
├── train/
│   ├── video_001/
│   │   ├── person_0/
│   │   │   ├── keypoints.npy    # (T, 17, 3)
│   │   │   ├── bboxes.npy       # (T, 4)
│   │   │   └── labels.npy       # (T,)
│   │   └── person_1/
│   │       └── ...
│   └── video_002/
│       └── ...
└── val/
    └── ...
```

---

## 模型配置

### Standard (推荐)
- Motion Feature: 48 维
- TCN: 5层, hidden=64, dilations=[1,2,4,8,16]
- 参数: ~60K

### Light (极致轻量)
- TCN: 4层, hidden=32, depthwise separable
- 参数: ~20K

### Large (高精度)
- Motion Feature: 64 维
- TCN: 6层, hidden=128, kernel=5
- 参数: ~200K

---

## 参考文献

1. LFD-YOLO: a lightweight fall detection network with enhanced feature extraction and fusion
2. BMR-YOLO: A deep learning approach for fall detection in complex environments
3. YOLO-fall: a YOLO-based fall detection model with high precision, shrunk size, and low latency
4. ByteTrack: Multi-Object Tracking by Associating Every Detection Box
5. Temporal Convolutional Networks: A Unified Approach to Action Segmentation
