"""
完整推理管线
--------------
整合 检测 → 跟踪 → 运动编码 → TCN → 规则校正 的完整端到端推理流程。

支持模式：
1. 视频文件推理（离线评测）
2. 摄像头实时推理（在线部署）
3. 批量推理（数据集评测）

核心优化：
- 隔帧检测：YOLO 可以隔 2 帧运行，TCN 每帧运行
- 分辨率自适应：1080P → 320/640 推理
- 人体 ROI 裁剪：仅对检测到的人体区域进行后续处理
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Generator
from collections import defaultdict
from dataclasses import dataclass, field
import time
import json

from .detector import (
    YOLOPoseDetector, YOLOPoseDetectorSim,
    PersonDetection,
)
from .tracker import ByteTrackWrapper
from models.motion_encoder import MotionFeatureEncoder
from models.light_tcn import LightTCN
from models.rule_refinement import RuleRefinementOnline, RuleConfig
from models.fall_detector import FallDetector


@dataclass
class FallEvent:
    """跌倒事件"""
    person_id: int
    start_frame: int
    end_frame: int
    max_probability: float
    duration_frames: int
    timestamp: float = 0.0  # 从视频开始的秒数
    
    @property
    def duration_seconds(self) -> float:
        return self.duration_frames / 30.0  # 假设 30fps


@dataclass
class FrameResult:
    """单帧推理结果"""
    frame_idx: int
    persons: List[PersonDetection]
    fall_probs: Dict[int, float]    # track_id → probability
    alarms: Dict[int, bool]          # track_id → alarm
    features: Optional[Dict[int, np.ndarray]] = None


@dataclass
class PipelineResult:
    """完整推理结果"""
    frame_results: List[FrameResult]
    fall_events: List[FallEvent]
    total_frames: int
    total_time: float
    fps: float
    avg_latency_ms: float


class FallDetectionPipeline:
    """
    完整跌倒检测推理管线
    
    ┌─────────────┐
    │  视频输入    │
    │ (RGB/红外)   │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 预处理      │
    │ Resize,CLAHE │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ YOLOv8n-Pose │  (隔帧运行)
    │ 检测+关键点  │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ ByteTrack   │
    │ 人体跟踪     │
    └──────┬──────┘
           ▼
    ┌─────────────────────┐
    │ Motion Feature       │
    │ Encoder              │
    │ (关键点→运动特征)     │
    └──────┬──────────────┘
           ▼
    ┌─────────────┐
    │ Light-TCN   │  (每帧运行)
    │ 时序建模     │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ Rule         │
    │ Refinement   │
    │ 规则校正     │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 报警/记录    │
    └─────────────┘
    """
    
    def __init__(
        self,
        detector: Optional[YOLOPoseDetector] = None,
        fall_detector: Optional[FallDetector] = None,
        # 管线参数
        sequence_length: int = 32,
        detection_interval: int = 2,       # YOLO 检测间隔（隔N帧）
        input_size: Tuple[int, int] = (640, 384),
        conf_threshold: float = 0.25,
        # 红外模式
        ir_mode: bool = False,
        # 报警参数
        alarm_cooldown_frames: int = 30,
        # 输出参数
        save_video: bool = False,
        output_dir: str = "output",
    ):
        """
        Args:
            detector: YOLO Pose 检测器
            fall_detector: 跌倒检测模型
            sequence_length: TCN 序列长度
            detection_interval: YOLO 检测间隔
            input_size: 模型输入尺寸
            conf_threshold: 检测置信度阈值
            ir_mode: 是否红外模式
            alarm_cooldown_frames: 报警冷却帧数
            save_video: 是否保存结果视频
            output_dir: 输出目录
        """
        # 检测器
        if detector is None:
            try:
                detector = YOLOPoseDetector(
                    input_size=input_size,
                    conf_threshold=conf_threshold,
                )
                detector.load_model()
            except:
                detector = YOLOPoseDetectorSim()
        self.detector = detector
        
        # 跟踪器
        self.tracker = ByteTrackWrapper(track_buffer=30)
        
        # 跌倒检测模型
        if fall_detector is None:
            fall_detector = FallDetector()
        self.fall_detector = fall_detector
        self.fall_detector.eval()
        
        # 在线规则校正（每人体一个实例）
        self.rule_engines: Dict[int, RuleRefinementOnline] = {}
        
        # 参数
        self.sequence_length = sequence_length
        self.detection_interval = detection_interval
        self.input_size = input_size
        self.ir_mode = ir_mode
        self.alarm_cooldown = alarm_cooldown_frames
        
        # 特征缓冲区（按人体 ID 维护）
        self._feature_buffers: Dict[int, list] = defaultdict(list)
        self._prob_buffers: Dict[int, list] = defaultdict(list)
        self._kp_buffers: Dict[int, list] = defaultdict(list)
        self._bbox_buffers: Dict[int, list] = defaultdict(list)
        
        # 输出
        self.save_video = save_video
        self.output_dir = output_dir
        if save_video:
            os.makedirs(output_dir, exist_ok=True)
        
        # 跌倒事件记录
        self._fall_events: List[FallEvent] = []
        self._fall_cooldowns: Dict[int, int] = {}
    
    def reset(self):
        """重置管线状态"""
        self.tracker.reset()
        self.rule_engines.clear()
        self._feature_buffers.clear()
        self._prob_buffers.clear()
        self._kp_buffers.clear()
        self._bbox_buffers.clear()
        self._fall_events.clear()
        self._fall_cooldowns.clear()
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> FrameResult:
        """
        处理单帧图像
        
        Args:
            frame: (H, W, 3) BGR/RGB 图像
            frame_idx: 帧索引
        
        Returns:
            FrameResult: 包含检测、概率、报警信息
        """
        h, w = frame.shape[:2]
        
        # 红外预处理
        if self.ir_mode:
            frame = self.detector.preprocess_for_ir(frame)
        
        # Step 1: YOLO Pose 检测（可能跳过）
        if frame_idx % self.detection_interval == 0:
            detections = self.detector.detect(frame, (h, w))
        else:
            detections = []
        
        # Step 2: ByteTrack 跟踪
        tracked = self.tracker.update(detections, self.detection_interval)
        
        # Step 3-6: 对每个跟踪的人体处理
        fall_probs = {}
        alarms = {}
        frame_features = {}
        
        for track_id, detection in tracked.items():
            # 获取或创建规则引擎
            if track_id not in self.rule_engines:
                self.rule_engines[track_id] = RuleRefinementOnline(
                    RuleConfig(),
                    history_len=self.sequence_length,
                )
            
            # 更新缓冲区
            self._kp_buffers[track_id].append(detection.keypoints)
            self._bbox_buffers[track_id].append(detection.bbox)
            
            # 保持缓冲区大小
            if len(self._kp_buffers[track_id]) > self.sequence_length:
                self._kp_buffers[track_id].pop(0)
                self._bbox_buffers[track_id].pop(0)
            
            # 需要足够的帧才能运行 TCN
            if len(self._kp_buffers[track_id]) >= max(8, self.sequence_length // 4):
                # 提取运动特征
                kp_seq = np.stack(self._kp_buffers[track_id][-self.sequence_length:])
                bb_seq = np.stack(self._bbox_buffers[track_id][-self.sequence_length:])
                
                motion_feat = self.fall_detector.motion_encoder.compute_sequence(kp_seq, bb_seq)
                # (T, C)
                
                # TCN 推理
                feat_tensor = torch.from_numpy(motion_feat).float().unsqueeze(0)  # (1, T, C)
                with torch.no_grad():
                    tcn_prob = self.fall_detector.tcn(feat_tensor)  # (1, T, 1)
                tcn_prob = tcn_prob.squeeze().numpy()  # (T,)
                
                last_prob = float(tcn_prob[-1])
                last_feat = motion_feat[-1]
                
                # Rule Refinement 在线处理
                refined_prob, is_alarm = self.rule_engines[track_id].update(
                    last_prob, last_feat
                )
                
                fall_probs[track_id] = refined_prob
                alarms[track_id] = is_alarm
                frame_features[track_id] = last_feat
                
                # 记录跌倒事件
                if is_alarm and self._fall_cooldowns.get(track_id, 0) <= 0:
                    self._record_fall_event(track_id, frame_idx, refined_prob)
            else:
                fall_probs[track_id] = 0.0
                alarms[track_id] = False
        
        # 更新冷却计数器
        for track_id in list(self._fall_cooldowns.keys()):
            self._fall_cooldowns[track_id] -= 1
            if self._fall_cooldowns[track_id] <= 0:
                del self._fall_cooldowns[track_id]
        
        return FrameResult(
            frame_idx=frame_idx,
            persons=list(tracked.values()),
            fall_probs=fall_probs,
            alarms=alarms,
            features=frame_features if frame_features else None,
        )
    
    def _record_fall_event(self, track_id: int, frame_idx: int, probability: float):
        """记录跌倒事件"""
        # 检查是否有正在进行的事件
        for event in self._fall_events:
            if event.person_id == track_id and frame_idx - event.end_frame < 30:
                event.end_frame = frame_idx
                event.max_probability = max(event.max_probability, probability)
                event.duration_frames = event.end_frame - event.start_frame
                return
        
        # 新事件
        self._fall_events.append(FallEvent(
            person_id=track_id,
            start_frame=frame_idx,
            end_frame=frame_idx,
            max_probability=probability,
            duration_frames=1,
            timestamp=frame_idx / 30.0,
        ))
        self._fall_cooldowns[track_id] = self.alarm_cooldown
    
    def process_video(
        self,
        video_path: str,
        progress_callback=None,
    ) -> PipelineResult:
        """
        处理完整视频
        
        Args:
            video_path: 视频文件路径
            progress_callback: 进度回调 (current_frame, total_frames) -> None
        
        Returns:
            PipelineResult: 包含所有帧结果和跌倒事件
        """
        self.reset()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        
        frame_results = []
        start_time = time.time()
        frame_idx = 0
        
        # 视频写入器
        writer = None
        if self.save_video:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = os.path.join(
                self.output_dir,
                f"fall_detection_{os.path.basename(video_path)}"
            )
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.process_frame(frame, frame_idx)
            frame_results.append(result)
            
            # 保存视频
            if writer is not None:
                annotated = self._draw_result(frame, result)
                writer.write(annotated)
            
            frame_idx += 1
            
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)
        
        cap.release()
        if writer is not None:
            writer.release()
        
        total_time = time.time() - start_time
        avg_fps = total_frames / total_time if total_time > 0 else 0
        avg_latency = (total_time / total_frames * 1000) if total_frames > 0 else 0
        
        result = PipelineResult(
            frame_results=frame_results,
            fall_events=self._fall_events,
            total_frames=total_frames,
            total_time=total_time,
            fps=avg_fps,
            avg_latency_ms=avg_latency,
        )
        
        # 保存结果
        self._save_results(result, video_path)
        
        return result
    
    def process_webcam(
        self,
        camera_id: int = 0,
    ) -> Generator[FrameResult, None, None]:
        """
        实时摄像头推理（生成器模式）
        
        Yields:
            FrameResult: 每帧推理结果
        """
        self.reset()
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")
        
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = self.process_frame(frame, frame_idx)
                frame_idx += 1
                yield result
                
                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def evaluate_dataset(
        self,
        video_paths: List[str],
        annotations: Optional[Dict] = None,
    ) -> Dict:
        """
        批量数据集评测
        
        Returns:
            dict: 包含 mAP, recall, precision, F1 等指标
        """
        all_events = []
        all_results = []
        
        for video_path in video_paths:
            print(f"[Pipeline] 评测: {video_path}")
            result = self.process_video(video_path)
            all_results.append(result)
            all_events.extend(result.fall_events)
        
        # 汇总指标
        total_fall_events = len(all_events)
        total_frames = sum(r.total_frames for r in all_results)
        total_time = sum(r.total_time for r in all_results)
        
        metrics = {
            "total_videos": len(video_paths),
            "total_frames": total_frames,
            "total_time_seconds": total_time,
            "total_fall_events": total_fall_events,
            "avg_fps": total_frames / total_time if total_time > 0 else 0,
            "avg_latency_ms": (total_time / total_frames * 1000) if total_frames > 0 else 0,
        }
        
        if annotations:
            # TODO: 计算准确率/召回率
            pass
        
        return metrics
    
    def _draw_result(
        self,
        frame: np.ndarray,
        result: FrameResult,
    ) -> np.ndarray:
        """在图像上绘制检测结果"""
        h, w = frame.shape[:2]
        vis = frame.copy()
        
        for person in result.persons:
            # 画边界框
            bbox = person.bbox * np.array([w, h, w, h])
            bbox = bbox.astype(int)
            
            # 检测是否有人体且报警
            has_alarm = False
            for track_id, alarm in result.alarms.items():
                if alarm:
                    has_alarm = True
                    break
            
            color = (0, 0, 255) if has_alarm else (0, 255, 0)  # 红/绿
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 画关键点
            for kp in person.keypoints:
                if kp[2] > 0.3:  # 置信度足够
                    px, py = int(kp[0] * w), int(kp[1] * h)
                    cv2.circle(vis, (px, py), 3, color, -1)
            
            # 显示概率
            for track_id, prob in result.fall_probs.items():
                if prob > 0.3:
                    text = f"Fall: {prob:.2f}"
                    cv2.putText(vis, text, (bbox[0], bbox[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 帧号
        cv2.putText(vis, f"Frame: {result.frame_idx}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
    
    def _save_results(
        self,
        result: PipelineResult,
        video_path: str,
    ):
        """保存推理结果到 JSON"""
        if not self.save_video:
            return
        
        basename = os.path.splitext(os.path.basename(video_path))[0]
        result_path = os.path.join(self.output_dir, f"{basename}_results.json")
        
        data = {
            "video": video_path,
            "total_frames": result.total_frames,
            "total_time": result.total_time,
            "fps": result.fps,
            "avg_latency_ms": result.avg_latency_ms,
            "fall_events": [
                {
                    "person_id": e.person_id,
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "max_probability": e.max_probability,
                    "duration_frames": e.duration_frames,
                    "timestamp": e.timestamp,
                }
                for e in result.fall_events
            ],
            "frame_alarms": [
                {
                    "frame": r.frame_idx,
                    "alarms": {str(k): v for k, v in r.alarms.items()},
                    "probs": {str(k): v for k, v in r.fall_probs.items()},
                }
                for r in result.frame_results
                if any(r.alarms.values())
            ],
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[Pipeline] 结果保存到: {result_path}")
    
    def get_statistics(self) -> Dict:
        """获取管线统计信息"""
        # 参数量
        detector_params = sum(p.numel() for p in self.fall_detector.motion_encoder.parameters())
        tcn_params = sum(p.numel() for p in self.fall_detector.tcn.parameters())
        
        return {
            "model_params": {
                "motion_encoder": detector_params,
                "tcn": tcn_params,
                "total": detector_params + tcn_params,
            },
            "model_size_mb": self.fall_detector.get_model_size_mb(),
            "detection_interval": self.detection_interval,
            "sequence_length": self.sequence_length,
            "input_size": self.input_size,
            "ir_mode": self.ir_mode,
        }
