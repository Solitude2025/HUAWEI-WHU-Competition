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
    is_detect_frame: bool = True     # 当前帧是否执行了 YOLO 检测
    raw_detections: List[PersonDetection] = field(default_factory=list)  # YOLO 原始检测结果


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
        # 规则配置（从 config.yaml 传入）
        rule_config: Optional[RuleConfig] = None,
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
        
        # 规则配置（从 config.yaml 传入）
        self.rule_config = RuleConfig()
        if rule_config is not None:
            self.rule_config = rule_config
        
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
            raw_detections = self.detector.detect(frame, (h, w))
            detections = raw_detections
            is_detect_frame = True
        else:
            raw_detections = []
            detections = []
            is_detect_frame = False
        
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
                    self.rule_config,
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
                
                # 记录跌倒事件（靠合并逻辑防重复，不依赖track_id冷却）
                if is_alarm:
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
            is_detect_frame=is_detect_frame,
            raw_detections=raw_detections,
        )
    
    def _record_fall_event(self, track_id: int, frame_idx: int, probability: float):
        """记录跌倒事件（按时间窗口合并，不区分track_id）"""
        # ByteTrack 可能频繁分配新 ID，所以按帧窗口合并而非 person_id
        for event in self._fall_events:
            if frame_idx - event.start_frame < 300:
                event.end_frame = frame_idx
                event.duration_frames = event.end_frame - event.start_frame
                event.max_probability = max(event.max_probability, probability)
                # 合并 person 范围
                if track_id != event.person_id:
                    event.person_id = -1  # 标记为多人/不稳定
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
            basename = os.path.basename(video_path)
            out_path = os.path.join(self.output_dir, basename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            print(f"[Pipeline] 输出视频: {out_path}")
        
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
        """在图像上绘制检测结果：骨架、边框、Fall 状态

        关键设计：只在 YOLO 检测帧画人体框/骨架，非检测帧只保留状态面板。
        这样不会出现"旧框残留"问题。
        """
        h, w = frame.shape[:2]
        vis = frame.copy()

        # ── 判断跌倒状态 ──
        is_any_alarm = any(result.alarms.values())

        # ── 左上角：Fall 状态面板（每帧都画） ──
        panel_x, panel_y = 15, 15
        panel_w, panel_h = 260, 70
        overlay = vis.copy()
        if is_any_alarm:
            cv2.rectangle(overlay, (panel_x, panel_y),
                         (panel_x + panel_w, panel_y + panel_h), (0, 0, 60), -1)
        else:
            cv2.rectangle(overlay, (panel_x, panel_y),
                         (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        border_color = (0, 0, 255) if is_any_alarm else (100, 100, 100)
        cv2.rectangle(vis, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), border_color, 2)
        status_text = "Fall: YES" if is_any_alarm else "Fall: Not"
        status_color = (0, 0, 255) if is_any_alarm else (0, 255, 0)
        cv2.putText(vis, status_text, (panel_x + 12, panel_y + 38),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, status_color, 2, cv2.LINE_AA)
        if result.fall_probs:
            max_prob = max(result.fall_probs.values())
            cv2.putText(vis, f"Prob: {max_prob:.3f}", (panel_x + 12, panel_y + 62),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # ── 只在 YOLO 检测帧画人体框/骨架 ──
        if result.is_detect_frame and result.raw_detections:
            from utils.visualization import SKELETON_CONNECTIONS, SKELETON_COLORS

            # 用 YOLO 原始检测结果画框和骨架（不画 tracker 推演的位置）
            for det in result.raw_detections:
                color_bgr = (0, 0, 255) if is_any_alarm else (0, 255, 0)

                # 边界框
                bbox = det.bbox * np.array([w, h, w, h])
                bbox = bbox.astype(int)
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_bgr, 2)

                # 骨架连线
                kp_pixel = det.keypoints[:, :2] * np.array([w, h])
                kp_pixel = kp_pixel.astype(int)
                kp_conf = det.keypoints[:, 2]
                for conn_idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
                    if kp_conf[i] > 0.3 and kp_conf[j] > 0.3:
                        line_clr = SKELETON_COLORS[conn_idx % len(SKELETON_COLORS)]
                        cv2.line(vis, tuple(kp_pixel[i]), tuple(kp_pixel[j]),
                                line_clr, 2, cv2.LINE_AA)

                # 关键点圆点
                for i in range(17):
                    if kp_conf[i] > 0.3:
                        px, py = kp_pixel[i]
                        cv2.circle(vis, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(vis, (px, py), 5, color_bgr, 2, cv2.LINE_AA)

                # 人体框上方显示概率
                prob_val = max(result.fall_probs.values()) if result.fall_probs else 0.0
                label = f"{'FALL' if is_any_alarm else 'Normal'} {prob_val:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(vis, (bbox[0], bbox[1] - th - 8),
                             (bbox[0] + tw + 6, bbox[1]), color_bgr, -1)
                cv2.putText(vis, label, (bbox[0] + 3, bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # ── 右下角：帧号 ──
        frame_text = f"Frame: {result.frame_idx}"
        (fw, fh), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (w - fw - 25, h - fh - 20),
                     (w - 10, h - 8), (0, 0, 0), -1)
        cv2.putText(vis, frame_text, (w - fw - 20, h - 14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return vis
    
    def _save_results(
        self,
        result: PipelineResult,
        video_path: str,
    ):
        """保存推理结果到 JSON"""
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
