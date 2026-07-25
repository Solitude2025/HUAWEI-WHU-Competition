"""
ByteTrack 人体跟踪封装
------------------------
轻量级多目标跟踪，保持人体 ID 连续性。

作用：
1. 维持同一人体的时序 ID，确保关键点序列属于同一个人
2. 支持隔帧检测优化：无新人进入时不重新检测，仅更新跟踪
3. 减少 YOLO 推理次数（可隔2帧运行），提升整体速度
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .detector import PersonDetection


@dataclass
class TrackedPerson:
    """跟踪中的人体"""
    track_id: int
    detection: PersonDetection
    age: int = 0                     # 跟踪持续的帧数
    lost_count: int = 0              # 连续丢失帧数
    history: List[PersonDetection] = field(default_factory=list)
    
    def update(self, detection: PersonDetection):
        """更新跟踪状态"""
        self.detection = detection
        self.age += 1
        self.lost_count = 0
        self.history.append(detection)
        if len(self.history) > 60:
            self.history.pop(0)
    
    def mark_lost(self):
        """标记为丢失"""
        self.lost_count += 1


class ByteTrackWrapper:
    """
    ByteTrack 轻量封装
    
    ByteTrack 优势：
    - 基于 IoU 匹配，计算极轻量
    - 支持低分框关联，减少 ID Switch
    - 适合端侧部署
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,         # 高分框阈值
        match_thresh: float = 0.8,          # 首次匹配 IoU 阈值
        second_match_thresh: float = 0.5,   # 二次匹配 IoU 阈值
        track_buffer: int = 30,             # 丢失后保留帧数
        frame_rate: int = 30,               # 帧率
        max_age: int = 60,                  # 最大跟踪寿命
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.max_age = max_age
        
        # 内部状态
        self._next_id = 0
        self._tracks: Dict[int, TrackedPerson] = {}
        self._frame_count = 0
    
    def reset(self):
        """重置跟踪器状态"""
        self._next_id = 0
        self._tracks.clear()
        self._frame_count = 0
    
    def update(
        self,
        detections: List[PersonDetection],
        detection_skip: int = 1,
    ) -> Dict[int, PersonDetection]:
        """
        更新跟踪状态
        
        Args:
            detections: 当前帧检测结果
            detection_skip: 检测间隔（如为2则隔帧检测）
        
        Returns:
            Dict[track_id, PersonDetection]: 当前帧的跟踪结果
        """
        self._frame_count += 1
        
        # 如果隔帧检测且当前帧没有新检测
        if detection_skip > 1 and self._frame_count % detection_skip != 0 and len(detections) == 0:
            return self._predict_only()
        
        # 分离高低分框
        high_score_dets = [d for d in detections if d.confidence >= self.track_thresh]
        low_score_dets = [d for d in detections if d.confidence < self.track_thresh]
        
        # 获取活跃跟踪
        active_tracks = self._get_active_tracks()
        
        # Step 1: 高分框与所有活跃跟踪匹配
        matched, unmatched_dets, unmatched_trks = self._match_detections(
            high_score_dets, active_tracks, self.match_thresh
        )
        
        # Step 2: 更新已匹配的跟踪
        for track_id, det_idx in matched:
            self._tracks[track_id].update(high_score_dets[det_idx])
        
        # Step 3: 低分框与剩余未匹配跟踪匹配
        remaining_dets = [high_score_dets[i] for i in unmatched_dets] + low_score_dets
        
        if len(remaining_dets) > 0:
            remaining_trks = [active_tracks[i] for i in unmatched_trks]
            matched2, _, _ = self._match_detections(
                remaining_dets, remaining_trks, self.second_match_thresh
            )
            
            # Map back to original indices
            for track_id_remap, det_idx in matched2:
                orig_trk_id = unmatched_trks[track_id_remap]
                self._tracks[orig_trk_id].update(remaining_dets[det_idx])
        
        # Step 4: 未匹配的高分框 → 新跟踪
        newly_matched = set(i for i, _ in matched)
        newly_matched2 = set()
        for trk_remap, det_idx in matched2:
            newly_matched2.add(det_idx)
        
        # 即没有在匹配1也没有在匹配2中的高分框 → 创建新跟踪
        for i, det in enumerate(high_score_dets):
            if i not in newly_matched and i not in newly_matched2:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = TrackedPerson(
                    track_id=track_id,
                    detection=det,
                    age=0,
                    history=[det],
                )
        
        # Step 5: 标记丢失的跟踪
        matched_trk_ids = set(i for i, _ in matched)
        for track_id, track in self._tracks.items():
            if track_id not in matched_trk_ids:
                track.mark_lost()
        
        # Step 6: 清理过期跟踪
        self._cleanup()
        
        # 返回当前帧结果
        return self._get_current_tracks()
    
    def _predict_only(self) -> Dict[int, PersonDetection]:
        """仅预测（无新检测时的跟踪维持）"""
        # 保持现有跟踪
        return {
            tid: t.detection
            for tid, t in self._tracks.items()
            if t.lost_count < self.track_buffer
        }
    
    def _get_active_tracks(self) -> List['TrackedPerson']:
        """获取活跃的跟踪目标"""
        return [
            t for t in self._tracks.values()
            if t.lost_count < self.track_buffer and t.age > 0
        ]
    
    def _match_detections(
        self,
        detections: List[PersonDetection],
        tracks: List['TrackedPerson'],
        iou_threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        IoU 匹配
        
        Returns:
            matched: [(track_index, det_index), ...]
            unmatched_dets: [det_index, ...]
            unmatched_tracks: [track_index, ...]
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # 计算 IoU 矩阵
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for i, det in enumerate(detections):
            for j, trk in enumerate(tracks):
                iou_matrix[i, j] = self._compute_iou(
                    det.bbox, trk.detection.bbox
                )
        
        # 贪婪匹配（简化版匈牙利算法）
        matched = []
        used_dets = set()
        used_trks = set()
        
        # 按 IoU 降序排列
        indices = np.argsort(-iou_matrix, axis=None)
        for idx in indices:
            i = idx // len(tracks)
            j = idx % len(tracks)
            if iou_matrix[i, j] >= iou_threshold and i not in used_dets and j not in used_trks:
                matched.append((j, i))
                used_dets.add(i)
                used_trks.add(j)
        
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_trks = [j for j in range(len(tracks)) if j not in used_trks]
        
        return matched, unmatched_dets, unmatched_trks
    
    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """计算两个边界框的 IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter_area = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return inter_area / (area1 + area2 - inter_area + 1e-6)
    
    def _get_current_tracks(self) -> Dict[int, PersonDetection]:
        """获取当前帧的有效跟踪结果"""
        return {
            tid: t.detection
            for tid, t in self._tracks.items()
            if t.lost_count < self.track_buffer
        }
    
    def _cleanup(self):
        """清理过期跟踪"""
        to_remove = [
            tid for tid, t in self._tracks.items()
            if t.lost_count > self.track_buffer * 2
        ]
        for tid in to_remove:
            del self._tracks[tid]
    
    def get_track_history(
        self,
        track_id: int,
        max_len: int = 32,
    ) -> List[PersonDetection]:
        """获取指定跟踪 ID 的历史检测"""
        if track_id not in self._tracks:
            return []
        return self._tracks[track_id].history[-max_len:]
    
    @property
    def active_count(self) -> int:
        """当前活跃跟踪数"""
        return sum(1 for t in self._tracks.values() if t.lost_count < self.track_buffer)
