"""自动标注核心类 — 环形缓冲 + 命中检测 + clip 保存.

通过 HealthBarTracker 检测 is_hit 事件，回溯缓冲区保存攻击 clip。
"""

import json
import os
import time
from collections import deque

import cv2

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    AUTO_LABEL_DIR, BUFFER_SIZE, CAPTURE_FPS,
    LOOKBACK_SEC, LOOKFORWARD_SEC, HIT_COOLDOWN_SEC, INPUT_SIZE,
)


class AutoLabeler:
    """从实时帧流中自动检测命中事件并保存攻击 clip.

    核心流程:
        add_frame(frame, ts) → check_hit_event(health_state, ts)
        命中 → 回溯 LOOKBACK_SEC + 继续录制 LOOKFORWARD_SEC → 保存 clip
    """

    def __init__(self, output_dir=None, session_name=None):
        """初始化 AutoLabeler.

        Args:
            output_dir: clip 输出根目录, 默认 AUTO_LABEL_DIR.
            session_name: session 子目录名, 默认用时间戳.
        """
        self.output_dir = output_dir or AUTO_LABEL_DIR

        if session_name is None:
            session_name = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = os.path.join(self.output_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)

        # 环形缓冲: 存 (frame, timestamp) 元组
        self._buffer = deque(maxlen=BUFFER_SIZE)

        # clip 计数
        self._clip_count = 0
        self._clips_meta = []  # session 级汇总

        # 冷却 & lookforward 状态
        self._last_hit_time = -999.0
        self._lookforward_active = False
        self._lookforward_frames = []
        self._lookforward_remaining = 0
        self._pending_lookback = None  # 暂存 lookback 帧 + hit 信息

        # 计算帧数
        self._lookback_frames = int(LOOKBACK_SEC * CAPTURE_FPS)
        self._lookforward_frame_count = int(LOOKFORWARD_SEC * CAPTURE_FPS)

    @property
    def clip_count(self):
        return self._clip_count

    def add_frame(self, frame, timestamp):
        """添加一帧到环形缓冲，并处理 lookforward 录制.

        Args:
            frame: BGR 图像 (已裁剪 ROI 并 resize 到 INPUT_SIZE).
            timestamp: 帧时间戳 (秒).
        """
        self._buffer.append((frame.copy(), timestamp))

        # 如果正在 lookforward 录制
        if self._lookforward_active:
            self._lookforward_frames.append(frame.copy())
            self._lookforward_remaining -= 1
            if self._lookforward_remaining <= 0:
                self._save_clip()
                self._lookforward_active = False

    def check_hit_event(self, health_state, timestamp):
        """检测命中事件并触发 clip 提取.

        Args:
            health_state: HealthBarTracker.update() 返回的状态字典.
            timestamp: 当前时间戳 (秒).

        Returns:
            True 如果触发了新 clip 提取.
        """
        if not health_state["is_hit"]:
            return False

        # 冷却去重
        if timestamp - self._last_hit_time < HIT_COOLDOWN_SEC:
            return False

        # 如果正在 lookforward 录制中，先保存当前 clip
        if self._lookforward_active:
            self._save_clip()
            self._lookforward_active = False

        self._last_hit_time = timestamp
        self._extract_clip(health_state, timestamp)
        return True

    def _extract_clip(self, health_state, hit_time):
        """从缓冲区提取 lookback 窗口，开始 lookforward 录制."""
        # 从缓冲区取最近 lookback_frames 帧
        buf_list = list(self._buffer)
        lookback_count = min(self._lookback_frames, len(buf_list))
        lookback = buf_list[-lookback_count:]

        # 暂存 lookback 帧和元信息
        self._pending_lookback = {
            "frames": [f for f, _ in lookback],
            "timestamps": [t for _, t in lookback],
            "hit_time": hit_time,
            "health_delta": health_state.get("health_delta", 0.0),
            "hit_frame_idx": len(lookback) - 1,  # 最后一帧 ≈ 命中帧
        }

        # 开始 lookforward 录制
        self._lookforward_frames = []
        self._lookforward_remaining = self._lookforward_frame_count
        self._lookforward_active = True

    def _save_clip(self):
        """保存当前 clip (lookback + lookforward) 到磁盘."""
        if self._pending_lookback is None:
            return

        self._clip_count += 1
        clip_name = f"clip_{self._clip_count:05d}"
        clip_dir = os.path.join(self.session_dir, clip_name)
        frames_dir = os.path.join(clip_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # 合并 lookback + lookforward 帧
        all_frames = self._pending_lookback["frames"] + self._lookforward_frames

        # 保存帧图像
        for i, frame in enumerate(all_frames):
            path = os.path.join(frames_dir, f"{i:03d}.jpg")
            cv2.imwrite(path, frame)

        # 生成 metadata
        meta = {
            "clip_name": clip_name,
            "total_frames": len(all_frames),
            "hit_frame_idx": self._pending_lookback["hit_frame_idx"],
            "hit_time": round(self._pending_lookback["hit_time"], 3),
            "health_delta": self._pending_lookback["health_delta"],
            "lookback_frames": len(self._pending_lookback["frames"]),
            "lookforward_frames": len(self._lookforward_frames),
            "fps": CAPTURE_FPS,
            "attack": "unknown",
        }

        meta_path = os.path.join(clip_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self._clips_meta.append(meta)
        self._pending_lookback = None
        self._lookforward_frames = []

        print(f"  保存 {clip_name}: {len(all_frames)} 帧, "
              f"health_delta={meta['health_delta']:.4f}")

    def save_session(self):
        """保存 session 级 labels.json 汇总."""
        # 如果还有未完成的 lookforward clip，强制保存
        if self._lookforward_active:
            self._save_clip()
            self._lookforward_active = False

        labels = {
            "session_dir": self.session_dir,
            "total_clips": self._clip_count,
            "fps": CAPTURE_FPS,
            "clips": self._clips_meta,
        }

        labels_path = os.path.join(self.session_dir, "labels.json")
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)

        print(f"Session 保存完成: {self._clip_count} clips → {labels_path}")
        return labels_path
