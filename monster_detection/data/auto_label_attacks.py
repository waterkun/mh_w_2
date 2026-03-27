"""自动标注攻击 clip — 实时 / 离线双模式主入口.

使用方式:
  实时模式: python data/auto_label_attacks.py --realtime
  离线模式: python data/auto_label_attacks.py --video path/to/video.avi

依赖:
  - HealthBarTracker (health_bar_module)
  - AutoLabeler (本模块)
  - roi_config.json (通过 select_roi.py 生成)
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ROI_CONFIG_PATH, CAPTURE_FPS, INPUT_SIZE, AUTO_LABEL_DIR,
    FRAME_INTERVAL_MS,
)
from auto_labeler import AutoLabeler

# health_bar_module 路径
_HB_MODULE = os.path.join(os.path.dirname(__file__),
                          "..", "..", "health_bar_module",
                          "mh_w_2_health_bar")
sys.path.insert(0, _HB_MODULE)
from health_bar_tracker import HealthBarTracker
from health_bar_detector import HealthBarDetector


def _load_roi():
    """从 roi_config.json 加载怪物活动区域 ROI."""
    if not os.path.exists(ROI_CONFIG_PATH):
        print(f"错误: 找不到 {ROI_CONFIG_PATH}")
        print("请先运行 select_roi.py 选择 ROI 区域。")
        sys.exit(1)
    with open(ROI_CONFIG_PATH) as f:
        cfg = json.load(f)
    return cfg["x"], cfg["y"], cfg["w"], cfg["h"]


def _crop_and_resize(frame, roi):
    """从全屏帧中裁剪 ROI 并 resize 到 INPUT_SIZE."""
    x, y, w, h = roi
    crop = frame[y:y + h, x:x + w]
    return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))


def _draw_info(display, info_dict, panel_h=120):
    """在图像底部绘制信息面板."""
    h, w = display.shape[:2]
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)

    lines = [
        f"HP: {info_dict.get('health_pct', 0):.1%}  "
        f"Delta: {info_dict.get('health_delta', 0):.4f}  "
        f"Hit: {info_dict.get('is_hit', False)}",
        f"Clips: {info_dict.get('clip_count', 0)}  "
        f"Hits: {info_dict.get('hit_count', 0)}  "
        f"FPS: {info_dict.get('fps', 0):.1f}",
        f"Status: {info_dict.get('status', '')}",
    ]

    for i, line in enumerate(lines):
        color = (0, 255, 0) if not info_dict.get("is_hit", False) else (0, 0, 255)
        cv2.putText(panel, line, (10, 25 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return np.vstack([display, panel])


def run_realtime(session_name=None):
    """实时模式: mss 截屏 → 检测命中 → 保存 clip."""
    import mss

    monster_roi = _load_roi()
    health_tracker = HealthBarTracker()  # 使用默认血条 ROI
    labeler = AutoLabeler(session_name=session_name)

    sct = mss.mss()
    monitor = sct.monitors[1]  # 主屏幕

    frame_interval = 1.0 / CAPTURE_FPS
    frame_count = 0
    paused = False
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0.0

    print(f"实时自动标注已启动")
    print(f"  怪物 ROI: {monster_roi}")
    print(f"  Session: {labeler.session_dir}")
    print(f"  按 Q 退出, P 暂停/继续")

    try:
        while True:
            t0 = time.perf_counter()

            if paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord("p"):
                    paused = False
                    print("继续...")
                elif key == ord("q"):
                    break
                continue

            # 截屏
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            timestamp = time.time()

            # 血条检测 (全屏帧，HealthBarDetector 内部裁剪血条 ROI)
            health_state = health_tracker.update(frame)

            # 裁剪怪物 ROI → resize → 送入 labeler
            roi_frame = _crop_and_resize(frame, monster_roi)
            labeler.add_frame(roi_frame, timestamp)

            # 检测命中事件
            triggered = labeler.check_hit_event(health_state, timestamp)

            frame_count += 1
            fps_counter += 1

            # 计算 FPS
            now = time.time()
            if now - fps_timer >= 1.0:
                current_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # 显示预览
            preview = cv2.resize(roi_frame, (400, 400))
            display = _draw_info(preview, {
                "health_pct": health_state["health_pct"],
                "health_delta": health_state["health_delta"],
                "is_hit": health_state["is_hit"],
                "clip_count": labeler.clip_count,
                "hit_count": health_state["hit_count"],
                "fps": current_fps,
                "status": "CLIP!" if triggered else "监控中",
            })
            cv2.imshow("Auto Labeler", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = True
                print("暂停")

            # 帧率控制
            dt = time.perf_counter() - t0
            sleep_time = frame_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    labeler.save_session()
    print(f"共处理 {frame_count} 帧, 保存 {labeler.clip_count} 个 clip")


def run_offline(video_path, session_name=None):
    """离线模式: 从视频文件中检测命中并保存 clip."""
    if not os.path.exists(video_path):
        print(f"错误: 找不到视频 {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or CAPTURE_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if session_name is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        session_name = f"session_{video_name}"

    health_tracker = HealthBarTracker()
    labeler = AutoLabeler(session_name=session_name)

    # 离线模式: 视频可能是全屏录制或纯 ROI 录制
    # 尝试加载怪物 ROI，如果存在则裁剪
    monster_roi = None
    if os.path.exists(ROI_CONFIG_PATH):
        monster_roi = _load_roi()
        print(f"使用怪物 ROI: {monster_roi}")
    else:
        print("未找到 roi_config.json, 将整个视频帧作为怪物区域处理")

    print(f"离线标注: {video_path}")
    print(f"  总帧数: {total_frames}, 视频 FPS: {video_fps:.1f}")
    print(f"  Session: {labeler.session_dir}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 模拟时间戳
            timestamp = frame_idx / video_fps

            # 血条检测
            health_state = health_tracker.update(frame)

            # 裁剪怪物区域
            if monster_roi is not None:
                roi_frame = _crop_and_resize(frame, monster_roi)
            else:
                roi_frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

            labeler.add_frame(roi_frame, timestamp)
            triggered = labeler.check_hit_event(health_state, timestamp)

            frame_idx += 1

            # 进度显示
            if frame_idx % 100 == 0 or triggered:
                pct = frame_idx / total_frames * 100
                status = " ← CLIP!" if triggered else ""
                print(f"  [{pct:5.1f}%] frame {frame_idx}/{total_frames}, "
                      f"clips: {labeler.clip_count}{status}")

    except KeyboardInterrupt:
        print("\n中断")

    cap.release()
    labeler.save_session()
    print(f"离线标注完成: {frame_idx} 帧, {labeler.clip_count} clips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动标注攻击 clip")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--realtime", action="store_true",
                       help="实时截屏模式")
    group.add_argument("--video", type=str,
                       help="离线视频文件路径")
    parser.add_argument("--session", type=str, default=None,
                        help="Session 名称 (默认自动生成)")
    args = parser.parse_args()

    if args.realtime:
        run_realtime(session_name=args.session)
    else:
        run_offline(args.video, session_name=args.session)
