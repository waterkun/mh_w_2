"""录屏工具 — mss 截取 ROI 区域，保存为 AVI 视频."""

import json
import os
import sys
import time

import cv2
import numpy as np
import mss

# 允许从 attack_prediction/ 根目录 import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ROI_CONFIG_PATH, RAW_VIDEO_DIR, CAPTURE_FPS, INPUT_SIZE


def load_roi():
    """从 roi_config.json 加载 ROI 坐标."""
    if not os.path.exists(ROI_CONFIG_PATH):
        print(f"错误: 找不到 {ROI_CONFIG_PATH}")
        print("请先运行 select_roi.py 选择 ROI 区域。")
        sys.exit(1)
    with open(ROI_CONFIG_PATH) as f:
        cfg = json.load(f)
    return cfg["x"], cfg["y"], cfg["w"], cfg["h"]


def record(duration_sec=60, output_name=None):
    """录制指定时长的 ROI 区域视频.

    Args:
        duration_sec: 录制时长 (秒), 0 表示手动按 q 停止.
        output_name: 输出文件名 (不含后缀), None 则用时间戳.
    """
    roi_x, roi_y, roi_w, roi_h = load_roi()

    # mss 的 monitor 格式
    monitor = {
        "left": roi_x,
        "top": roi_y,
        "width": roi_w,
        "height": roi_h,
    }

    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
    if output_name is None:
        output_name = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RAW_VIDEO_DIR, f"{output_name}.avi")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, CAPTURE_FPS,
                             (roi_w, roi_h))

    sct = mss.mss()
    frame_interval = 1.0 / CAPTURE_FPS
    frame_count = 0
    start_time = time.time()

    print(f"开始录制 ROI ({roi_w}x{roi_h}) @ {CAPTURE_FPS} FPS")
    print(f"输出: {out_path}")
    if duration_sec > 0:
        print(f"时长: {duration_sec}s (按 q 提前停止)")
    else:
        print("按 q 停止录制")

    try:
        while True:
            t0 = time.perf_counter()

            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            writer.write(frame)
            frame_count += 1

            # 显示预览 (缩小)
            preview = cv2.resize(frame, (min(roi_w, 640),
                                         min(roi_h, 480)))
            elapsed = time.time() - start_time
            cv2.putText(preview, f"REC {elapsed:.1f}s  F:{frame_count}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
            cv2.imshow("Recording", preview)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if duration_sec > 0 and elapsed >= duration_sec:
                break

            # 控制帧率
            dt = time.perf_counter() - t0
            sleep_time = frame_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    writer.release()
    cv2.destroyAllWindows()

    actual_fps = frame_count / max(time.time() - start_time, 0.001)
    print(f"录制完成: {frame_count} 帧, 实际 FPS: {actual_fps:.1f}")
    print(f"保存至: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="录制游戏 ROI 区域")
    parser.add_argument("-d", "--duration", type=int, default=60,
                        help="录制时长 (秒), 0=手动停止")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出文件名 (不含后缀)")
    args = parser.parse_args()
    record(duration_sec=args.duration, output_name=args.output)
