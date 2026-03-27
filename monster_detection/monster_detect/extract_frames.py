"""从游戏视频抽帧 — 保存到 unlabeled/.

用法:
  python -m monster_detect.extract_frames --video path/to/video.mp4
  python -m monster_detect.extract_frames --video path/to/video.mp4 --interval 2.0
  python -m monster_detect.extract_frames --video path/to/video.mp4 --no-crop
"""

import argparse
import json
import os

import cv2


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_AP_DIR = os.path.join(_MODULE_DIR, "..")
_ROI_CONFIG = os.path.join(_AP_DIR, "roi_config.json")
_OUTPUT_DIR = os.path.join(_MODULE_DIR, "monster_yolo_data", "images", "unlabeled")

YOLO_IMG_SIZE = 640


def _load_roi():
    with open(_ROI_CONFIG) as f:
        cfg = json.load(f)
    return cfg["x"], cfg["y"], cfg["w"], cfg["h"]


def extract(video_path: str, interval: float = 0.5, no_crop: bool = False,
            prefix: str = "frame"):
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 {video_path}")
        return

    if no_crop:
        print("模式: 全屏 (原始分辨率)")
    else:
        roi = _load_roi()
        x, y, w, h = roi
        print(f"模式: 裁剪 ROI  x={x}, y={y}, w={w}, h={h}")

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # 统计同前缀已有帧数，避免覆盖
    existing = [f for f in os.listdir(_OUTPUT_DIR)
                if f.startswith(prefix + "_") and f.endswith(".jpg")]
    start_idx = len(existing)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps * interval))

    print(f"视频: {video_path}")
    print(f"  FPS: {fps:.1f}, 总帧数: {total_frames}")
    print(f"  抽帧间隔: {interval}s (每 {frame_step} 帧取 1 帧)")
    print(f"  输出目录: {_OUTPUT_DIR}")
    print(f"  已有 {start_idx} 张图片, 从 {prefix}_{start_idx:05d} 开始编号")
    print()

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            if no_crop:
                out_img = frame
            else:
                out_img = frame[y:y + h, x:x + w]

            out_name = f"{prefix}_{start_idx + saved:05d}.jpg"
            out_path = os.path.join(_OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, out_img)
            saved += 1

            if saved % 50 == 0:
                print(f"  已保存 {saved} 帧...")

        frame_idx += 1

    cap.release()
    print(f"\n完成! 共保存 {saved} 帧到 {_OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="从游戏视频抽帧")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="抽帧间隔 (秒), 默认 0.5")
    parser.add_argument("--no-crop", action="store_true",
                        help="不裁剪 ROI, 保存全屏原始分辨率")
    parser.add_argument("--prefix", default="frame",
                        help="文件名前缀 (默认 frame)")
    args = parser.parse_args()
    extract(args.video, args.interval, args.no_crop, args.prefix)


if __name__ == "__main__":
    main()
