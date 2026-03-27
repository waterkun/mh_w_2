"""实时 YOLO 怪物检测可视化 — 显示检测框和置信度.

使用方式:
  python -m farming_bot.find_monster.realtime_viewer

按 Q 或 ESC 退出窗口.
"""

import os
import sys
import time

import cv2
import mss
import numpy as np

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "monster_detection"))

from ultralytics import YOLO

_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "monster_detection", "monster_detect",
    "runs", "monster_detect", "weights", "best.pt"
)

# 颜色 (BGR)
COLORS = {
    "body": (0, 255, 0),    # 绿色
    "head": (0, 0, 255),    # 红色
}
COLOR_DEFAULT = (255, 255, 0)  # 青色

CONFIDENCE_THRESHOLD = 0.3  # 低阈值, 显示更多框帮助调试
DISPLAY_SCALE = 0.75         # 显示缩放
FPS = 5


def main():
    print("=" * 50)
    print("YOLO 怪物检测 — 实时可视化")
    print("按 Q 或 ESC 退出")
    print("=" * 50)

    model = YOLO(_MODEL_PATH)
    class_names = model.names
    print(f"模型: {_MODEL_PATH}")
    print(f"类别: {class_names}")
    print(f"显示阈值: {CONFIDENCE_THRESHOLD}")
    print()

    sct = mss.mss()
    monitor = sct.monitors[1]
    frame_interval = 1.0 / FPS

    cv2.namedWindow("Monster Detection", cv2.WINDOW_NORMAL)

    frame_count = 0
    fps_start = time.time()
    actual_fps = 0

    try:
        while True:
            t0 = time.perf_counter()

            # 截屏
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # YOLO 推理
            results = model(frame, verbose=False)
            boxes = results[0].boxes

            # 画检测框
            display = frame.copy()
            detected_count = 0

            for box in boxes:
                conf = box.conf.item()
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                detected_count += 1
                cls_id = int(box.cls.item())
                cls_name = class_names.get(cls_id, f"cls{cls_id}")
                color = COLORS.get(cls_name, COLOR_DEFAULT)

                # bbox
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                thickness = 3 if conf >= 0.6 else 1
                cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

                # 标签
                label = f"{cls_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(display, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), color, -1)
                cv2.putText(display, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

            # 状态栏
            frame_count += 1
            if frame_count % 10 == 0:
                actual_fps = 10 / (time.time() - fps_start)
                fps_start = time.time()

            status = f"FPS: {actual_fps:.1f} | Detections: {detected_count}"
            if detected_count > 0:
                max_conf = max(box.conf.item() for box in boxes
                               if box.conf.item() >= CONFIDENCE_THRESHOLD)
                status += f" | Max conf: {max_conf:.3f}"
                if max_conf >= 0.6:
                    status += " | MONSTER FOUND"

            cv2.putText(display, status, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)

            # 缩放显示
            h, w = display.shape[:2]
            display_resized = cv2.resize(display,
                                         (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
            cv2.imshow("Monster Detection", display_resized)

            # 按键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break

            # 帧率控制
            dt = time.perf_counter() - t0
            sleep_time = frame_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    print(f"\n结束, 共 {frame_count} 帧")


if __name__ == "__main__":
    main()
