"""实时攻击预测 demo — 屏幕截取 + 模型推理 + 可视化."""

import sys
import time

import cv2
import numpy as np
import mss

from config import ATTACK_CLASSES, FRAME_INTERVAL_MS, CAPTURE_FPS
from attack_detector import AttackDetector
from attack_tracker import AttackTracker


# 攻击类别颜色 (BGR)
ATTACK_COLORS = {
    "idle":           (128, 128, 128),
    "pounce":         (0, 165, 255),
    "beam":           (0, 0, 255),
    "tail_sweep":     (0, 255, 255),
    "flying_attack":  (255, 0, 0),
    "claw_swipe":     (0, 255, 0),
    "charge":         (255, 0, 255),
    "nova":           (0, 200, 255),
}


def main(model_path):
    detector = AttackDetector(model_path)
    tracker = AttackTracker(detector)

    sct = mss.mss()
    monitor = sct.monitors[1]

    print("Attack Predictor — 按 q 退出")
    print(f"屏幕: {monitor['width']}x{monitor['height']}")
    print(f"ROI: {detector.roi}")
    print(f"目标帧率: {CAPTURE_FPS} FPS ({FRAME_INTERVAL_MS} ms/帧)")

    frame_count = 0
    fps_counter = 0
    fps_timer = time.time()
    actual_fps = 0.0

    while True:
        t0 = time.perf_counter()

        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        state = tracker.update(frame)
        reward = tracker.get_reward_signal()
        frame_count += 1

        # FPS 计算
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            actual_fps = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        # 显示 ROI 预览
        if detector.roi:
            x, y, w, h = detector.roi
            roi_crop = frame[y:y+h, x:x+w]
            preview = cv2.resize(roi_crop, (min(w, 640), min(h, 480)))
        else:
            preview = cv2.resize(frame, (640, 360))

        # 信息面板
        color = ATTACK_COLORS.get(state["current_attack"], (128, 128, 128))
        panel_h = 140
        panel = np.zeros((panel_h, preview.shape[1], 3), dtype=np.uint8)

        lines = [
            f"Attack: {state['current_attack'].upper()}  "
            f"Conf: {state['confidence']:.2%}  "
            f"FPS: {actual_fps}",

            f"Prev: {state['prev_attack']}  "
            f"Started: {state['attack_just_started']}  "
            f"Ended: {state['attack_just_ended']}",

            f"Duration: {state['time_in_attack']:.1f}s  "
            f"Reward: {reward:+.4f}  "
            f"Ready: {state['ready']}",
        ]

        # 概率条
        if state["probs"]:
            prob_parts = [f"{name[:5]}:{p:.0%}"
                          for name, p in state["probs"].items()
                          if p >= 0.05]
            lines.append("  ".join(prob_parts))

        for i, line in enumerate(lines):
            cv2.putText(panel, line, (10, 22 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        display = np.vstack([preview, panel])
        cv2.imshow("Attack Predictor", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # 帧率控制
        dt = time.perf_counter() - t0
        sleep_ms = (FRAME_INTERVAL_MS / 1000.0) - dt
        if sleep_ms > 0:
            time.sleep(sleep_ms)

    cv2.destroyAllWindows()
    print(f"总帧数: {frame_count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="实时攻击预测")
    parser.add_argument("model", type=str, help="模型 checkpoint 路径")
    args = parser.parse_args()
    main(args.model)
