"""实时血量检测调试工具 + ROI 校准.

同时显示旧规则检测器和 AI 检测器的结果进行对比。

使用方式:
  python -m health_bar_module.test_health_realtime

操作:
  方向键 ← → ↑ ↓  移动 ROI 位置 (步进 5px)
  A / D            缩小 / 放大 ROI 宽度
  W / S            缩小 / 放大 ROI 高度
  P                打印当前 ROI 值 (复制到代码)
  R                重置为默认 ROI
  Q                退出
"""

import os
import time

import cv2
import mss
import numpy as np

from health_bar_module.mh_w_2_health_bar.health_bar_detector import HealthBarDetector

# Try to load AI detector
AI_MODEL_PATH = os.path.join(os.path.dirname(__file__), "runs", "best.pt")
ai_detector = None
try:
    from health_bar_module.mh_w_2_health_bar.health_bar_detector_ai import HealthBarDetectorAI
    if os.path.exists(AI_MODEL_PATH):
        ai_detector = HealthBarDetectorAI(AI_MODEL_PATH)
        print(f"AI detector loaded: {AI_MODEL_PATH}")
except Exception as e:
    print(f"AI detector not available: {e}")

STEP = 5


def main():
    print("=" * 50)
    print("实时血量检测调试 + ROI 校准")
    if ai_detector:
        print("  [AI 检测器已加载 — 同时显示对比]")
    print("=" * 50)
    print()
    print("操作:")
    print("  方向键       移动 ROI (步进 5px)")
    print("  A/D          缩小/放大宽度 (步进 5px)")
    print("  W/S          缩小/放大高度 (步进 5px)")
    print("  P            打印当前 ROI (复制到代码)")
    print("  R            重置为默认 ROI")
    print("  Q            退出")
    print()

    sct = mss.mss()
    monitor = sct.monitors[1]
    detector = HealthBarDetector()
    default_roi = detector.DEFAULT_ROI

    roi = list(detector.roi)

    print(f"  屏幕: {monitor['width']}x{monitor['height']}")
    print(f"  默认 ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print()

    cv2.namedWindow("Health Debug", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ROI + Debug", cv2.WINDOW_NORMAL)

    while True:
        t0 = time.perf_counter()

        # 更新 detector ROI
        detector.roi = tuple(roi)
        if ai_detector:
            ai_detector.roi = tuple(roi)

        # 截屏
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 规则检测
        result, roi_bgr, brightness_vis, gradient_vis = detector.detect_debug(frame)
        health_pct = result["health_pct"]
        damage_pct = result["damage_pct"]
        is_hit = result["is_hit"]

        # AI 检测
        ai_result = None
        ai_health_vis = None
        ai_damage_vis = None
        if ai_detector:
            ai_result, _, ai_health_vis, ai_damage_vis = ai_detector.detect_debug(frame)

        # 在全屏画面上画 ROI 框
        display = frame.copy()
        rx, ry, rw, rh = roi
        cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)

        # 血量条可视化 (右上角)
        h, w = display.shape[:2]
        bar_x, bar_y, bar_w, bar_h = w - 400, 20, 360, 40
        cv2.rectangle(display, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        fill_w = int(bar_w * health_pct)
        if health_pct < 0.20:
            bar_color = (0, 0, 255)
        elif health_pct < 0.50:
            bar_color = (0, 165, 255)
        else:
            bar_color = (0, 200, 0)
        cv2.rectangle(display, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(display, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)

        # 文字信息 — 规则检测器
        info = f"RULE: HP={health_pct:.1%}  DMG={damage_pct:.1%}  HIT={is_hit}"
        cv2.putText(display, info, (bar_x, bar_y + bar_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # AI 检测器结果
        if ai_result:
            # AI 血量条 (右上角, 第二行)
            ai_bar_y = bar_y + bar_h + 50
            cv2.rectangle(display, (bar_x, ai_bar_y),
                          (bar_x + bar_w, ai_bar_y + bar_h), (40, 40, 40), -1)
            ai_fill = int(bar_w * ai_result["health_pct"])
            ai_hp = ai_result["health_pct"]
            if ai_hp < 0.20:
                ai_color = (0, 0, 255)
            elif ai_hp < 0.50:
                ai_color = (0, 165, 255)
            else:
                ai_color = (0, 200, 0)
            cv2.rectangle(display, (bar_x, ai_bar_y),
                          (bar_x + ai_fill, ai_bar_y + bar_h), ai_color, -1)
            # Damage overlay
            ai_dmg_start = ai_fill
            ai_dmg_end = min(bar_w, ai_dmg_start + int(bar_w * ai_result["damage_pct"]))
            if ai_dmg_end > ai_dmg_start:
                cv2.rectangle(display, (bar_x + ai_dmg_start, ai_bar_y),
                              (bar_x + ai_dmg_end, ai_bar_y + bar_h), (0, 0, 200), -1)
            cv2.rectangle(display, (bar_x, ai_bar_y),
                          (bar_x + bar_w, ai_bar_y + bar_h), (200, 200, 200), 2)

            ai_info = (f"AI:   HP={ai_result['health_pct']:.1%}  "
                       f"DMG={ai_result['damage_pct']:.1%}  "
                       f"HIT={ai_result['is_hit']}")
            cv2.putText(display, ai_info, (bar_x, ai_bar_y + bar_h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        roi_info = f"ROI: ({rx},{ry}) {rw}x{rh}"
        y_offset = bar_y + bar_h + (140 if ai_result else 60)
        cv2.putText(display, roi_info, (bar_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(display, "Arrows=move A/D=width W/S=height P=print Q=quit",
                    (bar_x - 200, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # 缩放显示
        scale = 0.5
        resized = cv2.resize(display, (int(w * scale), int(h * scale)))
        cv2.imshow("Health Debug", resized)

        # ROI 放大 + 可视化
        if roi_bgr.size > 0:
            panels = [roi_bgr, brightness_vis, gradient_vis]
            labels = ["RULE: ROI", "BRIGHTNESS", "GRADIENT"]

            if ai_health_vis is not None and ai_damage_vis is not None:
                # Merge AI health+damage into one vis
                ai_combined = ai_health_vis.copy()
                mask = ai_damage_vis > 0
                ai_combined[mask] = ai_damage_vis[mask]
                panels.append(ai_combined)
                labels.append("AI: HP+DMG")

            combined = np.hstack(panels)
            ch, cw = combined.shape[:2]
            scale_roi = max(1, min(5, 600 // max(ch, 1)))
            combined_big = cv2.resize(combined, (cw * scale_roi, ch * scale_roi),
                                      interpolation=cv2.INTER_NEAREST)
            sec_w = cw * scale_roi // len(panels)
            colors = [(0, 255, 255), (200, 200, 200), (0, 100, 255), (0, 255, 0)]
            for i, label in enumerate(labels):
                cv2.putText(combined_big, label, (sec_w * i + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i % len(colors)], 2)
            cv2.imshow("ROI + Debug", combined_big)

        dt = time.perf_counter() - t0

        key = cv2.waitKeyEx(1)
        step = STEP

        if key == ord("q"):
            break
        elif key == 2424832:  # Left
            roi[0] = max(0, roi[0] - step)
        elif key == 2555904:  # Right
            roi[0] = min(monitor["width"] - roi[2], roi[0] + step)
        elif key == 2490368:  # Up
            roi[1] = max(0, roi[1] - step)
        elif key == 2621440:  # Down
            roi[1] = min(monitor["height"] - roi[3], roi[1] + step)
        elif key == ord("a"):
            roi[2] = max(20, roi[2] - step)
        elif key == ord("d"):
            roi[2] = min(monitor["width"] - roi[0], roi[2] + step)
        elif key == ord("w"):
            roi[3] = max(10, roi[3] - step)
        elif key == ord("s"):
            roi[3] = min(monitor["height"] - roi[1], roi[3] + step)
        elif key == ord("r"):
            roi = list(default_roi)
            print(f"  重置 ROI: {tuple(roi)}")
        elif key == ord("p"):
            print(f"\n  当前 ROI: DEFAULT_ROI = ({roi[0]}, {roi[1]}, {roi[2]}, {roi[3]})")
            print(f"  复制到 health_bar_detector.py 的 DEFAULT_ROI\n")

        sleep_time = 0.1 - dt
        if sleep_time > 0:
            time.sleep(sleep_time)

    cv2.destroyAllWindows()
    print(f"\n最终 ROI: DEFAULT_ROI = ({roi[0]}, {roi[1]}, {roi[2]}, {roi[3]})")
    print("退出")


if __name__ == "__main__":
    main()
