"""
实时综合测试脚本 — 同时运行 health_bar / hit_number / gauge / sharpness 四模块

功能:
  - 实时截取游戏画面，大画面预览 (可调缩放比例)
  - 顶部叠加: 血条 ROI + HSV 遮罩
  - 左上角 HUD: RL 状态 (HP、命中、奖励)
  - 伤害数字: 直接在画面上画检测框和识别结果
  - 左下角 HUD: 气刃槽状态 + 斩味状态
  - 底部统计条: FPS、帧数、检测次数、累计伤害

用法:
  python test_combined_realtime.py
  python test_combined_realtime.py --monitor 1
  python test_combined_realtime.py --scale 0.5        # 缩放比例 (默认 0.45)
  python test_combined_realtime.py --no-damage        # 禁用伤害数字
  python test_combined_realtime.py --no-health        # 禁用血条
  python test_combined_realtime.py --no-gauge         # 禁用气刃槽
  python test_combined_realtime.py --no-sharpness     # 禁用斩味

按 Q 退出 | 按 R 重置追踪器 | 按 S 截图保存
按 +/- 调整缩放比例
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import mss

# ---------- 路径设置 ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEALTH_MODULE = os.path.join(PROJECT_ROOT, "health_bar_module", "mh_w_2_health_bar")
DAMAGE_MODULE = os.path.join(PROJECT_ROOT, "hit_number_detection")
GAUGE_MODULE = os.path.join(PROJECT_ROOT, "Gauge")
SHARPNESS_MODULE = os.path.join(PROJECT_ROOT, "Sharpness")

sys.path.insert(0, HEALTH_MODULE)
sys.path.insert(0, DAMAGE_MODULE)
sys.path.insert(0, GAUGE_MODULE)
sys.path.insert(0, SHARPNESS_MODULE)

from health_bar_detector import HealthBarDetector
from health_bar_tracker import HealthBarTracker
from gauge_detector import GaugeDetector
from gauge_tracker import GaugeTracker
from sharpness_detector import SharpnessDetector
from sharpness_tracker import SharpnessTracker


def load_damage_detector(yolo_path, crnn_path, conf):
    """延迟加载伤害数字检测器 (模型较大，按需加载)。"""
    from inference import DamageNumberDetector
    return DamageNumberDetector(
        yolo_path=yolo_path,
        crnn_path=crnn_path,
        conf_threshold=conf,
    )


def draw_damage_detections(frame, detections, scale):
    """在画面上直接绘制伤害数字检测框 (缩放后坐标)。"""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        sx1, sy1 = int(x1 * scale), int(y1 * scale)
        sx2, sy2 = int(x2 * scale), int(y2 * scale)

        # 检测框
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)

        # 数字标签 (较大字体)
        label = f"{det['number']} ({det['confidence']:.2f})"
        font_scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (sx1, sy1 - th - 8), (sx1 + tw + 4, sy1),
                      (0, 255, 255), -1)
        cv2.putText(frame, label, (sx1 + 2, sy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


def draw_health_hud(frame, state, reward):
    """在左上角绘制血条 HUD 叠加层。"""
    lines = [
        f"HP: {state['health_pct']:.1%}   DMG: {state['damage_pct']:.1%}",
        f"Hit: {state['is_hit']}   Hits: {state['hit_count']}   Delta: {state['health_delta']:+.2%}",
        f"Alive: {state['is_alive']}   Reward: {reward:+.4f}",
        f"Last hit: {state['time_since_last_hit']:.1f}s ago",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    line_height = 28
    padding = 12
    x0, y0 = 15, 15

    # 计算背景大小
    max_tw = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_tw = max(max_tw, tw)
    bg_w = max_tw + padding * 2
    bg_h = len(lines) * line_height + padding * 2

    # 半透明黑色背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bg_w, y0 + bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 文字
    color = (0, 255, 0) if state['is_alive'] else (0, 0, 255)
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (x0 + padding, y0 + padding + 20 + i * line_height),
                    font, font_scale, color, thickness)


def draw_health_bar_preview(frame, roi_bgr, green_mask, red_mask, y_offset):
    """在画面顶部右侧绘制血条 ROI 和遮罩预览。返回占用高度。"""
    preview_w = 400
    roi_h = 35
    mask_h = 25

    # ROI 预览
    roi_resized = cv2.resize(roi_bgr, (preview_w, roi_h))
    # 遮罩
    green_vis = cv2.resize(green_mask, (preview_w // 2, mask_h))
    red_vis = cv2.resize(red_mask, (preview_w // 2, mask_h))
    masks_row = np.hstack([green_vis, red_vis])
    masks_bgr = cv2.cvtColor(masks_row, cv2.COLOR_GRAY2BGR)

    # 组合
    preview = np.vstack([roi_resized, masks_bgr])
    ph, pw = preview.shape[:2]

    # 放到画面右上角
    x0 = frame.shape[1] - pw - 15
    y0 = y_offset + 15

    # 边框
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + pw + 2, y0 + ph + 2),
                  (100, 100, 100), 1)
    frame[y0:y0 + ph, x0:x0 + pw] = preview

    # 标签
    cv2.putText(frame, "Health Bar ROI", (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return ph + 25  # 占用高度 (含标签和间距)


def draw_roi_preview(frame, label, roi_bgr, mask_a, mask_b, y_offset):
    """在画面右侧绘制通用 ROI + 双遮罩预览。返回占用高度。"""
    preview_w = 400
    roi_h = 35
    mask_h = 25

    roi_resized = cv2.resize(roi_bgr, (preview_w, roi_h))
    vis_a = cv2.resize(mask_a, (preview_w // 2, mask_h))
    vis_b = cv2.resize(mask_b, (preview_w // 2, mask_h))
    masks_row = np.hstack([vis_a, vis_b])
    masks_bgr = cv2.cvtColor(masks_row, cv2.COLOR_GRAY2BGR)

    preview = np.vstack([roi_resized, masks_bgr])
    ph, pw = preview.shape[:2]

    x0 = frame.shape[1] - pw - 15
    y0 = y_offset + 15

    # 边界检查
    if y0 + ph > frame.shape[0] or x0 < 0:
        return ph + 25

    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + pw + 2, y0 + ph + 2),
                  (100, 100, 100), 1)
    frame[y0:y0 + ph, x0:x0 + pw] = preview
    cv2.putText(frame, label, (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return ph + 25


# ---------- 气刃槽 / 斩味颜色映射 ----------
GAUGE_STATE_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 230, 255),
    "white": (255, 255, 255),
    "empty": (128, 128, 128),
}

SHARPNESS_COLORS = {
    "purple": (200, 50, 200),
    "white": (255, 255, 255),
    "blue": (255, 150, 0),
    "green": (0, 200, 0),
    "yellow": (0, 230, 255),
    "orange": (0, 140, 255),
    "red": (0, 0, 255),
    "unknown": (128, 128, 128),
}


def draw_gauge_hud(frame, state, reward, y_base):
    """在左下方绘制气刃槽 HUD。返回占用高度。"""
    lines = [
        f"Gauge: {state['gauge_state'].upper()}   Red%: {state['red_pct']:.0%}",
        f"InRed: {state['time_in_red']:.1f}s   Reward: {reward:+.3f}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    line_height = 24
    padding = 8
    x0 = 15

    max_tw = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_tw = max(max_tw, tw)
    bg_w = max_tw + padding * 2
    bg_h = len(lines) * line_height + padding * 2

    y0 = y_base - bg_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bg_w, y0 + bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = GAUGE_STATE_COLORS.get(state['gauge_state'], (200, 200, 200))
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (x0 + padding, y0 + padding + 16 + i * line_height),
                    font, font_scale, color, thickness)

    return bg_h + 5


def draw_sharpness_hud(frame, state, reward, y_base):
    """在左下方绘制斩味 HUD (在气刃槽 HUD 上方)。返回占用高度。"""
    alert_tag = " >>> SHARPEN! <<<" if state['sharpen_alert'] else ""
    flash_tag = " [FLASH]" if state['is_flashing'] else ""

    lines = [
        f"Sharp: {state['sharpness_color'].upper()}  {state['sharpness_pct']:.0%}{flash_tag}{alert_tag}",
        f"Reward: {reward:+.3f}   SinceSharp: {state['time_since_last_sharpen']:.0f}s",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    line_height = 24
    padding = 8
    x0 = 15

    max_tw = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_tw = max(max_tw, tw)
    bg_w = max_tw + padding * 2
    bg_h = len(lines) * line_height + padding * 2

    y0 = y_base - bg_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bg_w, y0 + bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = SHARPNESS_COLORS.get(state['sharpness_color'], (200, 200, 200))
    if state['sharpen_alert']:
        color = (0, 0, 255)  # 红色警告
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (x0 + padding, y0 + padding + 16 + i * line_height),
                    font, font_scale, color, thickness)

    return bg_h + 5


def draw_stats_bar(frame, fps, total_damage, detection_count, frame_count):
    """在画面底部绘制统计条。"""
    bar_h = 35
    h, w = frame.shape[:2]

    # 半透明黑色背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    text = (f"FPS: {fps:.1f}   |   "
            f"Frame: {frame_count}   |   "
            f"Damage detected: {detection_count}   |   "
            f"Total damage: {total_damage}")
    cv2.putText(frame, text, (15, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


def parse_args():
    p = argparse.ArgumentParser(description="MH Wilds 实时综合测试")
    p.add_argument("--monitor", type=int, default=1,
                   help="mss 显示器编号 (默认 1 = 主显示器)")
    p.add_argument("--scale", type=float, default=0.45,
                   help="画面缩放比例 (默认 0.45, 可按 +/- 调整)")
    p.add_argument("--yolo",
                   default=os.path.join(DAMAGE_MODULE,
                                        "runs", "detect", "runs",
                                        "damage_number", "weights", "best.pt"),
                   help="YOLO 模型路径")
    p.add_argument("--crnn",
                   default=os.path.join(DAMAGE_MODULE, "runs", "crnn", "best.pt"),
                   help="CRNN 模型路径")
    p.add_argument("--conf", type=float, default=0.5,
                   help="YOLO 检测置信度阈值")
    p.add_argument("--no-damage", action="store_true",
                   help="禁用伤害数字检测")
    p.add_argument("--no-health", action="store_true",
                   help="禁用血条检测")
    p.add_argument("--no-gauge", action="store_true",
                   help="禁用气刃槽检测")
    p.add_argument("--no-sharpness", action="store_true",
                   help="禁用斩味检测")
    p.add_argument("--screenshot-dir", default="screenshots",
                   help="截图保存目录")
    return p.parse_args()


def main():
    args = parse_args()
    scale = args.scale

    # ---------- 初始化模块 ----------
    enable_health = not args.no_health
    enable_damage = not args.no_damage
    enable_gauge = not args.no_gauge
    enable_sharpness = not args.no_sharpness

    if enable_health:
        detector = HealthBarDetector()
        tracker = HealthBarTracker(detector)
        print(f"[血条模块] ROI = {detector.roi}")

    damage_detector = None
    if enable_damage:
        if os.path.exists(args.yolo) and os.path.exists(args.crnn):
            damage_detector = load_damage_detector(args.yolo, args.crnn, args.conf)
        else:
            print(f"[警告] 模型文件不存在，伤害数字检测已禁用:")
            if not os.path.exists(args.yolo):
                print(f"  YOLO: {args.yolo}")
            if not os.path.exists(args.crnn):
                print(f"  CRNN: {args.crnn}")
            enable_damage = False

    if enable_gauge:
        gauge_det = GaugeDetector()
        gauge_trk = GaugeTracker(gauge_det)
        print(f"[气刃槽模块] ROI = {gauge_det.roi}")

    if enable_sharpness:
        sharp_det = SharpnessDetector()
        sharp_trk = SharpnessTracker(sharp_det)
        print(f"[斩味模块] ROI = {sharp_det.roi}")

    if not any([enable_health, enable_damage, enable_gauge, enable_sharpness]):
        print("[错误] 所有模块都未启用，退出。")
        return

    # ---------- 截屏初始化 ----------
    sct = mss.mss()
    monitor = sct.monitors[args.monitor]
    print(f"[截屏] 显示器 #{args.monitor}: {monitor['width']}x{monitor['height']}")
    print(f"[缩放] {scale:.0%} (按 +/- 调整)")
    print("按 Q 退出 | R 重置 | S 截图 | +/- 缩放")

    # ---------- 统计变量 ----------
    frame_count = 0
    total_damage = 0
    damage_det_count = 0
    fps_counter = 0
    fps_timer = time.time()
    fps = 0.0
    screenshot_dir = os.path.join(os.path.dirname(__file__), args.screenshot_dir)

    # ---------- 主循环 ----------
    while True:
        # 截屏
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 缩放画面
        disp_w = int(frame.shape[1] * scale)
        disp_h = int(frame.shape[0] * scale)
        display = cv2.resize(frame, (disp_w, disp_h))

        # ---- 伤害数字检测 (在原始帧上检测，在缩放帧上画框) ----
        detections = []
        if enable_damage and damage_detector is not None:
            detections = damage_detector.detect(frame)
            damage_det_count += len(detections)
            for det in detections:
                try:
                    total_damage += int(det["number"])
                except (ValueError, TypeError):
                    pass
            draw_damage_detections(display, detections, scale)

        # ---- 血条检测 ----
        preview_y_offset = 0
        if enable_health:
            state = tracker.update(frame)
            reward = tracker.get_reward_signal()
            _, roi_bgr, green_mask, red_mask = detector.detect_debug(frame)

            # HUD 叠加 (左上角)
            draw_health_hud(display, state, reward)
            # 血条 ROI 预览 (右上角)
            h_used = draw_health_bar_preview(display, roi_bgr, green_mask, red_mask, preview_y_offset)
            preview_y_offset += h_used

        # ---- 气刃槽检测 ----
        if enable_gauge:
            gauge_state = gauge_trk.update(frame)
            gauge_reward = gauge_trk.get_reward_signal()
            _, gauge_roi, gauge_red_mask, gauge_combined = gauge_det.detect_debug(frame)

            # ROI 预览 (右侧，叠在血条下方)
            h_used = draw_roi_preview(display, "Gauge ROI",
                                      gauge_roi, gauge_red_mask, gauge_combined,
                                      preview_y_offset)
            preview_y_offset += h_used

        # ---- 斩味检测 ----
        if enable_sharpness:
            sharp_state = sharp_trk.update(frame)
            sharp_reward = sharp_trk.get_reward_signal()
            _, sharp_roi, sharp_color_mask, sharp_combined = sharp_det.detect_debug(frame)

            # ROI 预览 (右侧，继续叠加)
            h_used = draw_roi_preview(display, "Sharpness ROI",
                                      sharp_roi, sharp_color_mask, sharp_combined,
                                      preview_y_offset)
            preview_y_offset += h_used

        # ---- 左下角 HUD: 气刃槽 + 斩味 ----
        stats_bar_h = 35
        bottom_y = display.shape[0] - stats_bar_h  # 统计条上方

        if enable_gauge:
            gh = draw_gauge_hud(display, gauge_state, gauge_reward, bottom_y)
            bottom_y -= gh
        if enable_sharpness:
            draw_sharpness_hud(display, sharp_state, sharp_reward, bottom_y)

        # ---- 底部统计条 ----
        frame_count += 1
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        draw_stats_bar(display, fps, total_damage, damage_det_count, frame_count)

        cv2.imshow("MH Wilds - Realtime Test", display)

        # ---- 按键处理 ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if enable_health:
                tracker.reset()
                print("[重置] 血条追踪器已重置")
            if enable_gauge:
                gauge_trk.reset()
                print("[重置] 气刃槽追踪器已重置")
            if enable_sharpness:
                sharp_trk.reset()
                print("[重置] 斩味追踪器已重置")
            total_damage = 0
            damage_det_count = 0
            frame_count = 0
            print("[重置] 统计数据已清零")
        elif key == ord("s"):
            os.makedirs(screenshot_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(screenshot_dir, f"test_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"[截图] 保存到 {path}")
        elif key == ord("+") or key == ord("="):
            scale = min(scale + 0.05, 1.0)
            print(f"[缩放] {scale:.0%}")
        elif key == ord("-"):
            scale = max(scale - 0.05, 0.15)
            print(f"[缩放] {scale:.0%}")

    cv2.destroyAllWindows()
    print("\n===== 测试结束 =====")
    print(f"总帧数: {frame_count}")
    print(f"平均 FPS: {fps:.1f}")
    if enable_damage:
        print(f"伤害数字检测次数: {damage_det_count}")
        print(f"累计伤害值: {total_damage}")
    if enable_health:
        print(f"总受击次数: {tracker.hit_count}")
    if enable_gauge:
        print(f"气刃槽最终状态: {gauge_trk.gauge_state}")
    if enable_sharpness:
        print(f"斩味最终状态: {sharp_trk.sharpness_color}")


if __name__ == "__main__":
    main()