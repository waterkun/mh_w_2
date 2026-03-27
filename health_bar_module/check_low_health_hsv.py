"""分析低血量截图的 HSV 值，验证检测阈值是否覆盖闪烁颜色.

使用方式:
  python -m health_bar_module.check_low_health_hsv
"""

import os
import cv2
import numpy as np

from health_bar_module.mh_w_2_health_bar.health_bar_detector import HealthBarDetector

PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "health_bar_photos")

LOW_HEALTH_FILES = [
    "low_health_bar_green.png",
    "low_health_bar_yellow.png",
    "low_health_bar_red.png",
]


def analyze_image(detector, filepath, filename):
    img = cv2.imread(filepath)
    if img is None:
        print(f"  [ERROR] 无法读取 {filename}")
        return

    # 裁剪 ROI
    rx, ry, rw, rh = detector.roi
    roi = img[ry:ry + rh, rx:rx + rw]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 找到有颜色的像素 (排除暗色背景: S>50, V>50)
    mask_colored = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
    colored_pixels = hsv[mask_colored]

    if len(colored_pixels) == 0:
        print(f"  ROI 内无彩色像素 — ROI 位置可能不对!")
        return

    h_vals = colored_pixels[:, 0]
    s_vals = colored_pixels[:, 1]
    v_vals = colored_pixels[:, 2]

    print(f"  彩色像素数: {len(colored_pixels)}")
    print(f"  H (色相):  min={h_vals.min():3d}  max={h_vals.max():3d}  "
          f"mean={h_vals.mean():.1f}  median={np.median(h_vals):.1f}")
    print(f"  S (饱和度): min={s_vals.min():3d}  max={s_vals.max():3d}  "
          f"mean={s_vals.mean():.1f}")
    print(f"  V (亮度):  min={v_vals.min():3d}  max={v_vals.max():3d}  "
          f"mean={v_vals.mean():.1f}")

    # H 值分布 (直方图)
    h_hist, _ = np.histogram(h_vals, bins=[0, 10, 15, 20, 35, 85, 100, 170, 180])
    labels = ["0-9(红)", "10-14(橙红)", "15-19(橙)", "20-34(黄)",
              "35-84(绿)", "85-99(青)", "100-169(蓝紫)", "170-179(深红)"]
    print(f"  H 分布:")
    for label, count in zip(labels, h_hist):
        if count > 0:
            bar = "#" * min(count // 5, 40)
            print(f"    {label:16s}: {count:5d} {bar}")

    # 运行检测器
    result = detector.detect(img)
    print(f"  检测结果: health_pct={result['health_pct']:.1%}  "
          f"damage_pct={result['damage_pct']:.1%}  is_hit={result['is_hit']}")

    # 检查当前阈值覆盖情况
    in_green = np.sum((h_vals >= 35) & (h_vals <= 85))
    in_yellow = np.sum((h_vals >= 15) & (h_vals < 35))
    in_flash_red = np.sum((h_vals >= 0) & (h_vals < 15))
    in_deep_red = np.sum((h_vals >= 170) & (h_vals <= 180))
    uncovered = len(h_vals) - in_green - in_yellow - in_flash_red - in_deep_red
    total = len(h_vals)

    print(f"  阈值覆盖:")
    print(f"    绿 (H 35-85):     {in_green:5d} ({in_green/total:.1%})")
    print(f"    黄 (H 15-34):     {in_yellow:5d} ({in_yellow/total:.1%})")
    print(f"    闪红 (H 0-14):    {in_flash_red:5d} ({in_flash_red/total:.1%})")
    print(f"    深红 (H 170-180): {in_deep_red:5d} ({in_deep_red/total:.1%})")
    if uncovered > 0:
        print(f"    *** 未覆盖:       {uncovered:5d} ({uncovered/total:.1%}) ***")
        uncov_mask = ~((h_vals >= 35) & (h_vals <= 85)) & \
                     ~((h_vals >= 15) & (h_vals < 35)) & \
                     ~((h_vals >= 0) & (h_vals < 15)) & \
                     ~((h_vals >= 170) & (h_vals <= 180))
        uncov_h = h_vals[uncov_mask]
        print(f"    未覆盖 H 值: {np.unique(uncov_h)}")


def main():
    detector = HealthBarDetector()
    rx, ry, rw, rh = detector.roi
    print(f"ROI: x={rx}, y={ry}, w={rw}, h={rh}")
    print(f"当前阈值:")
    print(f"  GREEN:     H {detector.GREEN_LOWER[0]}-{detector.GREEN_UPPER[0]}")
    print(f"  YELLOW:    H {detector.YELLOW_LOWER[0]}-{detector.YELLOW_UPPER[0]}")
    print(f"  FLASH_RED: H {detector.FLASH_RED_LOWER[0]}-{detector.FLASH_RED_UPPER[0]}")
    print(f"  DAMAGE:    H {detector.RED_LOWER_1[0]}-{detector.RED_UPPER_1[0]} "
          f"+ H {detector.RED_LOWER_2[0]}-{detector.RED_UPPER_2[0]}")
    print()

    for filename in LOW_HEALTH_FILES:
        filepath = os.path.join(PHOTOS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[SKIP] {filename} — 文件不存在")
            continue

        print(f"{'='*60}")
        print(f"文件: {filename}")
        analyze_image(detector, filepath, filename)
        print()

    # 也测试其他图片作为对比
    print(f"\n{'='*60}")
    print("对比: 正常血量图片")
    for filename in ["health_bar_full_health.png", "health_bar_not_full_health.png",
                     "Health_bar_below_half_health.png"]:
        filepath = os.path.join(PHOTOS_DIR, filename)
        if not os.path.exists(filepath):
            continue
        print(f"\n--- {filename} ---")
        analyze_image(detector, filepath, filename)


if __name__ == "__main__":
    main()
