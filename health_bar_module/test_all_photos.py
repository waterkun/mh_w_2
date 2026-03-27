"""用所有 health_bar_photos 测试检测器.

使用方式:
  python -m health_bar_module.test_all_photos
"""

import os
import cv2
import numpy as np
from health_bar_module.mh_w_2_health_bar.health_bar_detector import HealthBarDetector


PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "health_bar_photos")

# (filename, expected description)
TEST_CASES = [
    ("health_bar_full_health.png", "满血 ~100%"),
    ("health_bar_not_full_health.png", "不满血 ~70-80%"),
    ("health_bar_get_hit_lost_health.png", "刚被打 有伤害段"),
    ("Health_bar_below_half_health.png", "半血以下 ~40-50%"),
    ("low_health_bar_green.png", "低血-绿闪 ~15-25%"),
    ("low_health_bar_yellow.png", "低血-黄闪 ~15-25%"),
    ("low_health_bar_red.png", "低血-红闪 ~15-25%"),
    ("wrong1.png", "debug截图 (有游戏特效)"),
    ("wrong2.png", "debug截图 (不满血但显示100%)"),
]


def test_image(detector, filepath, filename, desc):
    img = cv2.imread(filepath)
    if img is None:
        print(f"  [ERROR] 无法读取")
        return None

    h_img, w_img = img.shape[:2]

    # 检查图片是否足够大来使用默认 ROI
    rx, ry, rw, rh = detector.roi
    if ry + rh > h_img or rx + rw > w_img:
        # 图片太小 (裁剪图), 用整张图作为 ROI
        detector_local = HealthBarDetector(roi=(0, 0, w_img, h_img))
        result = detector_local.detect(img)
        roi_note = f"(整图 {w_img}x{h_img})"
    else:
        # 重置 history 避免帧间污染
        detector._hp_history.clear()
        detector._dmg_history.clear()
        result = detector.detect(img)
        roi_note = f"(ROI {rx},{ry} {rw}x{rh})"

    hp = result["health_pct"]
    dmg = result["damage_pct"]

    # 简单状态条
    bar_len = 30
    filled = int(bar_len * hp)
    bar = "#" * filled + "-" * (bar_len - filled)

    print(f"  HP={hp:6.1%}  |{bar}|  DMG={dmg:.1%}  {roi_note}")
    return result


def main():
    print("=" * 70)
    print("Health Bar 检测器 — 全图片测试")
    print("=" * 70)

    detector = HealthBarDetector()
    print(f"ROI: {detector.roi}")
    print()

    for filename, desc in TEST_CASES:
        filepath = os.path.join(PHOTOS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[SKIP] {filename}")
            continue

        print(f"--- {filename} ---")
        print(f"  期望: {desc}")
        test_image(detector, filepath, filename, desc)
        print()

    # 也扫描目录中其他未列出的图片
    all_files = set(os.listdir(PHOTOS_DIR))
    listed_files = set(f for f, _ in TEST_CASES)
    extra = sorted(all_files - listed_files)
    if extra:
        print(f"--- 其他图片 ---")
        for filename in extra:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            filepath = os.path.join(PHOTOS_DIR, filename)
            print(f"  {filename}:")
            test_image(detector, filepath, filename, "")
            print()

    print("=" * 70)
    print("完成")


if __name__ == "__main__":
    main()
