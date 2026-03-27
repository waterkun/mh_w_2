"""ROI 选择工具 — 截屏后两步缩放选择怪物活动区域."""

import json
import os
import cv2
import numpy as np
import mss

from config import ROI_CONFIG_PATH


def select_roi():
    """两步 ROI 选择: 粗选 → 放大精选."""
    sct = mss.mss()
    monitor = sct.monitors[1]

    print(f"截取屏幕: {monitor['width']}x{monitor['height']}")
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # --- Step 1: 粗选 (缩放到 1920 宽) ---
    full_h, full_w = frame.shape[:2]
    scale1 = 1.0
    if full_w > 1920:
        scale1 = 1920 / full_w
        display1 = cv2.resize(frame, (int(full_w * scale1), int(full_h * scale1)))
    else:
        display1 = frame.copy()

    print("[Step 1] 拖选怪物活动的大致区域，然后按 Enter 确认。")
    rough = cv2.selectROI("Step 1: 粗选区域", display1,
                          fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if rough[2] == 0 or rough[3] == 0:
        print("未选择区域，退出。")
        return None

    # 映射回原始分辨率
    rx = int(rough[0] / scale1)
    ry = int(rough[1] / scale1)
    rw = int(rough[2] / scale1)
    rh = int(rough[3] / scale1)

    # 加 padding
    pad = 50
    rx0 = max(rx - pad, 0)
    ry0 = max(ry - pad, 0)
    rx1 = min(rx + rw + pad, full_w)
    ry1 = min(ry + rh + pad, full_h)
    crop_rough = frame[ry0:ry1, rx0:rx1]

    # --- Step 2: 精选 (放大) ---
    crop_h, crop_w = crop_rough.shape[:2]
    target_w = 1200
    scale2 = max(target_w / crop_w, 1.0)
    zoomed = cv2.resize(crop_rough,
                        (int(crop_w * scale2), int(crop_h * scale2)),
                        interpolation=cv2.INTER_LINEAR)

    print(f"[Step 2] 放大 {scale2:.1f}x，精确选择怪物活动区域。")
    precise = cv2.selectROI("Step 2: 精选 (放大)", zoomed,
                            fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if precise[2] == 0 or precise[3] == 0:
        print("未选择区域，退出。")
        return None

    # 映射回原始分辨率
    x = int(rx0 + precise[0] / scale2)
    y = int(ry0 + precise[1] / scale2)
    w = int(precise[2] / scale2)
    h = int(precise[3] / scale2)

    roi_dict = {"x": x, "y": y, "w": w, "h": h}
    with open(ROI_CONFIG_PATH, "w") as f:
        json.dump(roi_dict, f, indent=2)

    print(f"ROI 已保存到 {ROI_CONFIG_PATH}: {roi_dict}")

    # 预览
    crop = frame[y:y + h, x:x + w]
    scale_preview = max(400 / crop.shape[1], 1.0)
    preview = cv2.resize(crop,
                         (int(crop.shape[1] * scale_preview),
                          int(crop.shape[0] * scale_preview)),
                         interpolation=cv2.INTER_LINEAR)
    cv2.imshow("最终 ROI 预览", preview)
    print("按任意键关闭预览。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (x, y, w, h)


if __name__ == "__main__":
    select_roi()
