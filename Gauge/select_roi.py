"""ROI selection tool — screenshot then drag-select the gauge region."""

import json
import os
import cv2
import numpy as np
import mss

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "roi_config.json")


def select_roi():
    """Two-step ROI selection: rough select then zoom-in for precise select."""
    sct = mss.mss()
    monitor = sct.monitors[1]

    print(f"Capturing monitor: {monitor['width']}x{monitor['height']}")
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # --- Step 1: rough select on downscaled image ---
    full_h, full_w = frame.shape[:2]
    scale1 = 1.0
    if full_w > 1920:
        scale1 = 1920 / full_w
        display1 = cv2.resize(frame, (int(full_w * scale1), int(full_h * scale1)))
    else:
        display1 = frame.copy()

    print("[Step 1] Drag a ROUGH region around the gauge area, then ENTER.")
    rough = cv2.selectROI("Step 1: Rough Select", display1,
                          fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if rough[2] == 0 or rough[3] == 0:
        print("No region selected. Exiting.")
        return None

    # Map rough selection back to original resolution
    rx = int(rough[0] / scale1)
    ry = int(rough[1] / scale1)
    rw = int(rough[2] / scale1)
    rh = int(rough[3] / scale1)

    # Add padding around rough selection for comfort
    pad = 50
    rx0 = max(rx - pad, 0)
    ry0 = max(ry - pad, 0)
    rx1 = min(rx + rw + pad, full_w)
    ry1 = min(ry + rh + pad, full_h)
    crop_rough = frame[ry0:ry1, rx0:rx1]

    # --- Step 2: precise select on zoomed-in crop ---
    crop_h, crop_w = crop_rough.shape[:2]
    # Scale up so the crop fills ~1200px wide for easy selection
    target_w = 1200
    scale2 = max(target_w / crop_w, 1.0)
    zoomed = cv2.resize(crop_rough,
                        (int(crop_w * scale2), int(crop_h * scale2)),
                        interpolation=cv2.INTER_LINEAR)

    print(f"[Step 2] Zoomed in {scale2:.1f}x. Now PRECISELY select the gauge bar.")
    print("  -> Only box the bar itself, skip the hilt decoration on the left.")
    precise = cv2.selectROI("Step 2: Precise Select (zoomed)",
                            zoomed, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if precise[2] == 0 or precise[3] == 0:
        print("No region selected. Exiting.")
        return None

    # Map precise selection back to original resolution
    x = int(rx0 + precise[0] / scale2)
    y = int(ry0 + precise[1] / scale2)
    w = int(precise[2] / scale2)
    h = int(precise[3] / scale2)

    roi_dict = {"x": x, "y": y, "w": w, "h": h}
    with open(CONFIG_PATH, "w") as f:
        json.dump(roi_dict, f, indent=2)

    print(f"ROI saved to {CONFIG_PATH}: {roi_dict}")

    # Preview the selected region at zoom
    crop = frame[y:y + h, x:x + w]
    preview = cv2.resize(crop, (max(crop.shape[1] * 4, 400),
                                max(crop.shape[0] * 4, 100)),
                         interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Final ROI (4x zoom)", preview)
    print("Press any key to close preview.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (x, y, w, h)


if __name__ == "__main__":
    select_roi()
