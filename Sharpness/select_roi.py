"""ROI selection tool — screenshot then drag-select the sharpness bar region."""

import json
import os
import cv2
import numpy as np
import mss

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "roi_config.json")


def select_roi():
    """Capture screen, let user drag-select ROI, save to roi_config.json."""
    sct = mss.mss()
    monitor = sct.monitors[1]

    print(f"Capturing monitor: {monitor['width']}x{monitor['height']}")
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Downscale for display if too large
    h, w = frame.shape[:2]
    scale = 1.0
    if w > 1920:
        scale = 1920 / w
        display = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        display = frame.copy()

    print("Drag to select the sharpness bar region, then press ENTER or SPACE to confirm.")
    print("Press C to cancel and retry.")

    roi = cv2.selectROI("Select Sharpness ROI", display, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        print("No region selected. Exiting.")
        return None

    # Scale back to original resolution
    x = int(roi[0] / scale)
    y = int(roi[1] / scale)
    w = int(roi[2] / scale)
    h = int(roi[3] / scale)

    roi_dict = {"x": x, "y": y, "w": w, "h": h}
    with open(CONFIG_PATH, "w") as f:
        json.dump(roi_dict, f, indent=2)

    print(f"ROI saved to {CONFIG_PATH}: {roi_dict}")

    # Preview the selected region
    crop = frame[y:y + h, x:x + w]
    cv2.imshow("Selected ROI", crop)
    print("Press any key to close preview.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (x, y, w, h)


if __name__ == "__main__":
    select_roi()
