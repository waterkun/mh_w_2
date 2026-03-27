"""Live screen capture — displays sharpness detection and sharpen alerts in real time."""

import cv2
import numpy as np
import mss
from sharpness_detector import SharpnessDetector
from sharpness_tracker import SharpnessTracker


# Color mapping for each sharpness tier (BGR)
TIER_COLORS = {
    "purple": (200, 50, 200),
    "white": (255, 255, 255),
    "blue": (255, 180, 50),
    "green": (50, 200, 50),
    "yellow": (0, 220, 255),
    "orange": (0, 140, 255),
    "red": (0, 0, 255),
    "unknown": (128, 128, 128),
}


def main():
    detector = SharpnessDetector()
    tracker = SharpnessTracker(detector)

    sct = mss.mss()
    monitor = sct.monitors[1]

    print("Sharpness Live Detector — press 'q' to quit")
    print(f"Monitor: {monitor['width']}x{monitor['height']}")
    print(f"ROI: {detector.roi}")

    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Update tracker
        state = tracker.update(frame)
        reward = tracker.get_reward_signal()

        # Get debug visuals
        _, roi_bgr, current_mask, combined_mask = detector.detect_debug(frame)

        # Status color
        color = TIER_COLORS.get(state["sharpness_color"], (128, 128, 128))

        flash_tag = " [FLASHING]" if state["is_flashing"] else ""
        status_lines = [
            f"Sharpness: {state['sharpness_color'].upper()}{flash_tag}  "
            f"Pct: {state['sharpness_pct']:.1%}",
            f"Should Sharpen: {state['should_sharpen']}  "
            f"NEED Sharpen: {state['need_sharpen']}  "
            f"Alert: {state['sharpen_alert']}",
            f"Prev: {state['prev_color']}  "
            f"Dropped: {state['color_just_dropped']}  "
            f"Reward: {reward:+.4f}",
        ]

        # Draw ROI preview
        display_roi = cv2.resize(roi_bgr, (600, 60))

        # Info panel
        info_panel = np.zeros((100, 600, 3), dtype=np.uint8)
        for i, line in enumerate(status_lines):
            cv2.putText(
                info_panel, line, (10, 25 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1,
            )

        # Alert overlay — two levels
        if state["need_sharpen"]:
            cv2.putText(
                info_panel, ">>> SHARPEN NOW! <<<", (350, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
        elif state["should_sharpen"]:
            cv2.putText(
                info_panel, "Sharpen when possible", (350, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1,
            )

        # Stack masks side by side
        cur_vis = cv2.resize(current_mask, (300, 40))
        comb_vis = cv2.resize(combined_mask, (300, 40))
        masks_row = np.hstack([cur_vis, comb_vis])
        masks_bgr = cv2.cvtColor(masks_row, cv2.COLOR_GRAY2BGR)

        # Combine all panels
        display = np.vstack([display_roi, masks_bgr, info_panel])

        cv2.imshow("Sharpness Detector", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
