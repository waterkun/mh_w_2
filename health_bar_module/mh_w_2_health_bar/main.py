"""Live screen capture demo — displays health bar detection in real time."""

import cv2
import numpy as np
import mss
from health_bar_detector import HealthBarDetector
from health_bar_tracker import HealthBarTracker


def main():
    detector = HealthBarDetector()
    tracker = HealthBarTracker(detector)

    sct = mss.mss()
    # Capture the primary monitor
    monitor = sct.monitors[1]

    print("Health Bar Live Detector — press 'q' to quit")
    print(f"Monitor: {monitor['width']}x{monitor['height']}")
    print(f"ROI: {detector.roi}")

    while True:
        # Grab screen
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        # mss returns BGRA, convert to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Update tracker (calls detector internally)
        state = tracker.update(frame)
        reward = tracker.get_reward_signal()

        # Get debug visuals
        _, roi_bgr, green_mask, red_mask = detector.detect_debug(frame)

        # Build status text overlay
        status_lines = [
            f"HP: {state['health_pct']:.1%}  DMG: {state['damage_pct']:.1%}",
            f"Hit: {state['is_hit']}  Hits: {state['hit_count']}  "
            f"Delta: {state['health_delta']:+.2%}",
            f"Alive: {state['is_alive']}  Reward: {reward:+.4f}",
            f"Last hit: {state['time_since_last_hit']:.1f}s ago",
        ]

        # Draw status on ROI preview
        display_roi = cv2.resize(roi_bgr, (600, 60))
        info_panel = np.zeros((120, 600, 3), dtype=np.uint8)
        for i, line in enumerate(status_lines):
            cv2.putText(
                info_panel, line, (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
            )

        # Stack masks side by side
        green_vis = cv2.resize(green_mask, (300, 40))
        red_vis = cv2.resize(red_mask, (300, 40))
        masks_row = np.hstack([green_vis, red_vis])
        masks_bgr = cv2.cvtColor(masks_row, cv2.COLOR_GRAY2BGR)

        # Combine all panels
        display = np.vstack([display_roi, masks_bgr, info_panel])

        cv2.imshow("Health Bar Detector", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
