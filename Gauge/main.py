"""Live screen capture demo — displays gauge detection in real time."""

import cv2
import numpy as np
import mss
from gauge_detector import GaugeDetector
from gauge_tracker import GaugeTracker


def main():
    detector = GaugeDetector()
    tracker = GaugeTracker(detector)

    sct = mss.mss()
    monitor = sct.monitors[1]

    print("Gauge Live Detector — press 'q' to quit")
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
        _, roi_bgr, red_mask, gauge_mask = detector.detect_debug(frame)

        # Build status text
        state_color = {
            "red": (0, 0, 255),
            "yellow": (0, 200, 255),
            "white": (255, 255, 255),
            "empty": (128, 128, 128),
        }.get(state["gauge_state"], (128, 128, 128))

        status_lines = [
            f"State: {state['gauge_state'].upper()}  "
            f"Red: {state['is_red']}  RedPct: {state['red_pct']:.1%}",
            f"Prev: {state['prev_gauge_state']}  "
            f"Activated: {state['red_just_activated']}  "
            f"Expired: {state['red_just_expired']}",
            f"Time in Red: {state['time_in_red']:.1f}s  Reward: {reward:+.4f}",
        ]

        # Draw ROI preview
        display_roi = cv2.resize(roi_bgr, (600, 60))

        # Info panel
        info_panel = np.zeros((100, 600, 3), dtype=np.uint8)
        for i, line in enumerate(status_lines):
            cv2.putText(
                info_panel, line, (10, 25 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 1,
            )

        # Stack masks side by side
        red_vis = cv2.resize(red_mask, (300, 40))
        gauge_vis = cv2.resize(gauge_mask, (300, 40))
        masks_row = np.hstack([red_vis, gauge_vis])
        masks_bgr = cv2.cvtColor(masks_row, cv2.COLOR_GRAY2BGR)

        # Combine all panels
        display = np.vstack([display_roi, masks_bgr, info_panel])

        cv2.imshow("Gauge Detector", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
