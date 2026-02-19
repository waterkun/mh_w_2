"""Test health bar detector against static screenshots in health_bar_photos/."""

import os
import cv2
from health_bar_detector import HealthBarDetector


PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "..", "health_bar_photos")

# (filename, expected_description)
TEST_CASES = [
    ("health_bar_full_health.png", "Full health — expect ~100% health, 0% damage"),
    ("health_bar_not_full_health.png", "Partial health — expect <100% health, 0% damage"),
    ("health_bar_get_hit_lost_health.png", "Just got hit — expect red damage segment"),
    ("Health_bar_below_half_health.png", "Below half — expect <50% health, red segment"),
]


def main():
    detector = HealthBarDetector()

    # The screenshots are cropped to the health bar region already,
    # so we use the full image as ROI.
    for filename, desc in TEST_CASES:
        path = os.path.join(PHOTOS_DIR, filename)
        if not os.path.exists(path):
            print(f"[SKIP] {filename} — file not found")
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] {filename} — could not read image")
            continue

        # Override ROI to full image since these are pre-cropped screenshots
        h, w = img.shape[:2]
        detector.roi = (0, 0, w, h)

        result = detector.detect(img)

        print(f"\n{'=' * 60}")
        print(f"File: {filename}")
        print(f"Desc: {desc}")
        print(f"  health_pct : {result['health_pct']:.2%}")
        print(f"  damage_pct : {result['damage_pct']:.2%}")
        print(f"  is_hit     : {result['is_hit']}")

    print(f"\n{'=' * 60}")
    print("Done. Review results above to verify detection accuracy.")


if __name__ == "__main__":
    main()
