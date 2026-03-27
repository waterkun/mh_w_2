"""Test gauge detector against static screenshots in gauge photo/."""

import os
import cv2
import numpy as np
from gauge_detector import GaugeDetector


def imread_unicode(path):
    """Read image with unicode path (cv2.imread fails on Windows with CJK)."""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "gauge photo")

# (filename, expected_state, expected_description)
TEST_CASES = [
    ("红刃.png", "red", "Red blade full — expect is_red=True, red_pct ~1.0"),
    ("红刃2.png", "red", "Red blade partial — expect is_red=True, red_pct ~0.5-0.7"),
    ("红刃3.png", "red", "Red blade low — expect is_red=True, red_pct ~0.2-0.3"),
    ("yelloGauge.png", "yellow", "Yellow gauge — expect is_red=False, gauge_state=yellow"),
    ("whiteGauge.png", "white", "White gauge — expect is_red=False, gauge_state=white"),
    ("NoGauge.png", "empty", "No gauge — expect is_red=False, gauge_state=empty"),
]


def main():
    detector = GaugeDetector()

    passed = 0
    failed = 0

    for filename, expected_state, desc in TEST_CASES:
        path = os.path.join(PHOTOS_DIR, filename)
        if not os.path.exists(path):
            print(f"[SKIP] {filename} — file not found")
            continue

        img = imread_unicode(path)
        if img is None:
            print(f"[ERROR] {filename} — could not read image")
            continue

        # Photos are pre-cropped gauge regions, override ROI to full image
        h, w = img.shape[:2]
        detector.roi = (0, 0, w, h)

        result = detector.detect(img)

        # Check state match
        state_ok = result["gauge_state"] == expected_state
        status = "PASS" if state_ok else "FAIL"
        if state_ok:
            passed += 1
        else:
            failed += 1

        print(f"\n{'=' * 60}")
        print(f"[{status}] {filename}")
        print(f"  Desc       : {desc}")
        print(f"  gauge_state: {result['gauge_state']} (expected: {expected_state})")
        print(f"  is_red     : {result['is_red']}")
        print(f"  red_pct    : {result['red_pct']:.2%}")

        # Show debug masks
        _, roi_bgr, red_mask, gauge_mask = detector.detect_debug(img)
        display_h = 80
        roi_vis = cv2.resize(roi_bgr, (400, display_h))
        red_vis = cv2.resize(red_mask, (400, display_h))
        gauge_vis = cv2.resize(gauge_mask, (400, display_h))
        red_bgr = cv2.cvtColor(red_vis, cv2.COLOR_GRAY2BGR)
        gauge_bgr = cv2.cvtColor(gauge_vis, cv2.COLOR_GRAY2BGR)

        combined = cv2.vconcat([roi_vis, red_bgr, gauge_bgr])
        cv2.imshow(f"{filename} | ROI / Red / All masks", combined)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("Press any key to close windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
