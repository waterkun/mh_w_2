"""Test sharpness detector against static screenshots in photo/."""

import os
import cv2
from sharpness_detector import SharpnessDetector


PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "photo")

# (filename, expected_color, description)
TEST_CASES = [
    ("blue_sharp.png", "blue", "Full blue sharpness — expect blue, high pct"),
    ("blue_sharp1.png", "blue", "Blue sharpness partial — expect blue"),
    ("white_sharp.png", "white", "Full white sharpness — expect white, high pct"),
    ("white_sharp1.png", "white", "White sharpness — expect white"),
    ("white_sharpness_almost_end.png", "white", "White almost depleted — expect white, low pct"),
]


def main():
    detector = SharpnessDetector()

    passed = 0
    failed = 0

    for filename, expected_color, desc in TEST_CASES:
        path = os.path.join(PHOTOS_DIR, filename)
        if not os.path.exists(path):
            print(f"[SKIP] {filename} — file not found")
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] {filename} — could not read image")
            continue

        # Photos are pre-cropped sharpness bar regions, override ROI to full image
        h, w = img.shape[:2]
        detector.roi = (0, 0, w, h)

        result = detector.detect(img)

        state_ok = result["sharpness_color"] == expected_color
        status = "PASS" if state_ok else "FAIL"
        if state_ok:
            passed += 1
        else:
            failed += 1

        print(f"\n{'=' * 60}")
        print(f"[{status}] {filename}")
        print(f"  Desc           : {desc}")
        print(f"  sharpness_color: {result['sharpness_color']} (expected: {expected_color})")
        print(f"  sharpness_pct  : {result['sharpness_pct']:.2%}")
        print(f"  need_sharpen   : {result['need_sharpen']}")
        print(f"  color_fracs    : ", end="")
        for color, frac in result["color_fracs"].items():
            if frac > 0.001:
                print(f"{color}={frac:.3f} ", end="")
        print()

        # Show debug masks
        _, roi_bgr, current_mask, combined_mask = detector.detect_debug(img)
        display_h = 80
        roi_vis = cv2.resize(roi_bgr, (400, display_h))
        cur_vis = cv2.resize(current_mask, (400, display_h))
        comb_vis = cv2.resize(combined_mask, (400, display_h))
        cur_bgr = cv2.cvtColor(cur_vis, cv2.COLOR_GRAY2BGR)
        comb_bgr = cv2.cvtColor(comb_vis, cv2.COLOR_GRAY2BGR)

        combined_display = cv2.vconcat([roi_vis, cur_bgr, comb_bgr])
        cv2.imshow(f"{filename} | ROI / Current / All masks", combined_display)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("Press any key to close windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
