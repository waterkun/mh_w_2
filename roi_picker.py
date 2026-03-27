"""Reusable ROI picker tool — screenshot + mouse drag to select region.

Usage:
    python roi_picker.py                     # screenshot and pick
    python roi_picker.py --image path.png    # pick from existing image

Output: prints (x, y, w, h) to console for copy-paste.

Can also be imported:
    from roi_picker import pick_roi
    roi = pick_roi()           # from live screenshot
    roi = pick_roi("img.png")  # from file
"""

import argparse
import sys

import cv2
import mss
import numpy as np


WINDOW_NAME = "ROI Picker — drag to select, Enter=confirm, R=reset, Q=quit"


def _grab_screenshot():
    """Capture primary monitor screenshot, return BGR image."""
    sct = mss.mss()
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


class _RoiPicker:
    def __init__(self, image):
        self.image = image
        self.clone = image.copy()
        self.drawing = False
        self.x0 = self.y0 = 0
        self.x1 = self.y1 = 0
        self.roi = None
        # Scale for display if image is large
        h, w = image.shape[:2]
        self.scale = min(1.0, 1920 / w, 1080 / h)
        self.display_w = int(w * self.scale)
        self.display_h = int(h * self.scale)

    def _to_orig(self, x, y):
        """Convert display coordinates to original image coordinates."""
        return int(x / self.scale), int(y / self.scale)

    def _mouse_callback(self, event, x, y, flags, param):
        ox, oy = self._to_orig(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x0, self.y0 = ox, oy
            self.x1, self.y1 = ox, oy

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x1, self.y1 = ox, oy

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.x1, self.y1 = ox, oy
            self._update_roi()

    def _update_roi(self):
        x = min(self.x0, self.x1)
        y = min(self.y0, self.y1)
        w = abs(self.x1 - self.x0)
        h = abs(self.y1 - self.y0)
        if w > 2 and h > 2:
            self.roi = (x, y, w, h)

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.display_w, self.display_h)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        print("Drag to select ROI, Enter=confirm, R=reset, Q=quit")

        while True:
            display = self.clone.copy()

            # Draw current selection rectangle
            if self.x0 != self.x1 or self.y0 != self.y1:
                x = min(self.x0, self.x1)
                y = min(self.y0, self.y1)
                w = abs(self.x1 - self.x0)
                h = abs(self.y1 - self.y0)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Show ROI text
                label = f"({x}, {y}, {w}, {h})"
                cv2.putText(display, label, (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Draw zoomed ROI preview in corner
                if w > 2 and h > 2:
                    roi_crop = self.image[y:y+h, x:x+w]
                    preview_scale = max(1, min(8, 400 // max(w, 1)))
                    preview = cv2.resize(roi_crop,
                                         (w * preview_scale, h * preview_scale),
                                         interpolation=cv2.INTER_NEAREST)
                    ph, pw = preview.shape[:2]
                    # Place at top-right
                    img_h, img_w = display.shape[:2]
                    px = max(0, img_w - pw - 10)
                    py = 10
                    if px + pw <= img_w and py + ph <= img_h:
                        display[py:py+ph, px:px+pw] = preview
                        cv2.rectangle(display, (px-1, py-1),
                                      (px+pw+1, py+ph+1), (0, 255, 0), 1)

            resized = cv2.resize(display, (self.display_w, self.display_h))
            cv2.imshow(WINDOW_NAME, resized)

            key = cv2.waitKey(30) & 0xFF

            if key == 13:  # Enter
                if self.roi:
                    cv2.destroyAllWindows()
                    return self.roi
                print("No ROI selected yet — drag first")

            elif key == ord("r"):
                self.x0 = self.y0 = self.x1 = self.y1 = 0
                self.roi = None
                print("Reset")

            elif key == ord("q"):
                cv2.destroyAllWindows()
                return self.roi


def pick_roi(image_path=None):
    """Pick ROI interactively. Returns (x, y, w, h) or None.

    Args:
        image_path: path to image file. If None, captures screenshot.
    """
    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: cannot read {image_path}")
            return None
    else:
        print("Capturing screenshot...")
        image = _grab_screenshot()

    picker = _RoiPicker(image)
    roi = picker.run()

    if roi:
        print(f"\nROI = ({roi[0]}, {roi[1]}, {roi[2]}, {roi[3]})")
    else:
        print("\nNo ROI selected")

    return roi


def main():
    parser = argparse.ArgumentParser(description="Pick ROI from screenshot or image")
    parser.add_argument("--image", default=None,
                        help="Image file path (default: capture screenshot)")
    args = parser.parse_args()

    roi = pick_roi(args.image)
    if roi:
        print(f"\nCopy this to your code:")
        print(f"  ROI = ({roi[0]}, {roi[1]}, {roi[2]}, {roi[3]})")


if __name__ == "__main__":
    main()
