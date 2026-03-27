"""Sharpness detector — single-frame, stateless sharpness gauge detection via HSV.

Detects the current sharpness color level (purple/white/blue/green/yellow/orange/red)
and estimates the remaining percentage of the current color segment.
"""

import json
import os
import cv2
import numpy as np


class SharpnessDetector:
    """Detect weapon sharpness from a single frame using HSV color masking.

    The sharpness bar in Monster Hunter World shows colored segments.
    Higher colors = more damage. When a color depletes, it drops to the next tier.

    Tier order (best to worst): purple > white > blue > green > yellow > orange > red
    """

    DEFAULT_ROI = (100, 50, 400, 25)
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "roi_config.json")

    # --- HSV ranges for each sharpness color ---
    # Blue sharpness (bright cyan-blue glow)
    BLUE_LOWER = np.array([90, 80, 120])
    BLUE_UPPER = np.array([130, 255, 255])

    # White sharpness (low saturation, very high value)
    WHITE_LOWER = np.array([0, 0, 180])
    WHITE_UPPER = np.array([180, 60, 255])

    # Green sharpness
    GREEN_LOWER = np.array([35, 60, 80])
    GREEN_UPPER = np.array([85, 255, 255])

    # Yellow sharpness
    YELLOW_LOWER = np.array([20, 80, 100])
    YELLOW_UPPER = np.array([34, 255, 255])

    # Orange sharpness
    ORANGE_LOWER = np.array([10, 80, 100])
    ORANGE_UPPER = np.array([19, 255, 255])

    # Red sharpness (wraps around H=0/180)
    RED_LOWER_1 = np.array([0, 80, 100])
    RED_UPPER_1 = np.array([9, 255, 255])
    RED_LOWER_2 = np.array([165, 80, 100])
    RED_UPPER_2 = np.array([180, 255, 255])

    # Purple sharpness
    PURPLE_LOWER = np.array([130, 50, 80])
    PURPLE_UPPER = np.array([165, 255, 255])

    # Minimum pixel ratio per column to count as having color
    COL_FILL_THRESHOLD = 0.05

    # Margin to trim decorative borders (weapon icon on left, cap on right)
    LEFT_MARGIN_FRAC = 0.15
    RIGHT_MARGIN_FRAC = 0.02

    # Minimum pixel fraction to consider a color "present"
    MIN_COLOR_AREA_FRAC = 0.01

    # Sharpness tiers ordered from best to worst
    TIER_ORDER = ["purple", "white", "blue", "green", "yellow", "orange", "red"]

    # Tiers that need sharpening
    WARN_TIERS = {"blue"}                              # soft — sharpen when possible
    LOW_TIERS = {"green", "yellow", "orange", "red"}   # hard — must sharpen now

    def __init__(self, roi=None):
        if roi is not None:
            self.roi = roi
        else:
            self.roi = self._load_roi()

    def _load_roi(self):
        if os.path.exists(self.CONFIG_PATH):
            with open(self.CONFIG_PATH) as f:
                cfg = json.load(f)
            return (cfg["x"], cfg["y"], cfg["w"], cfg["h"])
        return self.DEFAULT_ROI

    def _crop_roi(self, frame):
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    # Morphological kernel to remove stray pixels from masks
    _MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))

    def _clean_mask(self, mask):
        """Remove small noise from a mask using morphological opening."""
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._MORPH_KERNEL)

    def _make_masks(self, hsv):
        """Create color masks for all sharpness tiers."""
        blue_mask = self._clean_mask(cv2.inRange(hsv, self.BLUE_LOWER, self.BLUE_UPPER))
        white_mask = self._clean_mask(cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER))
        green_mask = self._clean_mask(cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER))
        yellow_mask = self._clean_mask(cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER))
        orange_mask = self._clean_mask(cv2.inRange(hsv, self.ORANGE_LOWER, self.ORANGE_UPPER))

        red_mask_1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
        red_mask_2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
        red_mask = self._clean_mask(cv2.bitwise_or(red_mask_1, red_mask_2))

        purple_mask = self._clean_mask(cv2.inRange(hsv, self.PURPLE_LOWER, self.PURPLE_UPPER))

        return {
            "purple": purple_mask,
            "white": white_mask,
            "blue": blue_mask,
            "green": green_mask,
            "yellow": yellow_mask,
            "orange": orange_mask,
            "red": red_mask,
        }

    def _color_area_frac(self, mask):
        total = mask.size
        if total == 0:
            return 0.0
        return np.count_nonzero(mask) / total

    def _column_has_color(self, mask):
        h = mask.shape[0]
        if h == 0:
            return np.array([], dtype=bool)
        ratio = np.count_nonzero(mask, axis=0) / h
        return ratio >= self.COL_FILL_THRESHOLD

    def _detect_from_masks(self, masks, bar_width, bar_start, bar_end):
        """Core detection: find current sharpness color and remaining percentage."""
        if bar_width <= 0:
            return {
                "sharpness_color": "unknown",
                "sharpness_pct": 0.0,
                "need_sharpen": True,
                "color_fracs": {},
            }

        # Compute area fraction for each color in the bar region
        color_fracs = {}
        for color, mask in masks.items():
            bar_mask = mask[:, bar_start:bar_end]
            color_fracs[color] = self._color_area_frac(bar_mask)

        # Find the dominant color (highest area fraction above threshold)
        current_color = "unknown"
        best_frac = 0.0
        for color, frac in color_fracs.items():
            if frac >= self.MIN_COLOR_AREA_FRAC and frac > best_frac:
                best_frac = frac
                current_color = color

        # Estimate remaining percentage using the largest contiguous color run.
        # The sharpness bar fills from the left; stray pixels on the right
        # (from decorations) should not inflate the percentage.
        sharpness_pct = 0.0
        if current_color != "unknown":
            bar_mask = masks[current_color][:, bar_start:bar_end]
            color_cols = self._column_has_color(bar_mask)

            # Find the largest contiguous run of True columns
            best_run_end = -1
            best_run_len = 0
            run_start = -1
            run_len = 0
            for i, has_color in enumerate(color_cols):
                if has_color:
                    if run_len == 0:
                        run_start = i
                    run_len += 1
                    if run_len > best_run_len:
                        best_run_len = run_len
                        best_run_end = i
                else:
                    run_len = 0

            if best_run_len > 0:
                # Percentage = right edge of the largest run / total bar width
                sharpness_pct = round((best_run_end + 1) / bar_width, 4)
                sharpness_pct = min(sharpness_pct, 1.0)

        need_sharpen = current_color in self.LOW_TIERS or current_color == "unknown"
        should_sharpen = need_sharpen or current_color in self.WARN_TIERS

        return {
            "sharpness_color": current_color,
            "sharpness_pct": sharpness_pct,
            "should_sharpen": should_sharpen,  # blue — sharpen when possible
            "need_sharpen": need_sharpen,       # green/yellow/orange/red — must sharpen
            "color_fracs": color_fracs,
        }

    def detect(self, frame) -> dict:
        """Detect sharpness from a single frame (BGR image).

        Returns:
            dict with keys:
                sharpness_color — current tier: "purple"/"white"/"blue"/"green"/"yellow"/"orange"/"red"/"unknown"
                sharpness_pct   — estimated fill ratio of current color [0, 1]
                need_sharpen    — True if sharpness is in a low tier
                color_fracs     — dict of area fractions for each color (for debugging)
        """
        roi = self._crop_roi(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        masks = self._make_masks(hsv)

        total_w = roi.shape[1]
        bar_start = int(total_w * self.LEFT_MARGIN_FRAC)
        bar_end = int(total_w * (1.0 - self.RIGHT_MARGIN_FRAC))
        bar_width = bar_end - bar_start

        return self._detect_from_masks(masks, bar_width, bar_start, bar_end)

    def detect_debug(self, frame) -> tuple:
        """Like detect(), but also returns debug images.

        Returns:
            (result_dict, roi_bgr, current_color_mask, combined_mask)
        """
        roi = self._crop_roi(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        masks = self._make_masks(hsv)

        total_w = roi.shape[1]
        bar_start = int(total_w * self.LEFT_MARGIN_FRAC)
        bar_end = int(total_w * (1.0 - self.RIGHT_MARGIN_FRAC))
        bar_width = bar_end - bar_start

        result = self._detect_from_masks(masks, bar_width, bar_start, bar_end)

        # Current color mask
        current_color = result["sharpness_color"]
        if current_color in masks:
            current_mask = masks[current_color]
        else:
            current_mask = np.zeros(roi.shape[:2], dtype=np.uint8)

        # Combined mask for visualization
        combined = np.zeros(roi.shape[:2], dtype=np.uint8)
        for mask in masks.values():
            combined = cv2.bitwise_or(combined, mask)

        return result, roi, current_mask, combined
