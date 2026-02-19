import cv2
import numpy as np


class HealthBarDetector:
    """Detect health bar from a single frame using HSV color masking.

    Strategy: The health bar has a wavy/animated green fill, so we use
    span-based detection — find the rightmost green and red columns
    relative to the total bar width — rather than counting filled columns.
    """

    # Default ROI: top-left region where health bar lives (x, y, w, h)
    # Calibrated for 3440x1440 ultrawide — left monitor
    DEFAULT_ROI = (143, 63, 716, 48)

    # HSV ranges for green (current health)
    GREEN_LOWER = np.array([35, 80, 80])
    GREEN_UPPER = np.array([85, 255, 255])

    # HSV ranges for red (recent damage) — red wraps around in HSV
    RED_LOWER_1 = np.array([0, 80, 80])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([170, 80, 80])
    RED_UPPER_2 = np.array([180, 255, 255])

    # Minimum pixel ratio per column to count as having color
    COL_FILL_THRESHOLD = 0.05

    # Margin (fraction of width) to trim decorative borders on left/right
    LEFT_MARGIN_FRAC = 0.04
    RIGHT_MARGIN_FRAC = 0.04

    def __init__(self, roi=None):
        """
        Args:
            roi: (x, y, w, h) region of interest for the health bar.
                 If None, uses DEFAULT_ROI.
        """
        self.roi = roi or self.DEFAULT_ROI

    def _crop_roi(self, frame):
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    def _make_masks(self, hsv):
        green_mask = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)
        red_mask_1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
        red_mask_2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        return green_mask, red_mask

    def _column_has_color(self, mask):
        """Return boolean array: True if column has enough colored pixels."""
        h = mask.shape[0]
        if h == 0:
            return np.array([], dtype=bool)
        ratio = np.count_nonzero(mask, axis=0) / h
        return ratio >= self.COL_FILL_THRESHOLD

    def _find_bar_span(self, green_cols, red_cols, total_w):
        """Determine the usable bar span, trimming decorative margins.

        Returns:
            (bar_start, bar_end) — column indices defining the full bar.
        """
        bar_start = int(total_w * self.LEFT_MARGIN_FRAC)
        bar_end = int(total_w * (1.0 - self.RIGHT_MARGIN_FRAC))
        return bar_start, bar_end

    def _detect_from_masks(self, green_mask, red_mask):
        """Core detection logic from pre-computed masks."""
        total_w = green_mask.shape[1]
        if total_w == 0:
            return {"health_pct": 0.0, "damage_pct": 0.0, "is_hit": False}

        green_cols = self._column_has_color(green_mask)
        red_cols = self._column_has_color(red_mask)

        bar_start, bar_end = self._find_bar_span(green_cols, red_cols, total_w)
        bar_width = bar_end - bar_start
        if bar_width <= 0:
            return {"health_pct": 0.0, "damage_pct": 0.0, "is_hit": False}

        # Restrict to bar span
        green_in_bar = green_cols[bar_start:bar_end]
        red_in_bar = red_cols[bar_start:bar_end]

        # Find rightmost green column (= end of current health)
        green_indices = np.where(green_in_bar)[0]
        red_indices = np.where(red_in_bar)[0]

        if len(green_indices) == 0:
            health_pct = 0.0
        else:
            rightmost_green = green_indices[-1]
            health_pct = (rightmost_green + 1) / bar_width

        if len(red_indices) == 0:
            damage_pct = 0.0
        else:
            # Red span: from end of green to end of red
            rightmost_red = red_indices[-1]
            leftmost_red = red_indices[0]
            # Only count red that's beyond the green region
            if len(green_indices) > 0:
                red_beyond_green = red_indices[red_indices > green_indices[-1]]
                if len(red_beyond_green) > 0:
                    damage_pct = len(red_beyond_green) / bar_width
                else:
                    damage_pct = 0.0
            else:
                damage_pct = (rightmost_red - leftmost_red + 1) / bar_width

        health_pct = round(min(health_pct, 1.0), 4)
        damage_pct = round(min(damage_pct, 1.0), 4)
        is_hit = damage_pct > 0.005

        return {
            "health_pct": health_pct,
            "damage_pct": damage_pct,
            "is_hit": is_hit,
        }

    def detect(self, frame) -> dict:
        """Detect health bar state from a single frame (BGR image).

        Returns:
            dict with keys:
                health_pct  — green portion ratio [0, 1]
                damage_pct  — red portion ratio [0, 1]
                is_hit      — True if red segment detected
        """
        roi = self._crop_roi(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green_mask, red_mask = self._make_masks(hsv)
        return self._detect_from_masks(green_mask, red_mask)

    def detect_debug(self, frame) -> tuple:
        """Like detect(), but also returns debug images.

        Returns:
            (result_dict, roi_bgr, green_mask, red_mask)
        """
        roi = self._crop_roi(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green_mask, red_mask = self._make_masks(hsv)
        result = self._detect_from_masks(green_mask, red_mask)
        return result, roi, green_mask, red_mask
