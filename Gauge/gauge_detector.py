"""Gauge detector — single-frame, stateless spirit gauge detection via HSV."""

import json
import os
import cv2
import numpy as np


class GaugeDetector:
    """Detect spirit gauge state from a single frame using HSV color masking.

    Uses span-based detection (same approach as health_bar_detector) to measure
    how far the colored fill extends across the gauge bar.

    The gauge always has a golden decorative frame (~15-17% yellow pixels).
    We use per-color thresholds to distinguish actual fill from frame.
    """

    # Default ROI — will be overridden by roi_config.json or constructor arg
    DEFAULT_ROI = (100, 130, 400, 30)

    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "roi_config.json")

    # --- HSV ranges ---
    # Red (spirit gauge red glow): wraps around H=0/180
    # Stricter S/V to avoid ambient red glow in background
    RED_LOWER_1 = np.array([0, 100, 120])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([160, 100, 120])
    RED_UPPER_2 = np.array([180, 255, 255])

    # Yellow / gold fill (same hue as frame but distinguished by area threshold)
    YELLOW_LOWER = np.array([15, 60, 80])
    YELLOW_UPPER = np.array([35, 255, 255])

    # White (low saturation, high value)
    WHITE_LOWER = np.array([0, 0, 180])
    WHITE_UPPER = np.array([180, 50, 255])

    # Minimum pixel ratio per column to count as having color
    COL_FILL_THRESHOLD = 0.10

    # Brightness-drop method for red fill percentage:
    # Window size (columns) for computing average brightness on each side
    BRIGHT_DROP_WINDOW = 7
    # Minimum brightness drop to consider a real fill boundary
    BRIGHT_DROP_THRESHOLD = 40

    # Margin to trim decorative borders (hilt on left, cap on right)
    LEFT_MARGIN_FRAC = 0.15
    RIGHT_MARGIN_FRAC = 0.02

    # Vertical margin to trim top/bottom gauge border
    TOP_MARGIN_FRAC = 0.20
    BOTTOM_MARGIN_FRAC = 0.20

    # Per-color area thresholds (fraction of bar-region pixels)
    # Red is distinctive (never in frame) → low threshold
    # Yellow needs high threshold to exceed the golden frame background (~15-17%)
    # White is rare → moderate threshold
    MIN_RED_AREA_FRAC = 0.02
    MIN_YELLOW_AREA_FRAC = 0.55
    MIN_WHITE_AREA_FRAC = 0.03

    def __init__(self, roi=None):
        if roi is not None:
            self.roi = roi
        else:
            self.roi = self._load_roi()

    def _load_roi(self):
        """Load ROI from config file, fall back to default."""
        if os.path.exists(self.CONFIG_PATH):
            with open(self.CONFIG_PATH) as f:
                cfg = json.load(f)
            return (cfg["x"], cfg["y"], cfg["w"], cfg["h"])
        return self.DEFAULT_ROI

    def _crop_roi(self, frame):
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    def _crop_bar_region(self, mask):
        """Crop mask to the inner bar region (trim hilt, edges, top/bottom)."""
        h, w = mask.shape[:2]
        x0 = int(w * self.LEFT_MARGIN_FRAC)
        x1 = int(w * (1.0 - self.RIGHT_MARGIN_FRAC))
        y0 = int(h * self.TOP_MARGIN_FRAC)
        y1 = int(h * (1.0 - self.BOTTOM_MARGIN_FRAC))
        return mask[y0:y1, x0:x1]

    def _make_masks(self, hsv):
        """Create color masks for red, yellow, and white."""
        red_mask_1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
        red_mask_2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        yellow_mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
        white_mask = cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER)

        return red_mask, yellow_mask, white_mask

    def _column_has_color(self, mask):
        """Return boolean array: True if column has enough colored pixels."""
        h = mask.shape[0]
        if h == 0:
            return np.array([], dtype=bool)
        ratio = np.count_nonzero(mask, axis=0) / h
        return ratio >= self.COL_FILL_THRESHOLD

    def _color_area_frac(self, mask):
        """Fraction of pixels in the mask that are set."""
        total = mask.size
        if total == 0:
            return 0.0
        return np.count_nonzero(mask) / total

    def _find_fill_boundary(self, roi_bgr):
        """Find the fill boundary column using brightness-drop detection.

        Operates on the full ROI (only trims top/bottom borders, NOT
        left/right) so the boundary is detectable even at very low fill.

        Returns:
            Fill ratio [0, 1].  1.0 if no boundary found (gauge is full).
        """
        roi_h, roi_w = roi_bgr.shape[:2]
        if roi_w <= 0 or roi_h <= 0:
            return 0.0

        # Trim only top/bottom decorative borders
        y0 = int(roi_h * self.TOP_MARGIN_FRAC)
        y1 = int(roi_h * (1.0 - self.BOTTOM_MARGIN_FRAC))
        strip = roi_bgr[y0:y1, :]

        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        col_v = np.mean(hsv[:, :, 2].astype(float), axis=0)

        # Smooth brightness
        col_v = np.convolve(col_v, np.ones(5) / 5, mode='same')

        # Compute brightness drop at each column:
        # avg brightness in [c-win, c) minus avg brightness in (c, c+win]
        win = self.BRIGHT_DROP_WINDOW
        drop = np.zeros(roi_w)
        for c in range(win, roi_w - win):
            left_avg = np.mean(col_v[c - win:c])
            right_avg = np.mean(col_v[c + 1:c + 1 + win])
            drop[c] = left_avg - right_avg

        # Only skip a small right-edge margin (decorative tip)
        right_margin = max(int(roi_w * 0.03), 1)
        drop[roi_w - right_margin:] = 0

        peak_col = int(np.argmax(drop))
        peak_val = drop[peak_col]

        if peak_val >= self.BRIGHT_DROP_THRESHOLD:
            # Compensate: the drop peak is shifted right of the actual
            # boundary by roughly half the window size
            boundary = max(peak_col - win // 2, 0)
            return round(min(boundary / roi_w, 1.0), 4)
        return 1.0  # no boundary → gauge is full

    def _detect_from_masks(self, red_mask, yellow_mask, white_mask,
                           roi_bgr=None):
        """Core detection logic from pre-computed masks."""
        # Crop to inner bar region (trim hilt, edges, top/bottom border)
        bar_red = self._crop_bar_region(red_mask)
        bar_yellow = self._crop_bar_region(yellow_mask)
        bar_white = self._crop_bar_region(white_mask)

        bar_h, bar_w = bar_red.shape[:2]
        if bar_w <= 0 or bar_h <= 0:
            return {"is_red": False, "red_pct": 0.0, "gauge_state": "empty"}

        red_frac = self._color_area_frac(bar_red)
        yellow_frac = self._color_area_frac(bar_yellow)
        white_frac = self._color_area_frac(bar_white)

        # Determine gauge state with per-color thresholds
        if red_frac >= self.MIN_RED_AREA_FRAC:
            gauge_state = "red"
        elif yellow_frac >= self.MIN_YELLOW_AREA_FRAC:
            gauge_state = "yellow"
        elif white_frac >= self.MIN_WHITE_AREA_FRAC:
            gauge_state = "white"
        else:
            gauge_state = "empty"

        is_red = gauge_state == "red"

        # Estimate red fill percentage using brightness-drop boundary detection.
        # Finds the bright vertical line where fill ends — immune to red flash.
        # Uses the full ROI (not bar region) so low-fill boundaries aren't
        # trimmed away by LEFT_MARGIN_FRAC.
        red_pct = 0.0
        if is_red and roi_bgr is not None:
            red_pct = self._find_fill_boundary(roi_bgr)

        return {
            "is_red": is_red,
            "red_pct": red_pct,
            "gauge_state": gauge_state,
        }

    def detect(self, frame) -> dict:
        """Detect gauge state from a single frame (BGR image).

        Returns:
            dict with keys:
                is_red      — True if red blade active
                red_pct     — red fill ratio [0, 1], 0 if not red
                gauge_state — "red" / "yellow" / "white" / "empty"
        """
        roi = self._crop_roi(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask, yellow_mask, white_mask = self._make_masks(hsv)
        return self._detect_from_masks(red_mask, yellow_mask, white_mask,
                                       roi_bgr=roi)

    def detect_debug(self, frame) -> tuple:
        """Like detect(), but also returns debug images.

        Returns:
            (result_dict, roi_bgr, red_mask, gauge_mask)
            where gauge_mask = red | yellow | white combined
        """
        roi = self._crop_roi(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask, yellow_mask, white_mask = self._make_masks(hsv)
        result = self._detect_from_masks(red_mask, yellow_mask, white_mask,
                                         roi_bgr=roi)

        # Combined mask for visualization
        gauge_mask = cv2.bitwise_or(red_mask, cv2.bitwise_or(yellow_mask, white_mask))

        return result, roi, red_mask, gauge_mask
