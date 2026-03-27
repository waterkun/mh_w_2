import collections

import cv2
import numpy as np


class HealthBarDetector:
    """Detect health bar via column-wise Y-axis projection.

    The MH Wilds health bar has three visual elements:
    - Green ECG wave (left) = current health, always green
    - Red bar/wave (right of green) = recently lost health (damage trail)
    - At low health: left wave flashes colors, right red wave becomes sparse

    Strategy (Y-axis projection — squash vertically, read horizontally):
    1. Extract the "bar band" (middle rows of ROI, excluding borders).
    2. GREEN channel max per column → green wave becomes a solid bar.
       Rightmost green column = X_current = current health boundary.
    3. R-G difference max per column → red damage trail.
       Rightmost red-dominant column = X_damage = pre-hit health boundary.
    4. Flash detection for low health (dense flashing wave).
    5. Temporal smoothing (median) for noise reduction.
    """

    # Default ROI: top-left region where health bar lives (x, y, w, h)
    # Calibrated for 3440x1440 ultrawide
    DEFAULT_ROI = (146, 71, 728, 31)

    # Margin (fraction of width) to trim decorative borders
    LEFT_MARGIN_FRAC = 0.06
    RIGHT_MARGIN_FRAC = 0.12   # exclude right decoration/arrow

    # Bar band: vertical slice of ROI that contains the actual bar content
    # (excludes top border line and bottom decorations)
    BAR_BAND_TOP_FRAC = 0.35
    BAR_BAND_BOT_FRAC = 0.80

    # ECG wave detection (GREEN channel only — ignores red damage)
    WAVE_THRESHOLD = 180       # green channel peak of ECG wave
    WAVE_SMOOTH_WINDOW = 25    # columns to smooth over

    # Damage trail detection (R-G dominance)
    DAMAGE_RG_THRESHOLD = 50   # R-G difference to count as "red damage"

    # Flash detection (low health flashing)
    FLASH_VAR_STD_THRESHOLD = 180   # column variance std below this = flash
    FLASH_HEALTH_ESTIMATE = 0.15    # assumed health during flash

    # Temporal smoothing
    SMOOTH_WINDOW = 7

    def __init__(self, roi=None):
        self.roi = roi or self.DEFAULT_ROI
        self._hp_history = collections.deque(maxlen=self.SMOOTH_WINDOW)
        self._dmg_history = collections.deque(maxlen=self.SMOOTH_WINDOW)
        self._prev_damage_pct = 0.0

    def _crop_roi(self, frame):
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    def _extract_bar_band(self, roi_bgr):
        """Extract the middle rows containing the actual bar content."""
        h = roi_bgr.shape[0]
        r1 = int(h * self.BAR_BAND_TOP_FRAC)
        r2 = int(h * self.BAR_BAND_BOT_FRAC)
        if r2 <= r1:
            r1, r2 = 0, h
        return roi_bgr[r1:r2]

    def _is_flashing(self, bar_band, bar_start, bar_end):
        """Detect low-health flash: whole bar uniformly colored, no wave."""
        gray = cv2.cvtColor(bar_band, cv2.COLOR_BGR2GRAY).astype(float)
        col_var = np.var(gray, axis=0)
        var_std = np.std(col_var[bar_start:bar_end])
        return var_std < self.FLASH_VAR_STD_THRESHOLD

    def _column_projections(self, bar_band):
        """Compute per-column projections for green wave and red damage.

        Returns:
            (smoothed_green, smoothed_rg)
            smoothed_green: per-column max green intensity (smoothed)
            smoothed_rg: per-column max R-G difference (smoothed)
        """
        green = bar_band[:, :, 1].astype(float)
        red = bar_band[:, :, 2].astype(float)

        col_max_g = np.max(green, axis=0)
        col_max_rg = np.max(red - green, axis=0)

        kernel = np.ones(self.WAVE_SMOOTH_WINDOW) / self.WAVE_SMOOTH_WINDOW
        smoothed_g = np.convolve(col_max_g, kernel, mode="same")
        smoothed_rg = np.convolve(col_max_rg, kernel, mode="same")

        return smoothed_g, smoothed_rg

    def _find_boundary(self, profile, bar_start, bar_end, threshold):
        """Find rightmost column where profile >= threshold."""
        for c in range(bar_end, bar_start, -1):
            if profile[c] >= threshold:
                return c, True
        return bar_start, False

    def _detect_core(self, roi_bgr):
        """Core detection logic."""
        h, total_w = roi_bgr.shape[:2]
        if total_w == 0 or h == 0:
            return {"health_pct": 0.0, "damage_pct": 0.0,
                    "_valid": False, "_flash": False}

        bar_start = int(total_w * self.LEFT_MARGIN_FRAC)
        bar_end = int(total_w * (1.0 - self.RIGHT_MARGIN_FRAC))
        bar_width = bar_end - bar_start
        if bar_width <= 0:
            return {"health_pct": 0.0, "damage_pct": 0.0,
                    "_valid": False, "_flash": False}

        bar_band = self._extract_bar_band(roi_bgr)
        smoothed_g, smoothed_rg = self._column_projections(bar_band)

        # Step 1: find green wave boundary (current health)
        x_health, green_found = self._find_boundary(
            smoothed_g, bar_start, bar_end, self.WAVE_THRESHOLD)

        if green_found:
            if x_health >= bar_end - 5:
                health_pct = 1.0
            else:
                health_pct = (x_health - bar_start) / bar_width
                health_pct = max(0.0, min(1.0, health_pct))

            # Step 2: find red damage trail (R-G dominance after health boundary)
            x_damage, red_found = self._find_boundary(
                smoothed_rg, bar_start, bar_end, self.DAMAGE_RG_THRESHOLD)

            if red_found and x_damage > x_health + 10:
                damage_pct = (x_damage - x_health) / bar_width
                damage_pct = max(0.0, min(1.0 - health_pct, damage_pct))
            else:
                damage_pct = 0.0

            return {
                "health_pct": round(health_pct, 4),
                "damage_pct": round(damage_pct, 4),
                "_valid": True,
                "_flash": False,
                "_x_health": x_health,
                "_x_damage": x_damage if red_found else x_health,
            }

        # Step 3: no green wave — check flash (low health)
        flash = self._is_flashing(bar_band, bar_start, bar_end)
        if flash:
            # At low health, the red wave shows total lost health
            x_damage, red_found = self._find_boundary(
                smoothed_rg, bar_start, bar_end, self.DAMAGE_RG_THRESHOLD)
            damage_pct = (x_damage - bar_start) / bar_width if red_found else 0.0

            return {
                "health_pct": self.FLASH_HEALTH_ESTIMATE,
                "damage_pct": round(min(damage_pct, 0.85), 4),
                "_valid": True,
                "_flash": True,
                "_x_health": bar_start,
                "_x_damage": x_damage if red_found else bar_start,
            }

        # Step 4: game effects blocking view — invalid frame
        return {"health_pct": 0.0, "damage_pct": 0.0,
                "_valid": False, "_flash": False,
                "_x_health": bar_start, "_x_damage": bar_start}

    def _smooth(self, raw_result):
        """Temporal smoothing with median filter."""
        health_pct = raw_result["health_pct"]
        damage_pct = raw_result["damage_pct"]
        valid = raw_result["_valid"]

        if valid:
            self._hp_history.append(health_pct)
            self._dmg_history.append(damage_pct)

        if self._hp_history:
            health_pct = float(np.median(list(self._hp_history)))
        if self._dmg_history:
            damage_pct = float(np.median(list(self._dmg_history)))

        # Detect hit: damage_pct increased significantly
        is_hit = damage_pct > self._prev_damage_pct + 0.03
        self._prev_damage_pct = damage_pct

        return {
            "health_pct": round(health_pct, 4),
            "damage_pct": round(damage_pct, 4),
            "is_hit": is_hit,
        }

    def detect(self, frame) -> dict:
        """Detect health bar state from a single frame (BGR image).

        Returns:
            dict with keys:
                health_pct: current health (0.0 - 1.0)
                damage_pct: visible damage trail (0.0 - 1.0)
                is_hit: True if damage increased this frame
        """
        roi = self._crop_roi(frame)
        raw = self._detect_core(roi)
        return self._smooth(raw)

    def detect_debug(self, frame) -> tuple:
        """Like detect(), but also returns debug images.

        Returns:
            (result_dict, roi_bgr, wave_vis, rg_vis)
        """
        roi = self._crop_roi(frame)
        raw = self._detect_core(roi)
        result = self._smooth(raw)

        total_w = roi.shape[1]
        rh = roi.shape[0]
        bar_band = self._extract_bar_band(roi)
        bar_start = int(total_w * self.LEFT_MARGIN_FRAC)
        bar_end = int(total_w * (1.0 - self.RIGHT_MARGIN_FRAC))
        x_health = raw.get("_x_health", bar_start)
        x_damage = raw.get("_x_damage", bar_start)
        flash = raw.get("_flash", False)
        valid = raw.get("_valid", False)

        smoothed_g, smoothed_rg = self._column_projections(bar_band)

        # Wave visualization: green projection + red-dominance
        wave_vis = np.zeros((rh, total_w, 3), dtype=np.uint8)
        if total_w > 0 and rh > 0:
            g_max = max(smoothed_g.max(), 1)
            rg_max = max(smoothed_rg.max(), 1)
            for x in range(total_w):
                # Green wave bar (bottom-up)
                g_h = int(smoothed_g[x] / g_max * rh * 0.5)
                if g_h > 0:
                    color = (0, 255, 0) if smoothed_g[x] >= self.WAVE_THRESHOLD \
                        else (50, 80, 50)
                    wave_vis[rh - g_h:, x] = color

                # Red damage bar (top-down, in upper half)
                r_h = int(max(0, smoothed_rg[x]) / rg_max * rh * 0.5)
                if r_h > 0:
                    color = (0, 0, 255) if smoothed_rg[x] >= self.DAMAGE_RG_THRESHOLD \
                        else (50, 50, 80)
                    wave_vis[:r_h, x] = color

        # R-G difference visualization
        rg_norm = np.clip(smoothed_rg / max(smoothed_rg.max(), 1) * 255,
                          0, 255).astype(np.uint8)
        rg_vis = np.zeros((rh, total_w, 3), dtype=np.uint8)
        rg_row = np.tile(rg_norm, (rh, 1))
        rg_vis[:, :, 2] = rg_row  # show in red channel

        # Mark health boundary (cyan)
        if valid and 0 <= x_health < total_w:
            cv2.line(roi, (x_health, 0), (x_health, rh), (0, 255, 255), 2)
            cv2.line(wave_vis, (x_health, 0), (x_health, rh), (0, 255, 255), 2)
            cv2.line(rg_vis, (x_health, 0), (x_health, rh), (0, 255, 255), 2)

        # Mark damage boundary (magenta)
        if valid and x_damage > x_health + 10 and x_damage < total_w:
            cv2.line(roi, (x_damage, 0), (x_damage, rh), (255, 0, 255), 2)
            cv2.line(wave_vis, (x_damage, 0), (x_damage, rh), (255, 0, 255), 2)
            cv2.line(rg_vis, (x_damage, 0), (x_damage, rh), (255, 0, 255), 2)

        # Status indicator
        if flash:
            cv2.putText(wave_vis, "FLASH", (5, rh - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        elif not valid:
            cv2.putText(wave_vis, "N/A", (5, rh - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 255), 1)

        return result, roi, wave_vis, rg_vis
