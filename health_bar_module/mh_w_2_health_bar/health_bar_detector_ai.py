"""AI-based health bar detector using CNN regression.

Drop-in replacement for HealthBarDetector — same interface.
"""

import collections

import cv2
import numpy as np
import torch

try:
    from health_bar_model import HealthBarNet
except ImportError:
    from health_bar_module.mh_w_2_health_bar.health_bar_model import HealthBarNet


class HealthBarDetectorAI:
    DEFAULT_ROI = (146, 71, 728, 31)
    INPUT_H, INPUT_W = 32, 256
    SMOOTH_WINDOW = 7

    def __init__(self, model_path, roi=None):
        self.roi = roi or self.DEFAULT_ROI
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HealthBarNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self._hp_history = collections.deque(maxlen=self.SMOOTH_WINDOW)
        self._dmg_history = collections.deque(maxlen=self.SMOOTH_WINDOW)
        self._prev_damage_pct = 0.0

    def _crop_roi(self, frame):
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    def _preprocess(self, roi_bgr):
        """Resize and convert to tensor."""
        img = cv2.resize(roi_bgr, (self.INPUT_W, self.INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _infer(self, tensor):
        """Run model inference, returns (health_pct, damage_pct)."""
        output = self.model(tensor).cpu().numpy()[0]
        return float(output[0]), float(output[1])

    def _smooth(self, health_pct, damage_pct):
        """Temporal smoothing with median filter."""
        self._hp_history.append(health_pct)
        self._dmg_history.append(damage_pct)

        health_pct = float(np.median(list(self._hp_history)))
        damage_pct = float(np.median(list(self._dmg_history)))

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
            dict with keys: health_pct, damage_pct, is_hit
        """
        roi = self._crop_roi(frame)
        tensor = self._preprocess(roi)
        health_pct, damage_pct = self._infer(tensor)
        return self._smooth(health_pct, damage_pct)

    def detect_debug(self, frame) -> tuple:
        """Like detect(), but also returns debug visuals.

        Returns:
            (result_dict, roi_bgr, health_vis, damage_vis)
        """
        roi = self._crop_roi(frame)
        tensor = self._preprocess(roi)
        health_pct, damage_pct = self._infer(tensor)
        result = self._smooth(health_pct, damage_pct)

        h, w = roi.shape[:2]

        # Health visualization: green bar
        health_vis = np.zeros((h, w, 3), dtype=np.uint8)
        fill_w = int(w * result["health_pct"])
        if fill_w > 0:
            health_vis[:, :fill_w] = (0, 200, 0)

        # Damage visualization: red bar after health
        damage_vis = np.zeros((h, w, 3), dtype=np.uint8)
        dmg_start = fill_w
        dmg_end = min(w, dmg_start + int(w * result["damage_pct"]))
        if dmg_end > dmg_start:
            damage_vis[:, dmg_start:dmg_end] = (0, 0, 200)

        return result, roi, health_vis, damage_vis
