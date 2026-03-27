"""Sharpness tracker — multi-frame state tracking with sharpen alert."""

import time
from sharpness_detector import SharpnessDetector


class SharpnessTracker:
    """Track sharpness over time, detect drops, and signal when to sharpen."""

    # Consecutive frames needed to confirm a color change
    CONFIRM_FRAMES = 3

    # Sharpness percentage threshold — below this, warn even if color is ok
    LOW_PCT_THRESHOLD = 0.10

    TIER_ORDER = ["purple", "white", "blue", "green", "yellow", "orange", "red"]
    WARN_TIERS = {"blue"}                              # soft — sharpen when possible
    LOW_TIERS = {"green", "yellow", "orange", "red"}   # hard — must sharpen now

    def __init__(self, detector=None):
        self.detector = detector or SharpnessDetector()
        self.reset()

    # Flashing detection: if we see unknown/color alternating this many times
    # within FLASHING_WINDOW_SEC, consider the bar as flashing (nearly depleted)
    FLASHING_COUNT_THRESHOLD = 4
    FLASHING_WINDOW_SEC = 2.0

    def reset(self):
        """Reset tracker state."""
        self.sharpness_color = "unknown"
        self.prev_color = "unknown"
        self.sharpness_pct = 0.0
        self.need_sharpen = False

        self.color_just_dropped = False
        self.sharpen_alert = False
        self.is_flashing = False
        self.time_since_last_sharpen = 0.0

        self._last_sharpen_time = time.time()
        self._last_update_time = time.time()

        # State confirmation buffer
        self._pending_color = None
        self._pending_count = 0

        # Flashing detection: track recent unknown detections
        self._unknown_timestamps = []

    def _tier_index(self, color):
        """Get tier index (0=best, 6=worst). Returns 99 for unknown."""
        try:
            return self.TIER_ORDER.index(color)
        except ValueError:
            return 99

    def update(self, frame) -> dict:
        """Process a new frame and return the full state dict.

        Args:
            frame: BGR image (full screen).

        Returns:
            dict with sharpness state for the RL agent.
        """
        now = time.time()
        det = self.detector.detect(frame)

        new_color = det["sharpness_color"]
        self.sharpness_pct = det["sharpness_pct"]

        # Flashing detection: track "unknown" frames within a time window.
        # When the bar flashes (nearly depleted), detector alternates between
        # the real color and "unknown" rapidly.
        if new_color == "unknown" and self.sharpness_color != "unknown":
            self._unknown_timestamps.append(now)
        # Prune old timestamps outside the window
        cutoff = now - self.FLASHING_WINDOW_SEC
        self._unknown_timestamps = [t for t in self._unknown_timestamps if t > cutoff]
        self.is_flashing = len(self._unknown_timestamps) >= self.FLASHING_COUNT_THRESHOLD

        # When flashing (unknown detected but we have a known color),
        # keep the current color instead of switching to unknown
        if new_color == "unknown" and self.sharpness_color != "unknown":
            new_color = self.sharpness_color

        # State confirmation
        if new_color != self.sharpness_color:
            if new_color == self._pending_color:
                self._pending_count += 1
            else:
                self._pending_color = new_color
                self._pending_count = 1

            if self._pending_count >= self.CONFIRM_FRAMES:
                self.prev_color = self.sharpness_color
                self.sharpness_color = new_color
                self._pending_color = None
                self._pending_count = 0
        else:
            self._pending_color = None
            self._pending_count = 0

        # Detect color drop (tier got worse)
        self.color_just_dropped = (
            self.sharpness_color != self.prev_color
            and self._tier_index(self.sharpness_color) > self._tier_index(self.prev_color)
        )

        # should_sharpen (soft): blue — sharpen when possible
        # need_sharpen (hard): green/yellow/orange/red — must sharpen now
        self.should_sharpen = det["should_sharpen"] or self.is_flashing
        self.need_sharpen = det["need_sharpen"] or self.is_flashing
        self.sharpen_alert = (
            self.sharpness_color in self.LOW_TIERS
            or self.sharpness_color == "unknown"
            or self.is_flashing
            or (self.sharpness_pct < self.LOW_PCT_THRESHOLD and self.sharpness_color != "unknown")
        )

        # Detect if player sharpened (color jumped to a better tier)
        if (self._tier_index(self.sharpness_color) < self._tier_index(self.prev_color)
                and self.sharpness_color != self.prev_color):
            self._last_sharpen_time = now

        self.time_since_last_sharpen = round(now - self._last_sharpen_time, 3)
        self._last_update_time = now

        return {
            "sharpness_color": self.sharpness_color,
            "prev_color": self.prev_color,
            "sharpness_pct": self.sharpness_pct,
            "should_sharpen": self.should_sharpen,  # soft: blue
            "need_sharpen": self.need_sharpen,       # hard: green/yellow/orange/red
            "sharpen_alert": self.sharpen_alert,
            "is_flashing": self.is_flashing,
            "color_just_dropped": self.color_just_dropped,
            "time_since_last_sharpen": self.time_since_last_sharpen,
        }

    def get_reward_signal(self) -> float:
        """Return RL reward signal based on sharpness state.

        Rewards:
            -1.0  — sharpness dropped to a low tier (should have sharpened)
            -0.5  — color just dropped one tier
            +0.5  — maintaining high sharpness (blue/white/purple)
            -0.01 — each frame in low sharpness (penalty for not sharpening)
        """
        if self.color_just_dropped and self.sharpness_color in self.LOW_TIERS:
            return -1.0
        if self.color_just_dropped:
            return -0.5
        if self.sharpness_color in {"purple", "white", "blue"}:
            return +0.01
        if self.sharpness_color in self.LOW_TIERS:
            return -0.01
        return 0.0
