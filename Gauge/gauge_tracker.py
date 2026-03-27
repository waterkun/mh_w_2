"""Gauge tracker — multi-frame state tracking for spirit gauge."""

import time
from gauge_detector import GaugeDetector


class GaugeTracker:
    """Track spirit gauge state over time for RL agent consumption."""

    # Number of consecutive frames to confirm a state transition
    CONFIRM_FRAMES = 3

    def __init__(self, detector=None):
        self.detector = detector or GaugeDetector()
        self.reset()

    def reset(self):
        """Reset tracker state for a new episode."""
        self.is_red = False
        self.red_pct = 0.0
        self.gauge_state = "empty"
        self.prev_gauge_state = "empty"

        self.red_just_activated = False
        self.red_just_expired = False
        self.time_in_red = 0.0

        self._red_start_time = None
        self._last_update_time = time.time()
        self._last_reward = 0.0

        # State confirmation buffer
        self._pending_state = None
        self._pending_count = 0

    def update(self, frame) -> dict:
        """Process a new frame and return the full state dict.

        Args:
            frame: BGR image (full screen).

        Returns:
            dict with state fields for the RL agent.
        """
        now = time.time()
        det = self.detector.detect(frame)

        new_state = det["gauge_state"]
        self.red_pct = det["red_pct"]

        # State confirmation: require CONFIRM_FRAMES consecutive frames
        # of the same new state before switching
        if new_state != self.gauge_state:
            if new_state == self._pending_state:
                self._pending_count += 1
            else:
                self._pending_state = new_state
                self._pending_count = 1

            if self._pending_count >= self.CONFIRM_FRAMES:
                self.prev_gauge_state = self.gauge_state
                self.gauge_state = new_state
                self._pending_state = None
                self._pending_count = 0
        else:
            self._pending_state = None
            self._pending_count = 0

        # Track red blade transitions
        was_red = self.prev_gauge_state == "red" if self.gauge_state != self.prev_gauge_state else False
        self.is_red = self.gauge_state == "red"

        self.red_just_activated = (
            self.is_red and self.prev_gauge_state != "red"
            and self.gauge_state != self.prev_gauge_state
        )
        self.red_just_expired = (
            not self.is_red and was_red
        )

        # Track time in red
        if self.red_just_activated:
            self._red_start_time = now
        if self.is_red and self._red_start_time is not None:
            self.time_in_red = round(now - self._red_start_time, 3)
        if self.red_just_expired:
            self._red_start_time = None
            self.time_in_red = 0.0

        self._compute_reward()
        self._last_update_time = now

        # Reset transition flags after first read
        result = {
            "is_red": self.is_red,
            "red_pct": self.red_pct,
            "gauge_state": self.gauge_state,
            "prev_gauge_state": self.prev_gauge_state,
            "red_just_activated": self.red_just_activated,
            "red_just_expired": self.red_just_expired,
            "time_in_red": self.time_in_red,
        }

        return result

    def _compute_reward(self):
        """Compute RL reward signal based on gauge state."""
        reward = 0.0

        if self.red_just_expired:
            # Red blade expired without using it — missed opportunity
            reward = -1.0
        elif self.red_just_activated:
            # Red blade activated — positive signal
            reward = +0.5
        elif self.is_red:
            # Maintaining red blade — small positive
            reward = +0.01

        self._last_reward = round(reward, 4)

    def get_reward_signal(self) -> float:
        """Return the latest computed reward signal."""
        return self._last_reward
