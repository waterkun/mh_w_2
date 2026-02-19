import time
from health_bar_detector import HealthBarDetector


class HealthBarTracker:
    """Track health bar state over time for RL agent consumption."""

    # Thresholds
    ALIVE_THRESHOLD = 0.01       # below this = dead
    HIT_DELTA_THRESHOLD = -0.01  # health drop bigger than this = hit event

    # Reward weights
    REWARD_HIT_PENALTY = -1.0
    REWARD_SURVIVAL_BONUS = 0.01
    REWARD_DEATH_PENALTY = -5.0

    def __init__(self, detector=None):
        self.detector = detector or HealthBarDetector()
        self.reset()

    def reset(self):
        """Reset tracker state for a new episode."""
        self.health_pct = 1.0
        self.prev_health_pct = 1.0
        self.damage_pct = 0.0
        self.is_hit = False
        self.health_delta = 0.0
        self.hit_count = 0
        self.last_hit_time = None
        self.is_alive = True
        self._last_update_time = time.time()
        self._last_reward = 0.0

    def update(self, frame) -> dict:
        """Process a new frame and return the full RL-ready state dict.

        Args:
            frame: BGR image (full screen or pre-cropped to ROI).

        Returns:
            dict with all state fields for the RL agent.
        """
        now = time.time()
        det = self.detector.detect(frame)

        self.prev_health_pct = self.health_pct
        self.health_pct = det["health_pct"]
        self.damage_pct = det["damage_pct"]
        self.health_delta = round(self.health_pct - self.prev_health_pct, 4)

        # Detect hit: either red segment visible or significant health drop
        self.is_hit = det["is_hit"] or self.health_delta < self.HIT_DELTA_THRESHOLD

        if self.is_hit:
            self.hit_count += 1
            self.last_hit_time = now

        self.is_alive = self.health_pct >= self.ALIVE_THRESHOLD

        time_since_last_hit = 0.0
        if self.last_hit_time is not None:
            time_since_last_hit = round(now - self.last_hit_time, 3)

        self._compute_reward()
        self._last_update_time = now

        return {
            "health_pct": self.health_pct,
            "prev_health_pct": self.prev_health_pct,
            "damage_pct": self.damage_pct,
            "is_hit": self.is_hit,
            "health_delta": self.health_delta,
            "hit_count": self.hit_count,
            "time_since_last_hit": time_since_last_hit,
            "is_alive": self.is_alive,
        }

    def _compute_reward(self):
        """Compute RL reward signal based on current state."""
        reward = 0.0

        if not self.is_alive:
            reward += self.REWARD_DEATH_PENALTY
        elif self.is_hit:
            # Penalty proportional to damage taken
            reward += self.REWARD_HIT_PENALTY * max(-self.health_delta, 0.01)
        else:
            # Small survival bonus each frame
            reward += self.REWARD_SURVIVAL_BONUS

        self._last_reward = round(reward, 4)

    def get_reward_signal(self) -> float:
        """Return the latest computed reward signal.

        Positive = good (surviving), negative = bad (taking damage / dying).
        """
        return self._last_reward
