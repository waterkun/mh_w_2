"""AttackTracker — 状态确认 + RL 奖励信号."""

import time

from config import CONFIRM_FRAMES, ATTACK_CLASSES
from attack_detector import AttackDetector


class AttackTracker:
    """多帧状态确认 + RL 信号生成.

    模式与 Gauge/gauge_tracker.py 一致:
    - 需要连续 CONFIRM_FRAMES 帧同类别才确认状态切换
    - 生成攻击相关的 RL 奖励信号
    """

    def __init__(self, detector: AttackDetector):
        self.detector = detector
        self.reset()

    def reset(self):
        """重置 tracker 状态."""
        self.current_attack = "idle"
        self.prev_attack = "idle"
        self.confidence = 0.0
        self.attack_just_started = False
        self.attack_just_ended = False
        self.time_in_attack = 0.0

        self._attack_start_time = None
        self._last_reward = 0.0

        # 状态确认缓冲
        self._pending_state = None
        self._pending_count = 0

    def update(self, frame) -> dict:
        """处理一帧，返回攻击追踪状态.

        Args:
            frame: BGR 全屏图像

        Returns:
            dict: 攻击状态信息
        """
        det = self.detector.detect(frame)

        if not det["ready"]:
            return self._build_result(det)

        new_attack = det["attack"]
        self.confidence = det["confidence"]

        # 状态确认: 需要 CONFIRM_FRAMES 连续帧
        if new_attack != self.current_attack:
            if new_attack == self._pending_state:
                self._pending_count += 1
            else:
                self._pending_state = new_attack
                self._pending_count = 1

            if self._pending_count >= CONFIRM_FRAMES:
                self.prev_attack = self.current_attack
                self.current_attack = new_attack
                self._pending_state = None
                self._pending_count = 0
        else:
            self._pending_state = None
            self._pending_count = 0

        # 追踪攻击转换
        now = time.time()
        was_attacking = self.prev_attack != "idle"
        is_attacking = self.current_attack != "idle"

        self.attack_just_started = (
            is_attacking and self.prev_attack != self.current_attack
            and self.current_attack != self.prev_attack
        )
        self.attack_just_ended = (
            not is_attacking
            and self.prev_attack != "idle"
            and self.current_attack != self.prev_attack
        )

        # 计时
        if self.attack_just_started:
            self._attack_start_time = now
        if is_attacking and self._attack_start_time is not None:
            self.time_in_attack = round(now - self._attack_start_time, 3)
        if self.attack_just_ended:
            self._attack_start_time = None
            self.time_in_attack = 0.0

        self._compute_reward()

        return self._build_result(det)

    def _build_result(self, det):
        """构建返回结果 dict."""
        return {
            "ready": det["ready"],
            "current_attack": self.current_attack,
            "prev_attack": self.prev_attack,
            "confidence": self.confidence,
            "attack_just_started": self.attack_just_started,
            "attack_just_ended": self.attack_just_ended,
            "time_in_attack": self.time_in_attack,
            "probs": det.get("probs", {}),
        }

    def _compute_reward(self):
        """计算 RL 奖励信号.

        攻击开始 → 正信号 (提前预测到攻击有价值)
        攻击进行中 → 小正信号
        攻击结束 → 0
        """
        reward = 0.0
        if self.attack_just_started:
            reward = +1.0  # 成功预测到攻击
        elif self.current_attack != "idle":
            reward = +0.01  # 持续追踪
        self._last_reward = round(reward, 4)

    def get_reward_signal(self) -> float:
        """返回最新奖励信号."""
        return self._last_reward
