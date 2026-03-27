"""简单战斗模块 — 用 YOLO bbox 大小判断距离，靠近并攻击怪物.

使用方式:
  python -m farming_bot.attack_monster.attack_monster

三个状态:
  SEARCH   — 未检测到怪物 → 缓慢转向/等待
  APPROACH — bbox 面积小 → 推摇杆朝怪物走
  ATTACK   — bbox 面积大 → 按 Y 攻击, 偶尔 B 翻滚
"""

import os
import random
import time

import cv2
import mss
import numpy as np
import vgamepad as vg
import winsound
from ultralytics import YOLO

from health_bar_module.mh_w_2_health_bar.health_bar_detector_ai import HealthBarDetectorAI

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "monster_detection", "monster_detect",
    "runs", "monster_detect", "weights", "best.pt",
)
_HEALTH_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "health_bar_module", "runs", "best.pt",
)

# ── 检测参数 ──
MONSTER_CONFIDENCE = 0.6
FPS = 5

# ── 战斗参数 ──
CLOSE_ENOUGH_THRESHOLD = 0.08   # bbox 面积占屏幕 8% 以上 = 够近，可攻击
ATTACK_TIMEOUT = 180.0          # 战斗最大时长 (秒)
LOST_FRAMES = 30                # 连续 30 帧 (~6s @5fps) 检测不到 = 怪物丢失

# ── 吃药参数 ──
HEAL_THRESHOLD = 0.35           # 血量低于 35% 时立即吃药
HEAL_SAFE_THRESHOLD = 0.35      # 血量回到 35% 以上才恢复战斗
HEAL_COOLDOWN = 2.0             # 吃药冷却 (秒)

# ── 可视化 ──
_COLORS = {"body": (0, 255, 0), "head": (0, 0, 255)}
_STATE_COLORS = {
    "SEARCH": (0, 255, 255),    # 黄
    "APPROACH": (255, 165, 0),  # 橙
    "ATTACK": (0, 0, 255),      # 红
    "HEAL": (0, 255, 0),        # 绿
}

_BUTTON_MAP = {
    "X": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
    "Y": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
    "B": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    "LEFT_SHOULDER": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
}

LB_INTERVAL = 0.7  # 战斗中每 0.7 秒按 LB 修正视角


def beep(freq=1000, duration_ms=200):
    try:
        winsound.Beep(freq, duration_ms)
    except Exception:
        pass


class AttackMonster:
    """简单战斗: 靠近怪物 + Y/B 攻击."""

    def __init__(self, pad, sct, monitor, display_thread=None):
        """初始化.

        Args:
            pad: VX360Gamepad 实例.
            sct: mss 截屏实例.
            monitor: 截屏区域.
            display_thread: _DisplayThread 实例 (可选, 用于显示 YOLO 帧).
        """
        self._model = YOLO(_MODEL_PATH)
        self._class_names = self._model.names
        self._pad = pad
        self._sct = sct
        self._monitor = monitor
        self._display = display_thread
        self._frame_interval = 1.0 / FPS
        self._screen_area = monitor["width"] * monitor["height"]
        self._health_detector = HealthBarDetectorAI(_HEALTH_MODEL_PATH)
        self._last_heal_time = -999.0
        self._last_lb_time = -999.0

        print("AttackMonster 初始化完成")
        print(f"  模型: {_MODEL_PATH}")
        print(f"  靠近阈值: {CLOSE_ENOUGH_THRESHOLD:.1%} 屏幕面积")
        print(f"  吃药阈值: {HEAL_THRESHOLD:.0%} 血量")

    def _grab_frame(self):
        screenshot = self._sct.grab(self._monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def _press_button(self, name, duration=0.15):
        btn = _BUTTON_MAP.get(name)
        if btn is None:
            return
        self._pad.press_button(button=btn)
        self._pad.update()
        time.sleep(duration)
        self._pad.release_button(button=btn)
        self._pad.update()

    def _release_all(self):
        self._pad.reset()
        self._pad.update()

    def _set_left_stick(self, x, y):
        """设置左摇杆方向. x/y 范围 [-1, 1]."""
        self._pad.left_joystick_float(x_value_float=x, y_value_float=y)
        self._pad.update()

    def _detect(self, frame):
        """YOLO 推理, 返回检测结果.

        Returns:
            (boxes_info, display_frame)
            boxes_info: list of (x1, y1, x2, y2, conf, cls_name) 仅 conf >= 阈值
        """
        results = self._model(frame, verbose=False)
        boxes = results[0].boxes
        display = frame.copy()
        boxes_info = []

        for box in boxes:
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            cls_name = self._class_names.get(cls_id, f"cls{cls_id}")
            color = _COLORS.get(cls_name, (255, 255, 0))

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # 画框
            thickness = 3 if conf >= MONSTER_CONFIDENCE else 1
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            label = f"{cls_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(display, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(display, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

            if conf >= MONSTER_CONFIDENCE:
                boxes_info.append((x1, y1, x2, y2, conf, cls_name))

        return boxes_info, display

    def _largest_box(self, boxes_info):
        """返回面积最大的 bbox 及其面积比例."""
        best = None
        best_area = 0
        for x1, y1, x2, y2, conf, cls_name in boxes_info:
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2, conf, cls_name)
        area_ratio = best_area / self._screen_area if self._screen_area > 0 else 0
        return best, area_ratio

    def _approach(self, bbox):
        """推左摇杆朝 bbox 方向走."""
        x1, y1, x2, y2 = bbox[:4]
        cx = (x1 + x2) / 2
        screen_cx = self._monitor["width"] / 2

        # bbox 中心偏移 → 摇杆 x 方向 (归一化到 [-1, 1])
        offset_x = (cx - screen_cx) / screen_cx
        offset_x = max(-1.0, min(1.0, offset_x))

        # y 始终推前 (朝怪物走)
        self._set_left_stick(offset_x, 1.0)

    def _check_health(self, frame):
        """检测当前血量百分比."""
        result = self._health_detector.detect(frame)
        return result["health_pct"]

    def _try_heal(self):
        """低血量时按 X 吃药. 返回 True 如果触发了吃药."""
        now = time.time()
        if now - self._last_heal_time < HEAL_COOLDOWN:
            return False
        self._release_all()
        self._press_button("X", duration=0.15)
        self._last_heal_time = now
        print(f"  [战斗] 吃药! (X)")
        return True

    def _do_attack(self):
        """简单攻击: 70% Y, 20% B, 10% 停顿."""
        roll = random.random()
        if roll < 0.70:
            self._press_button("Y", duration=0.1)
        elif roll < 0.90:
            self._press_button("B", duration=0.1)
        else:
            time.sleep(0.2)

    def _search_turn(self):
        """怪物不在视野 → 缓慢转向."""
        # 随机选方向, 推右摇杆转视角
        direction = random.choice([-0.5, 0.5])
        self._set_left_stick(direction, 0.3)

    def attack(self, timeout=ATTACK_TIMEOUT):
        """战斗主循环.

        Returns:
            "lost"    — 怪物丢失 (连续多帧检测不到)
            "timeout" — 超时
        """
        print(f"\n[战斗] 开始 (超时 {timeout}s)")
        start_time = time.time()
        search_count = 0
        state = "SEARCH"

        try:
            while True:
                elapsed = time.time() - start_time

                if elapsed > timeout:
                    print(f"  [战斗] 超时 ({timeout}s)")
                    self._release_all()
                    return "timeout"

                t0 = time.perf_counter()
                frame = self._grab_frame()

                # 优先检测血量，低血时持续吃药直到安全
                health_pct = self._check_health(frame)
                healed = False
                if health_pct < HEAL_THRESHOLD:
                    healed = self._try_heal()

                boxes_info, display = self._detect(frame)

                if healed or health_pct < HEAL_SAFE_THRESHOLD:
                    state = "HEAL"
                    self._release_all()
                elif not boxes_info:
                    # SEARCH — 先按 LB 锁定/面向怪物
                    search_count += 1
                    state = "SEARCH"
                    now = time.time()
                    if now - self._last_lb_time >= LB_INTERVAL:
                        self._press_button("LEFT_SHOULDER", duration=0.05)
                        self._last_lb_time = now
                    else:
                        self._search_turn()

                    if search_count >= LOST_FRAMES:
                        print(f"  [战斗] 怪物丢失 (连续 {search_count} 帧无检测)")
                        self._release_all()
                        return "lost"
                else:
                    search_count = 0
                    best_box, area_ratio = self._largest_box(boxes_info)

                    if area_ratio < CLOSE_ENOUGH_THRESHOLD:
                        # APPROACH
                        state = "APPROACH"
                        self._approach(best_box)
                    else:
                        # ATTACK
                        state = "ATTACK"
                        self._release_all()
                        now = time.time()
                        if now - self._last_lb_time >= LB_INTERVAL:
                            self._press_button("LEFT_SHOULDER", duration=0.05)
                            self._last_lb_time = now
                        self._do_attack()

                # 更新显示
                status_color = _STATE_COLORS.get(state, (255, 255, 255))
                if boxes_info:
                    _, area_ratio = self._largest_box(boxes_info)
                    status = (f"[{elapsed:.0f}s] {state} "
                              f"area={area_ratio:.2%} HP={health_pct:.0%}")
                else:
                    status = (f"[{elapsed:.0f}s] {state} "
                              f"lost={search_count}/{LOST_FRAMES} HP={health_pct:.0%}")
                cv2.putText(display, status, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, status_color, 3)

                if self._display:
                    self._display.set_yolo_frame(display)

                # 帧率控制
                dt = time.perf_counter() - t0
                sleep_time = self._frame_interval - dt
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            self._release_all()
            if self._display:
                self._display.clear_yolo_frame()


def main():
    print("=" * 50)
    print("MH Wilds 简单战斗测试")
    print("=" * 50)
    print()

    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始... 请切到游戏窗口")
        beep(800, 100)
        time.sleep(1)

    beep(1200, 500)
    print()

    pad = vg.VX360Gamepad()
    pad.update()
    time.sleep(0.3)

    sct = mss.mss()
    monitor = sct.monitors[1]

    attacker = AttackMonster(pad=pad, sct=sct, monitor=monitor)

    result = attacker.attack(timeout=60)
    print(f"\n战斗结束: {result}")


if __name__ == "__main__":
    main()
