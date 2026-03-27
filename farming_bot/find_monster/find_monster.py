"""找怪物 — 骑乘鹭鹰龙自动寻路到怪物.

使用方式:
  python -m farming_bot.find_monster.find_monster

流程:
  1. 按 DPAD_UP 召唤/骑上鹭鹰龙
  2. 鹭鹰龙自动沿导虫路线跑向怪物
  3. 每帧用 MonsterDetector (YOLO) 检测怪物
  4. 10 秒内没找到 → 再按 DPAD_UP 重新召唤
  5. 检测到怪物 → 按 A 跳下坐骑
"""

import os
import sys
import threading
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
    "runs", "monster_detect", "weights", "best.pt"
)
_HEALTH_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "health_bar_module", "runs", "best.pt"
)

# 可视化颜色 (BGR)
_COLORS = {"body": (0, 255, 0), "head": (0, 0, 255)}
_DISPLAY_SCALE = 0.75

# 检测参数
MONSTER_CONFIDENCE = 0.6
CONFIRM_FRAMES = 3      # 连续 N 帧检测到才算找到 (防误检)
SUMMON_INTERVAL = 10.0  # 每 10 秒没找到就重新按 UP
FIND_TIMEOUT = 120.0    # 最多找 2 分钟
FPS = 5                 # 检测帧率 (YOLO 推理较慢, 5fps 够用)

LB_INTERVAL = 1.0          # 每 1 秒按一次 LB

DISPLAY_FPS = 10           # 显示线程帧率

_BUTTON_MAP = {
    "A": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
    "B": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    "DPAD_UP": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
    "DPAD_DOWN": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
    "LEFT_SHOULDER": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
}


def beep(freq=1000, duration_ms=200):
    try:
        winsound.Beep(freq, duration_ms)
    except Exception:
        pass


class _DisplayThread:
    """独立线程持续截屏+显示, 不阻塞主线程."""

    def __init__(self, monitor):
        self._monitor = monitor
        self._lock = threading.Lock()
        self._label = "IDLE"
        self._yolo_frame = None  # 主线程放 YOLO 标注帧
        self._running = False
        self._thread = None
        self._health_detector = HealthBarDetectorAI(_HEALTH_MODEL_PATH)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def set_label(self, label):
        with self._lock:
            self._label = label

    def set_yolo_frame(self, frame):
        """主线程放入 YOLO 标注后的帧 (已画好检测框+状态文字)."""
        with self._lock:
            self._yolo_frame = frame

    def clear_yolo_frame(self):
        with self._lock:
            self._yolo_frame = None

    def _run(self):
        # 显示线程有自己的 mss 实例 (mss 不能跨线程)
        sct = mss.mss()
        cv2.namedWindow("Monster Detection", cv2.WINDOW_NORMAL)
        interval = 1.0 / DISPLAY_FPS

        while self._running:
            t0 = time.perf_counter()

            with self._lock:
                yolo_frame = self._yolo_frame
                label = self._label

            if yolo_frame is not None:
                # 找怪/战斗阶段: 显示 YOLO 标注帧
                display = yolo_frame
            else:
                # 非找怪阶段: 自己截屏, 显示原始画面 + 标签
                screenshot = sct.grab(self._monitor)
                frame = np.array(screenshot)
                display = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                cv2.putText(display, label, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (128, 128, 128), 3)

            # 血量检测 + 显示 (所有阶段)
            health_result = self._health_detector.detect(display)
            health_pct = health_result["health_pct"]
            h, w = display.shape[:2]
            bar_x, bar_y, bar_w, bar_h = w - 320, 20, 280, 30
            # 背景
            cv2.rectangle(display, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
            # 血量条
            fill_w = int(bar_w * health_pct)
            if health_pct < 0.20:
                bar_color = (0, 0, 255)      # 红
            elif health_pct < 0.50:
                bar_color = (0, 165, 255)    # 橙
            else:
                bar_color = (0, 200, 0)      # 绿
            cv2.rectangle(display, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
            # 边框
            cv2.rectangle(display, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
            # 百分比文字
            hp_text = f"HP {health_pct:.0%}"
            cv2.putText(display, hp_text, (bar_x + 8, bar_y + bar_h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            resized = cv2.resize(display,
                                 (int(w * _DISPLAY_SCALE), int(h * _DISPLAY_SCALE)))
            cv2.imshow("Monster Detection", resized)
            cv2.waitKey(1)

            dt = time.perf_counter() - t0
            sleep_time = interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

        cv2.destroyWindow("Monster Detection")


class FindMonster:
    """骑乘鹭鹰龙自动寻路找怪物."""

    def __init__(self, pad=None, sct=None, monitor=None):
        """初始化.

        Args:
            pad: 可复用的 VX360Gamepad 实例 (None 则新建).
            sct: 可复用的 mss 实例.
            monitor: 截屏区域.
        """
        self._model = YOLO(_MODEL_PATH)
        self._class_names = self._model.names

        if pad is None:
            self._pad = vg.VX360Gamepad()
            self._pad.update()
            time.sleep(0.3)
        else:
            self._pad = pad

        if sct is None:
            self._sct = mss.mss()
            self._monitor = self._sct.monitors[1]
        else:
            self._sct = sct
            self._monitor = monitor or self._sct.monitors[1]

        self._frame_interval = 1.0 / FPS
        self._display = None

        print("FindMonster 初始化完成")
        print(f"  模型: {_MODEL_PATH}")
        print(f"  置信度阈值: {MONSTER_CONFIDENCE}")

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

    def _detect_and_draw(self, frame):
        """YOLO 推理并在帧上画检测框.

        Returns:
            (visible: bool, max_conf: float, display_frame: ndarray)
        """
        results = self._model(frame, verbose=False)
        boxes = results[0].boxes

        display = frame.copy()
        max_conf = 0.0
        visible = False

        for box in boxes:
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            cls_name = self._class_names.get(cls_id, f"cls{cls_id}")
            color = _COLORS.get(cls_name, (255, 255, 0))

            if conf > max_conf:
                max_conf = conf
            if conf >= MONSTER_CONFIDENCE:
                visible = True

            # 画框 (>=0.6 粗框, <0.6 细框)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            thickness = 3 if conf >= MONSTER_CONFIDENCE else 1
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

            # 标签
            label = f"{cls_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(display, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(display, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        return visible, max_conf, display

    # ── 显示线程管理 ──

    def start_display(self):
        """启动独立显示线程 (整个循环期间保持)."""
        self._display = _DisplayThread(self._monitor)
        self._display.start()

    def stop_display(self):
        """停止显示线程."""
        if self._display:
            self._display.stop()
            self._display = None

    def set_display_label(self, label):
        """其他模块调用, 更新显示窗口的标签."""
        if self._display:
            self._display.set_label(label)

    # ── 找怪主逻辑 ──

    def find(self) -> bool:
        """召唤鹭鹰龙并寻找怪物.

        Returns:
            True 如果找到怪物并跳下坐骑.
        """
        print("\n[找怪物] 开始...")
        print(f"  需连续 {CONFIRM_FRAMES} 帧确认 (防误检)")

        start_time = time.time()
        last_summon = 0
        last_lb = 0
        consecutive_count = 0

        while True:
            elapsed = time.time() - start_time

            # 超时
            if elapsed > FIND_TIMEOUT:
                print(f"  找怪超时 ({FIND_TIMEOUT}s)")
                self._release_all()
                if self._display:
                    self._display.clear_yolo_frame()
                return False

            # 每 SUMMON_INTERVAL 秒按一次 UP 召唤/重新骑乘
            if elapsed - last_summon >= SUMMON_INTERVAL or last_summon == 0:
                self._press_button("DPAD_UP")
                print(f"  [{elapsed:.0f}s] 按 DPAD_UP 召唤鹭鹰龙")
                last_summon = elapsed
                last_lb = elapsed
                consecutive_count = 0
                time.sleep(2.0)

            # 每 LB_INTERVAL 秒按一次 LB
            if elapsed - last_lb >= LB_INTERVAL:
                self._press_button("LEFT_SHOULDER")
                last_lb = elapsed

            # 截屏 + YOLO 检测
            t0 = time.perf_counter()
            frame = self._grab_frame()
            visible, conf, display = self._detect_and_draw(frame)

            if visible:
                consecutive_count += 1
                print(f"  [{elapsed:.0f}s] 检测到怪物 conf={conf:.3f} "
                      f"({consecutive_count}/{CONFIRM_FRAMES})")

                if consecutive_count >= CONFIRM_FRAMES:
                    print(f"  连续 {CONFIRM_FRAMES} 帧确认, 怪物找到!")
                    beep(1500, 300)

                    # 按 A 跳下坐骑
                    self._press_button("A")
                    print("  按 A 跳下坐骑")
                    time.sleep(1.0)

                    self._release_all()
                    if self._display:
                        self._display.clear_yolo_frame()
                    print("  找怪完成!")
                    return True
            else:
                if consecutive_count > 0:
                    print(f"  [{elapsed:.0f}s] 怪物消失, 重置计数 "
                          f"(was {consecutive_count})")
                consecutive_count = 0

            # 把 YOLO 标注帧送给显示线程
            if self._display:
                status = f"[{elapsed:.0f}s] conf={conf:.3f} " \
                         f"confirm={consecutive_count}/{CONFIRM_FRAMES}"
                if visible:
                    status += " DETECTED"
                cv2.putText(display, status, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)
                self._display.set_yolo_frame(display)

            # 帧率控制
            dt = time.perf_counter() - t0
            sleep_time = self._frame_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    print("=" * 50)
    print("MH Wilds 找怪物")
    print("=" * 50)
    print()

    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始... 请切到游戏窗口")
        beep(800, 100)
        time.sleep(1)

    beep(1200, 500)
    print()

    finder = FindMonster()
    finder.start_display()

    try:
        success = finder.find()
        if success:
            print("\n找到怪物!")
        else:
            print("\n未找到怪物")
    finally:
        finder.stop_display()


if __name__ == "__main__":
    main()
