"""测试 ScreenAnalyzer 状态检测 — 在游戏画面前运行."""

import json
import os

import cv2
import mss
import numpy as np
import time

from bot.bot_config import MONSTER_DETECTOR_PATH
from monster_detect import MonsterDetector
from bot.screen_analyzer import ScreenAnalyzer

_AP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROI_CONFIG = os.path.join(_AP_DIR, "roi_config.json")


def _load_roi():
    with open(_ROI_CONFIG) as f:
        cfg = json.load(f)
    return cfg["x"], cfg["y"], cfg["w"], cfg["h"]


model_path = os.path.join(_AP_DIR, MONSTER_DETECTOR_PATH)
if not os.path.exists(model_path):
    print(f"警告: 模型文件不存在 {model_path}, 怪物检测不可用")
    md = None
else:
    md = MonsterDetector(model_path)

sct = mss.mss()
mon = sct.monitors[1]
sa = ScreenAnalyzer(md)
roi = _load_roi()
x, y, w, h = roi

print("开始检测游戏状态 (60帧, 每0.5s一帧)...")
print("请切换到游戏画面, 试试 野外/战斗 等场景\n")

for i in range(60):
    frame = np.array(sct.grab(mon))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    roi_frame = frame[y:y + h, x:x + w]

    state = sa.analyze(frame, roi_frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print(f"Frame {i:2d}: {state.name:<10s}  "
          f"bright={gray.mean():.0f}  "
          f"monster_conf={sa.last_monster_confidence:.3f}")
    time.sleep(0.5)
