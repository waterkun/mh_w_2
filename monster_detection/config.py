"""攻击预测模块 — 全局配置常量."""

import os

# ── 路径 ──────────────────────────────────────────────
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_CONFIG_PATH = os.path.join(MODULE_DIR, "roi_config.json")
DATA_DIR = os.path.join(MODULE_DIR, "data")
RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw")
RUNS_DIR = os.path.join(MODULE_DIR, "runs")

# ── 攻击类别 ──────────────────────────────────────────
ATTACK_CLASSES = [
    "idle",            # 0 — 无攻击 / 待机
    "pounce",          # 1 — 扑击
    "beam",            # 2 — 吐息光线
    "tail_sweep",      # 3 — 尾扫
    "flying_attack",   # 4 — 飞行攻击
    "claw_swipe",      # 5 — 爪击
    "charge",          # 6 — 冲锋
    "nova",            # 7 — 大爆发 (超新星)
]
NUM_CLASSES = len(ATTACK_CLASSES)
CLASS_TO_IDX = {name: i for i, name in enumerate(ATTACK_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(ATTACK_CLASSES)}

# ── 序列参数 ──────────────────────────────────────────
CAPTURE_FPS = 15          # 录屏 / 推理帧率
SEQ_LENGTH = 22           # 15 FPS × ~1.5s
INPUT_SIZE = 224          # 每帧 resize 到 224×224
FRAME_INTERVAL_MS = int(1000 / CAPTURE_FPS)  # 67 ms

# ── 数据集构建 ────────────────────────────────────────
SLIDE_STRIDE = 8          # 滑动窗口步长 (~0.5s)
CENTER_FRAME_IDX = SEQ_LENGTH // 2  # 第 11 帧 (0-indexed)
TRAIN_VAL_SPLIT = 0.8     # 80% train, 20% val

# ── 模型超参 ─────────────────────────────────────────
BACKBONE_FEAT_DIM = 576   # MobileNetV3-Small 最终特征维度
GRU_HIDDEN = 256
GRU_LAYERS = 2
DROPOUT = 0.3

# ── 训练超参 ─────────────────────────────────────────
BATCH_SIZE = 8
NUM_EPOCHS = 30
FREEZE_EPOCHS = 5         # 前 N epoch 冻结 CNN
LR = 1e-3
LR_FINETUNE = 1e-4        # 解冻 CNN 后的学习率
WEIGHT_DECAY = 1e-4

# ── 推理 / Tracker ───────────────────────────────────
CONFIRM_FRAMES = 3        # 连续 N 帧同类别才确认状态切换

# ── 自动标注参数 ─────────────────────────────────
AUTO_LABEL_DIR = os.path.join(DATA_DIR, "auto_labeled_clips")
LOOKBACK_SEC = 2.0           # 命中前回溯时长
LOOKFORWARD_SEC = 0.5        # 命中后延续时长
HIT_COOLDOWN_SEC = 2.5       # 连续命中最小间隔 (去重)
BUFFER_SIZE = int(CAPTURE_FPS * (LOOKBACK_SEC + 1))  # 环形缓冲大小

# ── 聚类参数 ──────────────────────────────────────
CLUSTER_METHOD = "dbscan"
KMEANS_K = 7
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 3
