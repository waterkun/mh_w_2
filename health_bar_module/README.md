# MH Wilds 2 血条检测器

怪物猎人荒野 2 实时血条检测与追踪系统，通过屏幕截图读取玩家血条状态，为强化学习（RL）智能体提供奖励信号。

## 功能特性

- **AI 检测器 (新)**: CNN 回归模型，输入 ROI 图像，输出 (health_pct, damage_pct)，对波纹/闪烁/特效更鲁棒
- **规则检测器 (旧)**: 基于列投影法的 BGR 通道逐列扫描，作为 fallback
- 追踪血量百分比、受到伤害、受击事件、存活/死亡状态
- 输出 RL 奖励信号（存活奖励、受击惩罚、死亡惩罚）

## 项目结构

```
health_bar_module/
├── health_bar_photos/                # 离线测试用截图
├── recording_health/                 # 游戏录像视频
├── health_bar_data/                  # AI 训练数据
│   ├── rois/                         #   ROI 裁剪图
│   ├── frames/                       #   全屏帧 (参考用)
│   └── labels.csv                    #   标注文件
├── runs/                             # 训练输出 (best.pt, last.pt)
├── extract_frames.py                 # 从视频抽帧 + 裁剪 ROI
├── label_health_bar.py               # 点击式标注工具
├── train_health_bar.py               # CNN 训练脚本
├── test_all_photos.py                # 批量测试
├── mh_w_2_health_bar/                # 核心代码
│   ├── health_bar_detector.py        #   规则检测器 (旧)
│   ├── health_bar_detector_ai.py     #   AI 检测器 (新)
│   ├── health_bar_model.py           #   HealthBarNet CNN 模型定义
│   ├── health_bar_tracker.py         #   时序状态追踪器 & RL 奖励
│   ├── main.py                       #   实时检测入口 (自动选 AI/规则)
│   └── test_detector.py              #   静态截图离线测试
└── README.md
```

## 环境要求

- Python 3.8+
- Windows（使用 `mss` 进行屏幕截取）

## 安装

```bash
cd mh_w_2_health_bar
pip install -r requirements.txt
```

## 使用方法

### 1. 校准 ROI

血条的 ROI（感兴趣区域）必须匹配你的屏幕分辨率和游戏 UI 布局。当前默认值针对 **3440x1440 超宽屏** 校准：

```python
# health_bar_detector.py
DEFAULT_ROI = (143, 63, 716, 48)  # (x, y, 宽, 高)
```

如需重新校准：
1. 在游戏中血条可见时截取全屏截图
2. 用图片编辑工具找到血条的边界框坐标（x, y, 宽, 高）
3. 更新 `health_bar_detector.py` 中的 `DEFAULT_ROI`

### 2. AI 检测器训练流程

#### Step 1: 抽帧
```bash
python extract_frames.py --video recording_health/1.mp4 --interval 10
```
从视频每 10 帧抽取一帧，裁剪 ROI 保存到 `health_bar_data/rois/`。

#### Step 2: 标注 (200-500 帧即可)
```bash
python label_health_bar.py --data health_bar_data
```
ROI 图放大 8 倍显示，点击标注边界：
- **左键** → 绿色血量右边界
- **右键** → 伤害段右边界（红色或黄色均可）
- **Enter** → 确认，进入下一帧
- **S** → 跳过  |  **Z** → 撤销  |  **Q** → 保存退出

#### Step 3: 训练
```bash
python train_health_bar.py --data health_bar_data --epochs 100
```
模型保存到 `runs/best.pt`。

#### Step 4: 实时测试
```bash
cd mh_w_2_health_bar
python main.py
```
自动检测 `runs/best.pt`，存在则用 AI 检测器，否则 fallback 到规则检测器。

按 **`q`** 退出。

### 3. RL 集成

在你的 RL 智能体代码中使用 `HealthBarTracker`：

```python
from health_bar_detector import HealthBarDetector
from health_bar_tracker import HealthBarTracker

detector = HealthBarDetector()
tracker = HealthBarTracker(detector)

# 在游戏循环中：
state = tracker.update(frame)   # 返回字典：health_pct, is_hit, is_alive 等
reward = tracker.get_reward_signal()  # 浮点数：存活 +0.01，受击 -1.0*伤害量，死亡 -5.0
```

## 奖励信号设计

| 事件 | 奖励值 |
|------|--------|
| 存活（每帧） | +0.01 |
| 受到攻击 | -1.0 * 伤害量 |
| 死亡（血量 < 1%） | -5.0 |