# MH Wilds 2 血条检测器

怪物猎人荒野 2 实时血条检测与追踪系统，通过屏幕截图读取玩家血条状态，为强化学习（RL）智能体提供奖励信号。

## 功能特性

- 基于 HSV 颜色遮罩的实时屏幕截取与血条检测
- 追踪血量百分比、受到伤害、受击事件、存活/死亡状态
- 输出 RL 奖励信号（存活奖励、受击惩罚、死亡惩罚）
- 调试可视化窗口，显示 ROI 预览、颜色遮罩和状态面板

## 项目结构

```
mh_w_2/
├── health_bar_photos/          # 离线测试用截图
│   ├── full-screen-shot.png
│   ├── health_bar_full_health.png
│   ├── health_bar_not_full_health.png
│   ├── health_bar_get_hit_lost_health.png
│   └── Health_bar_below_half_health.png
├── mh_w_2_health_bar/          # 核心代码
│   ├── health_bar_detector.py  # HSV 血条检测器
│   ├── health_bar_tracker.py   # 时序状态追踪器 & RL 奖励计算
│   ├── main.py                 # 实时检测入口
│   ├── test_detector.py        # 静态截图离线测试
│   └── requirements.txt
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

### 2. 离线测试

用静态截图验证检测准确性：

```bash
cd mh_w_2_health_bar
python test_detector.py
```

### 3. 实时检测

边玩游戏边运行实时检测：

```bash
cd mh_w_2_health_bar
python main.py
```

会弹出一个调试窗口，显示：
- **顶部**：血条 ROI 区域预览
- **中部**：绿色遮罩（当前血量）和红色遮罩（伤害）并排显示
- **底部**：状态面板，包含血量百分比、伤害百分比、受击次数、奖励信号

按 **`q`** 退出。

### 4. RL 集成

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