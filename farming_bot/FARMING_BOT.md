# Farming Bot — MH Wilds 自动刷怪

## 概述
自动循环执行：接任务 → 准备 → 找怪 → 战斗 → 结算 → 重启。

## 游戏循环
```
据点 → 接任务 → 加载 → 准备 → 野外骑乘找怪 → 战斗 → 结算 → 回据点 → 循环
```

## 各阶段

### 1. 接任务 — `start_mission/` [已完成]
- **传送回营地**: 录制的按键序列回放 (BACK → DPAD_RIGHT → A → A)
- **首次启动等 3 秒**: 给玩家切换窗口的时间
- **走到 NPC**: 录制的摇杆路线回放 + 模板匹配检测交互提示
- **选任务**: 按 A 对话 → 模板匹配导航到 "活动任务" → 确认出发
- **失败重试**: 任何阶段失败自动传送回营地无限重试
- 详见 `start_mission/START_MISSION.md`

### 1.5 任务准备 — `prepare_mission/` [已完成]
- **检查道具槽**: 右下角道具槽不为空则 LB+X 循环切换到空白
- **回放录制**: 任务加载后回放录制的准备操作 (跳对话、吃饭、准备装备等)
- **纯回放**: 无模板匹配，按时间戳回放按键和摇杆
- 详见 `prepare_mission/PREPARE_MISSION.md`

### 2. 找怪物 — `find_monster/` [已完成]
- **召唤坐骑**: 按 DPAD_UP 召唤鹭鹰龙，每 10 秒重新召唤
- **跟踪导虫**: 每 1 秒按 LB
- **YOLO 检测**: YOLOv10n 检测怪物 body/head，连续 5 帧确认 (防误检)
- **跳下坐骑**: 确认后按 A 跳下
- **实时显示**: 独立显示线程 10fps 持续刷新，不阻塞主逻辑
- 详见 `find_monster/FIND_MONSTER.md`

### 2.5 放弃任务 — `abort_mission/` [已完成]
- **打开菜单**: 键盘 Escape (vgamepad START 无效)
- **切到标签**: LB 重置 → RB 循环找 "使命/任务" tab (模板匹配)
- **选择放弃**: DOWN ×3 到 "从任务中返回" → A → LEFT → A
- **等待返回**: 等 10 秒 → 按 B 到黑屏 → 等加载结束
- 详见 `abort_mission/ABORT_MISSION.md`

### 3. 战斗 (IN_COMBAT) — 待开发
- **检测**: MonsterDetector 检测到怪物 body/head
- **操作**: 下马 → 执行简单连招 (X/Y combo) → 低血吃药 (LB) → 紧急翻滚 (B)
- **健康条**: HealthBarTracker 监控 HP

### 4. 结算 (RESULTS) — 待开发
- **检测**: 画面中央亮度高（白色结算屏）
- **操作**: 连续按 A 跳过所有确认画面

### 5. 重启 (RESULTS → IDLE) — 待开发
- **检测**: 结算画面消失 + 加载画面出现
- **操作**: 回到 start_mission 重新接任务

## 实时显示架构 (双线程)

```
主线程:    接任务 / 准备 / 找怪 / 放弃 (按键 + 检测逻辑)
显示线程:  独立 mss 截屏 → 缩放显示 → 10fps (永远不被主线程阻塞)
```

- 各模块通过 `set_display_label()` 更新窗口标签 (零开销，只改字符串)
- 找怪阶段主线程推送 YOLO 标注帧给显示线程
- 其他阶段显示线程自己截屏 + 显示当前阶段标签

## 检测模块依赖

| 模块 | 位置 | 用途 |
|---|---|---|
| YOLO (ultralytics) | `monster_detection/monster_detect/` | 判断是否看到怪物 |
| HealthBarTracker | `health_bar_module/mh_w_2_health_bar/` | 监控玩家血量 (待用) |
| 模板匹配 (cv2) | farming_bot 内部 | 交互提示/菜单/道具槽检测 |
| 屏幕亮度/颜色分析 | farming_bot 内部 | 加载/结算/昏厥检测 |

## 控制

| 模块 | 用途 |
|---|---|
| vgamepad (XInput) | 虚拟 Xbox 手柄输出 |
| XInput-Python | 读取真实手柄输入 (录制用) |
| ctypes keybd_event | 键盘模拟 (Escape 等 vgamepad 无效的按键) |

## 文件结构
```
farming_bot/
├── FARMING_BOT.md                    # 本文档
├── __init__.py
├── farm_loop.py                      # 主循环 (接任务 → 准备 → 找怪 → 放弃 → 重复)
├── start_mission/                    # 阶段1: 接任务 [已完成]
│   ├── START_MISSION.md
│   ├── start_mission.py              # 主脚本
│   ├── record_route.py               # 录制摇杆路线
│   ├── record_reset.py               # 录制传送回营地
│   ├── replay_route.py               # 测试路线回放
│   ├── templates/                    # 模板截图
│   └── recorded_route/               # 录制数据
├── prepare_mission/                  # 任务准备 [已完成]
│   ├── PREPARE_MISSION.md
│   ├── prepare_mission.py            # 回放脚本 (含道具槽检测)
│   ├── record_prepare.py             # 录制工具
│   ├── templates/                    # 道具槽模板截图
│   └── recorded_route/               # 录制数据
├── find_monster/                     # 阶段2: 找怪 [已完成]
│   ├── FIND_MONSTER.md
│   ├── find_monster.py               # 主脚本 (FindMonster + _DisplayThread)
│   ├── realtime_viewer.py            # 独立可视化工具 (调试)
│   └── __init__.py
├── abort_mission/                    # 放弃任务 [已完成]
│   ├── ABORT_MISSION.md
│   ├── abort_mission.py              # 主脚本
│   ├── record_abort.py               # 录制工具 (备用)
│   ├── templates/                    # 模板截图
│   └── recorded_route/               # 录制数据
├── combat/                           # 阶段3: 战斗 (待开发)
└── end_mission/                      # 阶段4+5: 结算+重启 (待开发)
```

## 快速开始

### 1. 录制 (首次设置)
```bash
# 录制走到 NPC 的路线
python -m farming_bot.start_mission.record_route

# 录制传送回营地的按键
python -m farming_bot.start_mission.record_reset

# 录制任务准备步骤
python -m farming_bot.prepare_mission.record_prepare
```

### 2. 运行
```bash
# 自动接任务
python -m farming_bot.start_mission.start_mission

# 回放任务准备
python -m farming_bot.prepare_mission.prepare_mission

# 找怪物 (单独测试)
python -m farming_bot.find_monster.find_monster

# 放弃任务
python -m farming_bot.abort_mission.abort_mission

# 完整循环 (接任务 → 准备 → 找怪 → 放弃 → 重复)
python -m farming_bot.farm_loop
```

## 依赖
- Python 3.10+
- opencv-python
- mss
- vgamepad
- XInput-Python
- numpy
- ultralytics (YOLO)
