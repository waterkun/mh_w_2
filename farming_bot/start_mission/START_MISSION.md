# Start Mission — 自动接任务

## 概述
从据点自动传送回营地、走到任务 NPC、选择活动任务、出发狩猎。全程无需玩家介入，失败自动重试。

## 完整流程

```
(首次等 3 秒) → 传送回营地 → 走到 NPC → 按 A 对话 → 导航到"活动任务" → 确认出发 → 等待加载
                    ↑                                                                    |
                    └──────────────── 任何阶段失败 ←─────────────────────────────────────┘
```

## 三个阶段

### 阶段 0: 传送回营地 (RESET_TO_CAMP)
- 首次启动等 3 秒 (给玩家切换窗口的时间)
- 回放录制的按键序列 (BACK → DPAD_RIGHT → A → A)
- 等待加载黑屏出现并结束
- 确保每次从相同位置出发

### 阶段 1: 走到 NPC (NAVIGATE_TO_NPC)
- 回放录制的摇杆路线 (~10 秒)
- 每帧模板匹配检测交互提示 (绿色圆圈 A 图标)
- 匹配阈值 > 0.75 → 停止移动，进入阶段 2
- 路线走完后额外等待 5 秒继续检测
- 失败 → 回到阶段 0 重试

### 阶段 2: 选任务并出发 (SELECT_MISSION)
- 按 A 与 NPC 对话 (1 次)
- 按 A 进入任务列表 (3 次，间隔 1 秒)
- 截图对比左侧菜单模板，检测 "活动任务" 是否选中
  - 选中项有向右偏移 + 金色边框，与模板匹配度 > 0.80
  - 未选中 → 按 DPAD_DOWN，重新截图对比 (最多 15 次)
  - 15 次都未匹配 → 按 B ×4 退出菜单 → 回到阶段 0
- 选中后按 A ×3 (进入任务列表 → 选择任务 → 确认出发)
- 等待加载黑屏 (连续 5 帧亮度 < 30 且方差 < 500)
- 超时 60 秒未加载 → 回到阶段 0

## 检测方式

| 检测目标 | 方法 | 模板/阈值 |
|---|---|---|
| 到达 NPC | 模板匹配 | `templates/npc_interact_prompt.png`, > 0.75 |
| 活动任务选中 | 模板匹配 (左侧菜单 ROI) | `templates/menu_active_mission_selected.png`, > 0.80 |
| 加载画面 | 亮度 + 方差 | brightness < 30, variance < 500, 连续 5 帧 |

## 文件结构

```
farming_bot/start_mission/
├── START_MISSION.md                  # 本文档
├── __init__.py
├── start_mission.py                  # 主脚本 (串联三个阶段)
├── record_route.py                   # 录制走到 NPC 的摇杆路线
├── record_reset.py                   # 录制传送回营地的按键序列
├── record_mission_select.py          # 录制选任务按键 (已弃用,改用固定逻辑)
├── replay_route.py                   # 单独测试回放路线
├── templates/
│   ├── npc_interact_prompt.png       # 交互提示图标模板 (绿色 A)
│   └── menu_active_mission_selected.png  # 活动任务选中时的左侧菜单模板
└── recorded_route/
    ├── route.json                    # 摇杆路线数据
    ├── reset_sequence.json           # 传送回营地按键序列
    ├── mission_select_sequence.json  # 选任务按键序列 (已弃用)
    └── npc_arrival_frame.png         # NPC 到达时的截图
```

## 使用方式

### 首次设置: 录制路线和模板

**1. 录制走到 NPC 的摇杆路线:**
```bash
python -m farming_bot.start_mission.record_route
```
- 倒计时 10 秒切到游戏
- 用左摇杆走到任务 NPC
- 到达 NPC 面前按手柄 B 停止
- 输出: `recorded_route/route.json` + `recorded_route/npc_arrival_frame.png`

**2. 录制传送回营地的按键序列:**
```bash
python -m farming_bot.start_mission.record_reset
```
- 倒计时 10 秒切到游戏
- 执行传送回营地操作 (BACK → 导航 → 确认)
- 传送完成后长按 B (0.8 秒) 停止
- 输出: `recorded_route/reset_sequence.json`

**3. 裁剪模板截图:**
- 从 `npc_arrival_frame.png` 裁剪交互提示图标 → `templates/npc_interact_prompt.png`
- 手动进入任务列表，选中 "活动任务"，截图裁剪左侧菜单 → `templates/menu_active_mission_selected.png`

### 运行自动接任务
```bash
python -m farming_bot.start_mission.start_mission
```
- 倒计时 10 秒切到游戏
- 首次启动等 3 秒后开始传送回营地
- 自动执行: 传送回营地 → 走到 NPC → 选活动任务 → 出发
- 失败自动重试，无限循环直到成功

### 单独测试路线回放
```bash
python -m farming_bot.start_mission.replay_route
```
- 仅回放摇杆路线 + 检测交互提示，不选任务

## 配置参数 (start_mission.py)

```python
MATCH_THRESHOLD = 0.75              # 交互提示模板匹配阈值
MENU_MATCH_THRESHOLD = 0.80         # 菜单模板匹配阈值
LOADING_BRIGHTNESS_THRESHOLD = 30   # 加载黑屏亮度阈值
LOADING_VARIANCE_THRESHOLD = 500    # 加载黑屏方差阈值
LOADING_CONFIRM_FRAMES = 5          # 连续 N 帧确认加载
FPS = 10                            # 检测帧率
```

## 已验证按键序列

**传送回营地:**
```
BACK → DPAD_RIGHT → A → A → 等待加载
```

**选任务 (NPC 对话后):**
```
A (对话) → A ×3 (进入任务列表) → DPAD_DOWN 直到活动任务选中 → A ×3 (选任务+确认)
```

## 注意事项
- 路线录制时请确保从营地出发点走到 NPC，路径尽量避开其他可交互 NPC
- 如果据点布局变更或 NPC 位置改变，需要重新录制路线
- 菜单结构如果更新（活动任务位置变化），需要重新截图菜单模板
- 分辨率: 当前基于 3440x1440 超宽屏，更换分辨率需要重新录制和截图
- 支持 `display_callback` 参数，循环中由显示线程自动更新窗口
