# Abort Mission — 放弃任务

## 概述
在任务中自动放弃当前任务，返回据点。用于 farming loop 中接任务后立即放弃再重新接。

## 完整流程

```
按 Escape 打开菜单 → LB 重置 tab → RB 找到"使命/任务"标签 → DOWN ×3 到"从任务中返回" → A → LEFT → A → 等 10 秒 → 按 B 到黑屏 → 等加载结束
```

## 各步骤

### 1. 打开菜单
- 按键盘 **Escape** (vgamepad START 对 MH Wilds 无效)
- 等待 2 秒

### 2. 重置 Tab 位置
- 按 **LB** 一次，重置高亮到第一个 tab
- 等待 2 秒

### 3. 切到"使命/任务"标签
- 按 **RB** 循环切换 tab (最多 8 次)
- 每次截图，模板匹配检测 tab 图标行
- 匹配阈值 > 0.90
- RB 未找到则尝试 LB 方向 (最多 8 次)
- 找不到 → 按 B ×4 退出，返回失败

### 4. 导航到"从任务中返回"
- 按 **DPAD_DOWN** × 3 (第 4 项)
- RB/LB 切 tab 会重置高亮到第 1 项，所以固定按 3 次

### 5. 确认放弃
- 按 **A** → 等 1 秒
- 按 **LEFT** → 等 1 秒
- 按 **A** → 等 1 秒

### 6. 等待返回
- 等待 10 秒 (游戏处理中)
- 按 **B** 每 0.5 秒一次，直到检测到黑屏 (最多 15 秒)
- 等加载黑屏结束
- 等 5 秒画面稳定

## 检测方式

| 检测目标 | 方法 | 模板/阈值 |
|---|---|---|
| 使命/任务 tab | 模板匹配 (tab 图标行 ROI) | `templates/abort_tab.png`, > 0.90 |
| 加载画面 | 亮度 + 方差 | brightness < 30, variance < 500, 连续 5 帧 |

## 文件结构

```
farming_bot/abort_mission/
├── ABORT_MISSION.md              # 本文档
├── __init__.py
├── abort_mission.py              # 主脚本 (AbortMission 类)
├── record_abort.py               # 录制工具 (备用)
├── recorded_route/               # 录制数据
└── templates/
    ├── abort_tab.png             # 使命/任务 tab 图标行模板
    ├── abort_item_return.png     # 从任务中返回高亮模板 (备用)
    └── menu_abort_mission.png    # 完整菜单模板 (备用)
```

## 使用方式

### 单独运行
```bash
python -m farming_bot.abort_mission.abort_mission
```

### 循环测试 (接任务 → 准备 → 找怪 → 放弃 → 重复)
```bash
python -m farming_bot.farm_loop
```

## 配置参数 (abort_mission.py)

```python
TAB_MATCH_THRESHOLD = 0.90          # Tab 模板匹配阈值
LOADING_BRIGHTNESS_THRESHOLD = 30   # 加载黑屏亮度阈值
LOADING_VARIANCE_THRESHOLD = 500    # 加载黑屏方差阈值
LOADING_CONFIRM_FRAMES = 5         # 连续 N 帧确认加载
FPS = 10                           # 检测帧率
```

## 注意事项
- vgamepad 的 START 按钮对 MH Wilds 无效，改用键盘 Escape
- "从任务中返回" 位置固定为第 4 项，如果菜单顺序变化需要调整 DOWN 次数
- 按键默认按压时间 0.15 秒，每次按键后等待 1 秒
- 支持 `display_callback` 参数，循环中由显示线程自动更新窗口
