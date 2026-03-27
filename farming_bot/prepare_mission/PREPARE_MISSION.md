# Prepare Mission — 任务准备

## 概述
任务加载完成后，先确保道具槽为空白，再回放录制的准备操作（跳过对话、吃饭、准备装备等）。

## 完整流程

```
任务加载完成 → 检查道具槽是否空白 → (不空白则 LB+X 切换) → 回放录制的按键/摇杆序列 → 准备完成
```

## 工作方式

### 1. 道具槽空白检测 (ensure_blank_item_slot)
- 先检查右下角道具槽是否为空白 (正常视角, 模板匹配 > 0.80)
- 如果不为空:
  - **按住 LB** 打开道具栏
  - 反复 **按 X** 切换道具槽
  - 每次检查 LB 道具栏中高亮格是否为空白模板
  - 空白 → 松开 LB，继续准备
  - 超时 30 秒 → 跳过准备
- LB 全程保持按住，只在找到空白时才松开

### 2. 回放准备操作
- 纯回放模式：按时间戳回放录制的所有按键和摇杆输入
- 包括按钮 press/release、左右摇杆、LT/RT 扳机
- 无模板匹配检测（准备步骤固定，不需要视觉反馈）

## 检测方式

| 检测目标 | 方法 | 模板/阈值 |
|---|---|---|
| 道具槽空白 (正常视角) | 模板匹配 ROI `[1250:1430, 3150:3420]` | `templates/item_slot_blank.png`, > 0.80 |
| 道具栏空白 (LB 视角) | 模板匹配 ROI `[1240:1430, 2850:3400]` | `templates/item_bar_blank.png`, > 0.80 |

## 录制

### 录制准备步骤
```bash
python -m farming_bot.prepare_mission.record_prepare
```

1. 10 秒倒计时切到游戏（任务已加载完成）
2. 手动执行所有准备操作
3. 完成后长按 B (0.8秒) 停止录制
4. 输出: `recorded_route/prepare_sequence.json` + `recorded_route/prepare_done_frame.png`

### 重新录制
如果准备步骤变了（换装备、换吃的），直接重新运行录制脚本覆盖即可。

## 回放

### 单独测试
```bash
python -m farming_bot.prepare_mission.prepare_mission
```

### 在循环中使用
```bash
python -m farming_bot.farm_loop
```
循环流程：接任务 → **准备** → 找怪 → 放弃 → 重复

## 文件结构

```
farming_bot/prepare_mission/
├── PREPARE_MISSION.md            # 本文档
├── __init__.py
├── prepare_mission.py            # 回放脚本 (PrepareMission 类)
├── record_prepare.py             # 录制工具
├── templates/
│   ├── item_slot_blank.png      # 右下角空白道具槽模板 (正常视角, 160x110)
│   └── item_bar_blank.png       # LB 道具栏空白高亮格模板 (170x130)
└── recorded_route/
    ├── prepare_sequence.json     # 录制的按键序列
    └── prepare_done_frame.png    # 准备完成时的截图
```

## 配置参数 (prepare_mission.py)

```python
ITEM_BLANK_THRESHOLD = 0.80     # 道具槽空白模板匹配阈值
ITEM_BLANK_TIMEOUT = 30.0       # 道具切换超时 (秒)
```

## 注意事项
- 录制用长按 B (0.8秒) 停止，短按 B 会被正常录制
- 录制的 B 长按事件会被自动移除，不影响回放
- 如果游戏更新改变了准备流程，需要重新录制
- PrepareMission 支持传入已有 pad 实例，避免创建多个虚拟手柄
- 支持 `display_callback` 参数，循环中由显示线程自动更新窗口
