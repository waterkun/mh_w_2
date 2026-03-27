# Find Monster — 找怪物

## 概述
任务准备完成后，召唤鹭鹰龙自动寻路到怪物位置，检测到怪物后跳下坐骑。

## 完整流程

```
按 DPAD_UP 召唤鹭鹰龙 → 自动寻路 (每 1 秒按 LB) → YOLO 检测怪物 → 连续 5 帧确认 → 按 A 跳下
         ↑                                                                    ↓
         └──── 10 秒内没找到 ← ───────────────────────────────────────────────┘
```

## 工作方式
1. **按 DPAD_UP** 召唤鹭鹰龙，等 2 秒上马
2. 鹭鹰龙自动沿导虫路线跑向怪物
3. **每 1 秒按 LB** 跟踪导虫
4. 每帧用 **YOLOv10n** 检测画面中是否有怪物 (body/head)
5. **连续 5 帧确认** (conf > 0.6) 才算找到怪物 (防误检，模型约 80% 准确率)
6. **10 秒内没找到** → 再按 DPAD_UP 重新召唤（可能没骑上或走丢了）
7. **确认找到** → 按 **A 跳下坐骑**
8. 最多找 **120 秒**，超时返回失败

## 实时显示 (双线程架构)

显示窗口在独立线程运行，不阻塞主线程的检测和按键逻辑：

```
主线程:    按键操作 + YOLO 检测 → 把标注帧推送给显示线程
显示线程:  独立 mss 截屏 → 缩放显示 → 10fps 持续刷新
```

- **找怪阶段**: 显示线程展示主线程推送的 YOLO 标注帧 (检测框 + 置信度 + 状态)
- **其他阶段**: 显示线程自己截屏，显示原始画面 + 当前阶段标签
- 显示线程有独立 `mss` 实例 (mss 不能跨线程共享)
- `DISPLAY_FPS = 10` 控制显示帧率

## 检测方式

| 检测目标 | 方法 | 阈值 |
|---|---|---|
| 怪物 (body/head) | YOLOv10n 目标检测 | confidence > 0.6, 连续 5 帧 |

## 依赖模块

| 模块 | 位置 | 用途 |
|---|---|---|
| YOLO (ultralytics) | `monster_detection/monster_detect/` | 怪物检测 |
| best.pt | `monster_detection/monster_detect/runs/monster_detect/weights/` | 训练好的模型 |

## 文件结构

```
farming_bot/find_monster/
├── FIND_MONSTER.md               # 本文档
├── __init__.py
├── find_monster.py               # 主脚本 (FindMonster + _DisplayThread)
└── realtime_viewer.py            # 独立 YOLO 可视化工具 (调试用)
```

## 使用方式

### 单独测试
```bash
python -m farming_bot.find_monster.find_monster
```

### 在循环中使用
```bash
python -m farming_bot.farm_loop
```

### 独立实时查看器 (调试)
```bash
python -m farming_bot.find_monster.realtime_viewer
```

## 配置参数 (find_monster.py)

```python
MONSTER_CONFIDENCE = 0.6    # YOLO 检测置信度阈值
CONFIRM_FRAMES = 5          # 连续 N 帧确认 (防误检)
SUMMON_INTERVAL = 10.0      # 每 N 秒重新按 UP 召唤
LB_INTERVAL = 1.0           # 每 N 秒按一次 LB
FIND_TIMEOUT = 120.0        # 最多找 N 秒
FPS = 5                     # YOLO 检测帧率
DISPLAY_FPS = 10            # 显示线程帧率
```
