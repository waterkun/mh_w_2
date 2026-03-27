# Attack Prediction Module — Progress

## Project Goal
实时预测怪物猎人中怪物的攻击动作类型，为 RL agent 提供攻击状态信号和奖励信号。

---

## Architecture Overview

```
屏幕截取 (mss) → ROI 裁剪 → 帧缓冲 (22帧 @15FPS ≈ 1.5s)
    → MobileNetV3-Small (CNN backbone, 576-d 特征)
    → GRU (2层, 256 hidden, 时序分类)
    → 8 类攻击分类
    → AttackTracker (多帧状态确认 + RL 奖励信号)
```

**8 个攻击类别:**
| ID | 类别 | 说明 |
|----|------|------|
| 0 | idle | 无攻击 / 待机 |
| 1 | pounce | 扑击 |
| 2 | beam | 吐息光线 |
| 3 | tail_sweep | 尾扫 |
| 4 | flying_attack | 飞行攻击 |
| 5 | claw_swipe | 爪击 |
| 6 | charge | 冲锋 |
| 7 | nova | 大爆发 (超新星) |

---

## Done (已完成)

### 1. 项目骨架搭建 — 全部代码已就位
- [x] `config.py` — 全局配置常量 (路径、类别、序列参数、模型超参、训练超参)
- [x] `select_roi.py` — 两步 ROI 选择工具 (粗选 → 放大精选)
- [x] `data/record_gameplay.py` — mss 录屏工具，按 ROI 区域录制 AVI
- [x] `data/label_attacks.py` — 交互式视频标注工具 (播放/暂停/逐帧/快进 + 标记攻击时间段)
- [x] `data/build_dataset.py` — 标注 JSON → 滑动窗口帧序列数据集 (含 train/val 分割)
- [x] `model/backbone.py` — MobileNetV3-Small 特征提取器 (ImageNet 预训练, 576-d)
- [x] `model/temporal_head.py` — GRU 时序分类头 (2层, 256 hidden)
- [x] `model/attack_model.py` — CNN + GRU 组合模型
- [x] `train.py` — 训练脚本 (WeightedRandomSampler 平衡采样, 分阶段冻结/解冻 CNN)
- [x] `evaluate.py` — 评估脚本 (混淆矩阵 + per-class precision/recall/F1)
- [x] `attack_detector.py` — 推理封装 (帧缓冲 deque + 单帧 detect 接口)
- [x] `attack_tracker.py` — 多帧状态确认 + RL 奖励信号生成
- [x] `main.py` — 实时 demo (屏幕截取 + 推理 + 可视化面板)

### 2. 自动标注系统
- [x] `config.py` — 新增自动标注参数 (LOOKBACK_SEC, LOOKFORWARD_SEC, HIT_COOLDOWN_SEC, BUFFER_SIZE) + 聚类参数
- [x] `data/auto_labeler.py` — 核心类: 环形缓冲 (deque 45帧) + 命中冷却去重 + lookforward 录制 + clip 保存
- [x] `data/auto_label_attacks.py` — 主入口: 实时模式 (mss 截屏) / 离线模式 (视频文件) 双模式
- [x] `data/cluster_attacks.py` — 聚类工具: MobileNetV3 特征提取 + DBSCAN/KMeans + t-SNE 可视化
- [x] `data/rename_clusters.py` — 簇命名: cluster_X → 实际攻击名批量替换
- [x] `data/build_dataset.py` — 新增 `--mode clips` 支持从自动标注 clip 构建训练数据集

### 3. 关键设计决策
- **序列长度:** 22 帧 × 15 FPS = ~1.5s 时间窗口
- **滑动窗口:** stride=8 (~0.5s)，中心帧标签
- **分阶段训练:** 前 5 epoch 冻结 CNN (lr=1e-3)，之后解冻 finetune (lr=1e-4)
- **类别平衡:** WeightedRandomSampler (1/count 权重)
- **状态确认:** 连续 3 帧同类别才切换状态 (防抖动)
- **RL 奖励:** 攻击开始 +1.0，攻击持续 +0.01
- **自动标注:** 血条下降 → 回溯 2s + 延续 0.5s → 保存 clip，2.5s 冷却去重
- **聚类:** hit_frame ±5 帧提取 MobileNetV3 576-d 特征取平均 → DBSCAN/KMeans

---

## TODO (待完成)

### Phase 1: 数据采集 — 自动标注流程 (推荐)
- [ ] **选择目标怪物** — 确定先针对哪个怪物采集数据 (建议从动作明显的 boss 开始)
- [ ] **运行 `select_roi.py`** — 选定怪物活动区域的 ROI，生成 `roi_config.json`
  ```bash
  python select_roi.py
  ```
- [ ] **自动收集攻击 clip** — 实时模式: 玩游戏，血条下降时自动回溯保存 clip
  ```bash
  python data/auto_label_attacks.py --realtime
  ```
  或离线模式: 从已有录像中提取
  ```bash
  python data/auto_label_attacks.py --video data/raw/video.avi
  ```
- [ ] **聚类分析** — 对收集到的 clip 自动发现攻击类型
  ```bash
  python data/cluster_attacks.py --session data/auto_labeled_clips/session_XXXXXXXX
  # 或聚类所有 sessions:
  python data/cluster_attacks.py --all --method dbscan
  ```
- [ ] **查看聚类结果** — 打开 `clusters_visualization.png`，观察 t-SNE 散点图
- [ ] **命名攻击类型** — 根据聚类结果给簇命名
  ```bash
  python data/rename_clusters.py --session <dir> \
    --mapping '{"cluster_0": "pounce", "cluster_1": "tail_sweep", "cluster_2": "beam"}'
  ```
- [ ] **构建帧序列数据集** — 从 clip 生成训练数据
  ```bash
  python data/build_dataset.py --mode clips --session data/auto_labeled_clips/session_XXXXXXXX
  ```

### Phase 1 (备选): 手动标注流程
- [ ] **录制游戏视频** — 使用 `record_gameplay.py` 录制多段战斗视频 (建议至少 5-10 分钟总时长)
  ```bash
  python data/record_gameplay.py -d 120  # 录制 2 分钟
  ```
- [ ] **标注攻击时间段** — 使用 `label_attacks.py` 逐段标注每个攻击动作
  ```bash
  python data/label_attacks.py data/raw/20260304_XXXXXX.avi
  ```
- [ ] **构建帧序列数据集**
  ```bash
  python data/build_dataset.py --mode video --video data/raw/video.avi --labels data/raw/video_labels.json
  ```

### Phase 2: 训练 & 评估
- [ ] **检查数据分布** — 确认 idle vs 攻击比例合理，不要过度 idle 主导
- [ ] **首轮训练** — 运行 `train.py` 训练模型
  ```bash
  python train.py data/sequences/session_name -e 30
  ```
- [ ] **评估模型** — 运行 `evaluate.py` 检查混淆矩阵
  ```bash
  python evaluate.py data/sequences/session_name runs/XXXXXX/best.pt
  ```

### Phase 3: 迭代优化
- [ ] **实时测试** — 使用 `main.py` 运行实时 demo，观察预测效果
- [ ] **收集更多数据** — 根据评估结果，补充弱势类别的数据
- [ ] **调整超参** — 根据过拟合/欠拟合情况调整 (学习率、dropout、序列长度)
- [ ] **多怪物泛化** — 扩展到更多怪物种类

### Phase 4: 模型改进 (可选)
- [ ] **数据增强** — 添加更多时序增强 (时间拉伸、帧间插值)
- [ ] **注意力机制** — 在 GRU 后添加 temporal attention
- [ ] **双向 GRU** — 离线标注场景可用双向，实时推理保持单向
- [ ] **多尺度特征** — 尝试不同序列长度组合
- [ ] **轻量化** — 知识蒸馏或量化，确保实时推理 <67ms/帧

### Phase 5: 系统集成
- [ ] **与 health_bar_module 整合** — 血量 + 攻击状态联合信号
- [ ] **与 hit_number_detection 整合** — 伤害数字 + 攻击类型关联分析
- [ ] **与 Gauge (斩味) 整合** — 完整的游戏状态感知系统
- [ ] **RL Agent 对接** — 将 AttackTracker 的奖励信号接入强化学习框架

---

## Current Status
**阶段: Phase 1 — 数据采集前 (自动标注系统已就绪)**

代码框架 100% 完成，自动标注流水线已就绪，尚无训练数据。下一步是：
1. 启动游戏
2. 运行 `select_roi.py` 选定怪物 ROI
3. 运行 `auto_label_attacks.py --realtime` 自动收集攻击 clip
4. 运行 `cluster_attacks.py` 聚类 → `rename_clusters.py` 命名
5. 运行 `build_dataset.py --mode clips` 生成训练数据
6. 开始训练

---

## File Structure
```
attack_prediction/
├── config.py                  # 全局配置 (含自动标注 + 聚类参数)
├── select_roi.py              # ROI 选择工具
├── main.py                    # 实时推理 demo
├── attack_detector.py         # 推理封装
├── attack_tracker.py          # 状态追踪 + RL 信号
├── train.py                   # 训练脚本
├── evaluate.py                # 评估脚本
├── model/
│   ├── backbone.py            # MobileNetV3-Small
│   ├── temporal_head.py       # GRU 时序头
│   └── attack_model.py        # CNN+GRU 组合
├── data/
│   ├── record_gameplay.py     # 录屏工具
│   ├── label_attacks.py       # 手动标注工具
│   ├── build_dataset.py       # 数据集构建 (video + clips 双模式)
│   ├── auto_labeler.py        # 自动标注核心类 (环形缓冲 + clip 保存)
│   ├── auto_label_attacks.py  # 自动标注入口 (实时/离线双模式)
│   ├── cluster_attacks.py     # 聚类工具 (特征提取 + DBSCAN/KMeans + t-SNE)
│   ├── rename_clusters.py     # 簇命名工具
│   ├── raw/                   # (待创建) 原始录制视频
│   ├── sequences/             # (待创建) 帧序列数据集
│   └── auto_labeled_clips/    # (待创建) 自动标注 clip 输出
├── runs/                      # (待创建) 训练输出
└── progress.md                # 本文件
```
