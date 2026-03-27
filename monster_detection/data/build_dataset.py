"""标注 → 帧序列训练数据 — 滑动窗口切片.

两种模式:
  video 模式: 标注 JSON + 原始视频 → 帧序列
  clips 模式: 自动标注 clip 目录 → 帧序列

输出格式:
  sequences/
    seq_00000/ (22 帧 jpg)
    seq_00001/
    ...
  labels.json  {seq_name: class_name}

自动 train/val 分割 (80/20).
"""

import json
import os
import random
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (ATTACK_CLASSES, CLASS_TO_IDX, SEQ_LENGTH, SLIDE_STRIDE,
                    CENTER_FRAME_IDX, TRAIN_VAL_SPLIT, INPUT_SIZE, DATA_DIR)


def build_frame_labels(annotations, total_frames):
    """从区间标注生成逐帧标签数组.

    未标注的帧标记为 'idle'.
    重叠区间: 后标注的优先.
    """
    labels = ["idle"] * total_frames
    for ann in annotations:
        s = ann["start_frame"]
        e = min(ann["end_frame"], total_frames - 1)
        attack = ann["attack"]
        for i in range(s, e + 1):
            labels[i] = attack
    return labels


def build_dataset(video_path, label_path, output_dir=None):
    """从一个视频 + 标注 JSON 生成帧序列数据集."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    with open(label_path) as f:
        label_data = json.load(f)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    annotations = label_data.get("annotations", [])
    frame_labels = build_frame_labels(annotations, total_frames)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, "sequences", video_name)
    os.makedirs(output_dir, exist_ok=True)

    # 预读所有帧
    print(f"读取视频帧... (共 {total_frames})")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    total_frames = len(frames)
    print(f"实际读取: {total_frames} 帧")

    # 滑动窗口切片
    all_sequences = {}
    seq_idx = 0

    for start in range(0, total_frames - SEQ_LENGTH + 1, SLIDE_STRIDE):
        center = start + CENTER_FRAME_IDX
        label = frame_labels[center]

        seq_name = f"seq_{seq_idx:05d}"
        seq_dir = os.path.join(output_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)

        for i in range(SEQ_LENGTH):
            frame = frames[start + i]
            resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            cv2.imwrite(os.path.join(seq_dir, f"{i:03d}.jpg"), resized)

        all_sequences[seq_name] = label
        seq_idx += 1

    print(f"生成 {seq_idx} 个序列")

    # 统计类别分布
    class_counts = {}
    for label in all_sequences.values():
        class_counts[label] = class_counts.get(label, 0) + 1
    print("类别分布:")
    for cls in ATTACK_CLASSES:
        print(f"  {cls}: {class_counts.get(cls, 0)}")

    # train/val 分割
    seq_names = list(all_sequences.keys())
    random.shuffle(seq_names)
    split_idx = int(len(seq_names) * TRAIN_VAL_SPLIT)
    train_seqs = seq_names[:split_idx]
    val_seqs = seq_names[split_idx:]

    train_labels = {s: all_sequences[s] for s in train_seqs}
    val_labels = {s: all_sequences[s] for s in val_seqs}

    # 保存标签
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump({
            "all": all_sequences,
            "train": train_labels,
            "val": val_labels,
        }, f, indent=2, ensure_ascii=False)

    print(f"Train: {len(train_labels)}, Val: {len(val_labels)}")
    print(f"标签已保存到 {labels_path}")


def build_from_clips(session_dir, output_dir=None):
    """从自动标注 clip 目录生成帧序列数据集.

    跳过 attack="unknown" 和 attack="noise" 的未聚类 clips。
    对每个 clip 做滑动窗口切片，输出与 build_dataset 一致的格式。

    Args:
        session_dir: session 目录路径 (包含 clip_XXXXX/ 子目录).
        output_dir: 输出目录, 默认 data/sequences/<session_name>.
    """
    session_name = os.path.basename(session_dir)
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, "sequences", session_name)
    os.makedirs(output_dir, exist_ok=True)

    all_sequences = {}
    seq_idx = 0
    clip_count = 0
    skipped = 0

    # 遍历所有 clip 目录
    clip_names = sorted(d for d in os.listdir(session_dir)
                        if d.startswith("clip_")
                        and os.path.isdir(os.path.join(session_dir, d)))

    for clip_name in clip_names:
        clip_dir = os.path.join(session_dir, clip_name)
        meta_path = os.path.join(clip_dir, "metadata.json")

        if not os.path.exists(meta_path):
            skipped += 1
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        attack = meta.get("attack", "unknown")
        if attack in ("unknown", "noise"):
            skipped += 1
            continue

        # 读取 clip 帧
        frames_dir = os.path.join(clip_dir, "frames")
        total_clip_frames = meta.get("total_frames", 0)

        frames = []
        for i in range(total_clip_frames):
            img_path = os.path.join(frames_dir, f"{i:03d}.jpg")
            if not os.path.exists(img_path):
                break
            img = cv2.imread(img_path)
            if img is None:
                break
            frames.append(img)

        if len(frames) < SEQ_LENGTH:
            # clip 帧数不足一个完整序列, 用填充处理
            if len(frames) == 0:
                skipped += 1
                continue
            # 重复最后一帧填充到 SEQ_LENGTH
            while len(frames) < SEQ_LENGTH:
                frames.append(frames[-1].copy())

        # 滑动窗口切片
        for start in range(0, len(frames) - SEQ_LENGTH + 1, SLIDE_STRIDE):
            seq_name = f"seq_{seq_idx:05d}"
            seq_dir = os.path.join(output_dir, seq_name)
            os.makedirs(seq_dir, exist_ok=True)

            for i in range(SEQ_LENGTH):
                frame = frames[start + i]
                resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                cv2.imwrite(os.path.join(seq_dir, f"{i:03d}.jpg"), resized)

            all_sequences[seq_name] = attack
            seq_idx += 1

        clip_count += 1

    print(f"处理 {clip_count} 个 clip (跳过 {skipped}), 生成 {seq_idx} 个序列")

    if not all_sequences:
        print("警告: 未生成任何序列")
        return

    # 统计类别分布
    class_counts = {}
    for label in all_sequences.values():
        class_counts[label] = class_counts.get(label, 0) + 1
    print("类别分布:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")

    # train/val 分割
    seq_names = list(all_sequences.keys())
    random.shuffle(seq_names)
    split_idx = int(len(seq_names) * TRAIN_VAL_SPLIT)
    train_seqs = seq_names[:split_idx]
    val_seqs = seq_names[split_idx:]

    train_labels = {s: all_sequences[s] for s in train_seqs}
    val_labels = {s: all_sequences[s] for s in val_seqs}

    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump({
            "all": all_sequences,
            "train": train_labels,
            "val": val_labels,
        }, f, indent=2, ensure_ascii=False)

    print(f"Train: {len(train_labels)}, Val: {len(val_labels)}")
    print(f"标签已保存到 {labels_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="标注 → 帧序列数据集")
    parser.add_argument("--mode", type=str, default="video",
                        choices=["video", "clips"],
                        help="构建模式: video (视频+标注) 或 clips (自动标注 clip)")
    parser.add_argument("--video", type=str, default=None,
                        help="[video 模式] 视频文件路径")
    parser.add_argument("--labels", type=str, default=None,
                        help="[video 模式] 标注 JSON 路径")
    parser.add_argument("--session", type=str, default=None,
                        help="[clips 模式] Session 目录路径")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出目录")
    args = parser.parse_args()

    if args.mode == "video":
        if not args.video or not args.labels:
            parser.error("video 模式需要 --video 和 --labels 参数")
        build_dataset(args.video, args.labels, args.output)
    elif args.mode == "clips":
        if not args.session:
            parser.error("clips 模式需要 --session 参数")
        build_from_clips(args.session, args.output)
