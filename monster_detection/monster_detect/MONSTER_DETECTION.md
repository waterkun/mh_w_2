# Monster Detection Module

YOLO-based monster body part detection for the MHW bot.
Detects monster body parts (body, head) on screen.

## Classes

| ID | Name | Description |
|----|------|-------------|
| 0  | body | Monster torso/main body |
| 1  | head | Monster head |

## Current Model Performance (2026-03-23)

Trained on 944 images (686 labeled + 258 negative), 1118 bboxes, 100 epochs YOLOv10n.

| Metric | All | body | head |
|--------|-----|------|------|
| Precision | 0.885 | 0.858 | 0.911 |
| Recall | 0.829 | 0.847 | 0.810 |
| mAP50 | 0.899 | 0.883 | 0.915 |
| mAP50-95 | 0.586 | 0.532 | 0.640 |

## Data Sources

| Batch | Images | Labeled | Source | Prefix |
|-------|--------|---------|--------|--------|
| batch1 | 270 | 141 | CVAT (r2_00401-r2_00670) | r2 |
| batch2 | 270 | 165 | CVAT (r2_00015-r2_00399) | r2 |
| batch3 | 404 | 380 | CVAT (r3_00189-r3_00592) | r3 |
| **Total** | **944** | **686** | | |

---

## Pipeline Overview

```
Record Video → Extract Frames → Upload to CVAT → Label → Export XML
    → Convert CVAT XML to YOLO txt → Split Dataset → Train
```

---

## Step 1: Extract Frames from Video

```bash
cd monster_detection
python -m monster_detect.extract_frames --video game_recording\1.mp4 --interval 2.0 --no-crop
python -m monster_detect.extract_frames --video game_recording\2.mp4 --interval 2.0 --no-crop --prefix r2
```

| Flag | Description |
|------|-------------|
| `--video` | Video file path (required) |
| `--interval` | Seconds between frames (default 0.5) |
| `--no-crop` | Save full screen at original resolution (no ROI crop) |
| `--prefix` | Filename prefix (default `frame`), use `r2`, `r3` etc. for different rounds |

Output: `monster_detect/monster_yolo_data/images/unlabeled/`

## Step 2: Label with CVAT

1. Upload frames to [CVAT](https://app.cvat.ai/)
2. Create task with labels: `body`, `head` (body has `back` checkbox attribute)
3. Draw bounding boxes
4. Export as **CVAT for images 1.1** (XML format)
5. Unzip to `monster_yolo_data/cvat_batchN_labels/annotations.xml`

## Step 3: Convert CVAT XML to YOLO

```bash
cd monster_detection

# Auto-detect all cvat_batch*_labels/annotations.xml
python -m monster_detect.convert_cvat_to_yolo

# Or specify XML files manually
python -m monster_detect.convert_cvat_to_yolo --xml path1.xml path2.xml
```

Reads CVAT XML, writes YOLO `.txt` labels to `images/unlabeled/`.

**Important:** If batch images are in `labeled_backup/images/` (not in `unlabeled/`), copy them first:
```bash
cp monster_detect/monster_yolo_data/labeled_backup/images/*.jpg monster_detect/monster_yolo_data/images/unlabeled/
```

## Step 4: Split Dataset

```bash
cd monster_detection
python -m monster_detect.split_dataset
python -m monster_detect.split_dataset --ratio 0.8 --neg-ratio 0.3
```

- 80/20 train/val split + optional negative samples
- Generates `monster_yolo_data/dataset.yaml`

## Step 5: Train

```bash
cd monster_detection
python -m monster_detect.train_monster_yolo --epochs 100 --batch 16
```

- Base model: YOLOv10n
- Output: `monster_detect/runs/monster_detect/weights/best.pt`

## Step 6: Prelabel New Data (Optional)

```bash
cd monster_detection
python -m monster_detect.prelabel
python -m monster_detect.prelabel --limit 200 --conf 0.2
```

---

## Quick Retrain from CVAT Annotations

```bash
cd monster_detection

# 1. Copy old batch images to unlabeled (if needed)
cp monster_detect/monster_yolo_data/labeled_backup_round2/images/*.jpg monster_detect/monster_yolo_data/images/unlabeled/

# 2. Convert all CVAT XML to YOLO labels
python -m monster_detect.convert_cvat_to_yolo

# 3. Split train/val
python -m monster_detect.split_dataset

# 4. Train
python -m monster_detect.train_monster_yolo --epochs 100 --batch 16
```

---

## Backup Structure

```
monster_yolo_data/
├── labeled_backup/                # Old round 1 backup (540 images only, labels deleted)
│   └── images/                    # 540 r2_* images
├── labeled_backup_round2/         # Full backup after CVAT relabeling (2026-03-23)
│   ├── images/                    # 944 images (r2 + r3)
│   ├── labels/                    # 945 YOLO .txt labels
│   ├── cvat_batch1_labels/        # CVAT XML (270 images, r2_00401-r2_00670)
│   ├── cvat_batch2_labels/        # CVAT XML (270 images, r2_00015-r2_00399)
│   └── cvat_batch3_labels/        # CVAT XML (404 images, r3_00189-r3_00592)
```

## File Structure

```
monster_detect/
├── __init__.py              # Exports MonsterDetector
├── monster_detector.py      # YOLO inference wrapper: predict(frame) → (bool, float)
├── extract_frames.py        # Video → frames
├── split_dataset.py         # Split train/val + generate dataset.yaml
├── train_monster_yolo.py    # YOLOv10n training
├── convert_cvat_to_yolo.py  # CVAT XML → YOLO txt
├── convert_voc_to_yolo.py   # PascalVOC XML → YOLO txt (legacy)
├── prelabel.py              # Auto-prelabel with trained model
├── MONSTER_DETECTION.md     # This file
├── monster_yolo_data/
│   ├── images/
│   │   ├── unlabeled/       # Frames + .txt labels (working dir)
│   │   ├── train/           # After split
│   │   └── val/             # After split
│   ├── labels/
│   │   ├── train/           # After split
│   │   └── val/             # After split
│   ├── cvat_batch*_labels/  # CVAT annotation XMLs
│   ├── labeled_backup/      # Old round 1 backup
│   ├── labeled_backup_round2/ # Full backup (2026-03-23)
│   └── dataset.yaml         # Generated by split_dataset.py
└── runs/
    └── monster_detect/
        └── weights/
            └── best.pt      # Trained model
```

## Changelog

### 2026-03-23: CVAT Relabeling + Retrain
- Relabeled all data using CVAT (3 batches: batch1, batch2, batch3)
- Reduced classes from 4 (body, chain, head, arm) to 2 (body, head)
- Batch 1+2: 540 r2_* images from old labeled_backup
- Batch 3: 404 r3_* images (new round)
- Total: 944 images, 686 labeled, 1118 bboxes
- Model: mAP50=0.899, Precision=0.885, Recall=0.829
- Backup saved to `labeled_backup_round2/`
