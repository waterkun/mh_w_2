"""Prepare training data: copy labeled ROIs and labels into a clean backup folder.

Only includes frames that have actual labels in labels.csv.

Usage:
    python prepare_training_data.py [--data health_bar_data] [--output health_bar_training]
"""

import argparse
import csv
import os
import shutil


def prepare(data_dir, output_dir):
    labels_path = os.path.join(data_dir, "labels.csv")
    rois_dir = os.path.join(data_dir, "rois")

    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} not found")
        return

    # Read all labels
    rows = []
    with open(labels_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Total entries in labels.csv: {len(rows)}")

    # Filter: only keep entries where the ROI image exists
    valid = []
    missing_img = 0
    for row in rows:
        roi_path = os.path.join(rois_dir, row["filename"])
        if os.path.exists(roi_path):
            valid.append(row)
        else:
            missing_img += 1

    if missing_img:
        print(f"Skipped {missing_img} entries (ROI image missing)")

    print(f"Valid labeled frames: {len(valid)}")

    # Create output structure
    out_rois = os.path.join(output_dir, "rois")
    os.makedirs(out_rois, exist_ok=True)

    # Copy ROI images
    for row in valid:
        src = os.path.join(rois_dir, row["filename"])
        dst = os.path.join(out_rois, row["filename"])
        shutil.copy2(src, dst)

    # Write clean labels.csv
    out_labels = os.path.join(output_dir, "labels.csv")
    with open(out_labels, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "health_pct", "damage_pct"])
        writer.writeheader()
        writer.writerows(valid)

    print(f"\nSaved to {output_dir}/:")
    print(f"  {len(valid)} ROI images in rois/")
    print(f"  labels.csv with {len(valid)} entries")


def main():
    parser = argparse.ArgumentParser(description="Prepare clean training data")
    parser.add_argument("--data", default="health_bar_data",
                        help="Source data directory")
    parser.add_argument("--output", default="health_bar_training",
                        help="Output directory for clean training data")
    args = parser.parse_args()
    prepare(args.data, args.output)


if __name__ == "__main__":
    main()
