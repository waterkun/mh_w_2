"""Migrate labels.csv from old margin-based percentages to new full-width percentages.

Old formula: pct = (click_x - bar_start) / bar_width   (with clamp 0~1)
New formula: pct = click_x / roi_width                  (no margins)

Conversion: click_x = old_pct * bar_width + bar_start
            new_pct = click_x / roi_width

Frames where old health_pct == 0 (clamped) cannot be recovered and are removed.
"""

import csv
import os
import shutil

# Old margin constants
LEFT_MARGIN_FRAC = 0.06
RIGHT_MARGIN_FRAC = 0.12
ROI_W = 728  # current ROI width


def migrate(labels_path):
    bar_start = ROI_W * LEFT_MARGIN_FRAC
    bar_end = ROI_W * (1.0 - RIGHT_MARGIN_FRAC)
    bar_width = bar_end - bar_start

    rows = []
    removed = []
    with open(labels_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            old_hp = float(row["health_pct"])
            old_dmg = float(row["damage_pct"])

            # Cannot recover clamped-to-zero entries
            if old_hp == 0.0 and old_dmg > 0.0:
                removed.append(fname)
                continue

            # Reverse old formula to get original pixel position
            hp_x = old_hp * bar_width + bar_start
            new_hp = hp_x / ROI_W

            # Damage boundary was at health + damage in old system
            dmg_x = (old_hp + old_dmg) * bar_width + bar_start
            new_dmg = dmg_x / ROI_W - new_hp

            new_hp = max(0.0, min(1.0, new_hp))
            new_dmg = max(0.0, min(1.0 - new_hp, new_dmg))

            rows.append({
                "filename": fname,
                "health_pct": f"{new_hp:.4f}",
                "damage_pct": f"{new_dmg:.4f}",
            })

    # Backup original
    backup = labels_path + ".bak"
    shutil.copy2(labels_path, backup)
    print(f"Backup saved to {backup}")

    # Write migrated labels
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "health_pct", "damage_pct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Migrated {len(rows)} labels")
    if removed:
        print(f"Removed {len(removed)} unrecoverable frames (health_pct=0 clamped):")
        for fname in removed:
            print(f"  {fname}")
        print("Please re-label these frames.")


if __name__ == "__main__":
    labels_path = os.path.join("health_bar_data", "labels.csv")
    migrate(labels_path)
