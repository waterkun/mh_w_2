"""簇命名工具 — 将 cluster_X 批量替换为实际攻击名.

使用方式:
  python data/rename_clusters.py --session <dir> \
    --mapping '{"cluster_0": "pounce", "cluster_1": "tail_sweep"}'
"""

import argparse
import json
import os
import sys


def rename_clusters(session_dir, mapping):
    """批量重命名 clip 的攻击类别.

    Args:
        session_dir: session 目录路径.
        mapping: {old_name: new_name} 字典, 如 {"cluster_0": "pounce"}.
    """
    renamed = 0
    skipped = 0

    for name in sorted(os.listdir(session_dir)):
        clip_dir = os.path.join(session_dir, name)
        if not os.path.isdir(clip_dir) or not name.startswith("clip_"):
            continue

        meta_path = os.path.join(clip_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        old_attack = meta.get("attack", "unknown")
        if old_attack in mapping:
            meta["attack"] = mapping[old_attack]
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            renamed += 1
        else:
            skipped += 1

    # 更新 labels.json
    labels_path = os.path.join(session_dir, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            labels = json.load(f)
        for clip_meta in labels.get("clips", []):
            old = clip_meta.get("attack", "unknown")
            if old in mapping:
                clip_meta["attack"] = mapping[old]
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"重命名完成: {renamed} 个 clip 已更新, {skipped} 个跳过")
    print(f"映射: {mapping}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="簇命名工具")
    parser.add_argument("--session", type=str, required=True,
                        help="Session 目录路径")
    parser.add_argument("--mapping", type=str, required=True,
                        help='JSON 映射, 如 \'{"cluster_0": "pounce"}\'')
    args = parser.parse_args()

    mapping = json.loads(args.mapping)
    rename_clusters(args.session, mapping)
