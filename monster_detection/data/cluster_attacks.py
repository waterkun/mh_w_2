"""攻击 clip 聚类工具 — CNN 特征提取 + DBSCAN/KMeans + t-SNE 可视化.

使用方式:
  python data/cluster_attacks.py --session <session_dir> --method dbscan
  python data/cluster_attacks.py --session <session_dir> --method kmeans --k 7
  python data/cluster_attacks.py --all  # 聚类所有 sessions
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    AUTO_LABEL_DIR, INPUT_SIZE, BACKBONE_FEAT_DIM,
    CLUSTER_METHOD, KMEANS_K, DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
)


class AttackClusterer:
    """对攻击 clip 进行特征提取和聚类分析."""

    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._build_feature_extractor()

    def _build_feature_extractor(self):
        """加载预训练 MobileNetV3-Small 作为特征提取器."""
        backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        # 去掉分类头, 保留到 adaptive avg pool
        self.feature_net = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
        ).to(self.device)
        self.feature_net.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, clip_dirs):
        """对每个 clip 提取 CNN 特征向量.

        对 hit_frame 附近 ±5 帧提取特征并取平均。

        Args:
            clip_dirs: clip 目录路径列表.

        Returns:
            features: (N, feat_dim) numpy 数组
            clip_ids: clip 名称列表
        """
        all_features = []
        clip_ids = []

        for clip_dir in clip_dirs:
            meta_path = os.path.join(clip_dir, "metadata.json")
            if not os.path.exists(meta_path):
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            frames_dir = os.path.join(clip_dir, "frames")
            hit_idx = meta.get("hit_frame_idx", 0)
            total = meta.get("total_frames", 0)

            # 取 hit_frame ± 5 帧范围
            start = max(0, hit_idx - 5)
            end = min(total, hit_idx + 6)

            frame_features = []
            for i in range(start, end):
                img_path = os.path.join(frames_dir, f"{i:03d}.jpg")
                if not os.path.exists(img_path):
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.feature_net(tensor)
                frame_features.append(feat.cpu().numpy().flatten())

            if frame_features:
                avg_feat = np.mean(frame_features, axis=0)
                all_features.append(avg_feat)
                clip_ids.append(os.path.basename(clip_dir))

        if not all_features:
            return np.array([]), []

        return np.array(all_features), clip_ids

    def cluster(self, features, method=None, k=None, eps=None,
                min_samples=None):
        """对特征向量进行聚类.

        Args:
            features: (N, feat_dim) numpy 数组.
            method: "dbscan" 或 "kmeans".
            k: KMeans 的簇数.
            eps: DBSCAN 的 epsilon.
            min_samples: DBSCAN 的 min_samples.

        Returns:
            labels: (N,) 聚类标签数组.
        """
        from sklearn.preprocessing import StandardScaler

        method = method or CLUSTER_METHOD
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if method == "dbscan":
            from sklearn.cluster import DBSCAN
            eps = eps or DBSCAN_EPS
            min_samples = min_samples or DBSCAN_MIN_SAMPLES
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(features_scaled)
        elif method == "kmeans":
            from sklearn.cluster import KMeans
            k = k or KMEANS_K
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(features_scaled)
        else:
            raise ValueError(f"未知聚类方法: {method}")

        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()
        print(f"聚类结果: {n_clusters} 簇, {n_noise} 噪声点")

        return labels

    def visualize(self, features, labels, clip_ids, output_path):
        """t-SNE 降维 + 散点图可视化.

        Args:
            features: (N, feat_dim) numpy 数组.
            labels: (N,) 聚类标签.
            clip_ids: clip 名称列表.
            output_path: 输出图片路径.
        """
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(features) < 2:
            print("样本太少，无法可视化")
            return

        perplexity = min(30, len(features) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = tsne.fit_transform(features)

        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab10", max(len(unique_labels), 1))

        plt.figure(figsize=(12, 8))
        for label in unique_labels:
            mask = labels == label
            if label == -1:
                plt.scatter(coords[mask, 0], coords[mask, 1],
                            c="gray", marker="x", s=30, alpha=0.5,
                            label="noise")
            else:
                plt.scatter(coords[mask, 0], coords[mask, 1],
                            c=[cmap(label)], s=50, alpha=0.7,
                            label=f"cluster_{label}")

        plt.legend(loc="best", fontsize=8)
        plt.title(f"Attack Clips Clustering ({len(features)} clips, "
                  f"{len(unique_labels)} clusters)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"可视化已保存: {output_path}")


def _find_clip_dirs(session_dir):
    """列出 session 下所有 clip 目录."""
    clip_dirs = []
    for name in sorted(os.listdir(session_dir)):
        d = os.path.join(session_dir, name)
        if os.path.isdir(d) and name.startswith("clip_"):
            clip_dirs.append(d)
    return clip_dirs


def _update_clip_labels(clip_dirs, labels, clip_ids):
    """更新每个 clip 的 metadata.json 中的 attack 字段."""
    id_to_label = dict(zip(clip_ids, labels))
    for clip_dir in clip_dirs:
        clip_name = os.path.basename(clip_dir)
        if clip_name not in id_to_label:
            continue
        meta_path = os.path.join(clip_dir, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        cluster_label = int(id_to_label[clip_name])
        meta["attack"] = f"cluster_{cluster_label}" if cluster_label >= 0 else "noise"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def cluster_session(session_dir, method=None, k=None, eps=None,
                    min_samples=None):
    """对一个 session 进行完整聚类流程."""
    print(f"聚类 session: {session_dir}")

    clip_dirs = _find_clip_dirs(session_dir)
    if not clip_dirs:
        print("  未找到 clip 目录")
        return

    print(f"  找到 {len(clip_dirs)} 个 clip")

    clusterer = AttackClusterer()
    features, clip_ids = clusterer.extract_features(clip_dirs)

    if len(features) == 0:
        print("  未提取到特征")
        return

    labels = clusterer.cluster(features, method=method, k=k, eps=eps,
                               min_samples=min_samples)

    # 更新 metadata
    _update_clip_labels(clip_dirs, labels, clip_ids)

    # 可视化
    viz_path = os.path.join(session_dir, "clusters_visualization.png")
    clusterer.visualize(features, labels, clip_ids, viz_path)

    # 打印簇分布
    print("\n簇分布:")
    for lbl in sorted(set(labels)):
        count = (labels == lbl).sum()
        name = f"cluster_{lbl}" if lbl >= 0 else "noise"
        print(f"  {name}: {count} clips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="攻击 clip 聚类分析")
    parser.add_argument("--session", type=str, default=None,
                        help="Session 目录路径")
    parser.add_argument("--all", action="store_true",
                        help="聚类所有 sessions")
    parser.add_argument("--method", type=str, default=None,
                        choices=["dbscan", "kmeans"],
                        help="聚类方法 (默认用 config 设置)")
    parser.add_argument("--k", type=int, default=None,
                        help="KMeans 簇数")
    parser.add_argument("--eps", type=float, default=None,
                        help="DBSCAN epsilon")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="DBSCAN min_samples")
    args = parser.parse_args()

    if args.all:
        if not os.path.exists(AUTO_LABEL_DIR):
            print(f"错误: {AUTO_LABEL_DIR} 不存在")
            sys.exit(1)
        for name in sorted(os.listdir(AUTO_LABEL_DIR)):
            session_path = os.path.join(AUTO_LABEL_DIR, name)
            if os.path.isdir(session_path) and name.startswith("session_"):
                cluster_session(session_path, method=args.method, k=args.k,
                                eps=args.eps, min_samples=args.min_samples)
    elif args.session:
        cluster_session(args.session, method=args.method, k=args.k,
                        eps=args.eps, min_samples=args.min_samples)
    else:
        print("请指定 --session <dir> 或 --all")
