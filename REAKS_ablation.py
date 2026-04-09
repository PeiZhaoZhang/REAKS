import numpy as np
import shutil
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
import faiss
from tqdm import tqdm
from typing import List, Dict, Tuple
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import math
from pathlib import Path
import os
import random

os.environ['OMP_NUM_THREADS'] = '4'


class GSImageSelector:
    def __init__(self, device='cuda', feature_dim=128):
        self.device = device
        self.model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.model.classifier = torch.nn.Identity()
        self.model.eval().to(device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.Resize((256, 256)),
            transforms.GaussianBlur(3, sigma=(0.1, 1.0))
        ])

        self.pca = PCA(n_components=feature_dim)
        self.feature_dim = feature_dim

    def extract_enhanced_features(self, image_paths: List[str]) -> np.ndarray:
        hybrid_features = []
        for path in tqdm(image_paths, desc="Extracting features"):
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_pil = Image.fromarray(img)
            with torch.no_grad():
                deep_feat = self.model(self.transform(img_pil).unsqueeze(0).to(self.device))
                deep_feat = deep_feat.cpu().numpy().flatten()

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            lbp = local_binary_pattern(gray, 8, 1)
            hist = np.histogram(lbp, bins=32, range=(0, 256))[0]

            blur = cv2.Laplacian(gray, cv2.CV_64F).var()

            hybrid_feat = np.concatenate([
                deep_feat,
                hist / hist.sum(),
                [blur]
            ])
            hybrid_features.append(hybrid_feat)

        hybrid_features = np.array(hybrid_features)
        self.pca.fit(hybrid_features)
        return self.pca.transform(hybrid_features)

    def build_strict_similarity_graph(self, features: np.ndarray, alpha: float = 0.1,
                                      percentile: float = 90, use_dynamic_threshold: bool = True, fixed_threshold: float = 0.85) -> Dict[int, List[Tuple[int, float]]]:
        """
        消融支持：如果 use_dynamic_threshold 为 False，则使用固定的 fixed_threshold
        """
        n_samples = features.shape[0]
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(features.shape[1])
        index.add(features)

        k = max(5, int(np.sqrt(n_samples)))
        similarities, indices = index.search(features, k)

        if use_dynamic_threshold:
            nonzero_sims = similarities[similarities > -np.inf].flatten()
            q_percentile = np.percentile(nonzero_sims, percentile)
            threshold = q_percentile * alpha
        else:
            threshold = fixed_threshold # 消融 DQ-SMC 时使用固定阈值

        graph = {}
        for i in range(len(features)):
            graph[i] = [
                (j, sim) for j, sim in zip(indices[i], similarities[i]) if sim > threshold
            ]
        return graph

    def kmeans_clustering(self, features: np.ndarray) -> List[List[int]]:
        """消融 VASC 时使用的基线聚类方法 (标准的 K-Means)"""
        n_samples = features.shape[0]
        # 经验法则：簇数定为样本数的平方根
        n_clusters = max(2, int(math.sqrt(n_samples))) 
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        
        cluster_dict = {}
        for idx, label in enumerate(labels):
            cluster_dict.setdefault(label, []).append(idx)
        return [c for c in cluster_dict.values() if len(c) >= 2]

    def strict_clustering(self, graph: Dict[int, List[Tuple[int, float]]]) -> List[List[int]]:
        """原本的 VASC: 严格谱聚类 - 使用交叉验证确定最佳簇数"""
        n = len(graph)
        adj_matrix = np.zeros((n, n))
        for i in graph:
            for j, sim in graph[i]:
                adj_matrix[i, j] = sim
                adj_matrix[j, i] = sim

        min_clusters = 1
        max_clusters = min(100, n)
        cluster_range = list(range(min_clusters, max_clusters + 1))

        if n < min_clusters * 2:
            n_clusters = min(100, max(10, math.ceil(math.sqrt(n))))
        else:
            indices = np.arange(n)
            np.random.seed(42) # 固定种子保证复现
            np.random.shuffle(indices)
            split_idx = int(n * 0.8)
            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]

            if len(val_idx) < 5:
                train_idx = indices[5:]
                val_idx = indices[:5]

            train_matrix = adj_matrix[np.ix_(train_idx, train_idx)]
            val_matrix = adj_matrix[np.ix_(val_idx, train_idx)]

            best_score = -np.inf
            best_k = min_clusters

            for k in cluster_range:
                try:
                    clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
                    train_labels = clustering.fit_predict(train_matrix)

                    val_labels = []
                    for i in range(len(val_idx)):
                        cluster_similarities = []
                        for c in range(k):
                            cluster_indices = np.where(train_labels == c)[0]
                            if len(cluster_indices) == 0:
                                cluster_similarities.append(-np.inf)
                                continue
                            avg_sim = np.mean(val_matrix[i, cluster_indices])
                            cluster_similarities.append(avg_sim)
                        val_labels.append(np.argmax(cluster_similarities))

                    if len(set(val_labels)) > 1:
                        val_adj = adj_matrix[np.ix_(val_idx, val_idx)]
                        silhouette_avg = self._calculate_silhouette(val_adj, val_labels)

                        if silhouette_avg > best_score:
                            best_score = silhouette_avg
                            best_k = k
                except Exception:
                    continue

            n_clusters = best_k

        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        labels = clustering.fit_predict(adj_matrix)

        cluster_dict = {}
        for idx, label in enumerate(labels):
            cluster_dict.setdefault(label, []).append(idx)

        return [c for c in cluster_dict.values() if len(c) >= 2]

    def _calculate_silhouette(self, adj_matrix, labels):
        n = len(labels)
        if n <= 1: return 0
        dist_matrix = (1 - adj_matrix) / 2
        silhouette_sum, valid_samples = 0, 0

        for i in range(n):
            label_i = labels[i]
            same_cluster = np.where(labels == label_i)[0]
            if len(same_cluster) <= 1: continue

            same_indices = same_cluster[same_cluster != i]
            if len(same_indices) == 0: continue
            a_i = np.mean(dist_matrix[i, same_indices])

            min_b_i = np.inf
            other_labels = set(labels) - {label_i}
            for other_label in other_labels:
                other_indices = np.where(labels == other_label)[0]
                if len(other_indices) == 0: continue
                b_i_other = np.mean(dist_matrix[i, other_indices])
                if b_i_other < min_b_i:
                    min_b_i = b_i_other

            if min_b_i == np.inf: continue
            s_i = (min_b_i - a_i) / max(a_i, min_b_i)
            silhouette_sum += s_i
            valid_samples += 1

        return silhouette_sum / valid_samples if valid_samples > 0 else 0

    def select_keyframes(self, clusters: List[List[int]], features: np.ndarray,
                         target_keep_ratio: float = 0.5, min_keep_num: int = 3,
                         max_keep_percentile: float = 0.9, variance_influence: float = 0.3,
                         use_pridiff: bool = True) -> List[int]:
        """
        消融支持：如果 use_pridiff 为 False，则在簇内随机选择，而不是差异最大化。
        """
        total_images = sum(len(c) for c in clusters)
        target_keep_count = max(min_keep_num * len(clusters), int(total_images * target_keep_ratio))

        cluster_variances = []
        for cluster in clusters:
            if len(cluster) < 2:
                cluster_variances.append(0)
                continue
            variance = np.var(features[cluster], axis=0).mean()
            cluster_variances.append(variance)

        if cluster_variances and max(cluster_variances) > 0:
            max_variance = max(cluster_variances)
            normalized_variances = [v / max_variance for v in cluster_variances]
        else:
            normalized_variances = [0.5] * len(clusters)

        initial_ratios = []
        for i, cluster in enumerate(clusters):
            base_ratio = target_keep_ratio * (1 + variance_influence * (normalized_variances[i] - 0.5))
            base_ratio = min(max_keep_percentile, max(min_keep_num / len(cluster), base_ratio))
            initial_ratios.append(base_ratio)

        keep_counts = [max(min_keep_num, int(round(len(c) * r))) for c, r in zip(clusters, initial_ratios)]
        current_total = sum(keep_counts)

        if current_total != target_keep_count:
            delta = target_keep_count - current_total
            if delta != 0:
                priorities = [(normalized_variances[i] * len(cluster), i) for i, cluster in enumerate(clusters)]
                priorities.sort(reverse=(delta > 0))

                for _, i in priorities:
                    if delta == 0: break
                    cluster_size = len(clusters[i])
                    current_keep = keep_counts[i]
                    if delta > 0:
                        max_possible = min(cluster_size, int(cluster_size * max_keep_percentile))
                        if current_keep < max_possible:
                            increment = min(delta, max_possible - current_keep)
                            keep_counts[i] += increment
                            delta -= increment
                    else:
                        if current_keep > min_keep_num:
                            decrement = min(-delta, current_keep - min_keep_num)
                            keep_counts[i] -= decrement
                            delta += decrement

        selected = []
        for i, cluster in enumerate(clusters):
            keep_num = keep_counts[i]
            if len(cluster) <= keep_num:
                selected.extend(cluster)
                continue

            if not use_pridiff:
                # 消融 PriDiff: 簇内直接随机抽取
                random.seed(42)
                selected.extend(random.sample(cluster, keep_num))
                continue

            # 完整的 PriDiff-Select
            cluster_feats = features[cluster]
            center = np.mean(cluster_feats, axis=0)
            distances = np.linalg.norm(cluster_feats - center, axis=1)
            center_idx = np.argmin(distances)
            selected_in_cluster = [center_idx]

            if keep_num == 1:
                selected.append(cluster[center_idx])
                continue

            sim_matrix = cluster_feats @ cluster_feats.T
            remaining_indices = [j for j in range(len(cluster)) if j != center_idx]

            for _ in range(keep_num - 1):
                if not remaining_indices: break
                min_similarities = [
                    np.min([sim_matrix[idx, s] for s in selected_in_cluster])
                    for idx in remaining_indices
                ]
                best_idx = remaining_indices[np.argmin(min_similarities)]
                selected_in_cluster.append(best_idx)
                remaining_indices.remove(best_idx)

            selected.extend([cluster[j] for j in selected_in_cluster])

        return list(set(selected))

    def rename_selected_images(self, selected_paths: List[str], output_dir: str, start_index: int = 0):
        os.makedirs(output_dir, exist_ok=True)
        sorted_paths = sorted(selected_paths, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x).split('.')[0]))))
        for new_idx, src_path in enumerate(sorted_paths):
            try:
                file_ext = os.path.splitext(src_path)[1]
                new_name = f"{new_idx + start_index:04d}{file_ext}"
                shutil.copy2(src_path, os.path.join(output_dir, new_name))
            except Exception as e:
                pass

    def save_selected_images(self, selected_paths: List[str], output_dir: str, rename: bool = True):
        # 简化版保存：在消融实验中，直接保存改名后的图片即可，节约硬盘
        images_dir = Path(output_dir) / "images"
        os.makedirs(images_dir, exist_ok=True)
        if rename:
            self.rename_selected_images(selected_paths, images_dir)
        else:
            for src_path in selected_paths:
                shutil.copy2(src_path, os.path.join(images_dir, os.path.basename(src_path)))

    def run_ablation_experiment(self, image_paths: List[str], base_output_dir: str, retention_ratio: float = 0.5):
        """自动运行完整的架构消融实验"""
        print(f"提取特征中... (共 {len(image_paths)} 张图像)")
        features = self.extract_enhanced_features(image_paths)
        target_count = int(len(image_paths) * retention_ratio)

        experiments = [
            {"name": "1_Baseline_Random", "dq_smc": None, "vasc": None, "pridiff": None},
            {"name": "2_wo_DQ_SMC", "dq_smc": False, "vasc": True, "pridiff": True},
            {"name": "3_wo_VASC", "dq_smc": True, "vasc": False, "pridiff": True},
            {"name": "4_wo_PriDiff", "dq_smc": True, "vasc": True, "pridiff": False},
            {"name": "5_Full_REAKS", "dq_smc": True, "vasc": True, "pridiff": True}
        ]

        for exp in experiments:
            name = exp["name"]
            print(f"\n[{name}] 开始处理...")
            output_dir = os.path.join(base_output_dir, name)
            os.makedirs(output_dir, exist_ok=True)

            if exp["dq_smc"] is None:
                # Baseline: 纯随机
                random.seed(42)
                selected = random.sample(range(len(image_paths)), target_count)
            else:
                # 运行包含不同配置的管线
                graph = self.build_strict_similarity_graph(
                    features, use_dynamic_threshold=exp["dq_smc"], fixed_threshold=0.85
                )
                
                if exp["vasc"]:
                    clusters = self.strict_clustering(graph)
                else:
                    clusters = self.kmeans_clustering(features)
                    
                selected = self.select_keyframes(
                    clusters, features, target_keep_ratio=retention_ratio, use_pridiff=exp["pridiff"]
                )

            selected_paths = [image_paths[i] for i in selected]
            
            # 为了确保数量精确一致，如果最后挑出来的数量和 target_count 差一点，做截断或补充
            if len(selected_paths) > target_count:
                selected_paths = selected_paths[:target_count]
            elif len(selected_paths) < target_count:
                remaining = list(set(image_paths) - set(selected_paths))
                selected_paths.extend(random.sample(remaining, target_count - len(selected_paths)))

            print(f"[{name}] 筛选完成: {len(selected_paths)} 张")
            self.save_selected_images(selected_paths, output_dir, rename=False) # 为了 COLMAP 读取方便，消融实验通常保留原文件名
            print(f"[{name}] 数据已保存至: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated REAKS Ablation Runner')
    parser.add_argument('-s', '--source_path', required=True, help='Input directory containing images (e.g., .../truck/images)')
    parser.add_argument('-m', '--output_path', required=True, help='Base output directory for ablation datasets')
    parser.add_argument('-r', '--retention_ratio', type=float, default=0.5, help='Target retention ratio (default=0.5 for ablation)')
    args = parser.parse_args()

    image_dir = args.source_path
    base_output_dir = args.output_path
    retention_ratio = args.retention_ratio

    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg'))
    ])

    if not image_paths:
        print("错误：未找到PNG或JPG图片！")
        exit()

    print(f"====== 开始自动化消融实验生成 ======")
    print(f"输入路径: {image_dir}")
    print(f"输出路径: {base_output_dir}")
    print(f"目标保留率: {retention_ratio * 100}%")
    
    selector = GSImageSelector(device='cuda')
    # 这一行会自动跑完5个变体！
    selector.run_ablation_experiment(image_paths, base_output_dir, retention_ratio)
    print("\n====== 全部生成完毕！可以去跑 3DGS 了！ ======")
    # python ablation_runner.py -s data/truck -m data/ -r 0.5