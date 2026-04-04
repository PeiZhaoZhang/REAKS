import numpy as np
import shutil
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
import faiss
from tqdm import tqdm
from typing import List, Dict, Tuple
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import math
from pathlib import Path  # 新增：导入Path类
import os

os.environ['OMP_NUM_THREADS'] = '4'  # 设置线程数为1


class GSImageSelector:
    def __init__(self, device='cuda', feature_dim=128):  # 默认使用cuda，降维后的特征维度为128
        self.device = device

        # 使用轻量级模型组合
        self.model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.model.classifier = torch.nn.Identity()  # 移除分类头，只保留特征提取部分
        self.model.eval().to(device)

        # 图像预处理（增强内窥镜特征）：转换为张量，归一化，调整大小和高斯模糊
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.Resize((256, 256)),
            transforms.GaussianBlur(3, sigma=(0.1, 1.0))
        ])

        # 特征处理：初始化PCA对象，用于降维
        self.pca = PCA(n_components=feature_dim)
        self.feature_dim = feature_dim

    def extract_enhanced_features(self, image_paths: List[str]) -> np.ndarray:
        """提取结合深度特征和手工特征的混合特征"""
        hybrid_features = []
        # 遍历所有图像路径，读取图像
        for path in tqdm(image_paths, desc="Extracting features"):
            img = cv2.imread(path)
            if img is None:
                continue

            # 预处理：颜色空间转换和裁剪
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            img = img

            # 深度特征，使用预训练模型提取黏膜纹理的抽象特征
            img_pil = Image.fromarray(img)
            with torch.no_grad():
                deep_feat = self.model(self.transform(img_pil).unsqueeze(0).to(self.device))
                deep_feat = deep_feat.cpu().numpy().flatten()

            # 手工特征 (LBP纹理，局部二值模式)，量化组织表面微观结构变化的特征
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            lbp = local_binary_pattern(gray, 8, 1)
            hist = np.histogram(lbp, bins=32, range=(0, 256))[0]

            # 运动模糊特征，使用拉普拉斯算子识别运动模糊导致的无效帧
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 特征组合，将三种特征组合成混合特征，解决组织反光导致的特征失真
            hybrid_feat = np.concatenate([
                deep_feat,
                hist / hist.sum(),
                [blur]
            ])
            hybrid_features.append(hybrid_feat)

        # 使用PCA将混合特征降维到128维
        hybrid_features = np.array(hybrid_features)
        self.pca.fit(hybrid_features)
        return self.pca.transform(hybrid_features)

    def build_strict_similarity_graph(self, features: np.ndarray, alpha: float = 0.1,
                                      percentile: float = 90) -> Dict[int, List[Tuple[int, float]]]:
        """构建带权重的相似度图（相似度大于 分位数 × α）"""
        n_samples = features.shape[0]  # 新增：计算总图片数量
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(features.shape[1])
        index.add(features)

        k = max(5, int(np.sqrt(n_samples)))

        print(f"  总图片数量: {n_samples}")
        print(f"  动态近邻数 k = {k} (计算方式: max(5, √{n_samples:.1f}))")

        similarities, indices = index.search(features, k)

        # 计算相似度集合的 p 分位数
        nonzero_sims = similarities[similarities > -np.inf].flatten()
        q_percentile = np.percentile(nonzero_sims, percentile)

        # 动态阈值 = 分位数 × α
        threshold = q_percentile * 0.1

        graph = {}
        for i in range(len(features)):
            graph[i] = [
                (j, sim) for j, sim in zip(indices[i], similarities[i]) if sim > threshold
            ]
        return graph

    def strict_clustering(self, graph: Dict[int, List[Tuple[int, float]]]) -> List[List[int]]:
        """严格谱聚类 - 使用交叉验证确定最佳簇数"""
        # 根据相似度图，构建相似度邻接矩阵
        n = len(graph)
        adj_matrix = np.zeros((n, n))
        for i in graph:
            for j, sim in graph[i]:  # 假设graph存储相似度值
                adj_matrix[i, j] = sim
                adj_matrix[j, i] = sim  # 对称矩阵

        # 定义尝试的簇数范围
        min_clusters = 1
        max_clusters = min(100, n)  # 最大簇数不超过样本数
        cluster_range = list(range(min_clusters, max_clusters + 1))

        # 如果样本数太少，直接使用平方根取整作为簇数
        if n < min_clusters * 2:
            n_clusters = min(100, max(10, math.ceil(math.sqrt(n))))
            best_score = -np.inf
        else:
            # 分割训练集和验证集（使用留一法思想）
            indices = np.arange(n)
            np.random.shuffle(indices)
            split_idx = int(n * 0.8)  # 80%训练，20%验证
            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]

            # 确保验证集足够大
            if len(val_idx) < 5:
                train_idx = indices[5:]
                val_idx = indices[:5]

            # 训练集和验证集的邻接矩阵
            train_matrix = adj_matrix[np.ix_(train_idx, train_idx)]
            val_matrix = adj_matrix[np.ix_(val_idx, train_idx)]  # 验证集到训练集的连接

            best_score = -np.inf
            best_k = min_clusters

            # 交叉验证：尝试不同的簇数
            for k in cluster_range:
                try:
                    # 在训练集上聚类
                    clustering = SpectralClustering(
                        n_clusters=k,
                        affinity='precomputed',
                        random_state=42
                    )
                    train_labels = clustering.fit_predict(train_matrix)

                    # 为验证集样本分配簇（基于最近邻）
                    val_labels = []
                    for i in range(len(val_idx)):
                        # 计算与每个训练集簇的相似度
                        cluster_similarities = []
                        for c in range(k):
                            cluster_indices = np.where(train_labels == c)[0]
                            if len(cluster_indices) == 0:
                                cluster_similarities.append(-np.inf)
                                continue
                            # 计算验证样本与簇内样本的平均相似度
                            avg_sim = np.mean(val_matrix[i, cluster_indices])
                            cluster_similarities.append(avg_sim)
                        # 分配到相似度最高的簇
                        val_labels.append(np.argmax(cluster_similarities))

                    # 计算验证集的轮廓系数（作为评估指标）
                    if len(set(val_labels)) > 1:  # 确保有多个簇
                        # 构建验证集的邻接子矩阵
                        val_adj = adj_matrix[np.ix_(val_idx, val_idx)]
                        # 计算验证集的轮廓系数
                        silhouette_avg = self._calculate_silhouette(val_adj, val_labels)

                        if silhouette_avg > best_score:
                            best_score = silhouette_avg
                            best_k = k
                except Exception as e:
                    print(f"簇数 {k} 聚类失败: {str(e)}")
                    continue

            n_clusters = best_k
            print(f"交叉验证选择的最佳簇数: {n_clusters}, 验证集轮廓系数: {best_score:.4f}")

        # 使用最佳簇数对整个数据集进行聚类
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        labels = clustering.fit_predict(adj_matrix)

        # 过滤样本数小于3的小簇
        cluster_dict = {}
        for idx, label in enumerate(labels):
            cluster_dict.setdefault(label, []).append(idx)

        # 返回所有非空簇
        return [c for c in cluster_dict.values() if len(c) >= 2]  # 最小簇大小=2

    def _calculate_silhouette(self, adj_matrix, labels):
        n = len(labels)
        if n <= 1:
            return 0

        # 余弦相似度范围[-1,1]，转换为距离（0-1，值越大越不相似）
        dist_matrix = (1 - adj_matrix) / 2  # 最大值1（完全不相似），最小值0（完全相似）

        silhouette_sum = 0
        valid_samples = 0

        for i in range(n):
            label_i = labels[i]
            same_cluster = np.where(labels == label_i)[0]
            if len(same_cluster) <= 1:
                continue

            same_indices = same_cluster[same_cluster != i]
            if len(same_indices) == 0:
                continue

            a_i = np.mean(dist_matrix[i, same_indices])

            min_b_i = np.inf
            other_labels = set(labels) - {label_i}
            for other_label in other_labels:
                other_indices = np.where(labels == other_label)[0]
                if len(other_indices) == 0:
                    continue
                b_i_other = np.mean(dist_matrix[i, other_indices])
                if b_i_other < min_b_i:
                    min_b_i = b_i_other

            if min_b_i == np.inf:
                continue

            s_i = (min_b_i - a_i) / max(a_i, min_b_i)
            silhouette_sum += s_i
            valid_samples += 1

        return silhouette_sum / valid_samples if valid_samples > 0 else 0

    '''def select_keyframes(self, clusters: List[List[int]],
                         features: np.ndarray,
                         base_keep_ratio: float = 0.7,  # 降低基础保留比例
                         max_keep_ratio: float = 0.9,
                         min_keep_num: int = 3,
                         variance_weight: float = 0.8,  # 增大方差影响权重。0.8->201，0.6->204
                         full_retention_threshold: float = 0.95  # 提高全保留阈值
                         ) -> List[int]:
        """
        完全动态化的关键帧选择算法
        """
        selected = []
        all_variances = [np.var(features[cluster], axis=0).mean() for cluster in clusters if len(cluster) > 0]
        max_variance = max(all_variances) if all_variances else 1.0  # 避免除零错误
        i = 1
        for cluster in clusters:
            cluster_size = len(cluster)
            if cluster_size == 0:
                continue
            print(f"簇 {i}: {cluster_size} 张图片")
            i = i + 1
            cluster_feats = features[cluster]
            # 计算簇内样本的方差
            variance = np.var(cluster_feats, axis=0).mean()
            # 归一化方差
            normalized_variance = variance / max_variance
            normalized_variance = normalized_variance * 0.2
            # 使用反比函数的变体,更平滑的函数
            base_ratio = base_keep_ratio / (1 + np.exp(-0.1 * (cluster_size - 50)))
            # base_ratio = base_keep_ratio * (1 + np.log10(10 / cluster_size))
            print(f"基础保留比例: {base_ratio:.4f}")
            keep_ratio = max(base_ratio, base_keep_ratio)
            keep_ratio = keep_ratio + variance_weight * normalized_variance
            print(f" 方差影响调整后的保留比例: {keep_ratio:.4f}")

            keep_ratio = min(keep_ratio, max_keep_ratio)
            keep_ratio = max(keep_ratio, min_keep_num / cluster_size)
            print(f" 最终保留比例: {keep_ratio:.4f}")
            keep_num = max(min_keep_num, int(round(cluster_size * keep_ratio)))
            print(f" 保留数量为: {keep_num}")
            if cluster_size <= 3:
                selected.extend(cluster)
                continue
            elif keep_num >= cluster_size * full_retention_threshold:
                selected.extend(cluster)
                continue

            center = np.mean(cluster_feats, axis=0)
            distances = np.linalg.norm(cluster_feats - center, axis=1)
            selected_in_cluster = [np.argmin(distances)]

            if keep_num == 1:
                selected.append(cluster[selected_in_cluster[0]])
                continue

            sim_matrix = cluster_feats @ cluster_feats.T
            remaining_indices = [i for i in range(cluster_size) if i not in selected_in_cluster]

            for _ in range(keep_num - 1):
                if not remaining_indices:
                    break
                min_similarities = [np.min([sim_matrix[idx, s] for s in selected_in_cluster]) for idx in
                                    remaining_indices]
                best_idx = remaining_indices[np.argmin(min_similarities)]
                selected_in_cluster.append(best_idx)
                remaining_indices.remove(best_idx)

            selected.extend([cluster[i] for i in selected_in_cluster])

        return list(set(selected))  # 去重
'''

    def select_keyframes(self, clusters: List[List[int]],
                         features: np.ndarray,
                         target_keep_ratio: float = 0.5,  # 目标总保留比例 (0-1)
                         min_keep_num: int = 3,  # 每个簇至少保留的数量
                         max_keep_percentile: float = 0.9,  # 簇内最大保留比例上限
                         variance_influence: float = 0.3,  # 方差对保留比例的影响权重
                         ) -> List[int]:
        """
        基于目标保留比例的关键帧选择算法
        """
        total_images = sum(len(c) for c in clusters)
        target_keep_count = max(min_keep_num * len(clusters), int(total_images * target_keep_ratio))
        print(f"目标保留比例: {target_keep_ratio:.2%}, 目标保留数量: {target_keep_count}/{total_images}")

        # 计算每个簇的方差，用于动态调整保留策略
        cluster_variances = []
        for cluster in clusters:
            if len(cluster) < 2:
                cluster_variances.append(0)  # 单元素簇方差为0
                continue
            variance = np.var(features[cluster], axis=0).mean()
            cluster_variances.append(variance)

        # 归一化方差
        if cluster_variances and max(cluster_variances) > 0:
            max_variance = max(cluster_variances)
            normalized_variances = [v / max_variance for v in cluster_variances]
        else:
            normalized_variances = [0.5] * len(clusters)  # 默认中等方差

        # 基于方差分配每个簇的初始保留比例
        initial_ratios = []
        for i, cluster in enumerate(clusters):
            # 基础比例: 方差高的簇保留更多图像，方差低的簇保留更少
            base_ratio = target_keep_ratio * (1 + variance_influence * (normalized_variances[i] - 0.5))
            # 确保比例在合理范围
            base_ratio = min(max_keep_percentile, max(min_keep_num / len(cluster), base_ratio))
            initial_ratios.append(base_ratio)

        # 计算基于初始比例的保留数量
        keep_counts = [max(min_keep_num, int(round(len(c) * r))) for c, r in zip(clusters, initial_ratios)]
        current_total = sum(keep_counts)

        # 调整保留数量以匹配目标总数
        if current_total != target_keep_count:
            delta = target_keep_count - current_total
            if delta != 0:
                # 根据簇大小和方差调整
                priorities = []
                for i, cluster in enumerate(clusters):
                    # 优先调整方差大且数量多的簇
                    priority = normalized_variances[i] * len(cluster)
                    priorities.append((priority, i))

                # 按优先级排序
                priorities.sort(reverse=(delta > 0))  # 降序(增加)或升序(减少)

                # 逐个调整直到达到目标数量
                for _, i in priorities:
                    if delta == 0:
                        break
                    cluster_size = len(clusters[i])
                    current_keep = keep_counts[i]

                    if delta > 0:
                        # 增加保留数量
                        max_possible = min(cluster_size, int(cluster_size * max_keep_percentile))
                        if current_keep < max_possible:
                            increment = min(delta, max_possible - current_keep)
                            keep_counts[i] += increment
                            delta -= increment
                    else:
                        # 减少保留数量
                        if current_keep > min_keep_num:
                            decrement = min(-delta, current_keep - min_keep_num)
                            keep_counts[i] -= decrement
                            delta += decrement

        # 执行实际选择
        selected = []
        for i, cluster in enumerate(clusters):
            keep_num = keep_counts[i]
            if len(cluster) <= keep_num:
                # 簇大小小于等于保留数，全选
                selected.extend(cluster)
                continue

            cluster_feats = features[cluster]
            # 选择最接近簇中心的图像
            center = np.mean(cluster_feats, axis=0)
            distances = np.linalg.norm(cluster_feats - center, axis=1)
            center_idx = np.argmin(distances)
            selected_in_cluster = [center_idx]

            if keep_num == 1:
                selected.append(cluster[center_idx])
                continue

            # 多样性选择: 选择与已选图像最不相似的图像
            sim_matrix = cluster_feats @ cluster_feats.T
            remaining_indices = [j for j in range(len(cluster)) if j != center_idx]

            for _ in range(keep_num - 1):
                if not remaining_indices:
                    break
                # 计算每个剩余图像与已选图像的最小相似度
                min_similarities = [
                    np.min([sim_matrix[idx, s] for s in selected_in_cluster])
                    for idx in remaining_indices
                ]
                best_idx = remaining_indices[np.argmin(min_similarities)]
                selected_in_cluster.append(best_idx)
                remaining_indices.remove(best_idx)

            selected.extend([cluster[j] for j in selected_in_cluster])

        return list(set(selected))  # 去重

    def save_clusters_to_folders(self, clusters: List[List[int]], image_paths: List[str], output_dir: str):
        """将每个簇的图像保存到独立文件夹"""
        output_dir = Path(output_dir) / "clusters"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"开始保存簇数据到 {output_dir}")  # 修改：使用print替代logger

        for cluster_idx, cluster in enumerate(clusters):
            cluster_dir = output_dir / f"cluster_{cluster_idx:03d}"
            cluster_dir.mkdir(exist_ok=True)
            for img_idx in cluster:
                src_path = image_paths[img_idx]
                filename = os.path.basename(src_path)
                shutil.copy2(src_path, cluster_dir / filename)
            print(f"簇 {cluster_idx} 保存完成，包含 {len(cluster)} 张图像")  # 修改：使用print替代logger

    def run_strict_selection(self, image_paths: List[str], alpha: float = 1.0, percentile: float = 75, retention_ratio: float = 0.5) -> List[str]:
        """严格筛选流程，添加alpha和percentile参数"""
        features = self.extract_enhanced_features(image_paths)
        graph = self.build_strict_similarity_graph(features, alpha=alpha, percentile=percentile)
        clusters = self.strict_clustering(graph)

        # 新增簇保存逻辑
        # self.save_clusters_to_folders(clusters, image_paths, self.config['output_dir'])

        selected = self.select_keyframes(clusters, features, target_keep_ratio=retention_ratio)
        selected_paths = [image_paths[i] for i in selected]  # 直接转换为路径
        # return [image_paths[i] for i in selected]

        return clusters, selected_paths  # 修改：返回clusters和selected

    '''def rename_selected_images(self, selected_paths: List[str], output_dir: str, start_index: int = 0):
        """按原始文件名中的数字序号进行顺序重命名"""
        os.makedirs(output_dir, exist_ok=True)
        sorted_paths = sorted(selected_paths,
                              key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x).split('.')[0]))))
        for new_idx, src_path in enumerate(sorted_paths):
            try:
                file_ext = os.path.splitext(src_path)[1]
                new_name = f"{new_idx + start_index:04d}{file_ext}"
                shutil.copy2(src_path, os.path.join(output_dir, new_name))
                # print(f"Renamed: {os.path.basename(src_path)} -> {new_name}")
            except Exception as e:
                print(f"Error processing {src_path}: {e}")

    def save_selected_images(self, selected_paths: List[str], output_dir: str, rename: bool = False):
        """安全保存"""
        output_dir = Path(output_dir) / "input"
        if rename:
            self.rename_selected_images(selected_paths, output_dir)
        else:
            os.makedirs(output_dir, exist_ok=True)
            for src_path in selected_paths:
                shutil.copy2(src_path, os.path.join(output_dir, os.path.basename(src_path)))
'''

    def rename_selected_images(self, selected_paths: List[str], output_dir: str, start_index: int = 0):
        """按原始文件名中的数字序号进行顺序重命名"""
        os.makedirs(output_dir, exist_ok=True)
        sorted_paths = sorted(selected_paths,
                              key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x).split('.')[0]))))
        for new_idx, src_path in enumerate(sorted_paths):
            try:
                file_ext = os.path.splitext(src_path)[1]
                new_name = f"{new_idx + start_index:04d}{file_ext}"
                shutil.copy2(src_path, os.path.join(output_dir, new_name))
                # print(f"Renamed: {os.path.basename(src_path)} -> {new_name}")
            except Exception as e:
                print(f"Error processing {src_path}: {e}")

    def save_selected_images(self, selected_paths: List[str], output_dir: str, rename: bool = True):
        """
        安全保存图片到两个文件夹：
        1. original/input - 保留原始文件名
        2. renamed/input - 按序号重命名为四位数格式
        """
        # 创建两个子文件夹
        original_dir = Path(output_dir) / "original" / "input"
        renamed_dir = Path(output_dir) / "renamed" / "input"

        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(renamed_dir, exist_ok=True)

        if rename:
            # 重命名并保存到renamed文件夹
            self.rename_selected_images(selected_paths, renamed_dir)
            # 同时复制原始文件到original文件夹
            for src_path in selected_paths:
                try:
                    shutil.copy2(src_path, os.path.join(original_dir, os.path.basename(src_path)))
                except Exception as e:
                    print(f"Error copying original file {src_path}: {e}")
        else:
            # 只保存原始文件到original文件夹
            for src_path in selected_paths:
                try:
                    shutil.copy2(src_path, os.path.join(original_dir, os.path.basename(src_path)))
                except Exception as e:
                    print(f"Error copying original file {src_path}: {e}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Endoscopy Image Selection with Alpha and Percentile')
    parser.add_argument('-s', '--source_path', required=True, help='Input directory containing images')
    parser.add_argument('-m', '--output_path', required=True, help='Output directory for selected images')
    parser.add_argument('-a', '--alpha', type=float, default=1.0,
                        help='Alpha value for similarity threshold (0-1, default=1.0)')
    parser.add_argument('-p', '--percentile', type=int, default=75,
                        help='Percentile value for similarity threshold (0-100, default=75)')
    parser.add_argument('-r', '--retention_ratio', type=float, default=0.7,
                        help='Target retention ratio (0.0-1.0, default=0.7)')
    args = parser.parse_args()

    # 验证参数范围
    if not (0.0 <= args.alpha <= 2.0):
        raise ValueError("Alpha must be between 0 and 2")
    if not (0 <= args.percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100")
    if not (0.0 <= args.retention_ratio <= 1.0):
        raise ValueError("Retention ratio must be between 0.0 and 1.0")

    image_dir = args.source_path
    output_dir = args.output_path
    alpha = args.alpha
    percentile = args.percentile
    retention_ratio = args.retention_ratio  # 新增比例参数
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg'))
    ])

    if not image_paths:
        print("错误：未找到PNG或JPG图片！")
        exit()

    selector = GSImageSelector(device='cuda')
    clusters, selected_paths = selector.run_strict_selection(image_paths, alpha=alpha, percentile=percentile, retention_ratio=retention_ratio)
    selector.save_selected_images(selected_paths, output_dir, rename=True)
    # 新增：保存簇到文件夹
    selector.save_clusters_to_folders(clusters, image_paths, output_dir)
    print(f"从 {len(image_paths)} 张输入中筛选出 {len(selected_paths)} 张")
    # print(f"筛选率：{len(selected_paths) / len(image_paths) * 100:.1f}%")
    print(f"收缩系数为: {alpha}, 相似性百分位数：{percentile}")
    print(f"筛选率：{retention_ratio * 100:.1f}%")
