import os
import cv2
import shutil
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from PIL import Image
from tqdm import tqdm
import argparse

class BaselineSelectors:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.model.classifier = torch.nn.Identity()
        self.model.eval().to(device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((256, 256))
        ])

    def get_uniform_sampling(self, image_paths, target_count):
        print("[1/3] 正在运行 Uniform Sampling (均匀抽帧)...")
        indices = np.linspace(0, len(image_paths) - 1, target_count, dtype=int)
        return [image_paths[i] for i in indices]

    def get_blur_aware(self, image_paths, target_count):
        print("[2/3] 正在运行 Blur-aware Filtering (清晰度过滤)...")
        variances = []
        for path in tqdm(image_paths, desc="计算拉普拉斯方差"):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                variances.append(0)
                continue
            var = cv2.Laplacian(img, cv2.CV_64F).var()
            variances.append(var)
        
        sorted_indices = np.argsort(variances)[::-1]
        best_indices = sorted(sorted_indices[:target_count])
        return [image_paths[i] for i in best_indices]

    def get_deep_kmeans(self, image_paths, target_count):
        print("[3/3] 正在运行 Deep K-Means (深度特征聚类)...")
        features = []
        for path in tqdm(image_paths, desc="提取深度特征"):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            with torch.no_grad():
                feat = self.model(self.transform(img_pil).unsqueeze(0).to(self.device))
                features.append(feat.cpu().numpy().flatten())
        
        features = np.array(features)
        
        print(f"执行 K-Means 聚类 (K={target_count})...")
        kmeans = KMeans(n_clusters=target_count, random_state=42, n_init='auto')
        kmeans.fit(features)
        
        selected_indices = []
        for i in range(target_count):
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(features - center, axis=1)
            closest_idx = np.argmin(distances)
            selected_indices.append(closest_idx)
            features[closest_idx] = np.inf 
            
        selected_indices = sorted(list(set(selected_indices)))
        return [image_paths[i] for i in selected_indices]

    def save_images(self, selected_paths, output_dir):
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for path in selected_paths:
            # 直接使用传入的重命名后的文件名进行保存
            shutil.copy2(path, os.path.join(images_dir, os.path.basename(path)))

def prepare_renamed_source(source_dir, output_dir):
    """预处理：将原始图片按顺序重命名并统一存放，保护原始数据不被破坏"""
    renamed_source_dir = os.path.join(output_dir, "0_Renamed_Source", "images")
    os.makedirs(renamed_source_dir, exist_ok=True)
    
    raw_paths = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))])
    
    renamed_paths = []
    print(f"[0/0] 预处理：正在将 {len(raw_paths)} 张原图重命名并复制到 {renamed_source_dir}")
    for idx, src_path in enumerate(tqdm(raw_paths, desc="重命名原图")):
        ext = os.path.splitext(src_path)[1].lower()
        new_name = f"{idx:05d}{ext}"  # 生成 00000.jpg, 00001.jpg 等
        dst_path = os.path.join(renamed_source_dir, new_name)
        shutil.copy2(src_path, dst_path)
        renamed_paths.append(dst_path)
        
    return renamed_paths

def run_all_baselines(source_dir, output_dir, ratio=0.5):
    # 1. 首先生成重命名后的图库
    image_paths = prepare_renamed_source(source_dir, output_dir)
    
    if not image_paths:
        print("错误：未找到图片！")
        return
    
    target_count = int(len(image_paths) * ratio)
    print(f"\n✅ 数据集准备完毕！总图片数: {len(image_paths)}, 目标保留数: {target_count}")
    
    selector = BaselineSelectors(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 从重命名后的图库中挑选
    uniform_paths = selector.get_uniform_sampling(image_paths, target_count)
    selector.save_images(uniform_paths, os.path.join(output_dir, "1_Uniform"))
    
    blur_paths = selector.get_blur_aware(image_paths, target_count)
    selector.save_images(blur_paths, os.path.join(output_dir, "2_Blur_aware"))
    
    kmeans_paths = selector.get_deep_kmeans(image_paths, target_count)
    selector.save_images(kmeans_paths, os.path.join(output_dir, "3_Deep_KMeans"))
    
    print("\n🎉 所有 Baseline 数据生成完毕！")
    print(f"👉 原始（完整）的重命名图库位于: {os.path.join(output_dir, '0_Renamed_Source')}")
    print("👉 请将挑选出的三个文件夹输入到 COLMAP 和 3DGS 中进行训练对比。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Baseline Keyframes with Renaming')
    parser.add_argument('-s', '--source_path', required=True, help='输入原图文件夹路径')
    parser.add_argument('-m', '--output_path', required=True, help='输出总文件夹路径')
    parser.add_argument('-r', '--ratio', type=float, default=0.5, help='保留比例 (默认 0.5)')
    args = parser.parse_args()
    
    run_all_baselines(args.source_path, args.output_path, args.ratio)
#    python REAKS_comparison.py -s /root/project/data/360_v2/bicycle/images_4 -m /root/project/data/bicycle_baselines -r 0.5