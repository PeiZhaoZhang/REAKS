import os
import shutil
import re

def duplicate_images(folder_path):
    # 1. 获取所有jpg文件
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    # 提取编号并排序
    # 假设文件名格式为 000001.jpg
    pattern = re.compile(r'(\d+)')
    
    image_numbers = []
    for f in files:
        match = pattern.search(f)
        if match:
            image_numbers.append(int(match.group(1)))
    
    if not image_numbers:
        print("未在该目录下找到有效的图片文件。")
        return

    # 获取当前最大编号和原始文件列表（按编号排序）
    image_numbers.sort()
    max_num = max(image_numbers)
    # 按照编号顺序排列原始文件名，确保复制后的顺序对应
    sorted_files = [f for _, f in sorted(zip(image_numbers, files))]

    print(f"检测到 {len(sorted_files)} 张图片，当前最大编号为: {max_num}")
    print("开始复制并重新编号...")

    # 2. 复制并重命名
    count = 0
    for i, filename in enumerate(sorted_files):
        new_num = max_num + i + 1
        # 保持 6 位数字格式，如 000252.jpg
        new_filename = f"{new_num:06d}.jpg"
        
        src_path = os.path.join(folder_path, filename)
        dst_path = os.path.join(folder_path, new_filename)
        
        shutil.copy2(src_path, dst_path)
        count += 1

    print(f"处理完成！成功新增 {count} 张图片，最新编号至: {max_num + count:06d}.jpg")

if __name__ == "__main__":
    target_dir = "/root/project/gaussian-splatting/data/truck"
    duplicate_images(target_dir)