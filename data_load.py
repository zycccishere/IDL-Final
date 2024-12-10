import os
import cv2
import numpy as np

def load_yolo_data(base_dir):
    dataset = {'train': [], 'valid': [], 'test': []}

    # 遍历主文件夹中的子文件夹：train, validation, test
    for split in ['train', 'valid', 'test']:
        image_dir = os.path.join(base_dir, split, 'images')
        label_dir = os.path.join(base_dir, split, 'labels')

        # 检查图像目录和标签目录是否存在
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"Warning: {split} data directories are missing. Skipping.")
            continue

        # 获取所有图像路径
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # 获取所有标签路径
        label_paths = [
            os.path.join(label_dir, f.replace('.jpg', '.txt').replace('.png', '.txt'))
            for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')
        ]

        split_data = []

        for img_path, lbl_path in zip(image_paths, label_paths):
            # 检查图像和标签文件是否存在
            if not os.path.exists(img_path) or not os.path.exists(lbl_path):
                print(f"Warning: Missing image or label file for {img_path}. Skipping.")
                continue

            # 加载图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load image {img_path}. Skipping.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
            
            # 加载标签
            with open(lbl_path, 'r') as f:
                labels = f.readlines()

            label_data = []
            for label in labels:
                parts = label.strip().split()
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                label_data.append([class_id, center_x, center_y, width, height])
            
            split_data.append({'image': img, 'labels': label_data})

        dataset[split] = split_data

    return dataset

# 设置数据的根目录
base_dir = 'D:/chensiqi/24fall/intro_to_DL/dataset'

# 加载数据
dataset = load_yolo_data(base_dir)

# 输出示例
print(f"Loaded {len(dataset['train'])} training images.")
if len(dataset['train']) > 0:
    print(f"Example data from the first training image: {dataset['train'][0]}")
    # 输出image的大小
    print(f"Image shape: {dataset['train'][0]['image'].shape}")
    # 输出label的大小
    print(f"Number of labels: {len(dataset['train'][0]['labels'])}")
    # label中有

'''
读入train，valid，test数据集，以及样本的标签
图片大小是 (416, 416, 3)，RGB表示
标签代表的是识别出的区域，每个标签是一个包含5个数的列表，第一个应该都是0，代表识别的物体分类，后面四个数分别是中心点相对坐标和相对宽高
比如[0,0.5,0.4,0.3,0.2]代表在（0.5*宽，0.4*高）这个位置，识别出宽为0.3*宽，高为0.2*高的物体
'''