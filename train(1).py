import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda import is_available
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=416):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size

        # 获取所有图片文件名
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
        )

        # 图像预处理
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 加载图片
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert("RGB")

        # 应用转换
        img_tensor = self.transform(img)

        # 加载对应的标签文件
        label_file = self.img_files[idx].replace(".jpg", ".txt").replace(".png", ".txt")
        label_path = os.path.join(self.label_dir, label_file)

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    # YOLO格式: class_id x_center y_center width height
                    values = line.strip().split()
                    if len(values) == 5:
                        class_id = float(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                        boxes.append([class_id, x_center, y_center, width, height])

        # 转换为tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 5), dtype=torch.float32)

        return img_tensor, boxes, len(boxes)


def custom_collate_fn(batch):
    """自定义数据批处理函数"""
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    lengths = torch.tensor([item[2] for item in batch])

    # 对boxes进行padding，使每个样本的box数量相同
    max_boxes = max(len(b) for b in boxes)
    padded_boxes = []

    for box_list in boxes:
        if len(box_list) < max_boxes:
            # 填充到最大长度
            padding = torch.zeros((max_boxes - len(box_list), 5), dtype=torch.float32)
            padded_box = torch.cat([box_list, padding], dim=0)
        else:
            padded_box = box_list
        padded_boxes.append(padded_box)

    padded_boxes = torch.stack(padded_boxes)

    return images, padded_boxes, lengths


def save_yolo_dataset(dataset, output_base_dir):
    """保存数据集为YOLO格式"""
    for split in ["train", "valid", "test"]:
        # 创建目录
        img_dir = os.path.join(output_base_dir, split, "images")
        label_dir = os.path.join(output_base_dir, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        if split in dataset:
            for idx, sample in enumerate(dataset[split]):
                # 保存图片
                img_array = sample["image"]
                if isinstance(img_array, np.ndarray):
                    img = Image.fromarray(img_array.astype("uint8"))
                elif isinstance(img_array, torch.Tensor):
                    # 如果是tensor，先转换回numpy
                    img_array = img_array.numpy().transpose(1, 2, 0)
                    img_array = (img_array * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)

                img_path = os.path.join(img_dir, f"{idx:06d}.jpg")
                img.save(img_path)

                # 保存标签
                labels = sample["labels"]
                label_path = os.path.join(label_dir, f"{idx:06d}.txt")
                with open(label_path, "w") as f:
                    for label in labels:
                        # YOLO格式: class_id x_center y_center width height
                        label_str = " ".join([f"{x:.6f}" for x in label])
                        f.write(label_str + "\n")

    print(f"Dataset saved to {output_base_dir}")


def load_yolo_dataset(base_dir, img_size=416):
    """从YOLO格式加载数据集"""
    datasets = {}
    splits = ["train", "valid", "test"]

    for split in splits:
        img_dir = os.path.join(base_dir, split, "images")
        label_dir = os.path.join(base_dir, split, "labels")

        if os.path.exists(img_dir) and os.path.exists(label_dir):
            datasets[split] = CustomDataset(img_dir, label_dir, img_size)

    return datasets


# YOLO-like CNN Model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class YOLONAS_L(nn.Module):
    def __init__(self, num_classes=80, num_boxes=5):
        super(YOLONAS_L, self).__init__()
        self.num_boxes = num_boxes
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers with additional convolutions
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Neck (Feature Aggregation for Multiple Scales)
        self.neck = self._make_neck(512)

        # Head: Output layer for bounding box prediction (x, y, w, h), class scores, and objectness
        self.head = self._make_head(512)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(in_channels, out_channels))

        return nn.Sequential(*layers)

    def _make_neck(self, in_channels):
        # Neck structure for feature aggregation (PANet-like)
        return nn.Sequential(
            ConvBlock(in_channels, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 64),
        )

    def _make_head(self, in_channels):
        # Final detection head
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_boxes * 5, kernel_size=1)  # 5 = 4 for bbox + 1 for objectness
        )

    def forward(self, x):
        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Neck for feature aggregation
        x = self.neck(x)
        
        # Head for bounding box prediction
        x = self.head(x)

        return x.view(-1, self.num_boxes, 5)  # Reshape to (batch_size, num_boxes, 5)

class YOLOLikeCNN(nn.Module):
    def __init__(self, num_boxes=5):
        super(YOLOLikeCNN, self).__init__()
        self.num_boxes = num_boxes
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers with additional convolutions
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer1_extra = self._make_layer(64 * BasicBlock.expansion, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer2_extra = self._make_layer(128 * BasicBlock.expansion, 128, 1)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer3_extra = self._make_layer(256 * BasicBlock.expansion, 256, 1)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.layer4_extra = self._make_layer(512 * BasicBlock.expansion, 512, 1)
        
        self.dropout = nn.Dropout(0.5)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * BasicBlock.expansion, 512)
        self.fc2 = nn.Linear(512, num_boxes * 5)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_extra(x)
        x = self.layer2(x)
        x = self.layer2_extra(x)
        x = self.layer3(x)
        x = self.layer3_extra(x)
        x = self.layer4(x)
        x = self.layer4_extra(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x.view(-1, self.num_boxes, 5)

# 计算IoU
def calculate_box_iou(box1, box2):
    """Calculate IoU between two boxes"""
    # 转换为左上角和右下角坐标
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # 计算交集区域
    intersect_x1 = max(box1_x1, box2_x1)
    intersect_y1 = max(box1_y1, box2_y1)
    intersect_x2 = min(box1_x2, box2_x2)
    intersect_y2 = min(box1_y2, box2_y2)

    # 计算交集面积
    intersect_area = max(0, intersect_x2 - intersect_x1) * max(
        0, intersect_y2 - intersect_y1
    )

    # 计算两个框的面积
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # 计算IoU
    iou = intersect_area / (box1_area + box2_area - intersect_area + 1e-6)
    return iou


# Custom Loss Function
class YOLOLikeLoss(nn.Module):
    def __init__(self):
        super(YOLOLikeLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, targets, lengths):
        loss = 0
        batch_size = predictions.size(0)

        for i in range(batch_size):
            valid_targets = targets[i, : lengths[i]]
            pred_boxes = predictions[i]

            # 为每个目标框找到最佳匹配的预测框
            for target in valid_targets:
                best_iou = 0
                best_idx = 0

                # 找到最佳匹配的预测框
                for j, pred in enumerate(pred_boxes):
                    iou = calculate_box_iou(pred[1:], target[1:])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j

                # 计算损失
                loss += self.mse(pred_boxes[best_idx], target)

        return loss / batch_size


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda import is_available
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from tqdm import tqdm
import json
from pathlib import Path


def train_model(
    model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=25
):
    best_iou = 0.0
    history = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_iou = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels, lengths in train_pbar:
            images = images.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels, lengths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate average IoU
            batch_ious = []
            for i in range(len(lengths)):
                valid_targets = labels[i, : lengths[i]]
                pred_boxes = outputs[i]

                for target in valid_targets:
                    ious = [
                        calculate_box_iou(pred[1:].detach(), target[1:].detach())
                        for pred in pred_boxes
                    ]
                    batch_ious.append(max(ious))

            batch_iou = np.mean([iou.cpu().numpy() for iou in batch_ious])
            running_iou += batch_iou

            # Update progress bar
            train_pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "iou": f"{batch_iou:.4f}"}
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_iou = running_iou / len(train_loader)

        # Validation phase
        val_loss, val_iou = evaluate(
            model, valid_loader, criterion, device, desc="Validation"
        )

        # Update history
        history["train_loss"].append(epoch_loss)
        history["train_iou"].append(epoch_iou)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Save the best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "history": history,
                },
                "best_model.pth",
            )

    print("Training complete")
    return history


def evaluate(
    model, data_loader, criterion, device, desc="Evaluation", return_predictions=False
):
    """
    Evaluate the model on the given data loader
    Args:
        return_predictions: If True, also return predictions for each batch
    """
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    all_predictions = [] if return_predictions else None

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc)
        for images, labels, lengths in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels, lengths)
            running_loss += loss.item()

            # Calculate IoU
            batch_ious = []

            for i in range(len(lengths)):
                valid_targets = labels[i, : lengths[i]]
                pred_boxes = outputs[i]

                for target in valid_targets:
                    ious = [
                        calculate_box_iou(pred[1:], target[1:]) for pred in pred_boxes
                    ]
                    batch_ious.append(max(ious))

            batch_iou = (
                np.mean([iou.cpu().numpy() for iou in batch_ious])
                if batch_ious
                else 0.0
            )
            running_iou += batch_iou

            # Store predictions if requested
            if return_predictions:
                all_predictions.extend(
                    [
                        (output.cpu().numpy(), length.item())
                        for output, length in zip(outputs, lengths)
                    ]
                )

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{batch_iou:.4f}"})

    avg_loss = running_loss / len(data_loader)
    avg_iou = running_iou / len(data_loader)

    if return_predictions:
        return avg_loss, avg_iou, all_predictions
    return avg_loss, avg_iou

def test_model(model_path, test_loader, device, output_dir='visualization_results',boxes=5):
    """测试模型并可视化结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载最佳模型
    checkpoint = torch.load(model_path, map_location=device)
    model = YOLONAS_L(num_boxes=boxes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    criterion = YOLOLikeLoss()
    
    all_predictions = []
    running_loss = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        for batch_idx, (images, labels, lengths) in enumerate(tqdm(test_loader, desc='Testing')):
            images = images.to(device)
            labels = labels.to(device)
            
            # 获取预测结果
            outputs = model(images)
            loss = criterion(outputs, labels, lengths)
            running_loss += loss.item()
            
            # 对批次中的每张图片进行处理
            for i in range(len(images)):
                # 获取有效的真实框（去除padding）
                valid_labels = labels[i, :lengths[i]]
                
                # 获取预测框（应用阈值）
                pred_boxes = outputs[i]
                # 这里可以添加非极大值抑制(NMS)来过滤重叠的框
                
                # 计算IoU
                ious = []
                for target in valid_labels:
                    box_ious = [calculate_box_iou(pred[1:], target[1:]) for pred in pred_boxes]
                    ious.append(max(box_ious) if box_ious else 0)
                
                batch_iou = (
                np.mean([iou.cpu().numpy() for iou in ious])
                if ious
                else 0.0
                )
                running_iou += batch_iou
                
                # 保存可视化结果
                save_path = os.path.join(output_dir, f'test_img_{batch_idx}_{i}.png')
                visualize_predictions(
                    images[i],
                    pred_boxes.cpu().numpy(),
                    valid_labels.cpu().numpy(),
                    save_path
                )
                
                # 保存预测结果
                all_predictions.append({
                    'image_id': f'{batch_idx}_{i}',
                    'predictions': pred_boxes.cpu().numpy(),
                    'ground_truth': valid_labels.cpu().numpy(),
                    'iou': batch_iou
                })
    
    # 计算平均指标
    avg_loss = running_loss / len(test_loader)
    avg_iou = running_iou / len(test_loader.dataset)
    
    # 保存详细的测试结果
    results = {
        'loss': avg_loss,
        'iou': avg_iou,
        'predictions': all_predictions
    }
    
    # 保存结果到JSON文件
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'average_loss': avg_loss,
            'average_iou': avg_iou,
            'prediction_details': [
                {
                    'image_id': p['image_id'],
                    'iou': float(p['iou'])
                } for p in all_predictions
            ]
        }, f, indent=4)
    
    print(f"\nTest Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Visualization results saved to: {output_dir}")
    
    return results

def visualize_predictions(image_tensor, pred_boxes, true_boxes, save_path):
    """
    可视化预测结果和真实标注
    Args:
        image_tensor: 预处理后的图像张量 (C, H, W)
        pred_boxes: 预测的边界框 (N, 5) - (class_id, x_center, y_center, width, height)
        true_boxes: 真实的边界框 (M, 5)
        save_path: 保存可视化结果的路径
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # 将图像张量转换回numpy数组并反归一化
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    # 创建图形
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # 绘制预测框（红色）
    for box in pred_boxes:
        class_id, x_center, y_center, width, height = box
        # 转换为左上角坐标
        x = x_center - width/2
        y = y_center - height/2
        
        rect = patches.Rectangle(
            (x * 416, y * 416), width * 416, height * 416,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    # 绘制真实框（绿色）
    for box in true_boxes:
        class_id, x_center, y_center, width, height = box
        x = x_center - width/2
        y = y_center - height/2
        
        rect = patches.Rectangle(
            (x * 416, y * 416), width * 416, height * 416,
            linewidth=2, edgecolor='g', facecolor='none'
        )
        ax.add_patch(rect)

    # 添加图例
    ax.plot([], [], color='r', label='Predicted', linewidth=2)
    ax.plot([], [], color='g', label='Ground Truth', linewidth=2)
    ax.legend()

    # 保存图像
    plt.savefig(save_path)
    plt.close()

def run(e=1,boxes=5):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and data loaders
    train_dataset = CustomDataset("data/train/images", "data/train/labels")
    valid_dataset = CustomDataset("data/valid/images", "data/valid/labels")
    test_dataset = CustomDataset("data/test/images", "data/test/labels")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )

    # Initialize model, criterion, and optimizer
    model = YOLOLikeCNN(num_boxes=boxes).to(device)
    criterion = YOLOLikeLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # from super_gradients.training import models
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # MODEL_ARCH = 'yolo_nas_l'
    # #            'yolo_nas_m'
    # #            'yolo_nas_s'

    # model = models.get(MODEL_ARCH, pretrained_weights="coco").to(DEVICE)

    # Train model
    history = train_model(
        model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=e
    )

    # Test model
    test_results = test_model("best_model.pth", test_loader, device,boxes=boxes)

    # Plot training history
    try:
        import matplotlib.pyplot as plt

        # Plot loss
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot IoU
        plt.subplot(1, 2, 2)
        plt.plot(history["train_iou"], label="Train IoU")
        plt.plot(history["val_iou"], label="Validation IoU")
        plt.title("Training and Validation IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.close()

    except ImportError:
        print("Matplotlib not installed. Skipping plotting.")


def test(device='cuda',num=1):
    test_dataset = CustomDataset("data/test/images", "data/test/labels")
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )
    test_results = test_model(
        "best_model.pth", 
        test_loader, 
        device, 
        boxes=num,
        output_dir='test_visualization_results'
    )
    
    # Print detailed test results
    print("\nTest Results:")
    print(f"Final Test Loss: {test_results['loss']:.4f}")
    print(f"Final Test IoU: {test_results['iou']:.4f}")
    
    # Optionally, you can analyze the predictions
    predictions = test_results['predictions']
    print(f"\nTotal number of test predictions: {len(predictions)}")


if __name__ == "__main__":
    #run(e=100,boxes=1)
    test(num=1)