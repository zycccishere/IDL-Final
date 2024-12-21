import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
from symbol import test
#from networkx import optimize_graph_edit_distance
import torch
torch.cuda.set_device(2)
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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerCNN(nn.Module):
    def __init__(self):
        super(MultiLayerCNN, self).__init__()
        # 定义多层CNN网络结构
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # 多加一些卷积
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 5 * 5, 1024)  # 根据最后一个卷积层的输出尺寸调整
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)  # 假设我们有2个类别
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层

    def forward(self, x):
        # 第一层卷积和池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 第二层卷积和池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # 第三层卷积和池化
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # 第四层卷积和池化
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # 第五层卷积和池化
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        # 第六层卷积和池化
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        # 第七层卷积和池化
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        # 第八层卷积（不使用池化）
        x = F.relu(self.bn8(self.conv8(x)))

        # 展平特征图
        x = x.view(-1, 256 * 5 * 5)

        # 全连接层
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.dropout(x)  # 应用Dropout
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.dropout(x)  # 应用Dropout
        x = self.fc3(x)

        # 使用softmax进行分类
        return F.log_softmax(x, dim=1)


## 数据加载。每个数据样本包括图像和标签，图像为.jpg文件，rgb,大小为640*640；标签为txt文件，若其中有行以4开头，则为第0类；否则为第1类。数据分为训练集、验证集和测试集。已经分好，分别放在data/train, data/valid, data/test下。图像和标签分别在images和labels文件夹下。需要转成torch的Dataset和DataLoader。由于训练样本不平衡，需要以p的概率舍弃一些第1类样本，使得两类样本的数量相近。

def train_model(
    model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=25
):
    history = {"train_loss": [], "valid_loss": [], "valid_acc": []}
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        model.eval()
        valid_loss = 0.0
        valid_acc = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device).float()

                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                labels = torch.argmax(labels, dim=1)
                valid_acc += (preds == labels).sum().item()

        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_loader.dataset)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(
            f"epoch {epoch + 1}/{num_epochs}, "
            f"train_loss: {train_loss:.4f}, "
            f"valid_loss: {valid_loss:.4f}, "
            f"valid_acc: {valid_acc:.4f}"
        )

    return history

# evaluate_model函数用于评估模型在测试集上的性能
def evaluate_model(model, test_loader, device):
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            labels = torch.argmax(labels, dim=1)
            test_acc += (preds == labels).sum().item()

    test_acc /= len(test_loader.dataset)
    print(f"test_acc: {test_acc:.4f}")

    return test_acc

#可视化结果，将部分测试集的图像、标签和模型预测结果可视化存储为图像
def visualize_result(model, test_loader, device):
    model.eval()
    os.makedirs("results1", exist_ok=True) 
    with torch.no_grad():
        i=0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            i=i+1
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for image, label, pred in zip(images, labels, preds):
                image = image.cpu().numpy().transpose((1, 2, 0))
                label = torch.argmax(label).item()
                pred = pred.item()
                #存储图像
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                #为图像添加标签。若label为0，则标为“negative”；否则标为“positive".将标签画在图像的左上角
                label = "negative" if label == 0 else "positive"
                #预测标签
                pred = "negative" if pred == 0 else "positive"
                
                image.save(f"results1/pic{i}:{label}_predicted_as_{pred}.jpg")
            #     #只存储一部分图像
            #     if len(list(Path("results").glob("*.jpg"))) >= 10:
            #         break
            # if len(list(Path("results").glob("*.jpg"))) >= 10:
            #     break
            


class CustomDataset(Dataset):
    def __init__(self, data_dir, p=1):
        self.data = self.load_data(data_dir, p)

    def load_data(self, data_dir, p=1):
        image_dir = os.path.join(data_dir, "images")
        label_dir = os.path.join(data_dir, "labels")
        image_files = sorted(os.listdir(image_dir))
        label_files = sorted(os.listdir(label_dir))
        #获取文件数量
        len_files = len(image_files)
        data = []
        for image_file, label_file in tqdm(zip(image_files, label_files),desc="loading data from {}".format(data_dir),total=len_files,unit="files"):
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, label_file)

            image = Image.open(image_path).convert('RGB')
            image = np.array(image).transpose((2, 0, 1))  # 转换为CHW格式
            image = torch.tensor(image, dtype=torch.float32) / 255.0  # 归一化

            label = torch.tensor([1,0], dtype=torch.long) if label_file.startswith("5") else torch.tensor([0,1], dtype=torch.long)
            if(not label_file.startswith("5")):
                if np.random.rand() > p:
                    continue
            data.append((image, label))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
model = MultiLayerCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(), lr=0.001)
# 加载数据
print("loading data...")
train_data = CustomDataset("data/train",p=0.4)
print(f"train_data: {len(train_data)}")
valid_data = CustomDataset("data/valid",p=0.4)
print(f"valid_data: {len(valid_data)}")
test_data = CustomDataset("data/test",p=0.4)
print(f"test_data: {len(test_data)}")
#调用train_model函数进行训练
print("start training...")
batchsize=32
train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)    
test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False)
valid_loader = DataLoader(valid_data, batch_size=batchsize, shuffle=False)

history = train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=50)
##绘制训练过程中的loss曲线
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["valid_loss"], label="valid_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
##存储
plt.savefig("loss.jpg")
##评估模型在测试集上的性能
test_acc = evaluate_model(model, test_loader, device)
##可视化结果
visualize_result(model, test_loader, device)
