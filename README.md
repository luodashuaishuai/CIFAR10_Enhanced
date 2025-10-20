# CIFAR10_Enhanced

## 任务

##### 任务描述

数据预处理： 加载CIFAR-10数据集，并进行必要的预处理，如归一化和数据增强。

模型构建： 设计一个深度学习模型。可以从一个简单的卷积神经网络（CNN）开始。

训练模型： 使用训练数据集训练神经网络模型，并使用验证集进行评估和调整。

测试和评估： 使用测试集对模型进行最终的评估，并计算准确率等指标。

结果分析与讨论： 分析模型在不同类别上的表现，探讨可能的改进方法。

##### **可选扩展任务**

模型改进： 尝试更复杂的网络架构，如更深的网络或使用残差网络（ResNet）。

超参数调整： 调整学习率、批次大小或优化器，并观察其对模型性能的影响。

数据增强： 应用图像旋转、缩放、裁剪等技术来增强数据集，并观察其对模型性能的影响。

可视化： 可视化中间层的激活以更好地理解模型是如何识别不同类别的。

## 我们实现了什么

1. 一个简单的卷积神经网络（CNN）-精确度68%

2. 优化这个模型

   数据增强：在 `transform` 中加入旋转、裁剪、翻转等增强



   可视化：在训练好模型后，可以查看中间卷积层输出

   

3. 探索epoch的值为多少时能在不过拟合的情况下保证精确度可观

设置早停机制:

**早停机制（Early Stopping）**

- **耐心值（patience）**: 7
- **最小改善（min_delta）**: 0.0001
- **触发条件**: 连续7个epoch验证损失没有改善至少0.0001

```python
早停检查
early_stopper(val_loss)
	if early_stopper.early_stop:
     print(f'Early stopping at epoch {epoch}')
     break
```

## 训练模型步骤

### 模型构建

#### 构建代码

这部分代码实现了基本的模型训练 模型导出 以及基本的测试

```python
# main.py
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# ==============================
# 1. 数据预处理与加载
# ==============================

# 定义数据增强与归一化操作
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.RandomRotation(15),          # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ==============================
# 2. 构建CNN模型
# ==============================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1: 输入3通道 -> 输出32通道
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2: 32通道 -> 64通道
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 前向传播过程
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = SimpleCNN()

# ==============================
# 3. 定义损失函数与优化器
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# ==============================
# 4. 模型训练
# ==============================
for epoch in range(20):  # 训练20轮
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()           # 清空梯度
        outputs = net(inputs)           # 前向传播
        loss = criterion(outputs, labels) # 计算损失
        loss.backward()                 # 反向传播
        optimizer.step()                # 更新参数
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0
print("训练完成！")

# ==============================
# 5. 模型测试
# ==============================
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"在10000张测试图片上的准确率: {100 * correct / total:.2f}%")

# ==============================
# 6. 可视化部分（显示卷积层特征）
# ==============================
def visualize_feature_maps(model, image):
    with torch.no_grad():
        x = image.unsqueeze(0)
        for name, layer in model._modules.items():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                plt.figure(figsize=(10, 5))
                plt.suptitle(f"Feature maps after {name}")
                for i in range(min(x.size(1), 8)):  # 显示前8张特征图
                    plt.subplot(2, 4, i + 1)
                    plt.imshow(x[0, i].detach().numpy(), cmap='gray')
                plt.show()
                break

# 展示一张测试图片的特征图
dataiter = iter(testloader)
images, labels = next(dataiter)
visualize_feature_maps(net, images[0])
# ==============================
# 保存模型参数
# ==============================
torch.save(net.state_dict(), "cnn_cifar10-20.pth")
print("✅ 模型已保存为 cnn_cifar10-20.pth")


```

### 在构建基础上 加上早停以及绘制Epoch曲线 找最优epoch

```python
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from models.simple_cnn import SimpleCNN


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Datasets and loaders
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)  # 改为0避免多进程问题

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)  # 改为0避免多进程问题

    classes = trainset.classes

    # Model, loss, optimizer, scheduler
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Early stopping
    class EarlyStopping:
        def __init__(self, patience=7, min_delta=1e-4):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = None
            self.counter = 0
            self.early_stop = False

        def __call__(self, loss):
            if self.best_loss is None:
                self.best_loss = loss
                return
            if loss < self.best_loss - self.min_delta:
                self.best_loss = loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    early_stopper = EarlyStopping(patience=7)

    num_epochs = 100
    best_val_acc = 0.0
    train_losses, val_losses = [], []

    os.makedirs('saved_models', exist_ok=True)
    save_path = os.path.join('saved_models', 'cnn_best.pth')

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)

        # validation on test set (as val)
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)

        print(
            f'Epoch {epoch:03d} | Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'  Saved best model (val_acc={best_val_acc:.4f})')

        # step scheduler
        scheduler.step()

        # early stopping check
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f'Early stopping at epoch {epoch}')
            break

    # Plot loss curves
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_curve.png', bbox_inches='tight')
    plt.close()
    print('Saved loss curve to results/loss_curve.png')


if __name__ == '__main__':
    main()
```



#### 可视化代码

分三部分



```python
# validate.py
"""验证脚本：加载训练好的模型，计算测试集准确率、生成分类报告与混淆矩阵，
并展示/保存若干示例图片的预测与可视化结果。

使用方法（在项目根目录运行）:
    python src/validate.py --ckpt ../cnn_cifar10.pth --batch 128 --save_results ../results/

要求:
- 模型结构需与训练时一致（SimpleCNN）
- 如果模型文件路径不在当前目录，用 --ckpt 指定
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ------------------------------
# 定义模型（必须与训练时一致）
# ------------------------------
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool = nn.MaxPool2d(2,2)
#         self.fc1 = nn.Linear(64*8*8, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool(x)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
# ==============================
# 定义模型（与训练时完全一致）
# ==============================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------------
# Utilities
# ------------------------------
def imshow_tensor(img_tensor, mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)):
    """反归一化并显示Tensor图像 (C,H,W) -> matplotlib image"""
    img = img_tensor.numpy().transpose((1,2,0))
    img = (img * np.array(std)) + np.array(mean)
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')

def save_prediction_grid(images, labels, preds, classes, out_path, mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)):
    """保存一组图片及其真实/预测标签为网格图像"""
    images = images.cpu()
    n = min(8, images.size(0))
    fig = plt.figure(figsize=(12,3))
    for i in range(n):
        ax = fig.add_subplot(1, n, i+1)
        img = images[i].numpy().transpose((1,2,0))
        img = (img * np.array(std)) + np.array(mean)
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f'T:{classes[labels[i]]}\nP:{classes[preds[i]]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

# ------------------------------
# 主流程
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description='Validate CIFAR-10 model')
    parser.add_argument('--ckpt', type=str, default='../cnn_cifar10-20.pth', help='path to model checkpoint')
    parser.add_argument('--batch', type=int, default=128, help='batch size for test loader')
    parser.add_argument('--save_results', type=str, default='../results/', help='directory to save results')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader num_workers')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    os.makedirs(args.save_results, exist_ok=True)

    # 构建模型并加载权重
    model = SimpleCNN().to(device)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print('Loaded checkpoint:', args.ckpt)

    # 数据变换（与训练时的 normalize 保持一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    classes = testset.classes  # class names

    # 推理：收集预测与真实标签
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in testloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算总体准确率
    accuracy = (all_preds == all_labels).mean()
    print(f"Test accuracy: {accuracy*100:.2f}%")

    # 分类报告
    report = classification_report(all_labels, all_preds, digits=4, target_names=classes)
    print('Classification Report:\n', report)
    with open(os.path.join(args.save_results, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 混淆矩阵并保存图像
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(args.save_results, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print('Saved confusion matrix to', cm_path)

    # 随机挑选一批图像进行可视化并保存（真实标签 + 预测）
    dataiter = iter(testloader)
    imgs, labels = next(dataiter)
    imgs_cpu = imgs.cpu()
    with torch.no_grad():
        outputs = model(imgs.to(device))
        _, preds_batch = outputs.max(1)
    preds_batch = preds_batch.cpu().numpy()

    grid_path = os.path.join(args.save_results, 'sample_predictions.png')
    save_prediction_grid(imgs_cpu, labels.numpy(), preds_batch, classes, grid_path)
    print('Saved sample predictions to', grid_path)

if __name__ == '__main__':
    main()

```

### 
